from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass
import gc

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    MultimodalTransformer,
)
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower
import random
try:
    from transformers import (
        BeamSearchScorer,
        LogitsProcessorList,
        TopPLogitsWarper,
        TopKLogitsWarper,
        # TotalLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        MinLengthLogitsProcessor,
        MaxLengthCriteria,
        StoppingCriteriaList,

    )

    GENERATION_TYPES = {
        "top_k": TopKLogitsWarper,
        "top_p": TopPLogitsWarper,
        "beam_search": "beam_search"
    }
    _has_transformers = True
except ImportError as e:
    GENERATION_TYPES = {
        "top_k": None,
        "top_p": None,
        "beam_search": "beam_search"
    }
    _has_transformers = False


@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8


def _build_text_decoder_tower(
        embed_dim,
        multimodal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    decoder = MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return decoder


class CoCa(nn.Module):
    def __init__(
            self,
            embed_dim,
            multimodal_cfg: MultimodalCfg,
            text_cfg: CLIPTextCfg,
            vision_cfg: CLIPVisionCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            pad_id: int = 0,
    ):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        vocab_size = (
            text_cfg.vocab_size  # for hf models
            if hasattr(text_cfg, "hf_model_name") and text_cfg.hf_model_name is not None
            else text_cfg.vocab_size
        )

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.text_decoder = _build_text_decoder_tower(
            vocab_size,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pad_id = pad_id

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def _encode_image(self, images, normalize=True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, tokens_embs

    def _encode_text(self, text, normalize=True, embed_cls=True):
        text = text[:, :-1] if embed_cls else text # make space for CLS token
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize=True):
        image_latent, _ = self._encode_image(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, normalize=True, embed_cls=True):
        text_latent, _ = self._encode_text(text, normalize=normalize, embed_cls=embed_cls)
        return text_latent

    def forward(self, image, text, embed_cls=True, image_latent=None, image_embs=None):
        text_latent, token_embs = self._encode_text(text, embed_cls=embed_cls)
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self._encode_image(image)

        # TODO: add assertion to avoid bugs?
        labels = text[:, -token_embs.shape[1]:]

        logits = self.text_decoder(image_embs, token_embs)
        return {
            "image_features": image_latent,
            "text_features": text_latent,
            "logits": logits,
            "labels": labels,
            "logit_scale": self.logit_scale.exp()
        }

    def generate(
        self,
        image,
        text=None,
        seq_len=30, #
        max_seq_len=77,
        temperature=1., #
        generation_type="beam_search",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3, # original 3
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False, # if True output.shape == (batch_size, seq_len)
        dependent=False,
        specific_idx=False
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"

        with torch.no_grad():
            sot_token_id = 49406 if sot_token_id is None else sot_token_id
            eos_token_id = 49407 if eos_token_id is None else eos_token_id
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )
            # scores = [2 * N, 49408]

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]

            stopping_criteria = StoppingCriteriaList(
                stopping_criteria
            )

            device = image.device

            if generation_type == "beam_search":
                output = self._generate_beamsearch(
                    image_inputs = image,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    sot_token_id=sot_token_id,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    min_seq_len=min_seq_len,
                    stopping_criteria=stopping_criteria,
                    logit_processor=logit_processor,
                )

                if fixed_output_length and output.shape[1] < seq_len:
                    return torch.cat(
                        (output, torch.ones(output.shape[0], seq_len-output.shape[1], device=device, dtype=output.dtype) * self.pad_id),
                        dim=1
                    )
                return output

            elif generation_type == "top_p":
                logit_warper = GENERATION_TYPES[generation_type](top_p)

            elif generation_type == "top_k":
                logit_warper = GENERATION_TYPES[generation_type](top_k)
            else:
                raise ValueError(
                    f"generation_type has to be one of "
                    f"{'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}."
                )

            image_latent, image_embs = self._encode_image(image)

            if text is None:
                text = torch.ones((image.shape[0], 1), device=device, dtype=torch.long) * sot_token_id

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            cur_len = text.shape[1]
            self.eval()
            out = text

            if dependent == False:
                while True:
                    x = out[:, -max_seq_len:]
                    cur_len = x.shape[1]
                    logits = self(image, x, image_latent=image_latent, image_embs=image_embs, embed_cls=False)["logits"][:, -1]
                    mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
                    sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                    if mask.all():
                        if not fixed_output_length:
                            break
                    else:
                        logits = logits[~mask, :]
                        filtered_logits = logit_processor(x[~mask, :], logits)
                        filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                        probs = F.softmax(filtered_logits / temperature, dim=-1)

                        if (cur_len + 1 == seq_len):
                            sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                        else:
                            sample[~mask, :] = torch.multinomial(probs, 1)

                    out = torch.cat((out, sample), dim=-1)

                    cur_len += 1

                    if stopping_criteria(out, None):
                        break

                if num_dims == 1:
                    out = out.squeeze(0)

                self.train(was_training)
                return out


            elif dependent == True:
                '''
                Dependent
                '''

                # To-do: specify index of mask for generation
                while True:
                    x = out[:, -max_seq_len:]
                    cur_len = x.shape[1]

                    # SOT token
                    if cur_len == 1:
                        output = self(image, x, image_latent=image_latent, image_embs=image_embs, embed_cls=False) # my modification
                        logits = output["logits"][:, -1] # my modification

                        mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
                        sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                        if mask.all():
                            if not fixed_output_length:
                                break
                        else:
                            logits = logits[~mask, :]
                            filtered_logits = logit_processor(x[~mask, :], logits)

                            filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                            probs = F.softmax(filtered_logits / temperature, dim=-1)

                            if (cur_len + 1 == seq_len):
                                sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                            else:
                                sample[~mask, :] = torch.multinomial(probs, 1)

                        out = torch.cat((out, sample), dim=-1)

                        cur_len += 1

                        if stopping_criteria(out, None):
                            break

                    else:
                        mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
                        false_index = [false_idx.item() for false_idx in torch.where(~mask)[0]]

                        sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                        if mask.all():
                            if not fixed_output_length:
                                break

                        '''
                        Dependent caption generation
                        '''
                        for j, idx in enumerate(false_index):
                            if specific_idx:


                                if specific_idx not in false_index:

                                    out = torch.cat((out, sample), dim=-1)

                                    cur_len += 1

                                    if stopping_criteria(out, None):
                                        break

                                    if num_dims == 1:
                                        out = out.squeeze(0)

                                    self.train(was_training)

                                    return out[specific_idx]

                                if idx != specific_idx:
                                    continue


                            x_clone = x[idx].unsqueeze(0).repeat(torch.sum(~mask), 1)
                            output = self(image[~mask, :, :, :], x_clone, image_latent=image_latent[~mask,:], image_embs=image_embs[~mask,:,:], embed_cls=False) # my modification
                            logits = output["logits"][:, -1]

                            image_features = output["image_features"]
                            similarity_matrix = torch.mm(image_features, image_features.transpose(0, 1))[j]

                            filtered_logits = logit_processor(x_clone, logits)

                            filtered_logits = logit_warper(x_clone, filtered_logits, dependent=dependent,target_idx=j)
                            probs = F.softmax(filtered_logits / temperature, dim=-1)
                            original_prob = probs[j].unsqueeze(0)

                            if len(false_index) == 1:
                                probs = original_prob
                            else:
                                eot_prob = original_prob[:,eos_token_id]
                                target_prob = probs[j].unsqueeze(0)
                                other_prob = probs[torch.arange(probs.shape[0]) != j]
                                similarity_matrix = similarity_matrix[torch.arange(probs.shape[0]) != j]

                                other_prob = other_prob * similarity_matrix.unsqueeze(1)

                                target_prob = target_prob - torch.mean(other_prob, dim=0, keepdim=True)
                                target_prob = target_prob.clamp(min=0)


                                probs = target_prob
                                probs[:,eos_token_id] = eot_prob



                            if (cur_len + 1 == seq_len):
                                sample[idx, :] = torch.ones((1, 1), device=device, dtype=torch.long) * eos_token_id
                            else:
                                if torch.sum(probs) <= 0:
                                    probs = original_prob
                                    print('origianl')
                                sample[idx, :] = torch.multinomial(probs, 1)

                        out = torch.cat((out, sample), dim=-1)

                        cur_len += 1

                        if stopping_criteria(out, None):
                            break

                if num_dims == 1:
                    out = out.squeeze(0)

                self.train(was_training)
                return out

    def _generate_beamsearch(
            self,
            image_inputs,
            pad_token_id=None,
            eos_token_id=None,
            sot_token_id=None,
            num_beams=6,
            num_beam_groups=3,
            min_seq_len=5,
            stopping_criteria=None,
            logit_processor=None,
            logit_warper=None,
    ):
        device = image_inputs.device
        batch_size = image_inputs.shape[0]
        image_inputs = torch.repeat_interleave(image_inputs, num_beams, dim=0) # [image_inputs(Batch) * num_beams, 3, 224, 224]
        image_latent, image_embs = self._encode_image(image_inputs)
        # print(image_latent.shape)
        input_ids = torch.ones((batch_size * num_beams, 1), device=device, dtype=torch.long)
        input_ids = input_ids * sot_token_id

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            num_beam_groups=num_beam_groups,
        )
        # instantiate logits processors
        logits_processor = (
            LogitsProcessorList([MinLengthLogitsProcessor(min_seq_len, eos_token_id=eos_token_id)])
            if logit_processor is None
            else logit_processor
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_beam_size, cur_len = input_ids.shape
        beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:

            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = prepare_inputs_for_generation(input_ids=input_ids, image_inputs=image_inputs)
            outputs = self(
                model_inputs['images'],
                model_inputs['text'],
                embed_cls=False,
                image_latent=image_latent,
                image_embs=image_embs
            )

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of currentg group only
                next_token_logits = outputs['logits'][batch_group_indices, -1, :]
                vocab_size = next_token_logits.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids, next_token_logits, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch.div(beam_idx, group_size, rounding_mode="floor") + group_start_idx + (beam_idx % group_size)
                )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            # increase cur_len
            cur_len = cur_len + 1
            if beam_scorer.is_done or stopping_criteria(input_ids, None):
                break

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )
        return sequence_outputs['sequences']
        # # my modification
        # return sequence_outputs


    def multiple_generate(
        self,
        image,
        text=None,
        seq_len=30, #
        max_seq_len=77,
        temperature=1., #
        generation_type="beam_search",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3, # original 3
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False, # if True output.shape == (batch_size, seq_len)
        dependent=False,
        specific_idx=False
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"

        with torch.no_grad():
            sot_token_id = 49406 if sot_token_id is None else sot_token_id
            eos_token_id = 49407 if eos_token_id is None else eos_token_id
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )
            # scores = [2 * N, 49408]

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]

            stopping_criteria = StoppingCriteriaList(
                stopping_criteria
            )

            device = image.device

            num_images = image.shape[0]

            total_logit_warper = TotalLogitsWarper(top_k,top_p,num_images)

            top_k_p_len = len(top_k) + len(top_p)

            image_latent, image_embs = self._encode_image(image) # [5, 768], [5, 255, 768]

            # image_latent = torch.cat([latent.repeat(10,1) for latent in image_latent])
            # image_embs = torch.cat([embs.repeat(10,1,1) for embs in image_embs])

            image_latent = image_latent.repeat(top_k_p_len, 1) # [50, 768]
            image_embs = image_embs.repeat(top_k_p_len, 1, 1) # [50, 255, 768] , 1,2,3,4,5 1,2,3,4,5


            if text is None:
                text = torch.ones((image.shape[0] * top_k_p_len, 1), device=device, dtype=torch.long) * sot_token_id

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            cur_len = text.shape[1]
            self.eval()
            out = text

            while True:
                x = out[:, -max_seq_len:]
                cur_len = x.shape[1]

                text_latent, token_embs = self._encode_text(x, embed_cls=False)

                logits = self.text_decoder(image_embs, token_embs)[:,-1] # [50, 49408]

                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)

                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                if mask.all():
                    if not fixed_output_length:
                        break
                else:
                    logits = logits[~mask, :]
                    filtered_logits = logit_processor(x[~mask, :], logits)
                    filtered_logits = total_logit_warper(x[~mask, :], filtered_logits, mask)

                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                    if (cur_len + 1 == seq_len):
                        sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                    else:
                        sample[~mask, :] = torch.multinomial(probs, 1)

                        # print(sample[0] == sample[5])
                out = torch.cat((out, sample), dim=-1)

                cur_len += 1

                if stopping_criteria(out, None):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out

    def distinctive_caption_sampling(
        self,
        image,
        text=None,
        seq_len=30, #
        max_seq_len=77,
        temperature=1., #
        generation_type="beam_search",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3, # original 3
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False, # if True output.shape == (batch_size, seq_len)
        dependent=False,
        specific_idx=False,
        filter_value=-float("Inf"),
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"

        with torch.no_grad():
            sot_token_id = 49406 if sot_token_id is None else sot_token_id
            eos_token_id = 49407 if eos_token_id is None else eos_token_id
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )
            # scores = [2 * N, 49408]

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]

            stopping_criteria = StoppingCriteriaList(
                stopping_criteria
            )

            device = image.device

            num_images = image.shape[0]

            total_logit_warper = DependentTotalLogitsWarper(top_k,top_p,num_images)
            top_p_len = len(top_p)
            top_k_len = len(top_k)

            top_k_p_len = len(top_k) + len(top_p)

            image_latent, ori_image_embs = self._encode_image(image) # [5, 768], [5, 255, 768]

            # image_latent = torch.cat([latent.repeat(10,1) for latent in image_latent])
            # image_embs = torch.cat([embs.repeat(10,1,1) for embs in image_embs])

            similarity_matrix = torch.mm(image_latent, image_latent.T) # [5, 5]

            image_embs = ori_image_embs.repeat(top_k_p_len, 1, 1) # [50, 255, 768] , 1,2,3,4,5 1,2,3,4,5


            if text is None:
                text = torch.ones((image.shape[0] * top_k_p_len, 1), device=device, dtype=torch.long) * sot_token_id

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            cur_len = text.shape[1]
            self.eval()
            out = text

            while True:
                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id) # [50]
                if specific_idx:
                    for i in range(mask.shape[0]):
                        if i % num_images != specific_idx:
                            mask[i] = True


                x = out[~mask, -max_seq_len:]
                cur_len = x.shape[1]
                text_latent, token_embs = self._encode_text(x, embed_cls=False)
                logits = self.text_decoder(image_embs[~mask], token_embs)[:, -1]  # [50, 49408]
                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                if mask.all():
                    if not fixed_output_length:
                        break
                else:
                    # logits = logits[~mask, :]
                    filtered_logits = logit_processor(x, logits)
                    filtered_logits, indices_to_remove_list = total_logit_warper(x, filtered_logits, mask)


                probs = F.softmax(filtered_logits / temperature, dim=-1)
                probs_eot = probs[:, eos_token_id]
                original_probs = probs

                if 30 >= num_images >= 2:
                    non_zero_idx = torch.nonzero(~mask, as_tuple=True)[0] % num_images
                    dep_image_embs = [ori_image_embs[torch.arange(num_images) != int(idx)] for idx in non_zero_idx] # [50, 4, 255, 768]
                    dep_image_embs = torch.cat(dep_image_embs, dim=0) # [200, 255, 768]

                    dep_sim = [similarity_matrix[torch.arange(num_images) == int(idx), torch.arange(num_images) != int(idx)] for idx in non_zero_idx]  # [50, 4]
                    dep_sim = torch.stack(dep_sim, dim=0)  # [50, 4]

                    # dep_sim = F.softmax(dep_sim, dim=-1) # [50, 4], weights

                    # dep_token_embs = token_embs.repeat(4, 1, 1) # [200, 2, 768]

                    dep_token_embs = torch.cat([embs.repeat(num_images-1, 1, 1) for embs in token_embs]) # [200, context length, 768]

                    dep_logits = self.text_decoder(dep_image_embs, dep_token_embs)[:, -1] # [200, 49408]
                    dep_logits = dep_logits.view(-1, num_images-1, 49408) # [50, 4, 49408]
                    # dep_logits = torch.sum(dep_sim.unsqueeze(-1) * dep_logits, dim=1) # [50, 49408]
                    dep_logits = dep_logits.mul_(dep_sim.unsqueeze(-1)) # [50, 4, 49408]
                    dep_logits = torch.mean(dep_logits, dim=1) # [50, 49408]


                    # indices_to_remove_list = [50, 49408]
                    # dep_logits = dep_logits.masked_fill(indices_to_remove_list, filter_value) # [50, 49408]
                    dep_logits = filtered_logits - dep_logits # [50, 49408]


                    #### Prob version
                    # dep_prob = F.softmax(dep_logits / temperature, dim=-1) # [50, 4, 49408]


                    # dep_prob = dep_prob * dep_sim.unsqueeze(-1) # [50, 4, 49408]
                    # dep_prob = torch.mean(dep_prob, dim=1) # [50, 49408]

                    dep_logits = dep_logits.masked_fill(indices_to_remove_list, filter_value) # [50, 49408]
                    probs = F.softmax(dep_logits / temperature, dim=-1) # [50, 49408]

                    # probs = probs.clamp(min=0)
                    probs[:, eos_token_id] = probs_eot


                    del dep_image_embs
                    del dep_sim
                    del dep_token_embs



                if (cur_len + 1 == seq_len):
                    sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                else:
                    if torch.sum(probs) <= 0:
                        print('sum of probs is lower than 0')
                        probs = original_probs
                    sample[~mask, :] = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                cur_len += 1

                if stopping_criteria(out, None):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out

    def distinctive_short_sampling(
        self,
        image,
        text=None,
        seq_len=5, #
        max_seq_len=77,
        temperature=1., #
        generation_type="beam_search",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3, # original 3
        min_seq_len=1,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False, # if True output.shape == (batch_size, seq_len)
        dependent=False,
        specific_idx=False,
        filter_value=-float("Inf"),
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"

        with torch.no_grad():
            sot_token_id = 49406 if sot_token_id is None else sot_token_id
            eos_token_id = 49407 if eos_token_id is None else eos_token_id
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )
            # scores = [2 * N, 49408]

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]

            stopping_criteria = StoppingCriteriaList(
                stopping_criteria
            )

            device = image.device

            num_images = image.shape[0]

            total_logit_warper = DependentTotalLogitsWarper(top_k,top_p,num_images)
            top_p_len = len(top_p)
            top_k_len = len(top_k)

            top_k_p_len = len(top_k) + len(top_p)

            image_latent, ori_image_embs = self._encode_image(image) # [5, 768], [5, 255, 768]

            # image_latent = torch.cat([latent.repeat(10,1) for latent in image_latent])
            # image_embs = torch.cat([embs.repeat(10,1,1) for embs in image_embs])

            similarity_matrix = torch.mm(image_latent, image_latent.T) # [5, 5]

            image_embs = ori_image_embs.repeat(top_k_p_len, 1, 1) # [50, 255, 768] , 1,2,3,4,5 1,2,3,4,5


            if text is None:
                text = torch.ones((image.shape[0] * top_k_p_len, 1), device=device, dtype=torch.long) * sot_token_id

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            cur_len = text.shape[1]
            self.eval()
            out = text

            while True:
                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id) # [50]
                if specific_idx:
                    for i in range(mask.shape[0]):
                        if i % num_images != specific_idx:
                            mask[i] = True


                x = out[~mask, -max_seq_len:]
                cur_len = x.shape[1]
                text_latent, token_embs = self._encode_text(x, embed_cls=False)
                logits = self.text_decoder(image_embs[~mask], token_embs)[:, -1]  # [50, 49408]
                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                if mask.all():
                    if not fixed_output_length:
                        break
                else:
                    # logits = logits[~mask, :]
                    filtered_logits = logit_processor(x, logits)
                    filtered_logits, indices_to_remove_list = total_logit_warper(x, filtered_logits, mask)


                probs = F.softmax(filtered_logits / temperature, dim=-1)
                probs_eot = probs[:, eos_token_id]
                original_probs = probs

                if 30 >= num_images >= 2:
                    non_zero_idx = torch.nonzero(~mask, as_tuple=True)[0] % num_images
                    dep_image_embs = [ori_image_embs[torch.arange(num_images) != int(idx)] for idx in non_zero_idx] # [50, 4, 255, 768]
                    dep_image_embs = torch.cat(dep_image_embs, dim=0) # [200, 255, 768]

                    dep_sim = [similarity_matrix[torch.arange(num_images) == int(idx), torch.arange(num_images) != int(idx)] for idx in non_zero_idx]  # [50, 4]
                    dep_sim = torch.stack(dep_sim, dim=0)  # [50, 4]

                    # dep_sim = F.softmax(dep_sim, dim=-1) # [50, 4], weights

                    # dep_token_embs = token_embs.repeat(4, 1, 1) # [200, 2, 768]

                    dep_token_embs = torch.cat([embs.repeat(num_images-1, 1, 1) for embs in token_embs]) # [200, context length, 768]

                    dep_logits = self.text_decoder(dep_image_embs, dep_token_embs)[:, -1] # [200, 49408]
                    dep_logits = dep_logits.view(-1, num_images-1, 49408) # [50, 4, 49408]
                    # dep_logits = torch.sum(dep_sim.unsqueeze(-1) * dep_logits, dim=1) # [50, 49408]
                    dep_logits = dep_logits.mul_(dep_sim.unsqueeze(-1)) # [50, 4, 49408]
                    dep_logits = torch.mean(dep_logits, dim=1) # [50, 49408]


                    # indices_to_remove_list = [50, 49408]
                    # dep_logits = dep_logits.masked_fill(indices_to_remove_list, filter_value) # [50, 49408]
                    dep_logits = filtered_logits - dep_logits # [50, 49408]


                    #### Prob version
                    # dep_prob = F.softmax(dep_logits / temperature, dim=-1) # [50, 4, 49408]


                    # dep_prob = dep_prob * dep_sim.unsqueeze(-1) # [50, 4, 49408]
                    # dep_prob = torch.mean(dep_prob, dim=1) # [50, 49408]

                    dep_logits = dep_logits.masked_fill(indices_to_remove_list, filter_value) # [50, 49408]
                    probs = F.softmax(dep_logits / temperature, dim=-1) # [50, 49408]

                    # probs = probs.clamp(min=0)
                    probs[:, eos_token_id] = probs_eot


                    del dep_image_embs
                    del dep_sim
                    del dep_token_embs



                if (cur_len + 1 == seq_len):
                    sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                else:
                    if torch.sum(probs) <= 0:
                        print('sum of probs is lower than 0')
                        probs = original_probs
                    sample[~mask, :] = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                cur_len += 1
                if stopping_criteria(out, None):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out




    def distinctive_generate(
        self,
        image,
        text=None,
        seq_len=30, #
        max_seq_len=77,
        temperature=1., #
        generation_type="beam_search",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3, # original 3
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False, # if True output.shape == (batch_size, seq_len)
        dependent=False,
        specific_idx=False,
        filter_value=-float("Inf"),
        hyperparameter=0.01,
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"

        with torch.no_grad():
            sot_token_id = 49406 if sot_token_id is None else sot_token_id
            eos_token_id = 49407 if eos_token_id is None else eos_token_id
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )
            # scores = [2 * N, 49408]

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]

            stopping_criteria = StoppingCriteriaList(
                stopping_criteria
            )

            device = image.device

            num_images = image.shape[0]

            total_logit_warper = DependentTotalLogitsWarper(top_k,top_p,num_images)
            top_p_len = len(top_p)
            top_k_len = len(top_k)

            top_k_p_len = len(top_k) + len(top_p)

            image_latent, ori_image_embs = self._encode_image(image) # [5, 768], [5, 255, 768]

            # image_latent = torch.cat([latent.repeat(10,1) for latent in image_latent])
            # image_embs = torch.cat([embs.repeat(10,1,1) for embs in image_embs])

            similarity_matrix = torch.mm(image_latent, image_latent.T) # [5, 5]

            image_embs = ori_image_embs.repeat(top_k_p_len, 1, 1) # [50, 255, 768] , 1,2,3,4,5 1,2,3,4,5


            if text is None:
                text = torch.ones((image.shape[0] * top_k_p_len, 1), device=device, dtype=torch.long) * sot_token_id

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            cur_len = text.shape[1]
            self.eval()
            out = text

            while True:
                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id) # [50]
                if specific_idx:
                    for i in range(mask.shape[0]):
                        if i % num_images != specific_idx:
                            mask[i] = True


                x = out[~mask, -max_seq_len:]
                cur_len = x.shape[1]
                text_latent, token_embs = self._encode_text(x, embed_cls=False)
                logits = self.text_decoder(image_embs[~mask], token_embs)[:, -1]  # [50, 49408]
                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                if mask.all():
                    if not fixed_output_length:
                        break
                else:
                    # logits = logits[~mask, :]
                    filtered_logits = logit_processor(x, logits)
                    filtered_logits, indices_to_remove_list = total_logit_warper(x, filtered_logits, mask)


                probs = F.softmax(filtered_logits / temperature, dim=-1)
                probs_eot = probs[:, eos_token_id]
                original_probs = probs

                if 30 >= num_images >= 2:
                    non_zero_idx = torch.nonzero(~mask, as_tuple=True)[0] % num_images
                    dep_image_embs = [ori_image_embs[torch.arange(num_images) != int(idx)] for idx in non_zero_idx] # [50, 4, 255, 768]
                    dep_image_embs = torch.cat(dep_image_embs, dim=0) # [200, 255, 768]

                    dep_sim = [similarity_matrix[torch.arange(num_images) == int(idx), torch.arange(num_images) != int(idx)] for idx in non_zero_idx]  # [50, 4]
                    dep_sim = torch.stack(dep_sim, dim=0)  # [50, 4]

                    # dep_sim = F.softmax(dep_sim, dim=-1) # [50, 4], weights

                    # dep_token_embs = token_embs.repeat(4, 1, 1) # [200, 2, 768]

                    '''
                    Weighted Sum
                    '''
                    dep_token_embs = torch.cat([embs.repeat(num_images - 1, 1, 1) for embs in token_embs])
                    dep_logits = self.text_decoder(dep_image_embs, dep_token_embs)[:, -1] # [200, 49408]
                    dep_logits = dep_logits.view(-1, num_images-1, 49408) # [50, 4, 49408]

                    dep_sim = F.softmax(dep_sim, dim=-1)  # [50, 4], weights
                    dep_logits = torch.sum(dep_sim.unsqueeze(-1) * dep_logits, dim=1)  # [50, 49408]

                    dep_logits = filtered_logits - hyperparameter * dep_logits  # [50, 49408]

                    '''
                    Average
                    '''
                    # dep_token_embs = torch.cat([embs.repeat(num_images-1, 1, 1) for embs in token_embs]) # [200, context length, 768]
                    #
                    # dep_logits = self.text_decoder(dep_image_embs, dep_token_embs)[:, -1] # [200, 49408]
                    # dep_logits = dep_logits.view(-1, num_images-1, 49408) # [50, 4, 49408]
                    # # dep_logits = torch.sum(dep_sim.unsqueeze(-1) * dep_logits, dim=1) # [50, 49408]
                    # dep_logits = dep_logits.mul_(dep_sim.unsqueeze(-1)) # [50, 4, 49408]
                    # dep_logits = torch.mean(dep_logits, dim=1) # [50, 49408]
                    #
                    #
                    # # indices_to_remove_list = [50, 49408]
                    # # dep_logits = dep_logits.masked_fill(indices_to_remove_list, filter_value) # [50, 49408]
                    # dep_logits = filtered_logits - dep_logits # [50, 49408]/


                    #### Prob version
                    # dep_prob = F.softmax(dep_logits / temperature, dim=-1) # [50, 4, 49408]


                    # dep_prob = dep_prob * dep_sim.unsqueeze(-1) # [50, 4, 49408]
                    # dep_prob = torch.mean(dep_prob, dim=1) # [50, 49408]

                    dep_logits = dep_logits.masked_fill(indices_to_remove_list, filter_value) # [50, 49408]
                    probs = F.softmax(dep_logits / temperature, dim=-1) # [50, 49408]

                    # probs = probs.clamp(min=0)
                    probs[:, eos_token_id] = probs_eot


                    del dep_image_embs
                    del dep_sim
                    del dep_token_embs



                if (cur_len + 1 == seq_len):
                    sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                else:
                    if torch.sum(probs) <= 0:
                        print('sum of probs is lower than 0')
                        probs = original_probs
                    sample[~mask, :] = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                cur_len += 1

                if stopping_criteria(out, None):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out

def prepare_inputs_for_generation(input_ids, image_inputs, past=None, **kwargs):
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    else:
        position_ids = None
    return {
        "text": input_ids,
        "images": image_inputs,
        "past_key_values": past,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }

class TotalLogitsWarper:
    def __init__(self, top_k, top_p, num_images,filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        self.top_k = top_k
        self.top_p = top_p
        self.num_images = num_images
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

        self.top_k_len = len(top_k)
        self.top_p_len = len(top_p)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, mask):
        top_k_mask = mask[:self.top_k_len * self.num_images]
        top_p_mask = mask[-self.top_p_len * self.num_images:]

        contained_scores = [int(torch.sum(~mask[i * self.num_images: (i + 1) * self.num_images])) for i in range(self.top_k_len + self.top_p_len)]



        count = 0
        for i, c in enumerate(contained_scores):

            # print('c', c)
            if c == 0:
                continue
            if i < self.top_k_len and not top_k_mask.all():
                top_k = self.top_k[i]

                indices_to_remove = scores[count:count+c, :] < torch.topk(scores[count:count+c, :], top_k)[0][..., -1, None]

                scores[count:count+c, :] = scores[count:count+c, :].masked_fill(indices_to_remove, self.filter_value)


            elif i >= self.top_k_len and not top_p_mask.all():
                sorted_logits, sorted_indices = torch.sort(scores[count:count+c, :], descending=False)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                top_p = self.top_p[i - self.top_k_len]
                sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
                sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                scores[count:count+c, :] = scores[count:count+c, :].masked_fill(indices_to_remove, self.filter_value)

            count += c

        return scores



class DependentTotalLogitsWarper:
    def __init__(self, top_k, top_p, num_images,filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        self.top_k = top_k
        self.top_p = top_p
        self.num_images = num_images
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

        self.top_k_len = len(top_k)
        self.top_p_len = len(top_p)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, mask):
        top_k_mask = mask[:self.top_k_len * self.num_images]
        top_p_mask = mask[-self.top_p_len * self.num_images:]

        contained_scores = [int(torch.sum(~mask[i * self.num_images: (i + 1) * self.num_images])) for i in range(self.top_k_len + self.top_p_len)]
        indices_to_remove_list = []


        count = 0
        for i, c in enumerate(contained_scores):

            # print('c', c)
            if c == 0:
                continue
            if i < self.top_k_len and not top_k_mask.all():
                top_k = self.top_k[i]

                indices_to_remove = scores[count:count+c, :] < torch.topk(scores[count:count+c, :], top_k)[0][..., -1, None]
                indices_to_remove_list.append(indices_to_remove)
                # scores[count:count+c, :] = scores[count:count+c, :].masked_fill(indices_to_remove, self.filter_value)


            elif i >= self.top_k_len and not top_p_mask.all():
                sorted_logits, sorted_indices = torch.sort(scores[count:count+c, :], descending=False)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                top_p = self.top_p[i - self.top_k_len]
                sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
                sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                indices_to_remove_list.append(indices_to_remove)
                # scores[count:count+c, :] = scores[count:count+c, :].masked_fill(indices_to_remove, self.filter_value)

            count += c
        indices_to_remove_list = torch.cat(indices_to_remove_list, dim=0)
        scores = scores.masked_fill(indices_to_remove_list, self.filter_value)


        return scores, indices_to_remove_list

