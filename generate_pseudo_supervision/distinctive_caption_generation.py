import os
import sys
sys.path.append('./')
from my_utils.datasets_for_cutler import RefDataset_for_cutler
from torch.utils.data import DataLoader
import argparse
import warnings
import torch
import tqdm
import open_clip
import clip
import spacy
import pandas as pd
import numpy as np
import json
# import torchvision.transforms as T
import torchvision
import torchvision.transforms.functional as TF
import os
from my_utils.noun_extraction import noun_extraction
import statistics
from PIL import Image
import cv2
import collections
from torchvision.ops import masks_to_boxes

warnings.simplefilter(action='ignore', category=FutureWarning)
device = "cuda"


'''
CoCa
'''
# cc15m, cc12m, cc3m, mscoco, laion
fine_tune = 'cc3m'
if fine_tune in ['cc15m', 'cc12m', 'cc3m']:
    checkpoint = f'./third_party/open_clip/src/logs/laion_{fine_tune}/checkpoints/epoch_1.pt'
elif fine_tune == 'mscoco':
    checkpoint = "mscoco_finetuned_laion2B-s13B-b90k"
elif fine_tune == 'laion':
    checkpoint = "laion2B-s13B-b90k"
else:
    raise ValueError('fine_tune must be cc15m, cc12m, cc3m, mscoco, laion')

coca, _, coca_transform = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14", # coca_ViT-B-32, coca_ViT-L-14
  pretrained=checkpoint # laion2B-s13B-b90k, mscoco_finetuned_laion2B-s13B-b90k
)

coca = coca.to(device)


'''
CLIP
'''
clip_model_name = 'RN50'
clip_model, _ = clip.load(clip_model_name, device=device)
nlp = spacy.load('en_core_web_lg')


'''
Prepare dataset
'''
dataset = 'refcoco+'
split = 'train'
lmdb_dir = f'./datasets/lmdb/{dataset}/{split}.lmdb'
mask_dir = f'./datasets/masks/{dataset}/'
mode = 'train'
image_dir = f'./datasets/images/train2014/'

ref_dataset = RefDataset_for_cutler(lmdb_dir, mask_dir, dataset, split, mode)
dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)



'''
argment
'''
path2dir = f'./pseudo_supervision/cutler_sam'
os.makedirs(path2dir, exist_ok=True)
pseudo_name = f'distinctive_captions_{fine_tune}'
json_name = f'{pseudo_name}.json'
path2json = os.path.join(path2dir, f'{pseudo_name}.json')
path2csv = os.path.join(path2dir, f'{pseudo_name}.csv')
path2txt = os.path.join(path2dir, f'{pseudo_name}_text.txt')
path2doc = os.path.join(path2dir, f'{pseudo_name}_doc.txt')
path2mask_dir = f'./datasets/pseudo_masks/cutler_sam'

crop_margins = [1.0, 1.0, 1.1, 1.2]  # Masked [1.0] and Base Image [1.0, 1.1, 1.2]
clip_crop_margin = 1.2

mean = torch.tensor([0.48145466, 0.4578275,0.40821073]).reshape(1, 3, 1, 1).to(device)
std = torch.tensor([0.26862954, 0.26130258,0.27577711]).reshape(1, 3, 1, 1).to(device)

img_name_set = set()
uniqueness_list = []
correctness_list = []
doc_list = []
all_doc_count = 0
over_one_doc_count = 0
one_pred_mask_count = 0


t_bar = tqdm.tqdm(dataloader, ncols=100)
for i, data in enumerate(t_bar):
    img_name, img_id = data['img_name'], data['img_id']

    img_name, img_id = img_name[0], img_id.item()

    path2mask = os.path.join(path2mask_dir, str(img_id))

    exist = os.path.isdir(path2mask)
    if not exist:
        continue

    num_masks = len([f for f in os.listdir(path2mask) if os.path.isfile(os.path.join(path2mask, f))])

    if num_masks == 0:
        continue

    if img_name in img_name_set:
        continue
    img_name_set.add(img_name)

    pseudo_captions = collections.defaultdict(list)

    path2img = image_dir + img_name
    ori_img = cv2.imread(path2img)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_height, ori_width = ori_img.shape[:2]
    img_area = ori_height * ori_width

    '''
    Change original image to Tensor image
    '''
    img = torch.from_numpy(ori_img.transpose(2, 0, 1).copy()).to(device).unsqueeze(0)
    if not isinstance(img, torch.FloatTensor):
        img = img.float()
    img.div_(255.).sub_(mean).div_(std)
    img_h, img_w = img.shape[2:]


    '''
    Load Masks
    '''

    mask_list = []
    for mask_id in range(num_masks):
        path2mask = os.path.join(path2mask_dir, str(img_id), f'{mask_id}.png')
        mask = cv2.imread(path2mask, cv2.IMREAD_GRAYSCALE)
        if len(np.unique(mask)) == 1:
            continue
        mask = torch.from_numpy(mask).to(device).float() / 255.

        mask_list.append(mask)

    mask_list = torch.stack(mask_list, dim=0)
    num_masks = mask_list.shape[0]
    boxes = masks_to_boxes(mask_list)

    '''
    For Image Processing
    '''
    masked_imgs = img * mask_list[:, None, :, :] + (1. - mask_list[:, None, :, :]) * mean


    for j, crop_margin in enumerate(crop_margins):
        cropped_imgs = []

        for k, (m, b) in enumerate(zip(mask_list, boxes)):
            x_min, y_min, x_max, y_max = [_.item() for _ in b]
            w, h = crop_margin * (x_max - x_min), crop_margin * (y_max - y_min)

            if j == 0:
                masked_cropped_img = TF.crop(masked_imgs[k], int(y_min), int(x_min), int(h), int(w)).unsqueeze(0)

                max_wh = int(max(w, h))
                pad_left = (max_wh - masked_cropped_img.shape[3]) // 2
                pad_right = max_wh - masked_cropped_img.shape[3] - pad_left
                pad_top = (max_wh - masked_cropped_img.shape[2]) // 2
                pad_bottom = max_wh - masked_cropped_img.shape[2] - pad_top

                padded_image = torch.nn.functional.pad(masked_cropped_img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant')  # [1, 3, H, W]

                dummy_image = torch.ones(masked_cropped_img.shape).to(device)
                padded_dummy = torch.nn.functional.pad(dummy_image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant')

                padded_image = padded_image + (1 - padded_dummy) * mean

                resized_img = torchvision.transforms.Resize((224, 224))(padded_image)

                cropped_imgs.append(resized_img.squeeze(0))

            else:
                x_margin, y_margin = (w - (x_max - x_min)) / 2, (h - (y_max - y_min)) / 2
                x_min, y_min, x_max, y_max = max(x_min - x_margin, 0.0), max(y_min - y_margin, 0.0), min(x_max + x_margin, img_w), min(y_max + y_margin, img_h)
                w, h = x_max - x_min, y_max - y_min

                cropped_img = TF.crop(img.squeeze(0), int(y_min), int(x_min), int(h), int(w)).unsqueeze(0)

                max_wh = int(max(w, h))
                pad_left = (max_wh - cropped_img.shape[3]) // 2
                pad_right = max_wh - cropped_img.shape[3] - pad_left
                pad_top = (max_wh - cropped_img.shape[2]) // 2
                pad_bottom = max_wh - cropped_img.shape[2] - pad_top

                padded_image = torch.nn.functional.pad(cropped_img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant')  # [1, 3, H, W]

                dummy_image = torch.ones((cropped_img.shape)).to(device)
                padded_dummy = torch.nn.functional.pad(dummy_image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant')
                padded_image = padded_image + (1 - padded_dummy) * mean
                resized_img = torchvision.transforms.Resize((224, 224))(padded_image)

                cropped_imgs.append(resized_img.squeeze(0))

        if j == 0:
            masked_cropped_imgs = torch.stack(cropped_imgs, dim=0)  # [33, 3, 224, 224]
        cropped_imgs = torch.stack(cropped_imgs, dim=0)  # [33, 3, 224, 224]

        '''
        Generate Captions
        '''
        with torch.no_grad(), torch.cuda.amp.autocast():
            generation_type = 'beam_search'
            coca_generated = coca.generate(cropped_imgs, generation_type=generation_type)

        for idx in range(coca_generated.shape[0]):
            pseudo_caption = open_clip.decode(coca_generated[idx]).split("<end_of_text>")[0].replace("<start_of_text>", "")
            pseudo_captions[idx].append(pseudo_caption)

        # print(beam_search_captions)
        top_k = [5, 7, 9, 11, 13]
        top_p = [0.4, 0.5, 0.6, 0.7, 0.8]
        with torch.no_grad(), torch.cuda.amp.autocast():
            coca_generated = coca.distinctive_caption_sampling(cropped_imgs, top_k=top_k, top_p=top_p)

        for idx in range(coca_generated.shape[0]):
            pseudo_caption = open_clip.decode(coca_generated[idx]).split("<end_of_text>")[0].replace("<start_of_text>", "")
            idx = idx % num_masks
            pseudo_captions[idx].append(pseudo_caption)

    cropped_margin_imgs = cropped_imgs


    for seg_id in range(num_masks):
        pseudo_caption = pseudo_captions[seg_id]

        with open(f'{path2txt}', 'a') as f:
            f.write(f'\n\nimage id: {img_id} / seg id: {seg_id}')
            for c in pseudo_caption:
                f.write(f'\n{c}')

        '''
        DoC
        '''
        if num_masks == 1:
            one_pred_mask_count += 1
            noun = [noun_extraction(s, nlp) for s in pseudo_caption]
            p_noun = ['a photo of a ' + n if n.split()[0] not in ['a', 'an', 'the'] else 'a photo of ' + n for n in noun]
            noun_t = clip.tokenize(p_noun).to(device)

            with torch.no_grad():
                masked_clip_scores = clip_model(masked_cropped_imgs, noun_t)[0]
                masked_target_score = masked_clip_scores[seg_id]
                masked_target_score = masked_target_score.cpu().numpy()

            doc = [float(s) for s in masked_target_score]

            data_dict = {
                # 'ori_caption': sents,
                'img_name': img_name,
                'img_id': int(img_id),
                # 'seg_cat': int(cat_id),
                'seg_id': int(seg_id),
                'num_sent': len(pseudo_caption),
                'pseudo_caption': pseudo_caption,
                'doc': doc,
                'one_mask': 1,
            }

            with open(f'{path2json}', 'a') as f:
                json.dump(data_dict, f)
                f.write('\n')
        else:

            noun = [noun_extraction(s, nlp) for s in pseudo_caption]
            p_noun = ['a photo of a ' + n if n.split()[0] not in ['a', 'an', 'the'] else 'a photo of ' + n for n in noun]

            sent_t = clip.tokenize(pseudo_caption).to(device)
            noun_t = clip.tokenize(p_noun).to(device)

            with torch.no_grad():
                masked_clip_scores = clip_model(masked_cropped_imgs, noun_t)[0]
                cropped_clip_scores = clip_model(cropped_margin_imgs, sent_t)[0]

            masked_target_score = masked_clip_scores[seg_id]
            cropped_target_score = cropped_clip_scores[seg_id]
            target_score = masked_target_score * cropped_target_score


            masked_other_score = masked_clip_scores[torch.arange(masked_clip_scores.shape[0]) != int(seg_id)]
            cropped_other_score = cropped_clip_scores[torch.arange(cropped_clip_scores.shape[0]) != int(seg_id)]
            max_cropped_other_score = torch.max(cropped_other_score, dim=0)[0].unsqueeze(0)
            max_masked_other_score = torch.max(masked_other_score, dim=0)[0].unsqueeze(0)
            other_score = masked_other_score * cropped_other_score
            max_other_score = torch.max(other_score, dim=0)[0].unsqueeze(0)

            uniqueness = cropped_target_score / max_cropped_other_score  # [1, N] -> target mask and 4 captions
            correctness = masked_target_score  # [N]
            doc = target_score / max_other_score  # [1, N]

            uniqueness = uniqueness.squeeze(0)
            doc = doc.squeeze(0)

            uniqueness_list.extend(uniqueness.cpu().numpy())
            correctness_list.extend(correctness.cpu().numpy())
            doc_list.extend(doc.cpu().numpy())

            all_doc_count += int(doc.shape[0])
            over_one_doc_count += int(torch.sum(doc > 1).cpu())

            doc = doc.cpu().numpy()  # [44]
            doc = [float(d) for d in doc]

            data_dict = {
                # 'ori_caption': sents,
                'img_name': img_name,
                'img_id': int(img_id),
                # 'seg_cat': int(cat_id),
                'seg_id': int(seg_id),
                'num_sent': len(pseudo_caption),
                'pseudo_caption': pseudo_caption,
                'doc': doc,
                'one_mask': 0,
            }

            with open(f'{path2json}', 'a') as f:
                json.dump(data_dict, f)
                f.write('\n')


m_uniqueness = np.mean(uniqueness_list)
m_correctness = np.mean(correctness_list)
m_doc = np.mean(doc_list)
percent_doc_over_one = float(over_one_doc_count / all_doc_count)

with open(path2doc, 'a') as f:
    f.write(f'Pseudo name: {pseudo_name} / CLIP model: {clip_model_name}\n'
            f'New DoC / Uniqueness / Correctness\n')
    f.write(f'{m_doc} / {m_uniqueness} / {m_correctness}\n'
            f'All count / Over one count / percent\n'
            f'{all_doc_count} / {over_one_doc_count} / {percent_doc_over_one}\n'
            f'pred masks = 1 case: {one_pred_mask_count}\n\n')

df = pd.read_json(f'{path2json}', lines=True)
df = df.reset_index()
df.to_csv(f'{path2csv}', index=False)
