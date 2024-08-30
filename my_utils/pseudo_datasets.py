import sys
sys.path.append('./')
import os
from typing import List, Union
# from PIL import Image
# import io
import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from my_utils.simple_tokenizer import SimpleTokenizer as _Tokenizer
import pandas as pd
from pycocotools.coco import COCO
import random
from my_utils.noun_extraction import noun_extraction

_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)



class Pseudo_Dataset_with_refcoco_GT_masks_for_cris(Dataset):
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size,
                 word_length):
        super(Pseudo_Dataset_with_refcoco_GT_masks_for_cris, self).__init__()
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        # self.length = info[dataset][split]
        self.env = None
        self._init_db()

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir,
                             subdir=os.path.isdir(self.lmdb_dir),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)
        # img
        # image name, sents
        img_name = ref['img_name']
        seg_cat = ref['cat']
        ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8),
                               cv2.IMREAD_COLOR) # numpy
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]
        # mask
        seg_id = ref['seg_id']
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')
        # sentences
        idx = np.random.choice(2)
        # transform
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])


        if self.mode == 'train':

            sents = ref['pseudo_captions'][:2]
            # print('first',sents)
            # print('second', sents[:2])
            # mask transform
            mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            mask = cv2.warpAffine(mask,
                                  mat,
                                  self.input_size,
                                  flags=cv2.INTER_LINEAR,
                                  borderValue=0.)
            mask = mask / 255.
            # sentence -> vector
            sent = sents[idx]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img, mask = self.convert(img, mask)

            return img, word_vec, mask



        elif self.mode == 'val':
            # sentence -> vector
            sents = ref['sents']
            sent = sents[0]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img = self.convert(img)[0]
            params = {
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size)
            }
            return img, word_vec, params
        else:
            # sentence -> vector
            img = self.convert(img)[0]
            sents = ref['sents']
            params = {
                'ori_img': ori_img,
                'seg_id': seg_id,
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size),
                'sents': sents
            }
            return img, params

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def convert(self, img, mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"db_path={self.lmdb_dir}, " + \
            f"dataset={self.dataset}, " + \
            f"split={self.split}, " + \
            f"mode={self.mode}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length}"



class pseudo_captions(Dataset):
    def __init__(self, lmdb_dir, image_dir, mask_dir, input_size=480):
        self.lmdb_dir = lmdb_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = input_size

        self.env = None

        self._init_db()

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir,
                             subdir=os.path.isdir(self.lmdb_dir),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)
        original_captions = ref['original_captions']
        pseudo_captions = ref['pseudo_captions']
        pseudo_captions = [[pseudo_caption] for pseudo_caption in pseudo_captions]

        data = {'original_captions': original_captions, 'pseudo_captions': pseudo_captions}
        return data


class pseudo_dataset(Dataset):
    def __init__(self, csv_dir, image_dir, mask_dir, coco_instance_gt=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.coco_instance_gt = coco_instance_gt

        self.df = pd.read_csv(csv_dir)

        self.COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        if self.coco_instance_gt:
            path2ann = './datasets/annotations/instances_train2014.json'

            self.coco = COCO(path2ann)
            self.coco_instance_cat_dict = self.coco.cats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        ori_caption = eval(row['ori_caption'])
        img_name = eval(row['img_name'])
        img_id = row['img_id']
        seg_cat = row['seg_cat']
        seg_id = row['seg_id']
        num_sent = row['num_sent']
        pseudo_caption = eval(row['pseudo_caption'])

        seg_cat = self.COCO_CLASSES[seg_cat]

        img_dir = os.path.join(self.image_dir, img_name[0])
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        width, height = img.shape[1], img.shape[0]


        mask_dir = os.path.join(self.mask_dir, f'{seg_id}.png')
        mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.

        img, mask = self.transform(img, mask)




        if self.coco_instance_gt:
            coco_instance_target = self.coco.loadAnns(self.coco.getAnnIds(img_id))

            BoxAnns = []
            MaskAnns = []
            cat_names = []

            for t in coco_instance_target:
                BoxAnn = t['bbox']
                BoxAnns.append(BoxAnn)

                MaskAnn = self.coco.annToMask(t)
                MaskAnn = T.ToTensor()(MaskAnn)  # To_Tensor

                MaskAnns.append(MaskAnn)

                cat_id = t['category_id']
                cat_name = self.coco_instance_cat_dict[cat_id]['name']
                cat_names.append(cat_name)
            MaskAnns = torch.stack(MaskAnns, dim=1).squeeze(0).type(torch.bool) if len(MaskAnns) != 0 else []
        else:
            BoxAnns = []
            MaskAnns = []
            cat_names = []


        data = {'img':img, 'mask':mask, 'ori_caption': ori_caption, 'img_name': img_name, 'img_id': img_id, 'seg_cat': seg_cat,
                'seg_id': seg_id, 'num_sent': num_sent, 'pseudo_caption': pseudo_caption,
                'coco_BoxAnns': BoxAnns, 'coco_MaskAnns': MaskAnns, 'coco_cat_names': cat_names}

        return data

    def transform(self, img, mask=None):
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask


class pseudo_dataset_for_cris(Dataset):
    def __init__(self, csv_dir, image_dir, mask_dir, dataset, split, mode, input_size, word_length,
                 doc_mode=False, doc_filter='value', cutler=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.input_size = (input_size, input_size)
        self.word_length = word_length

        self.doc_mode = doc_mode

        self.doc_filter = doc_filter

        self.cutler = cutler

        if doc_filter not in ['value', 'top_k']:
            raise ValueError('doc_filter must be "value" or "top_k"')

        self.df = pd.read_csv(csv_dir)

        self.COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        ori_caption = eval(row['ori_caption'])
        img_name = row['img_name'] if self.cutler else eval(row['img_name'])
        img_id = row['img_id']
        seg_cat = row['seg_cat']
        seg_id = row['seg_id']
        num_sent = row['num_sent']
        pseudo_caption = eval(row['pseudo_caption'])

        # seg_cat = self.COCO_CLASSES[seg_cat]

        img_dir = os.path.join(self.image_dir, img_name) if self.cutler else os.path.join(self.image_dir, img_name[0])
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        width, height = img.shape[1], img.shape[0]
        img_size = img.shape[:2]

        mat, mat_inv = self.getTransformMat(img_size, True)

        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])

        mask_dir = os.path.join(self.mask_dir, f'{seg_id}.png')
        mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
        mask = cv2.warpAffine(mask,
                              mat,
                              self.input_size,
                              flags=cv2.INTER_LINEAR,
                              borderValue=0.)
        mask = mask / 255.

        img, mask = self.transform(img, mask)

        if self.mode == 'train':

            if self.doc_mode:
                doc = eval(row['doc'])


                if self.doc_filter == 'value':
                    doc_filter = np.array(doc) > 1.
                    filtered_idx = np.where(doc_filter)[0]

                    if len(filtered_idx) == 0:
                        random_idx = np.random.choice(num_sent)
                    else:
                        random_idx = random.choice(filtered_idx)

                elif self.doc_filter == 'top_k':
                    k = 30
                    indices = np.argsort(doc)[-k:]
                    random_idx = random.choice(indices)

            else:
                random_idx = np.random.choice(num_sent)

            pseudo_caption = pseudo_caption[random_idx]
            word_vec = tokenize(pseudo_caption, self.word_length, True).squeeze(0)

            return img, word_vec, mask

        else:
            raise NotImplementedError



    def transform(self, img, mask=None):
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask


    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None





    def transform(self, img, mask=None):
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask


    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None


class pseudo_dataset_with_pseudo_mask(Dataset):
    def __init__(self, csv_dir, image_dir, mask_dir, input_size, word_length, doc_mode=False,
                 doc_filter='value', doc_value = 1, seg_model='sam', beam_search=False, np_mode=False):
        self.csv_dir = csv_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.doc_mode = doc_mode
        self.doc_filter = doc_filter
        self.seg_model = seg_model
        self.beam_search = beam_search
        self.doc_value = float(doc_value)
        self.df = pd.read_csv(self.csv_dir)
        self.np_mode = np_mode

        if self.np_mode:
            import spacy
            self.nlp = spacy.load('en_core_web_lg')

        self.COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_name = row['img_name']
        img_id = row['img_id']
        seg_id = row['seg_id']
        num_sent = row['num_sent']
        pseudo_caption = eval(row['pseudo_caption'])

        if self.beam_search:
            pseudo_caption = [p for i,p in enumerate(pseudo_caption) if i % 11 == 1]

        try:
            one_mask = row['one_mask']
        except KeyError:
            one_mask = 0
        if self.seg_model == 'refcoco+':
            img_name = eval(img_name)[0]
        print(img_name)
        img_dir = os.path.join(self.image_dir, img_name)
        img = cv2.imread(img_dir)
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_size = img.shape[:2]
        mat, mat_inv = self.getTransformMat(img_size, inverse=True)

        img = cv2.warpAffine(img, mat, self.input_size, flags=cv2.INTER_CUBIC,
                             borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])

        if self.seg_model == 'refcoco+':
            mask_dir = os.path.join(self.mask_dir, f'{seg_id}.png')
        else:
            mask_dir = os.path.join(self.mask_dir, str(img_id), f'{seg_id}.png')

        # if not os.path.exists(mask_dir):
        #     print(f'No mask file: {mask_dir}')
            # return torch.zeros(3, self.input_size[0], self.input_size[1]), torch.zeros(self.word_length), torch.zeros(self.input_size[0], self.input_size[1])

        mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
        mask = cv2.warpAffine(mask, mat, self.input_size, flags=cv2.INTER_LINEAR, borderValue=0.)
        mask = mask / 255

        img, mask = self.transform(img, mask)

        if self.doc_mode:
            doc = eval(row['doc'])
            if one_mask == 0:
                doc_filter = np.array(doc) > self.doc_value
                filtered_idx = np.where(doc_filter)[0]
                random_idx = random.choice(filtered_idx) if len(filtered_idx) > 0 else np.random.choice(num_sent)

            elif one_mask == 1:
                sorted_idx = np.argsort(doc)[::-1]
                sorted_idx = sorted_idx[:num_sent // 2]
                random_idx = random.choice(sorted_idx)

        else:
            random_idx = np.random.choice(len(pseudo_caption))

        pseudo_caption = pseudo_caption[random_idx]
        if self.np_mode:
            noun_phrase = noun_extraction(pseudo_caption, self.nlp)
            pseudo_caption = noun_phrase

        word_vec = tokenize(pseudo_caption, self.word_length, True).squeeze(0)

        return img, word_vec, mask









    def transform(self, img, mask=None):
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask


    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None


class pseudo_dataset_raw(Dataset):
    def __init__(self, csv_dir, image_dir, mask_dir):
        self.csv_dir = csv_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.df = pd.read_csv(csv_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_name = row['img_name']
        img_id = row['img_id']
        seg_id = row['seg_id']
        num_sent = row['num_sent']
        pseudo_caption = eval(row['pseudo_caption'])
        doc = eval(row['doc'])
        return img_name, img_id, seg_id, num_sent, pseudo_caption, doc











if __name__ == '__main__':

    dataset = 'refcoco+'
    seg_model = 'sam'
    model_type = 'cc3m'
    csv_dir = f'./pseudo_supervision/{seg_model}/base_{model_type}.csv'
    image_dir = f'./datasets/images/train2014/'
    mask_dir = f'./datasets/pseudo_masks/{seg_model}'
    doc_mode = True

    # csv_dir, image_dir, mask_dir, dataset, split, mode, input_size, word_length, doc_mode=False
    dataset = pseudo_dataset_with_pseudo_mask(csv_dir=csv_dir,
                                              image_dir=image_dir,
                                              mask_dir=mask_dir,
                                              input_size=416,
                                              word_length=17,
                                              doc_mode=doc_mode)
    # dataset = pseudo_dataset_for_cris(csv_dir=csv_dir, image_dir=image_dir, mask_dir=mask_dir,
    #                                   dataset='refcoco+', split='train',mode='train',
    #                                   input_size=416, word_length=17, doc_mode=True, doc_filter='value', cutler=cutler)

    a = dataset[0]
    # print(a)

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # for i, data in enumerate(dataloader):
    #     print(data)
    #     # if i == 100:
    #     break
    #     # print(data)


    # for i, data in enumerate(dataloader):
    #     print(i)
    #     print(data)
    #     break