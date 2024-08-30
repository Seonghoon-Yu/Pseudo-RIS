import os
from typing import List, Union
import sys
sys.path.append('./')
sys.path.append('../')
import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset
from datasets.PhraseCutDataset.utils.refvg_loader import RefVGLoader
# from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from PIL import Image, ImageDraw





def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)



class PhraseCut_Dataset_lmdb(Dataset):
    def __init__(self, split='val', word_length=17,  input_size=416,
                 image_dir=f'./datasets/PhraseCutDataset/data/VGPhraseCut_v0/images/',
                 lmdb_dir=f'./datasets/PhraseCutDataset/lmdb/',
                 unseen_mode=False):
        # self.refvg_loader = RefVGLoader(split=split)
        self.word_length = word_length
        self.input_size = (input_size, input_size)
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.image_dir = image_dir
        self.lmdb_dir = os.path.join(lmdb_dir, f'phrasecut_{split}.lmdb')
        self.unseen_mode = unseen_mode
        self._init_db()

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
        image_id = ref['image_id']
        image_dir = os.path.join(self.image_dir, str(image_id) + '.jpg')
        mask = ref['mask']
        phrase = ref['phrase']
        cat_name = ref['cat_id']
        box = ref['box']
        if self.unseen_mode and cat_name in self.COCO_CLASSES:
            # if cat_name in self.COCO_CLASSES:
            # print(1)
                # print(type(self.input_size[0]))
            # print(cat_name)
            return torch.zeros((3,self.input_size[0], self.input_size[0])), torch.zeros((self.input_size[0], self.input_size[0])), torch.zeros((self.word_length))


        ori_img = cv2.imread(image_dir)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]

        # transform
        mat, mat_inv = self.getTransformMat(img_size, True)
        # img = cv2.warpAffine(img, mat, self.input_size, flags=cv2.INTER_CUBIC,
        #                      borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        # mask = cv2.warpAffine(mask, mat, self.input_size, flags=cv2.INTER_LINEAR, borderValue=0.)
        img, mask = self.convert(img, mask)

        data = {'img' : img, 'target': mask, 'sent': phrase, 'img_name': image_id, 'seg_cat':cat_name, 'seg_id': 0, 'box': box}

        return data

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


    def boxes_region(self, boxes):
        """
        :return: [x_min, y_min, x_max, y_max] of all boxes
        """

        boxes = np.array(boxes)
        min_xy = np.min(boxes[:, :2], axis=0)
        max_xy = np.max(boxes[:, 2:], axis=0)
        return [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]

    def polygons_to_mask(self, polygons, w, h):
        p_mask = np.zeros((h, w))
        for polygon in polygons:
            if len(polygon) < 2:
                continue
            p = []
            for x, y in polygon:
                p.append((int(x), int(y)))
            img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(img).polygon(p, outline=1, fill=1)
            mask = np.array(img)
            p_mask += mask
        p_mask = p_mask > 0
        return p_mask


    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"db_path={self.lmdb_dir}, " + \
            f"split={self.split}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length}"


class PhraseCut_Dataset(Dataset):
    def __init__(self, split='test', image_dir=f'./datasets/PhraseCutDataset/data/VGPhraseCut_v0/images/'):
        self.refvg_loader = RefVGLoader(split=split)
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.refvg_loader.img_ids)

    def __getitem__(self, index):
        img_id = self.refvg_loader.img_ids[index]
        img_ref_data = self.refvg_loader.get_img_ref_data(img_id)
        img_id = img_ref_data['image_id']
        img_dir = os.path.join(self.image_dir, str(img_id) + '.jpg')

        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        width = img_ref_data['width']
        height = img_ref_data['height']
        img_ins_cats = img_ref_data['img_ins_cats']

        count = 0
        category = []
        gt_masks = []
        sent = []
        for task_i, task_id in enumerate(img_ref_data['task_ids']):
            phrase = img_ref_data['phrases'][task_i]
            sent.append(phrase)

            gt_polygons = list()
            gt_Polygons = img_ref_data['gt_Polygons'][task_i]

            for ps in gt_Polygons:
                gt_polygons += ps

            gt_mask = self.polygons_to_mask(gt_polygons, width, height)

            gt_masks.append(gt_mask)

            num_obj = len(gt_Polygons)

            cat = img_ins_cats[count]

            category += [cat]

            count += num_obj

        gt_masks = np.array(gt_masks)

        img, gt_masks = self.convert(img, gt_masks)

        data = dict(img=img, target=gt_masks, sent=sent, cat=category, img_id=img_id, width=width, height=height)

        return data

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

    def boxes_region(self, boxes):
        """
        :return: [x_min, y_min, x_max, y_max] of all boxes
        """

        boxes = np.array(boxes)
        min_xy = np.min(boxes[:, :2], axis=0)
        max_xy = np.max(boxes[:, 2:], axis=0)
        return [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]

    def polygons_to_mask(self, polygons, w, h):
        p_mask = np.zeros((h, w))
        for polygon in polygons:
            if len(polygon) < 2:
                continue
            p = []
            for x, y in polygon:
                p.append((int(x), int(y)))
            img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(img).polygon(p, outline=1, fill=1)
            mask = np.array(img)
            p_mask += mask
        p_mask = p_mask > 0
        return p_mask


class PhraseCut_dataset_raw(Dataset):
    def __init__(self, split='test',
                 image_dir=f'./datasets/PhraseCutDataset/data/VGPhraseCut_v0/images/',
                 ):
        self.refvg_loader = RefVGLoader(split=split)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.refvg_loader.img_ids)

    def __getitem__(self, index):
        image_id = self.refvg_loader.img_ids[index]
        img_ref_data = self.refvg_loader.get_img_ref_data(image_id)
        image_id = img_ref_data['image_id']
        image_dir = os.path.join(self.image_dir, str(image_id) + '.jpg')

        img = cv2.imread(image_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]

        phrase_list = []
        for task_i, task_id in enumerate(img_ref_data['task_ids']):
            gt_Polygon = img_ref_data['gt_Polygons'][task_i]

            # num_obj = len(gt_Polygon)
            # cat = img_ins_cats[count]
            # count += num_obj

            # for ps in gt_Polygon:
            #     gt_polygons += ps

            phrase = img_ref_data['phrases'][task_i]
            phrase_list.append(phrase)

        return phrase_list



if __name__ == '__main__':
    dataset = PhraseCut_Dataset(split='test')
    dataset[0]

