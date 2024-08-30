import os
from torch.utils.data import Dataset, DataLoader
from datasets.PhraseCutDataset.utils.refvg_loader import RefVGLoader
import lmdb
import pyarrow as pa
import cv2
import numpy as np
import torch

def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)

class RefDataset_for_cutler(Dataset):
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode):
        super().__init__()
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.split = split

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
        if self.env is None:
            self._init_db()

        env = self.env

        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        ref = loads_pyarrow(byteflow)

        # img
        # ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8), cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        # height, width = img.shape[0], img.shape[1]

        # mask
        seg_id = ref['seg_id']

        # mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
        #                     cv2.IMREAD_GRAYSCALE)

        cat = ref['cat']
        img_name = ref['img_name']
        sents = ref['sents']
        num_sents = ref['num_sents']

        img_id = ref['img_id']


        data = {'img_name': img_name,
                'seg_id': seg_id,
                'cat': cat,
                'sents': sents,
                'num_sents': num_sents,
                'img_id': img_id,
                }

        return data
class PhraseCut_Dataset(Dataset):
    def __init__(self, split='val', word_length=17,  input_size=416,
                 image_dir=f'../datasets/PhraseCutDataset/data/VGPhraseCut_v0/images/',
                 unseen_mode=False,
                 seen_mode=False,
                 mask_gen_mode=False):
        self.refvg_loader = RefVGLoader(split=split)
        self.word_length = word_length
        self.input_size = (input_size, input_size)
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.image_dir = image_dir

        self.unseen_mode = unseen_mode
        self.seen_mode = seen_mode
        self.mask_gen_mode = mask_gen_mode
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
        return len(self.refvg_loader.img_ids)

    def __getitem__(self, index):
        image_id = self.refvg_loader.img_ids[index]
        img_ref_data = self.refvg_loader.get_img_ref_data(image_id)
        image_id = img_ref_data['image_id']
        image_dir = os.path.join(self.image_dir, str(image_id) + '.jpg')

        if self.mask_gen_mode:
            data = {'img_id': image_id}
            return data

        ori_img = cv2.imread(image_dir)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB) # 600, 800, 3, np.ndarray
        img_size = img.shape[:2]

        # transform
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(img, mat, self.input_size, flags=cv2.INTER_CUBIC,
                             borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])

        # if self.val_mode:
        #     data = {'img':img, 'mat': mat, 'img_ref_data'}

        gt_masks = []
        word_vecs = []

        gt_polygons = list()
        img_ins_cats = img_ref_data['img_ins_cats']

        count = 0

        for task_i, task_id in enumerate(img_ref_data['task_ids']):
            gt_Polygon = img_ref_data['gt_Polygons'][task_i]


            num_obj = len(gt_Polygon)
            cat = img_ins_cats[count]
            count += num_obj

            if self.unseen_mode and cat in self.COCO_CLASSES:
                continue
            elif self.seen_mode and cat not in self.COCO_CLASSES:
                continue


            for ps in gt_Polygon:
                gt_polygons += ps


            mask = self.polygons_to_mask(gt_polygons, img_ref_data['width'], img_ref_data['height'])
            mask = mask / 1.
            mask = cv2.warpAffine(mask, mat, self.input_size, flags=cv2.INTER_LINEAR, borderValue=0.)
            gt_masks.append(mask)

            phrase = img_ref_data['phrases'][task_i]
            word_vec = tokenize(phrase, self.word_length, True).squeeze(0) # [17]
            word_vecs.append(word_vec)

        gt_masks = np.array(gt_masks)

        if len(gt_masks) == 0:
            return torch.zeros((3,self.input_size[0], self.input_size[0])), torch.zeros((self.input_size[0], self.input_size[0])), torch.zeros((self.word_length))

        img, mask = self.convert(img, gt_masks)
        word_vecs = torch.stack(word_vecs, dim=0) # [N, 17]

        return img, mask, word_vecs


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

if __name__ == '__main__':
    dataset = 'refcoco+'
    split = 'val'
    lmdb_dir = f'./datasets/lmdb/{dataset}/{split}.lmdb'
    mask_dir = f'./datasets/masks/{dataset}/'
    mode = 'train'

    ref_dataset = RefDataset_for_cutler(lmdb_dir, mask_dir, dataset, split, mode)
    dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    for i, data in enumerate(dataloader):
        img_name, seg_id, cat, sents, num_sents = data['img_name'], data['seg_id'], data['cat'], data['sents'], data['num_sents']
        img_name = img_name[0]
        seg_id = int(seg_id)
        cat = int(cat)
        sents = [s[0] for s in sents]
        num_sents = int(num_sents)


        break