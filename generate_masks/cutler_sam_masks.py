import os
import sys
sys.path.append('./')
from my_utils.datasets_for_cutler import RefDataset_for_cutler
from torch.utils.data import DataLoader
import argparse
import warnings
import torch
import tqdm
# import open_clip
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
import cv2

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer


sys.path.append('./third_party/CutLER/cutler')
from modeling import build_model
from config import add_cutler_config
import data.transforms as T
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
warnings.simplefilter(action='ignore', category=FutureWarning)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_cutler_config(cfg)
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Disable the use of SyncBN normalization when running on a CPU
    # SyncBN is not supported on CPU and can cause errors, so we switch to BN instead
    if cfg.MODEL.DEVICE == 'cpu' and cfg.MODEL.RESNETS.NORM == 'SyncBN':
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.FPN.NORM = "BN"
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="third_party/CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def main(cfg):
    device = 'cuda'

    '''
    CutLER
    '''
    cutler = build_model(cfg)
    cutler.eval()

    checkpointer = DetectionCheckpointer(cutler)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )  # 800, 1333

    '''
    SAM
    '''
    model_type = 'vit_h'
    checkpoint = './third_party/segment-anything/segment_anything/checkpoints/sam_vit_h_4b8939.pth'
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam,
                                               pred_iou_thresh=0.5,)

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
    Arguments
    '''
    path2mask_dir = f'./datasets/pseudo_masks/cutler_sam'
    os.makedirs(path2mask_dir, exist_ok=True)

    img_name_set = set()

    t_bar = tqdm.tqdm(dataloader, ncols=100)
    for i, data in enumerate(t_bar):
        img_name, seg_id, cat_id, sents, num_sents, img_id = data['img_name'], data['seg_id'], data['cat'], data['sents'], data['num_sents'], data['img_id']

        img_name = img_name[0]
        img_id = img_id.item()
        seg_id = int(seg_id)
        cat = int(cat_id)
        sents = [s[0] for s in sents]
        num_sents = int(num_sents)

        if img_name in img_name_set:
            continue
        img_name_set.add(img_name)

        path2img = image_dir + img_name
        ori_img = read_image(path2img, format="BGR")  # numpy [H,W,3]
        ori_height, ori_width = ori_img.shape[:2]

        '''
        For Cutler prediction
        '''
        img = aug.get_transform(ori_img).apply_image(ori_img)  # [1199, 800, 3]
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))  # [3, 1199, 800]

        inputs = {'image': img, 'height': ori_height, 'width': ori_width}

        with torch.no_grad():
            pred = cutler([inputs])[0]  # keys = ['instances']

        detected_instances = len(pred['instances'])

        if detected_instances == 0:
            print('no instances detected')
            continue

        cutler_masks = pred['instances'].pred_masks  # [N, 640, 427]
        cutler_boxes = pred['instances'].pred_boxes.tensor  # [N, 4]
        cutler_scores = pred['instances'].scores  # [N]

        sam_masks = mask_generator.generate(ori_img)
        sam_masks = [torch.tensor(m['segmentation'], dtype=torch.bool) for m in sam_masks]
        sam_masks = torch.stack(sam_masks).to(device)

        max_overlap = torch.zeros(cutler_masks.shape[0])
        max_overlap_idx = torch.zeros(cutler_masks.shape[0], dtype=torch.long)

        for z, cutler_mask in enumerate(cutler_masks):
            overlap_list = []
            for j, sam_mask in enumerate(sam_masks):
                overlap_area = (cutler_mask & sam_mask).sum().item()
                overlap_list.append(overlap_area)

            overlap_list = torch.tensor(overlap_list)
            max_overlap[z] = overlap_list.max()
            max_overlap_idx[z] = overlap_list.argmax()

        final_masks = sam_masks[max_overlap_idx]


        path2mask = os.path.join(path2mask_dir, str(img_id))
        os.makedirs(path2mask, exist_ok=True)
        for j, m in enumerate(final_masks):
            m = m.cpu().numpy().astype(np.uint8) * 255

            cv2.imwrite(os.path.join(path2mask, f'{str(j)}.png'), m)


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    main(cfg)





