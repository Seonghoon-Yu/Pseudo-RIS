# Pseudo-RIS

> **[Pseudo-RIS: Distinctive Pseudo-supervision Generation for Referring Image Segmentation](https://arxiv.org/abs/2407.07412)** \
> [Seonghoon Yu](https://scholar.google.com/citations?user=VuIo1woAAAAJ&hl=ko), +[Paul Hongsuck Seo](https://phseo.github.io/), +[Jeany Son](https://jeanyson.github.io/) (+ corresponding authors) \
> AI graduate school, GIST and Korea University \
> ECCV 2024

<p align="center"> <img src="https://github.com/Seonghoon-Yu/Zero-shot-RIS/assets/75726938/21562645-62ed-4617-ad21-c88c267a62ab.PNG" width="500" align="center"> </p>

> **Abstract** \
> We propose a new framework that automatically generates high-quality segmentation masks with their referring expressions as pseudo supervisions for referring image segmentation (RIS). These pseudo supervisions allow the training of any supervised RIS methods without the cost of manual labeling. To achieve this, we incorporate existing segmentation and image captioning foundation models, leveraging their broad generalization capabilities. However, the naive incorporation of these models may generate non-distinctive expressions that do not distinctively refer to the target masks. To address this challenge, we propose two-fold strategies that generate distinctive captions: 1) 'distinctive caption sampling', a new decoding method for the captioning model, to generate multiple expression candidates with detailed words focusing on the target. 2) 'distinctiveness-based text filtering' to further validate the candidates and filter out those with a low level of distinctiveness. These two strategies ensure that the generated text supervisions can distinguish the target from other objects, making them appropriate for the RIS annotations. Our method significantly outperforms both weakly and zero-shot SoTA methods on the RIS benchmark datasets. It also surpasses fully supervised methods in unseen domains, proving its capability to tackle the open-world challenge within RIS. Furthermore, integrating our method with human annotations yields further improvements, highlighting its potential in semi-supervised learning applications.


## Installation
### 1. Environment
```shell
# create conda env
conda create -n pseudo_ris python=3.9

# activate the environment
conda activate pseudo_ris

# Install Pytorch
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# Install required package
pip install pydantic==1.10.11 --upgrade
conda install -c conda-forge spacy
python -m spacy download en_core_web_lg
conda install -c anaconda pandas
pip install opencv-python
pip install lmdb
pip install pyarrow==11.0.0
pip install colored
pip install pycocotools
```

### 2. Third Party
```shell
# Install CoCa in a dev mode, where distinctive caption sampling is implemented.
cd third_party/open_clip
pip install -e .

# Install detectron2 for CutLER 
cd third_party/detectron2
pip install -e .

# Install transformer in a dev mode
cd third_party/transformers
pip install -e .

# Install CLIP
cd third_party/CLIP
pip install -e .

# Install SAM in a dev mode
cd segment-anything
pip install -e .
```

### 3. Download pre-trained weights
We use the pre-trained weights for (1) CoCa, (2) SAM, and (3) CutLER.
#### For CoCa
We will provide CoCa pre-trained weights on LAION-2B and CC3M.

Note that, [official CoCa repository](https://github.com/mlfoundations/open_clip#fine-tuning-coca) offers pre-trained model on LAION-2B.

We fine-tune this on CC3M dataset.

#### For SAM
We use SAM ViT-H model.
```shell
# Download SAM ViT-H model.
cd segment-anything
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### For CutLER
We use [CutLER](https://github.com/facebookresearch/CutLER) to reduce the excessive number of SAM masks and over-segmented SAM masks to prevent OOM issues, as demonstrated in our supplementary and implementation details.

```
cd third_party/CuTLER/cutler/
mkdir checkpoints
cd checkpoints
wget http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth
```

## Dataset
We follow a [dataset setup](https://github.com/kkakkkka/ETRIS/blob/main/tools/prepare_datasets.md) in [ETRIS](https://github.com/kkakkkka/ETRIS) to get unlabeled images in the train set of refcoco+.

```
├── datasets
│   ├── images
│   │   └── train2014
│   │       ├── COCO_train2014_000000000009.jpg
│   │       └── ...
│   └── lmdb
│       └── refcoco+
│           ├── train.lmdb
│           └── ...
```

## Generate pseudo RIS annotations
Codes are coming soon.
### 1. Generate pseudo masks
We produce pseudo-masks using SAM and CutLER, as demonstrated in our implementation details and supplementary material.

Pseudo masks are saved in './datasets/pseudo_masks/cutler_sam' directory.
```
python generate_masks/cutler_sam_masks.py
```

### 2. Generate distinctive referring expressions on each pseudo mask.
Pseudo referring texts are saved in './pseudo_supervision/cutler_sam/distinctive_captions_cc3m.csv'

```
python generate_pseudo_supervision/distinctive_caption_generation.py
```

## Citation
```
@inproceedings{yu2024pseudoris,
    title={Pseudo-RIS: Distinctive Pseudo-supervision Generation for Referring Image Segmentation},
    author={Seonghoon Yu and Paul Hongsuck Seo and Jeany Son},
    booktitle={Proceedings of the European Conference on Computer Vision},
    year={2024}
}
```

## Acknowledgements
We are thanks to open-source foundation models ([CoCa](https://github.com/mlfoundations/open_clip), [SAM](https://github.com/facebookresearch/segment-anything), [CLIP](https://github.com/openai/CLIP)).


