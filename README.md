# EntitySAM [CVPR'25]

> [**EntitySAM: Segment Everything in Video**](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_EntitySAM_Segment_Everything_in_Video_CVPR_2025_paper.pdf)                 
> CVPR 2025  
> Adobe Research, EPFL, CMU

We propose **EntitySAM**, a novel framework extending SAM2 to the task of **Video Entity Segmentation**—segmenting every entity in a video without requiring category annotations. Our method achieves generalizable and exhaustive segmentation using **only image-level training data**, and demonstrates strong **zero-shot** performance across multiple benchmarks. Refer to our [paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_EntitySAM_Segment_Everything_in_Video_CVPR_2025_paper.pdf) for more details.

<img width="1000" alt="image" src='assets/Figure1.png'>

Updates
-----------------
🔥 2025/07/12: We release the training code. See [Training](#train) section.

2025/06/26: We release the evaluation code and checkpoints. See [Evaluation](#evaluation) section.

2025/06/02: Our paper [EntitySAM](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_EntitySAM_Segment_Everything_in_Video_CVPR_2025_paper.pdf) is online.

# Introduction
Automatically tracking and segmenting every video entity remains a significant challenge. Despite rapid advancements in video segmentation, even state-of-the-art models like SAM 2 struggle to consistently track all entities across a video—a task we refer to as Video Entity Segmentation. We propose EntitySAM, a framework for zero-shot video entity segmentation. EntitySAM extends SAM 2 by removing the need for explicit prompts, allowing automatic discovery and tracking of all entities, including those appearing in later frames. We incorporate query-based entity discovery and association into SAM 2, inspired by transformer-based object detectors. Specifically, we introduce an entity decoder to facilitate inter-object communication and an automatic prompt generator using learnable object queries. Additionally, we add a semantic encoder to enhance SAM 2’s semantic awareness, improving segmentation quality. Trained on image-level mask annotations without category information from the COCO dataset, EntitySAM demonstrates strong generalization on four zero-shot video segmentation tasks: Video Entity, Panoptic, Instance, and Semantic Segmentation. Results on six popular benchmarks show that EntitySAM outperforms previous unified video segmentation methods and strong baselines, setting new standards for zero-shot video segmentation.

# Method Overview
**(a) Overview of the EntitySAM framework**: EntitySAM utilizes the frozen encoder and memory parameters from SAM 2, incorporating a dual encoder design for enhanced semantic features. The PromptGenerator automatically generates prompts from Prompt Queries. The enhanced features and distinct query groups are processed by the EntityDecoder to produce video mask outputs. 

**(b) EntityDecoder** Self-attention and cross-attention mechanisms in EntityDecoder layers.
<img width="1096" alt="image" src='assets/Figure2.png'>


# Visualizations

## Zero-shot Video Entity Segmentation
Our EntitySAM segment every entity in a video. Results are compared with SAM 2 and DEVA. EntitySAM achieves zero-shot video entity segmentation without requiring  video-level training data or category annotations.

https://github.com/user-attachments/assets/ad9ef654-ca35-43c9-ac55-9778ac22a06f

https://github.com/user-attachments/assets/f4cfca96-80b2-4387-8cad-a46c10a23440

https://github.com/user-attachments/assets/761ddaa6-ad93-4c32-a44b-8984437a1584

https://github.com/user-attachments/assets/a2d69f01-e686-446d-8dc5-50e75e4cb816

# Installation
EntitySAM environment is based on SAM 2. The code requires `python>=3.10`, as well as `torch>=2.5.1` and `torchvision>=0.20.1`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install EntitySAM on a GPU machine using:

```bash
git clone https://github.com/ymq2017/entitysam && cd entitysam

pip install -e .

pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/facebookresearch/detectron2.git
```

# Datasets

## Training Dataset
For training EntitySAM, you can download the COCO dataset from the [official website](https://cocodataset.org/).

## Evaluation Datasets
For evaluation, you can download the VIPSeg dataset from the [official website](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset) or use [our processed version](https://huggingface.co/mqye/entitysam/tree/main) for convenience.

### Directory Structure
```
datasets/
├── VIPSeg_720P/
│   ├── images/
│   ├── panomasks/
│   ├── panomasksRGB/
│   ├── panoVIPSeg_categories.json
│   ├── panoptic_gt_VIPSeg_val.json
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── coco/
    ├── train2017/
    ├── val2017/
    ├── annotations/
    └── panoptic_train2017/
```

# Evaluation

For detailed evaluation instructions, please see [EVAL.md](EVAL.md).

## Pre-trained Checkpoints

We provide pre-trained EntitySAM checkpoints on [Hugging Face](https://huggingface.co/mqye/entitysam/tree/main). You can download the checkpoints using the following table:

| Model | Checkpoint Path | Download Link |
|-------|----------------|---------------|
| ViT-L | `./checkpoints/vit-l/model_0009999.pth` | [Download](https://huggingface.co/mqye/entitysam/blob/main/checkpoints/vit-l/model_0009999.pth) |
| ViT-S | `./checkpoints/vit-s/model_0009999.pth` | [Download](https://huggingface.co/mqye/entitysam/blob/main/checkpoints/vit-s/model_0009999.pth) |
<!-- 
Alternatively, you can clone the entire repository:
```bash
git lfs install
git clone https://huggingface.co/mqye/entitysam
``` -->

# Train

For training instructions, please see [TRAIN.md](TRAIN.md).

Citation
---------------
If you find EntitySAM useful in your research or refer to the provided baseline results, please star :star: this repository and consider citing :pencil::
```
@inproceedings{entitysam,
    title={EntitySAM: Segment Everything in Video},
    author={Ye, Mingqiao and Oh, Seoung Wug and Ke, Lei and Lee, Joon-Young},
    booktitle={CVPR},
    year={2025}
}
```
