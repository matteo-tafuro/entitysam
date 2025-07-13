import copy
import logging
import os
import random
import numpy as np
from typing import List, Union
import torch

try:
    import orjson as json
except:
    import json

from einops import rearrange
import matplotlib.pyplot as plt

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog

from pycocotools import mask as coco_mask

from train.data import detection_utils as utils
from fvcore.transforms.transform import HFlipTransform
from .augmentation import build_augmentation, build_pseudo_augmentation

import re

__all__ = ["YTVISDatasetMapper", "CocoClipDatasetMapper", ]


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())
    if instances.has("gt_classes"):
        r.append(instances.gt_classes != -1)

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def _get_dummy_anno():
    return {
        "iscrowd": 0,
        "category_id": -1,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }


def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obj["id"]) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class YTVISDatasetMapper:
    """
    Similar to YTVISDatasetMapper, only add text expressions
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_video_num: int = -1, 
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
        dataset_name: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num if sampling_frame_video_num == -1 else sampling_frame_video_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        self.dataset_name           = dataset_name

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, dataset_name=None):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_video_num = cfg.INPUT.SAMPLING_FRAME_VIDEO_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_video_num": sampling_frame_video_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "dataset_name": dataset_name,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict["length"]
        if self.is_train:
            ref_frame = random.randrange(video_length)

            if video_length == 1:
                selected_idx = [ref_frame] * self.sampling_frame_num
            else:
                start_idx = max(0, ref_frame-self.sampling_frame_range)
                end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

                selected_idx = np.random.choice(
                    np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                    self.sampling_frame_num - 1,
                )
                selected_idx = selected_idx.tolist() + [ref_frame]
                selected_idx = sorted(selected_idx)

            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        video_annos = dataset_dict["annotations"]
        file_names = dataset_dict["file_names"]

        if video_annos is not None:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["video_len"] = len(video_annos)
        dataset_dict["frame_idx"] = list(selected_idx)
        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])
            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore, it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            # if (video_annos is None) or (not self.is_train):
            if video_annos is None:
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]
            sorted_annos = [_get_dummy_anno() for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]
                if _anno['bbox'] is not None or _anno['segmentation'] is not None:
                    sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            instances = filter_empty_instances(instances)

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)

            if not instances.has("gt_masks"):
                if instances.has("gt_boxes"):
                    n_inst = instances.gt_boxes.tensor.shape[0]
                    instances.gt_masks = BitMasks(torch.empty((n_inst, *image_shape)))
                else:
                    instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            dataset_dict["instances"].append(instances)

        # remove empty objects from Instance
        gt_ids_per_video = []
        for f_i, targets_per_frame in enumerate(dataset_dict["instances"]):
            gt_ids_per_video.append(targets_per_frame.gt_ids)
        gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)
        valid_idxs = torch.nonzero((gt_ids_per_video >= 0).any(1)).reshape(-1)

        # to speed up training and save memory, there are so many objects in SA1B
        dataset_dict["instances"] = [
            targets_per_frame[valid_idxs] for targets_per_frame in dataset_dict["instances"]
        ]

        return dataset_dict


def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')


class CocoClipDatasetMapper:
    """
    A callable which takes a COCO image which converts into multiple frames,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        sampling_frame_num: int = 2,
        sampling_frame_video_num: int = -1,
        sampling_frame_range: int = 5,
        dataset_name: str = 'coco_2017_train',
        num_pos_queries: int = 20,
        eval_load_annotations: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.sampling_frame_num     = sampling_frame_num if sampling_frame_video_num == -1 else sampling_frame_video_num
        self.sampling_frame_range   = sampling_frame_range
        self.dataset_name = dataset_name
        self.num_pos_queries = num_pos_queries

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, dataset_name: str = ""):
        if cfg.INPUT.SAMPLING_FRAME_NUM == 1:
            augs = build_augmentation(cfg, is_train)
        else:
            augs = build_pseudo_augmentation(cfg, is_train)
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_video_num = cfg.INPUT.SAMPLING_FRAME_VIDEO_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_video_num": sampling_frame_video_num,
            "sampling_frame_range": sampling_frame_range,
            "dataset_name": dataset_name,
            "num_pos_queries": cfg.MODEL.UniVS.NUM_POS_QUERIES,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
            or dataset_dict (str): annotation file names
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        is_sa1b = False
        if isinstance(dataset_dict, str):
            is_sa1b = True
            # for SA-1B, where dataset_dict is the name of annotation file
            image_root = 'datasets/sa_1b/images'
            anno_root = 'datasets/sa_1b/annotations'

            file = open('/'.join([anno_root, dataset_dict]), 'r').read()
            annotations = json.loads(file)

            # {
            #     "image": image_info,
            #     "annotations": [annotation],
            # }
            #
            # image_info
            # {
            #     "image_id": int,  # Image id
            #     "width": int,  # Image width
            #     "height": int,  # Image height
            #     "file_name": str,  # Image filename
            # }
            #
            # annotation
            # {
            #     "id": int,  # Annotation id
            #     "segmentation": dict,  # Mask saved in COCO RLE format.
            #     "bbox": [x, y, w, h],  # The box around the mask, in XYWH format
            #     "area": int,  # The area in pixels of the mask
            #     "predicted_iou": float,  # The model's own prediction of the mask's quality
            #     "stability_score": float,  # A measure of the mask's quality
            #     "crop_box": [x, y, w, h],  # The crop of the image used to generate the mask, in XYWH format
            #     "point_coords": [[x, y]],  # The point coordinates input to the model to generate the mask
            # }

            dataset_dict = annotations["image"]
            dataset_dict["file_name"] = os.path.join(image_root, dataset_dict["file_name"])
            dataset_dict["annotations"] = annotations["annotations"]
            for anno_dict in dataset_dict["annotations"]:
                anno_dict["bbox_mode"] = BoxMode.XYWH_ABS

            dataset_dict["dataset_name"] = "sa_1b"
            dataset_dict["task"] = "sot"
            dataset_dict["has_stuff"] = True

        else:
            dataset_dict["has_stuff"] = False
            if "dataset_name" not in dataset_dict:
                if "pan_seg_file_name" in dataset_dict:
                    dataset_dict["dataset_name"] = "coco_panoptic"
                    dataset_dict["has_stuff"] = True
                else:
                    dataset_dict["dataset_name"] = "coco"
            dataset_dict["task"] = "detection"
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        img_annos = dataset_dict.pop("annotations", None)
        if is_sa1b and img_annos is not None and len(img_annos) > 100:
            # remove masks with low stability_score or predicted_iou
            img_annos= [anno for anno in img_annos if anno["stability_score"] > 0.97 and anno["predicted_iou"] > 0.9]

        file_name = dataset_dict.pop("file_name", None)
        original_image = utils.read_image(file_name, format=self.image_format)

        if self.is_train:
            video_length = random.randrange(16, 49)
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
        else:
            video_length = self.sampling_frame_num
            selected_idx = list(range(self.sampling_frame_num))

        dataset_dict["has_mask"] = True
        dataset_dict["video_len"] = video_length
        dataset_dict["frame_indices"] = selected_idx
        dataset_dict["image"] = []
        dataset_dict["image_padding_mask"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = [file_name] * self.sampling_frame_num
        if not self.is_train and "panoptic" in dataset_dict["dataset_name"]:
            # panoptic evaluator needs file_name
            dataset_dict["file_name"] = file_name

        for i in range(self.sampling_frame_num):
            utils.check_image_size(dataset_dict, original_image)
            image_padding_mask = np.ones_like(original_image)

            aug_input = T.AugInput(original_image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            image_shape = image.shape[:2]  # h, w

            image_padding_mask = transforms.apply_segmentation(image_padding_mask)

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore, it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))
            dataset_dict["image_padding_mask"].append(torch.as_tensor(
                np.ascontiguousarray(1 - image_padding_mask[:, :, 0])
            ))

            if (img_annos is None) or (not self.is_train):
                continue
                
            _img_annos = []
            for obj_i, anno in enumerate(img_annos):
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _img_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _img_annos
                if obj.get("iscrowd", 0) == 0
            ]

            _gt_ids = list(range(len(annos)))
            for idx in range(len(annos)):
                if len(annos[idx]["segmentation"]) == 0:
                    annos[idx]["segmentation"] = [np.array([0.0] * 6)]

            mask_format = "bitmask" if is_sa1b else "polygon"
            instances = utils.annotations_to_instances(annos, image_shape, mask_format)
            instances.gt_ids = torch.tensor(_gt_ids)
            if len(annos) == 0:
                dataset_dict["instances"].append(instances)
                continue

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if hasattr(instances, 'gt_masks'):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()  # NOTE we don't need boxes
            instances = filter_empty_instances(instances)

            h, w = instances.image_size
            if hasattr(instances, 'gt_masks'):
                if not is_sa1b:
                    gt_masks = instances.gt_masks
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                    instances.gt_masks = gt_masks

            # no classes in sa1b data
            if not instances.has("gt_classes"):
                instances.gt_classes = torch.ones_like(instances.gt_ids) * -1

            dataset_dict["instances"].append(instances)

        if self.is_train:
            # remove empty objects from Instance
            gt_ids_per_video, gt_masks_per_video = [], []
            for f_i, targets_per_frame in enumerate(dataset_dict["instances"]):
                gt_ids_per_video.append(targets_per_frame.gt_ids)
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks = targets_per_frame.gt_masks.tensor
                else:  # polygon
                    gt_masks = targets_per_frame.gt_masks
                gt_masks_per_video.append(gt_masks.sum((-1,-2)) > 0)
            gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)
            gt_masks_per_video = torch.stack(gt_masks_per_video, dim=1)
            valid_idxs = torch.nonzero((gt_ids_per_video >= 0).any(1) & gt_masks_per_video.any(1)).reshape(-1)
            # to speed up training and save memory, there are so many objects in SA1B
            dataset_dict["instances"] = [
                targets_per_frame[valid_idxs] for targets_per_frame in dataset_dict["instances"]
            ]

        # display_pseudo_clip_from_coco(dataset_dict["image"], file_name)
        return dataset_dict


def display_pseudo_clip_from_coco(images_list, file_name, output_path='output/pseudo_clip_from_coco/'):
    imgs = torch.stack(images_list)  # T, 3, H, W
    plt.imshow(rearrange(imgs.cpu().numpy(), 'T C H W -> H (T W) C'))
    plt.axis('off')

    os.makedirs(output_path, exist_ok=True)
    plt.savefig(output_path + file_name.split('/')[-1])
    plt.clf()



class OpenVocabularyCocoPanoClipDatasetMapper:
    """
    A callable which takes a COCO image which converts into multiple frames,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        is_tgt: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        reverse_agu: bool = False,
        src_dataset_name: str = "",
        tgt_dataset_name: str = "",
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        # fmt: off
        self.is_train               = is_train
        self.is_tgt                 = is_tgt
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.reverse_agu            = reverse_agu
        self.sampling_frame_ratio   = 1.0

        self.name = src_dataset_name

        if not is_tgt:
            self.src_metadata = MetadataCatalog.get(src_dataset_name)
            self.tgt_metadata = MetadataCatalog.get(tgt_dataset_name)
            if tgt_dataset_name.startswith("ytvis_2019"):
                src2tgt = COCO_TO_YTVIS_2019
            elif tgt_dataset_name.startswith("ytvis_2021"):
                src2tgt = COCO_TO_YTVIS_2021
            elif tgt_dataset_name.startswith("ovis"):
                src2tgt = COCO_TO_OVIS
            else:
                raise NotImplementedError

            self.src2tgt = {}
            for k, v in src2tgt.items():
                self.src2tgt[
                    self.src_metadata.thing_dataset_id_to_contiguous_id[k]
                ] = self.tgt_metadata.thing_dataset_id_to_contiguous_id[v]

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, is_tgt: bool = True):
        augs = build_pseudo_augmentation(cfg, is_train)
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
        reverse_agu = cfg.INPUT.REVERSE_AGU

        ret = {
            "is_train": is_train,
            "is_tgt": is_tgt,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "reverse_agu": reverse_agu,
            "tgt_dataset_name": cfg.DATASETS.TRAIN[-1],
        }

        return ret

    def select_frames(self, video_length):
        """
        Args:
            video_length (int): length of the video

        Returns:
            selected_idx (list[int]): a list of selected frame indices
        """
        if self.sampling_frame_ratio < 1.0:
            assert self.sampling_frame_num == 1, "only support subsampling for a single frame"
            subsampled_frames = max(int(np.round(video_length * self.sampling_frame_ratio)), 1)
            if subsampled_frames > 1:
                subsampled_idx = np.linspace(0, video_length, num=subsampled_frames, endpoint=False, dtype=int)
                ref_idx = random.randrange(subsampled_frames)
                ref_frame = subsampled_idx[ref_idx]
            else:
                ref_frame = video_length // 2  # middle frame

            selected_idx = [ref_frame]
        else:
            if self.sampling_frame_range * 2 + 1 == self.sampling_frame_num:
                if self.sampling_frame_num > video_length:
                    selected_idx = np.arange(0, video_length)
                    selected_idx_ = np.random.choice(selected_idx, self.sampling_frame_num - len(selected_idx))
                    selected_idx = selected_idx.tolist() + selected_idx_.tolist()
                    sorted(selected_idx)
                else:
                    if video_length == self.sampling_frame_num:
                        start_idx = 0
                    else:
                        start_idx = random.randrange(video_length - self.sampling_frame_num)
                    end_idx = start_idx + self.sampling_frame_num
                    selected_idx = np.arange(start_idx, end_idx).tolist()
                if self.reverse_agu and random.random() < 0.5:
                    selected_idx = selected_idx[::-1]
                return selected_idx

            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame - self.sampling_frame_range)
            end_idx = min(video_length, ref_frame + self.sampling_frame_range + 1)

            if end_idx - start_idx >= self.sampling_frame_num:
                replace = False
            else:
                replace = True
            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame + 1, end_idx))),
                self.sampling_frame_num - 1,
                replace=replace
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)

        return selected_idx

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        file_name = dataset_dict.pop("file_name", None)
        pano_file_name = dataset_dict.pop("pan_seg_file_name", None)
        segments_info = dataset_dict.pop("segments_info", None)
        if pano_file_name is not None:
            pan_seg_gt = utils.read_image(pano_file_name, "RGB")
        original_image = utils.read_image(file_name, format=self.image_format)

        if self.is_train:
            video_length = random.randrange(16, 49)
            selected_idx = self.select_frames(video_length)
        else:
            video_length = self.sampling_frame_num
            selected_idx = range(video_length)

        dataset_dict["dataset_name"] = self.name
        dataset_dict["pano"] = True
        dataset_dict["video_len"] = video_length
        dataset_dict["frame_indices"] = selected_idx
        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = [file_name] * self.sampling_frame_num
        dataset_dict["task"] = "detection"
        dataset_dict["has_mask"] = True
        dataset_dict["has_stuff"] = True
        
        for _ in range(self.sampling_frame_num):
            utils.check_image_size(dataset_dict, original_image)

            aug_input = T.AugInput(original_image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if not self.is_train:
                continue

            _pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
            from panopticapi.utils import rgb2id

            _pan_seg_gt = rgb2id(_pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                # NOTE class_id start from 1 
                class_id = segment_info["category_id"] + 1
                if class_id == 0:
                    import pdb;pdb.set_trace()
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(_pan_seg_gt == segment_info["id"])

            _gt_ids = list(range(len(classes)))
            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            instances.gt_ids = torch.tensor(_gt_ids)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, _pan_seg_gt.shape[-2], _pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

            if not self.is_tgt:
                instances.gt_classes = torch.tensor(
                    [self.src2tgt[c] if c in self.src2tgt else -1 for c in instances.gt_classes.tolist()]
                )
            # instances.gt_boxes = instances.gt_masks.get_bounding_boxes()  # NOTE we don't need boxes
            instances = filter_empty_instances_(instances)
            h, w = instances.image_size
            if hasattr(instances, 'gt_masks'):
                pass
            else:
                instances.gt_masks = torch.zeros((0, h, w), dtype=torch.uint8)
            dataset_dict["instances"].append(instances)
        return dataset_dict

def filter_empty_instances_(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.flatten(1, 2).sum(dim=-1) != 0)

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances







class EntitySegClipDatasetMapper:
    """
    A mapper for EntitySeg dataset that converts single images into pseudo video clips.
    Similar to CocoClipDatasetMapper but adapted for EntitySeg annotation format.
    """
    
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        dataset_name: str = None,
    ):
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.sampling_frame_num = sampling_frame_num
        self.sampling_frame_range = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.dataset_name = dataset_name
        self.fallback_dataset_dict = None

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[EntitySegDatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, dataset_name: str = None):
        """
        Uses the same config format as CocoClipDatasetMapper
        """

        augs = build_pseudo_augmentation(cfg, is_train)
        
        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "sampling_frame_num": cfg.INPUT.SAMPLING_FRAME_NUM,
            "sampling_frame_range": cfg.INPUT.SAMPLING_FRAME_RANGE,
            "sampling_frame_shuffle": cfg.INPUT.SAMPLING_FRAME_SHUFFLE,
            "dataset_name": dataset_name,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Dataset dict with EntitySeg format annotations
        Returns:
            dict: Processed data with pseudo video clip format
        """
        try:
            dataset_dict = copy.deepcopy(dataset_dict)
            img_annos = dataset_dict.pop("annotations", None)
            file_name = dataset_dict.pop("file_name", None)
            original_image = utils.read_image(file_name, format=self.image_format)

            if self.is_train:
                video_length = random.randrange(16, 49)
                ref_frame = random.randrange(video_length)

                start_idx = max(0, ref_frame-self.sampling_frame_range)
                end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

                selected_idx = np.random.choice(
                    np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                    self.sampling_frame_num - 1,
                )
                selected_idx = selected_idx.tolist() + [ref_frame]
                selected_idx = sorted(selected_idx)
            else:
                video_length = self.sampling_frame_num
                selected_idx = list(range(self.sampling_frame_num))

            dataset_dict["has_mask"] = True
            dataset_dict["video_len"] = video_length
            dataset_dict["frame_indices"] = selected_idx
            dataset_dict["image"] = []
            dataset_dict["instances"] = []
            dataset_dict["file_names"] = [file_name] * self.sampling_frame_num
            dataset_dict["dataset_name"] = self.dataset_name or "entityseg"
            dataset_dict["task"] = "detection"
            dataset_dict["has_stuff"] = True

            for i in range(self.sampling_frame_num):

                actual_height, actual_width = original_image.shape[:2]
                dataset_dict["width"] = actual_width
                dataset_dict["height"] = actual_height

                utils.check_image_size(dataset_dict, original_image)

                aug_input = T.AugInput(original_image)
                transforms = self.augmentations(aug_input)
                image = aug_input.image
                image_shape = image.shape[:2]  # h, w

                dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

                if (img_annos is None) or (not self.is_train):
                    continue

                _img_annos = []
                for obj_i, anno in enumerate(img_annos):
                    _anno = {}
                    for k, v in anno.items():
                        _anno[k] = copy.deepcopy(v)
                    # Convert RLE format segmentation to the expected format
                    if isinstance(_anno["segmentation"], dict):
                        _anno["segmentation"] = _anno["segmentation"]
                    _anno["bbox_mode"] = BoxMode.XYWH_ABS
                    _img_annos.append(_anno)

                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in _img_annos
                    if obj.get("iscrowd", 0) == 0
                ]

                _gt_ids = [anno.get("id", idx) for idx, anno in enumerate(annos)]
                for idx in range(len(annos)):
                    if len(annos[idx]["segmentation"]) == 0:
                        annos[idx]["segmentation"] = [np.array([0.0] * 6)]

                instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
                instances.gt_ids = torch.tensor(_gt_ids)

                if hasattr(instances, 'gt_masks'):
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)

                dataset_dict["instances"].append(instances)

            if self.is_train:
                # Remove empty objects from Instance
                gt_ids_per_video = []
                for f_i, targets_per_frame in enumerate(dataset_dict["instances"]):
                    gt_ids_per_video.append(targets_per_frame.gt_ids)
                gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)
                valid_idxs = torch.nonzero((gt_ids_per_video >= 0).any(1)).reshape(-1)

                dataset_dict["instances"] = [
                    targets_per_frame[valid_idxs] for targets_per_frame in dataset_dict["instances"]
                ]

            self.fallback_dataset_dict = dataset_dict
            return dataset_dict
        except:
            return copy.deepcopy(self.fallback_dataset_dict)

