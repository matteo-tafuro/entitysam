import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import register_coco_instances

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
)
from .burst import _get_burst_meta
from . import coco_panoptic_video_ov




# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train_sub.json"),  # 90% videos in training set
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid21.json"),
    "ytvis_2022_val": ("ytvis_2021/valid22/JPEGImages",
                       "ytvis_2021/valid22.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
    "ytvis_2021_dev0.01": ("ytvis_2021/train/JPEGImages",
                       "ytvis_2021/valid_sub_0.01.json"),  # 10% videos in training set
    "ytvis_2021_dev": ("ytvis_2021/train/JPEGImages",
                       "ytvis_2021/valid_sub.json"),  # 10% videos in training set
    "ytvis_2021_dev_merge": ("ytvis_2021/train/JPEGImages",
                             "ytvis_2021/valid_sub_merge_car_truck.json"),
    
}


# ====    Predefined splits for OVIS    ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train/JPEGImages",
                   "ovis/train_sub.json"),  # 90% videos in training set
    "ovis_dev": ("ovis/train/JPEGImages",
                 "ovis/valid_sub.json"),  # 10% videos in training set
    "ovis_dev0.01": ("ovis/train/JPEGImages",
                 "ovis/valid_sub_0.01.json"),  # 10% videos in training set
    "ovis_dev_merge": ("ovis/train/JPEGImages",
                       "ovis/valid_sub_merge_motorbike.json"),
    "ovis_val": ("ovis/valid/JPEGImages",
                 "ovis/valid.json"),
    "ovis_test": ("ovis/test/JPEGImages",
                  "ovis/test.json"),
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            evaluator_type='ytvis',
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            evaluator_type='ytvis',
        )




_PREDEFINED_SPLITS_SOT = {
    "sot_ytbvos18_train": ("ytbvos/train/JPEGImages", "ytbvos/train.json", "vos"),
    "sot_ytbvos18_val": ("ytbvos/valid/JPEGImages", "ytbvos/valid.json", "vos"),
    "sot_davis16_train": ("DAVIS/JPEGImages/Full-Resolution", "DAVIS/2016_train.json", "davis"),
    "sot_davis16_val": ("DAVIS/JPEGImages/Full-Resolution", "DAVIS/2016_val.json", "davis"),
    "sot_davis17_train": ("DAVIS/JPEGImages/Full-Resolution", "DAVIS/2017_train.json", "davis"),
    "sot_davis17_val": ("DAVIS/JPEGImages/Full-Resolution", "DAVIS/2017_val.json", "davis"),
}

# only one class for visual grounding
SOT_CATEGORIES = [{"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "object"}]


def _get_sot_meta():
    thing_ids = [k["id"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 1, len(thing_ids)

    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_all_sot(root):
    for key, (image_root, json_file, evaluator_type) in _PREDEFINED_SPLITS_SOT.items():
        has_mask = ("coco" in key) or ("vos" in key) or ("davis" in key)
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_sot_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=has_mask,
            sot=True,
            evaluator_type=evaluator_type, # "vos" or "davis"
        )

# ====    Predefined splits for MOSE    ===========
_PREDEFINED_SPLITS_MOSE = {
    "mots_mose_train": ("mose/train/JPEGImages", "mose/train/train.json"),
    "mots_mose_val": ("mose/valid/JPEGImages", "mose/valid/valid.json"),
    "mots_mose_dev": ("mose/valid/JPEGImages", "mose/valid/valid_sub.json"),
    "mots_mose_test": ("mose/test/JPEGImages", "mose/test/test.json"),
}

def register_all_mose(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_MOSE.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_sot_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=True,
            sot=True,
            evaluator_type="vos",
        )


# ==== Predefined splits for TAO/BURST  ===========
_PREDEFINED_SPLITS_BURST = {
    "mots_burst_train": ("burst/frames/train",
                         "burst/annotations/train_uni.json",
                         "vos"),
    "mots_burst_val_vos": ("burst/frames/val",
                          "burst/annotations/val_first_frame_uni.json",
                          "vos"),   # point, box, mask as visual prompts
    "mots_burst_val_det": ("burst/frames/val",
                           "burst/annotations/val_first_frame_uni.json",
                           "ytvis"), # category-guided common segmentation
}

def register_all_burst(root):
    for key, (image_root, json_file, evaluator) in _PREDEFINED_SPLITS_BURST.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_burst_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=True,
            sot=evaluator=='vos',
            evaluator_type=evaluator,
        )


# ====    Predefined splits for Referring YTVIS    ===========
_PREDEFINED_SPLITS_REFYTBVOS = {
    "rvos-refytb-train": ("ytbvos/train/JPEGImages", "ytbvos/train_ref.json", "vos"),
    "rvos-refytb-val": ("ytbvos/valid19/JPEGImages", "ytbvos/valid19_ref.json", "vos"),
    # unsupervised
    "rvos-refdavis-val-0": ("ref-davis/valid/JPEGImages", "ref-davis/valid_0.json", "davis"),
    "rvos-refdavis-val-1": ("ref-davis/valid/JPEGImages", "ref-davis/valid_1.json", "davis"),
    "rvos-refdavis-val-2": ("ref-davis/valid/JPEGImages", "ref-davis/valid_2.json", "davis"),
    "rvos-refdavis-val-3": ("ref-davis/valid/JPEGImages", "ref-davis/valid_3.json", "davis"),
}

def register_all_refytbvos_videos(root):
    for key, (image_root, json_file, evaluator_type) in _PREDEFINED_SPLITS_REFYTBVOS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_sot_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=True,
            has_expression=True,
            evaluator_type=evaluator_type,  # "vos"
        )

_PREDEFINED_SPLITS_RAW_VIDEOS_TEST = {
    # dataset_name: (video_root, annotation_file), evaluator_type
    "custom_videos": ("custom_videos/raw/", "custom_videos/raw/test.json", "none"),
    "custom_videos_text": ("custom_videos/raw_text/", "custom_videos/raw_text/test.json", "none"),
    "internvid-flt-1": ("internvid/raw/InternVId-FLT_1", "internvid/raw/InternVId-FLT_1.json", "none"),
}

def register_raw_videos(root):
    for key, (video_root, json_file, evaluator_type) in _PREDEFINED_SPLITS_RAW_VIDEOS_TEST.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_sot_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, video_root),
            has_mask=False,
            evaluator_type=evaluator_type, # "vos" or "davis"
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_burst(_root)

