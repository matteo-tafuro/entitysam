# Copyright (c) Adobe EntitySAM team.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import time

import numpy as np
import torch
from natsort import natsorted
from panopticapi.utils import IdGenerator, rgb2id
from PIL import Image
from torch.nn import functional as F

from sam2.build_sam import build_sam2_video_query_iou_predictor


def inference_video_vps_save_results(
    pred_ious,
    pred_stability_scores,
    pred_masks,
    out_size,
    overlap_threshold=0.7,
    mask_binary_threshold=0.5,
    object_mask_threshold=0.05,
    test_topk_per_image=100,
):
    """
    Post process of multi-mask into panoptic seg
    """

    scores = pred_ious
    # random label as class-agnostic for visualization
    labels = torch.randint(0, 123, (len(scores),)).to(pred_ious.device)
    pred_id = torch.arange(len(scores), device=pred_ious.device)

    keep = scores >= max(
        object_mask_threshold,
        scores.topk(k=min(len(scores), test_topk_per_image))[0][-1],
    )
    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_masks = pred_masks[keep]
    cur_ids = pred_id[keep]
    if pred_stability_scores:
        mask_quality_scores = pred_stability_scores[keep]
    else:
        from train.utils.comm import calculate_mask_quality_scores

        mask_quality_scores = calculate_mask_quality_scores(cur_masks[:, ::5])
    del pred_masks

    # initial panoptic_seg and segments infos
    h, w = cur_masks.shape[-2:]
    panoptic_seg = torch.zeros(
        (cur_masks.size(1), out_size[0], out_size[1]),
        dtype=torch.int32,
        device=cur_masks.device,
    )
    segments_infos = []
    out_ids = []
    current_segment_id = 0

    if cur_masks.shape[0] == 0:
        # We didn't detect any mask
        return {
            "image_size": out_size,
            "pred_masks": panoptic_seg.cpu(),
            "segments_infos": segments_infos,
            "pred_ids": out_ids,
            "task": "vps",
        }
    else:
        cur_scores = cur_scores + 0.5 * mask_quality_scores

        cur_masks = F.interpolate(
            cur_masks, size=out_size, mode="bilinear", align_corners=False
        )
        cur_masks = cur_masks.sigmoid()

        is_bg = (cur_masks < mask_binary_threshold).sum(0) == len(cur_masks)
        cur_prob_masks = cur_scores.view(-1, 1, 1, 1).to(cur_masks.device) * cur_masks
        cur_mask_ids = cur_prob_masks.argmax(0)  # (t, h, w)
        cur_mask_ids[is_bg] = -1

        del cur_prob_masks
        del is_bg

        stuff_memory_list = {}
        for k in range(cur_classes.shape[0]):
            cur_masks_k = cur_masks[k].squeeze(0)

            pred_class = int(cur_classes[k]) + 1  # start from 1
            # assume all are entities for class-agnostic
            isthing = True

            # filter out the unstable segmentation results
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks_k >= mask_binary_threshold).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks_k >= mask_binary_threshold)
            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < overlap_threshold:
                    continue

                # merge stuff regions
                if not isthing:
                    if pred_class in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[pred_class]
                        continue
                    else:
                        stuff_memory_list[pred_class] = current_segment_id + 1
                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                segments_infos.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": pred_class,
                    }
                )
                out_ids.append(cur_ids[k])

        del cur_masks

        return {
            "image_size": out_size,
            "pred_masks": panoptic_seg.cpu(),
            "segments_infos": segments_infos,
            "pred_ids": out_ids,
            "task": "vps",
        }


def process(video_id, frame_names, outputs, categories_dict, output_dir, video_dir):
    """
    save panoptic segmentation result as an image
    """
    color_generator = IdGenerator(categories_dict)

    img_shape = outputs["image_size"]
    pan_seg_result = outputs["pred_masks"]
    segments_infos = outputs["segments_infos"]
    segments_infos_ = []

    pan_format = np.zeros(
        (pan_seg_result.shape[0], img_shape[0], img_shape[1], 3), dtype=np.uint8
    )
    for segments_info in segments_infos:
        id = segments_info["id"]
        is_thing = segments_info["isthing"]
        sem = segments_info["category_id"]

        mask = pan_seg_result == id
        color = color_generator.get_color(sem)
        pan_format[mask] = color

        dts = []
        dt_ = {"category_id": int(sem) - 1, "iscrowd": 0, "id": int(rgb2id(color))}
        for i in range(pan_format.shape[0]):
            area = mask[i].sum()
            index = np.where(mask[i].numpy())
            if len(index[0]) == 0:
                dts.append(None)
            else:
                if area == 0:
                    dts.append(None)
                else:
                    x = index[1].min()
                    y = index[0].min()
                    width = index[1].max() - x
                    height = index[0].max() - y
                    dt = {
                        "bbox": [x.item(), y.item(), width.item(), height.item()],
                        "area": int(area),
                    }
                    dt.update(dt_)
                    dts.append(dt)
        segments_infos_.append(dts)

    #### save image
    annotations = []
    for i, image_name in enumerate(frame_names):
        image_ = Image.fromarray(pan_format[i])
        if not os.path.exists(os.path.join(output_dir, "pan_pred", video_id)):
            os.makedirs(os.path.join(output_dir, "pan_pred", video_id))
        image_.save(
            os.path.join(
                output_dir,
                "pan_pred",
                video_id,
                image_name.split("/")[-1].split(".")[0] + ".png",
            )
        )
        annotations.append(
            {
                "segments_info": [
                    item[i] for item in segments_infos_ if item[i] is not None
                ],
                "file_name": image_name.split("/")[-1],
            }
        )

    del outputs
    return {"annotations": annotations, "video_id": video_id}


if __name__ == "__main__":
    # VIPSeg uses 124 categories
    NUM_CATEGORIES = 124

    parser = argparse.ArgumentParser(
        description="Run the model with a specified checkpoint directory on a specified video."
    )
    parser.add_argument(
        "--video_frames_dir", type=str, required=True, help="Directory containing video frames"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="Checkpoint directory name"
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="configs/sam2.1_hiera_l.yaml",
        help="Base SAM 2 SModel config file",
    )
    parser.add_argument(
        "--mask_decoder_depth", type=int, default=8, help="Mask decoder depth"
    )
    args = parser.parse_args()

    # Log args
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("\n")

    # Validate input paths
    assert os.path.exists(args.video_frames_dir), (
        f"Video frames directory {args.video_frames_dir} does not exist."
    )
    assert os.path.exists(args.ckpt_dir), (
        f"Checkpoint directory {args.ckpt_dir} does not exist."
    )

    # Config file is taken from config search path, which includes /sam2
    # assert os.path.exists(args.model_cfg), f"Model config file {args.model_cfg} does not exist."

    # Use video directory name as video_id
    video_dir = os.path.dirname(args.video_frames_dir)
    video_id = os.path.basename(os.path.normpath(video_dir))

    ckpt_dir = args.ckpt_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_checkpoint = os.path.join(ckpt_dir, "model_0009999.pth")

    model_cfg = args.model_cfg
    predictor = build_sam2_video_query_iou_predictor(
        model_cfg, sam2_checkpoint, mask_decoder_depth=args.mask_decoder_depth
    )

    categories_dict = dict()
    for cat_id in range(1, NUM_CATEGORIES + 1):
        tmp_cat = {
            "id": cat_id,
            "isthing": 1,
            "color": [
                np.random.randint(0, 256),
                np.random.randint(0, 256),
                np.random.randint(0, 256),
            ],
        }
        categories_dict[cat_id] = tmp_cat

    predictions = []

    total_frames = 0
    total_time = 0
    peak_memory = 0

    ## ================ Per-video loop started here
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    # NOTE: Implemented natural sorting since our file names are 'frame_0001.jpg', ..., 'frame_0010.jpg', etc.
    frame_names = natsorted(
        [
            p
            for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
    )

    progress_order = [0, 0, 0] + list(range(0, len(frame_names)))
    padding_mask = [True] * 3 + [False] * len(frame_names)

    inference_state = predictor.init_state(
        video_path=video_dir, progress_order=progress_order
    )
    predictor.reset_state(inference_state)

    frame_idx = 0
    init_frame = Image.open(os.path.join(video_dir, frame_names[frame_idx]))
    init_frame = np.array(init_frame.convert("RGB"))
    out_size = (init_frame.shape[0], init_frame.shape[1])

    # run propagation throughout the video and collect the results in a dict
    pred_masks_list = []
    pred_eious = []
    for out_frame_idx, _, out_mask_logits, pred_eiou in predictor.propagate_in_video(
        inference_state, start_frame_idx=0
    ):
        pred_masks_list.append(out_mask_logits)
        pred_eious.append(pred_eiou)

    pred_masks = torch.cat(pred_masks_list, dim=1)
    pred_eious = torch.stack(pred_eious).mean(0)
    pred_stability_scores = None

    pred_masks = pred_masks[:, ~torch.tensor(padding_mask, device=pred_masks.device)]

    result_i = inference_video_vps_save_results(
        pred_eious, pred_stability_scores, pred_masks, out_size
    )
    anno = process(
        video_id, frame_names, result_i, categories_dict, output_dir, video_dir
    )
    predictions.append(anno)

    # After processing each video
    end_time = time.time()
    processing_time = end_time - start_time

    # Update running statistics
    total_frames += len(frame_names)
    total_time += processing_time
    current_peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    peak_memory = max(peak_memory, current_peak_memory)
    final_gpu_mem = torch.cuda.memory_allocated() / 1024**3

    print(f"\nVideo {video_id} Performance Metrics:")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Running Peak GPU Memory: {peak_memory:.2f} GB")
    print(f"Current final GPU Memory: {final_gpu_mem:.2f} GB")

    file_path = os.path.join(output_dir, "pred.json")
    with open(file_path, "w") as f:
        json.dump({"annotations": predictions}, f)
