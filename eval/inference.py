# Copyright (c) Adobe EntitySAM team.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from natsort import natsorted
from panopticapi.utils import IdGenerator, rgb2id
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from torch.nn import functional as F

from sam2.build_sam import build_sam2_video_query_iou_predictor

# VIPSeg uses 124 categories
NUM_CATEGORIES = 124


def save_generated_images(
    images: Iterable[Tuple[str, Image.Image]],
    output_dir: str,
    subdir: Optional[str] = None,
) -> None:
    """
    Save a collection of PIL images to disk.

    Args:
        images: Iterable of (filename, PIL Image) tuples WITH the desired file extension.
        output_dir: Base output directory.
        subdir: Optional subdirectory inside output_dir to save images.
    """
    out_dir = output_dir if subdir is None else os.path.join(output_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    for filename, img in images:
        # Use suffix to choose format (e.g., .jpg -> JPEG, .png -> PNG)
        extension = os.path.splitext(filename)[-1].lower()
        fname = os.path.basename(filename)
        fmt = (
            "PNG"
            if extension == ".png"
            else "JPEG"
            if extension in {".jpg", ".jpeg"}
            else None
        )
        if fmt is None:
            raise ValueError(
                f"Unsupported image extension for '{fname}'. Use .png/.jpg/.jpeg."
            )
        img.save(os.path.join(out_dir, fname), format=fmt)


def generate_entity_visual_prompts(
    panoptic_seg: torch.IntTensor,  # [T,H,W]
    segments_infos: list[dict],  # len N
    frame_dir: str,
    frame_names: list[str],
    crop_padding: int = 10,
    bbox_thickness: int = 2,
) -> List[Tuple[str, Image.Image]]:
    """
    Based on the original paper's supplementary material:
    https://openaccess.thecvf.com/content/CVPR2025/supplemental/Ye_EntitySAM_Segment_Everything_CVPR_2025_supplemental.pdf

    For each entity, generate and save a masked JPG of the frame where it had the highest confidence.
    It does the following:
      1. The red bounding box drawn around the entity.
      2. The rest of the image masked to black (after drawing the bbox).
      3. The final image cropped tightly around the entity mask.

    Args:
        panoptic_seg: torch.IntTensor with shape [T,H,W] containing the panoptic segmentation
            map for each frame. Each pixel value is an entity ID.
        segments_infos: list of dicts, one per entity segment. Each dict has the relevant keys:
            - "id": int, the entity ID matching pixel values in panoptic_seg
            - "best_conf_frame_idx": int, index of the frame where this entity had the highest confidence
        frame_dir: Directory containing the original video frames.
        frame_names: List of frame file names (str) in the video, in order.
        crop_padding: Number of pixels to pad around the entity mask when cropping.
        bbox_thickness: Thickness of the bounding box to draw around the entity.

    Returns:
        A list of (filename, PIL Image) tuples, one per entity
    """

    # Some checks
    assert len(frame_names) == panoptic_seg.shape[0], (
        f"Number of frame names ({len(frame_names)}) does not match number of frames "
        f"in panoptic_seg ({panoptic_seg.shape[0]})."
    )

    pan_np = panoptic_seg.cpu().numpy()  # [T,H,W]

    # Output dictionary that will contain (filename, PIL Image) tuples
    outputs = []
    for seg in segments_infos:
        seg_id = int(seg["id"])
        t = int(seg["best_conf_frame_idx"])

        mask = pan_np[t] == seg_id
        if not mask.any():
            continue

        # Extract boundaries
        bb_ys, bb_xs = np.where(mask)
        bb_y1, bb_y2 = int(bb_ys.min()), int(bb_ys.max())
        bb_x1, bb_x2 = int(bb_xs.min()), int(bb_xs.max())

        # Load frame and draw bbox
        frame_path = os.path.join(frame_dir, frame_names[t])
        with Image.open(frame_path) as im0:
            im0 = im0.convert("RGB")
            draw = ImageDraw.Draw(im0)
            draw.rectangle(
                [bb_x1, bb_y1, bb_x2 + 1, bb_y2 + 1],
                outline=(255, 0, 0),
                width=bbox_thickness,
            )

            # Slightly dilate the binary mask for masking
            dilated_mask = binary_dilation(mask, iterations=10).astype(mask.dtype)
            arr = np.array(im0)
            arr[~dilated_mask] = 0

            # For cropping, use the dilated bbox and add some padding (beware of im bounds)
            mask_ys, mask_xs = np.where(dilated_mask)
            mask_y1 = max(0, int(mask_ys.min()) - crop_padding)
            mask_y2 = min(im0.height - 1, int(mask_ys.max()) + crop_padding)
            mask_x1 = max(0, int(mask_xs.min()) - crop_padding)
            mask_x2 = min(im0.width - 1, int(mask_xs.max()) + crop_padding)
            im_cropped = Image.fromarray(arr).crop(
                (mask_x1, mask_y1, mask_x2 + 1, mask_y2 + 1)
            )

        save_name = f"entity_{seg_id:04d}_frame{t:05d}.png"
        outputs.append((save_name, im_cropped))

    return outputs


def post_process_results_for_vps(
    pred_ious: torch.Tensor,  # [T, N]
    pred_masks: torch.Tensor,  # [N, T, H, W]
    pred_stability_scores: Optional[torch.Tensor],  # [N] or None
    out_size: Tuple[int, int],  # (H, W)
    overlap_threshold: float = 0.7,
    mask_binary_threshold: float = 0.5,
    object_mask_threshold: float = 0.05,
    test_topk_per_image: int = 100,
    num_categories: int = NUM_CATEGORIES,
) -> Dict[str, object]:
    """
    Convert per-frame, per-candidate mask predictions into panoptic video segments.

    This function performs three main steps:
      1. Selects a best frame per candidate from `pred_ious` (per-frame IoU matrix).
      2. Aggregates per-candidate scores (mean over time), applies an absolute threshold
         and a top-K cap, and optionally incorporates stability scores to re-weight candidates.
      3. Runs pixel-wise competition (weighted by candidate scores) to produce a panoptic
         segmentation map for each frame and filters segments using spatio-temporal overlap criteria.

    Args:
        pred_ious: Float tensor with shape [T, N]. T is the number of frames (after removing any
            warmup/padding), N is the number of candidates.
        pred_masks: Float tensor with shape [N, T, H, W] containing raw mask logits for each
            candidate/time. These logits are sigmoid-ed and interpolated to `out_size` inside the function.
        pred_stability_scores: Optional 1D tensor [N] with an independently estimated mask quality/stability.
            If None, a fallback estimator is called on a temporal subsample.
        out_size: Tuple (height, width) of the output panoptic maps.
        overlap_threshold: Fraction in (0,1]. A candidate is discarded if the fraction of its original mask
            area that remains assigned after competition is less than this threshold.
        mask_binary_threshold: Threshold (0-1) applied to sigmoid(logits) to compute original mask
            area and intersections.
        object_mask_threshold: Absolute floor for average candidate score.
        test_topk_per_image: Cap on number of candidates to retain by score.
        num_categories: Number of categories used for randomized visualization labels (class-agnostic
            behavior uses random assignment in this demo/training script).

    Returns:
        A dictionary with the following keys:
          - "image_size": Tuple[int,int] (height, width)
          - "pred_masks": Tensor [T, H, W], A 2D map per frame where each pixel value is an entity ID
          - "segments_infos": list of dicts, one per surviving segment
          - "pred_ids": list[int] of original candidate ids surviving
          - "pred_best_frames": list[int] of best frame indices for each surviving candidate
          - "task": str, fixed to "vps"
    """

    # --- (1) get per-entity best frame index from pred_ious ---
    best_frame_idx = pred_ious.argmax(dim=0)  # [N]
    best_frame_score = pred_ious.max(dim=0).values

    # --- (2) keep the top-k scoring masks above a threshold ---
    average_scores = pred_ious.mean(dim=0)  # [N]

    # random label as class-agnostic for visualization
    labels = torch.randint(0, num_categories, (len(average_scores),)).to(
        pred_ious.device
    )
    pred_id = torch.arange(len(average_scores), device=pred_ious.device)

    keep = average_scores >= max(
        object_mask_threshold,
        average_scores.topk(k=min(len(average_scores), test_topk_per_image))[0][-1],
    )

    cur_scores = average_scores[keep]  # [N_kept]
    cur_classes = labels[keep]  # [N_kept]
    cur_masks = pred_masks[keep]  # [N_kept, T, H, W]
    cur_ids = pred_id[keep]  # [N_kept]
    cur_best_frame_idx = best_frame_idx[keep]  # [N_kept]
    cur_best_frame_score = best_frame_score[keep]  # [N_kept]

    # Use stability scores if provided
    if pred_stability_scores is not None:
        mask_quality_scores = pred_stability_scores[keep]
    else:
        from train.utils.comm import calculate_mask_quality_scores

        mask_quality_scores = calculate_mask_quality_scores(cur_masks[:, ::5])

    del pred_masks

    # --- (3) conversion to panoptic ---
    panoptic_seg = torch.zeros(
        (cur_masks.size(1), out_size[0], out_size[1]),
        dtype=torch.int32,
        device=cur_masks.device,
    )
    segments_infos = []
    out_ids = []
    out_best_frames = []
    current_segment_id = 0

    if cur_masks.shape[0] == 0:
        return {
            "image_size": out_size,
            "pred_masks": panoptic_seg.cpu(),
            "segments_infos": segments_infos,
            "pred_ids": out_ids,
            "pred_best_frames": out_best_frames,
            "task": "vps",
        }
    else:
        cur_scores = cur_scores + 0.5 * mask_quality_scores

        cur_masks = F.interpolate(cur_masks, size=out_size, mode="bilinear")
        cur_masks = cur_masks.sigmoid()
        is_bg = (cur_masks < mask_binary_threshold).sum(0) == len(cur_masks)
        cur_prob_masks = cur_scores.view(-1, 1, 1, 1).to(cur_masks.device) * cur_masks

        cur_mask_ids = cur_prob_masks.argmax(0)  # (T, H, W)
        cur_mask_ids[is_bg] = -1
        del cur_prob_masks, is_bg

        for k in range(cur_classes.shape[0]):  # N_kept
            cur_masks_k = cur_masks[k].squeeze(0)  # (T, H, W)

            pred_class = int(cur_classes[k]) + 1  # start from 1
            isthing = True  # class-agnostic entities

            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks_k >= mask_binary_threshold).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks_k >= mask_binary_threshold)

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < overlap_threshold:
                    continue

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                # Attach the peak frame info for this surviving entity
                peak_t = int(cur_best_frame_idx[k])
                peak_score = float(cur_best_frame_score[k])

                segments_infos.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": pred_class,
                        "best_conf_frame_idx": peak_t,
                        "best_conf_score": peak_score,
                    }
                )
                out_ids.append(int(cur_ids[k]))
                out_best_frames.append(peak_t)

        del cur_masks

        return {
            "image_size": out_size,
            "pred_masks": panoptic_seg.cpu(),
            "segments_infos": segments_infos,
            "pred_ids": out_ids,
            "pred_best_frames": out_best_frames,
            "task": "vps",
        }


def build_panoptic_frames_and_annotations(
    video_id: str,
    frame_names: List[str],
    panoptic_outputs: Dict[str, Any],
    categories_by_id: Dict[int, Dict[str, Any]],
) -> Tuple[List[Tuple[str, Image.Image]], Dict[str, Any]]:
    """
    Generates a colorized PNG for each frame (one PNG per input frame) using the panoptic
    segmentation map and a deterministic color generator keyed by `categories_by_id`.
    For each frame, it also builds a COCO-style list of segment annotations with
    bbox and area, aligned with VIPSeg conventions.

    Args:
        video_id: String identifier for the video.
        frame_names: List of input frame file names (str) in the video, in order.
        panoptic_outputs: Dict with keys:
            - "image_size": Tuple[int,int] (height, width)
            - "pred_masks": A 2D map per frame where each pixel value is an entity ID
            - "segments_infos": list of dicts, one per entity from `post_process_results_for_vps`.
        categories_by_id: Dict mapping category_id (int) to dict with keys:
            - "id": int, category ID
            - "isthing": int, 1 if thing, 0 if stuff
            - "color": List[int,int,int], RGB color for visualization

    Returns:
        A tuple (images, annotations) where:
            images: List of (png_filename, PIL Image) tuples for each frame.
            annotations: Dict with keys {"annotations": List[per-frame dict], "video_id": str}.
    """
    H, W = panoptic_outputs["image_size"]  # (H, W)
    pan_seg_result = panoptic_outputs["pred_masks"]  # [T,H,W]
    segments_infos = panoptic_outputs["segments_infos"]

    T = pan_seg_result.shape[0]
    # Sanity check
    assert len(frame_names) == T, (
        f"Number of frame names ({len(frame_names)}) does not match number "
        f"of frames in pan_seg_result ({T})."
    )

    color_generator = IdGenerator(categories_by_id)

    # Pre-initialize result array [T,H,W,3]
    panoptic_rgb = np.zeros((T, H, W, 3), dtype=np.uint8)

    # ---- Build per-frame annotations for VIPSeg/COCO format ----

    # List of length N (num segments) where each item is a list of length T (num frames).
    # Each inner list contains either None (if the segment is not present in that frame)
    # or a dict with keys: category_id, iscrowd, id, bbox, area
    per_segment_per_frame_ann = []
    for seg_info in segments_infos:
        entity_id = seg_info["id"]
        category_id = seg_info["category_id"]

        entity_mask = (
            (pan_seg_result == entity_id).cpu().numpy()
        )  # Binary mask over time for this entity ID, still [T,H,W]

        color = color_generator.get_color(category_id)
        panoptic_rgb[entity_mask] = color

        # VIPSeg/COCO format template
        base_ann = {
            "category_id": int(category_id) - 1,  # 0-indexed
            "iscrowd": 0,
            "id": int(rgb2id(color)),
        }
        # Build per-frame annotation list: bbox + area, or None if not on frame
        per_frame_ann = []
        for t in range(T):
            area = int(entity_mask[t].sum())
            if area == 0:
                per_frame_ann.append(None)
                continue

            # Compute tight bbox (x, y, w, h) for the mask on this frame
            ys, xs = np.where(entity_mask[t])
            x0, y0 = int(xs.min()), int(ys.min())
            w = int(xs.max() - x0)
            h = int(ys.max() - y0)

            ann = {
                "bbox": [x0, y0, w, h],
                "area": area,
            }
            ann.update(base_ann)
            per_frame_ann.append(ann)

        per_segment_per_frame_ann.append(per_frame_ann)

    image_outputs = []
    per_frame_outputs = []
    for t, image_name in enumerate(frame_names):
        pil_img = Image.fromarray(panoptic_rgb[t])
        save_name = image_name.split("/")[-1].split(".")[0] + ".png"
        image_outputs.append((save_name, pil_img))

        # Collect this frame's annotations by taking the t-th element for each segment
        frame_segments = [
            seg_ann[t]
            for seg_ann in per_segment_per_frame_ann
            if seg_ann[t] is not None
        ]
        per_frame_outputs.append(
            {
                # If None, the segment is not present in this frame
                "segments_info": frame_segments,
                "file_name": os.path.basename(image_name),
            }
        )

    del panoptic_outputs

    annotations = {"annotations": per_frame_outputs, "video_id": video_id}
    return image_outputs, annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model with a specified checkpoint directory on a specified video."
    )
    parser.add_argument(
        "--video_frames_dir",
        type=str,
        required=True,
        help="Directory containing video frames",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="Checkpoint directory name"
    )
    # Config file is taken from config search path, which includes /sam2
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="configs/sam2.1_hiera_l.yaml",
        help="Base SAM 2 SModel config file",
    )
    parser.add_argument(
        "--mask_decoder_depth", type=int, default=8, help="Mask decoder depth"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--save_images",
        dest="save_images",
        action="store_true",
        help="Save generated images.",
    )
    group.add_argument(
        "--no_save_images",
        dest="save_images",
        action="store_false",
        help="Do not save generated images.",
    )
    parser.set_defaults(save_images=True)
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

    # Use video directory name as video_id
    video_dir = os.path.dirname(args.video_frames_dir)
    video_id = os.path.basename(os.path.normpath(video_dir))

    ckpt_dir = args.ckpt_dir
    output_dir = os.path.join(
        args.output_dir, f"{os.path.basename(os.path.normpath(ckpt_dir))}_{video_id}"
    )

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

    # Prepare the inference state with padding frames
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
    pred_eious = torch.stack(pred_eious)

    pred_stability_scores = None

    # Remove padding frames
    pred_masks = pred_masks[:, ~torch.tensor(padding_mask, device=pred_masks.device)]
    pred_eious = pred_eious[~torch.tensor(padding_mask, device=pred_eious.device)]

    result_i = post_process_results_for_vps(
        pred_ious=pred_eious,
        pred_masks=pred_masks,
        out_size=out_size,
        pred_stability_scores=pred_stability_scores,
    )

    visual_prompt_images = generate_entity_visual_prompts(
        panoptic_seg=result_i["pred_masks"],
        segments_infos=result_i["segments_infos"],
        frame_dir=video_dir,
        frame_names=frame_names,
    )

    panoptic_images, anno = build_panoptic_frames_and_annotations(
        video_id, frame_names, result_i, categories_dict
    )
    predictions.append(anno)

    # After processing the video
    end_time = time.time()
    processing_time = end_time - start_time

    # Update running statistics
    total_frames += len(frame_names)
    total_time += processing_time
    current_peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    peak_memory = max(peak_memory, current_peak_memory)
    final_gpu_mem = torch.cuda.memory_allocated() / 1024**3

    print(f"\nVideo: {video_id}")
    print("Performance Metrics:")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Running Peak GPU Memory: {peak_memory:.2f} GB")
    print(f"Current final GPU Memory: {final_gpu_mem:.2f} GB")

    # Save images if needed
    if args.save_images:
        save_generated_images(
            visual_prompt_images, output_dir=output_dir, subdir="entity_prompts"
        )
        save_generated_images(
            panoptic_images, output_dir=output_dir, subdir="panoptic_images"
        )
        print(f"Saved images to {output_dir}")
    else:
        print("Skipping image saves. Set --save_images to enable.")

    # Save JSON annotations
    file_path = os.path.join(output_dir, "panoptic_preds.json")
    with open(file_path, "w") as f:
        json.dump({"annotations": predictions}, f)
