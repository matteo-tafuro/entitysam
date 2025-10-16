import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from panopticapi.utils import IdGenerator, rgb2id
from PIL import Image
from torch.nn import functional as F


def post_process_results_for_vps(
    pred_ious: torch.Tensor,  # [T, N]
    pred_masks: torch.Tensor,  # [N, T, H, W]
    pred_stability_scores: Optional[torch.Tensor],  # [N] or None
    out_size: Tuple[int, int],  # (H, W)
    overlap_threshold: float = 0.7,
    mask_binary_threshold: float = 0.5,
    object_mask_threshold: float = 0.05,
    test_topk_per_image: int = 100,
    num_categories: int = 124,
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
