import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from panopticapi.utils import IdGenerator
from PIL import Image
from torch.nn import functional as F


def post_process_results_for_vps(
    pred_ious: torch.Tensor,  # [1, N]
    pred_masks: torch.Tensor,  # [N, 1, H, W]
    out_size: Tuple[int, int],  # (H, W)
    overlap_threshold: float = 0.25,
    mask_binary_threshold: float = 0.025,
    query_to_category_map: Dict[int, int] = {},
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
        out_size: Tuple (height, width) of the output panoptic maps.
        overlap_threshold: Fraction in (0,1]. A candidate is discarded if the fraction of its original mask
            area that remains assigned after competition is less than this threshold.
        mask_binary_threshold: Threshold (0-1) applied to sigmoid(logits) to compute original mask
            area and intersections.
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

    # Entity-wise scores for the current frame
    frame_scores = pred_ious.squeeze(0)  # [N]

    # Map query ID (1..N) to category ID
    query_ids = torch.arange(frame_scores.size(0), device=frame_scores.device)
    category_ids = torch.tensor(
        [query_to_category_map[int(q.item())] for q in query_ids],
        device=frame_scores.device,
    )

    panoptic_seg = torch.zeros(
        (pred_masks.size(1), out_size[0], out_size[1]),
        dtype=torch.int32,
        device=pred_masks.device,
    )
    segments_infos = []
    out_ids = []
    out_best_frames = []
    current_segment_id = 0
    pred_masks = F.interpolate(pred_masks, size=out_size, mode="bilinear")
    pred_masks = pred_masks.sigmoid()
    is_bg = (pred_masks < mask_binary_threshold).sum(0) == len(pred_masks)
    cur_prob_masks = frame_scores.view(-1, 1, 1, 1).to(pred_masks.device) * pred_masks

    cur_mask_ids = cur_prob_masks.argmax(0)  # (1, H, W)
    cur_mask_ids[is_bg] = -1
    del cur_prob_masks, is_bg

    for k in range(category_ids.shape[0]):  # N
        cur_masks_k = pred_masks[k].squeeze(0)  # (1, H, W)

        cat_id = int(category_ids[k])
        isthing = True  # class-agnostic entities

        mask_area = (cur_mask_ids == k).sum().item()
        original_area = (cur_masks_k >= mask_binary_threshold).sum().item()
        mask = (cur_mask_ids == k) & (cur_masks_k >= mask_binary_threshold)

        if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
            if mask_area / original_area < overlap_threshold:
                current_segment_id += 1
                continue

            current_segment_id += 1
            panoptic_seg[mask] = current_segment_id

            segments_infos.append(
                {
                    "id": current_segment_id,
                    "isthing": bool(isthing),
                    "category_id": cat_id,
                    "score": frame_scores[k].item(),
                }
            )
            out_ids.append(int(query_ids[k]))

    del pred_masks

    return {
        "image_size": out_size,
        "pred_masks": panoptic_seg.cpu(),
        "segments_infos": segments_infos,
        "pred_ids": out_ids,
        "pred_best_frames": out_best_frames,
        "task": "vps",
    }


def build_panoptic_frame_and_annotations(
    video_id: str,
    frame_name: str,
    panoptic_outputs: Dict[str, Any],
    categories_by_id: Dict[int, Dict[str, Any]],
) -> Tuple[Tuple[str, Image.Image], Dict[str, Any]]:
    """
    Generate a colorized PNG for a single frame from a panoptic ID map and
    build a COCO/VIPSeg-style annotation dict for that frame.

    Args:
        video_id: Identifier for the video this frame belongs to.
        frame_name: Path or filename of the input frame.
        panoptic_outputs: Dict with keys:
            - "image_size": (height, width)
            - "pred_masks": Per-pixel entity IDs for the frame; shape (H, W) or (1, H, W).
            - "segments_infos": List of dicts from post-processing; each has keys like
                {"id": int, "category_id": int, ...}.
        categories_by_id: Mapping category_id -> {"id": int, "isthing": int, "color": [R,G,B]}.

    Returns:
        A tuple (image_output, annotations) where:
            image_output: (png_filename, PIL.Image) for this frame.
            annotations: {
                "annotations": {
                    "segments_info": List[dict],
                    "file_name": str
                },
                "video_id": str
            }.
    """
    height, width = panoptic_outputs["image_size"]
    seg_id_map = panoptic_outputs["pred_masks"]  # [1, H, W]
    seg_id_map = seg_id_map.squeeze(0).detach().cpu().numpy()  # [H, W]
    segments_info_list = panoptic_outputs["segments_infos"]

    # Colorize
    id_color_gen = IdGenerator(categories_by_id)
    panoptic_rgb = np.zeros((height, width, 3), dtype=np.uint8)

    frame_segments: list[dict] = []
    # Each item in segments_info_list contains either None (if the segment is not
    # present in that frame) or a dict with keys: category_id, iscrowd, id, bbox, area
    for seg_info in segments_info_list:
        if seg_info is None:
            continue

        entity_id = int(seg_info["id"])
        category_id = int(seg_info["category_id"])

        entity_mask = seg_id_map == entity_id
        area = int(entity_mask.sum())
        if area == 0:
            continue

        color = id_color_gen.get_color(category_id)
        panoptic_rgb[entity_mask] = color

        ys, xs = np.where(entity_mask)
        x0, y0 = int(xs.min()), int(ys.min())
        w = int(xs.max() - x0)
        h = int(ys.max() - y0)

        frame_segments.append(
            {
                "entity_id": entity_id,
                "category_id": category_id,
                "bbox": [x0, y0, w, h],
                "area": area,
            }
        )

    png_filename = os.path.splitext(os.path.basename(frame_name))[0]
    pil_image = Image.fromarray(panoptic_rgb)

    annotations = {
        "annotations": {
            "segments_info": frame_segments,
            "file_name": os.path.basename(frame_name),
        },
        "video_id": video_id,
    }

    return (png_filename, pil_image), annotations
