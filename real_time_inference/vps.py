from typing import Any, Dict, Literal, Optional, Tuple

import cv2
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
    frame_idx: int,
    panoptic_outputs: Dict[str, Any],
    categories_by_id: Dict[int, Dict[str, Any]],
    visualization: Literal["solid", "overlay"] = "solid",
    orig_bgr_frame: Optional[np.ndarray] = None,
    alpha: float = 0.55,
    contour_thickness: int = 2,
) -> Tuple[Tuple[str, Image.Image], list[Dict[str, Any]]]:
    """
    Generate a visualization PNG (solid or overlay) for a frame from a panoptic ID map
    and build a COCO/VIPSeg-style annotation dict.

    Args:
        frame_idx: Index of the current frame.
        panoptic_outputs: Dict with keys:
            - "image_size": (height, width)
            - "pred_masks": Per-pixel entity IDs; shape (H, W) or (1, H, W).
            - "segments_infos": List of dicts; each has {"id", "category_id", ...}.
        categories_by_id: Mapping category_id -> {"id", "isthing", "color": [R,G,B]}.
        visualization: "solid" for a colorized render, "overlay" to blend over the original.
        orig_bgr_frame: Required if visualization == "overlay". Original frame (H, W, 3) BGR.
        alpha: Blend factor for overlay (mask areas only).
        contour_thickness: Thickness for entity contours.

    Returns:
        image_output: (png_filename, PIL.Image)
        frame_segments: List of dictionaries containing segment annotations (entity_id,
            category_id, bbox, area) for the current frame.
    """
    height, width = panoptic_outputs["image_size"]
    seg_id_map = (
        torch.as_tensor(panoptic_outputs["pred_masks"])
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )  # [H, W]
    segments_info_list = panoptic_outputs["segments_infos"]

    # Compute segments_info and a solid color render (panoptic_rgb)
    id_color_gen = IdGenerator(categories_by_id)
    panoptic_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    frame_segments: list[dict] = []

    # We fill colors here, but we reuse it for the overlay source if needed
    for seg_info in segments_info_list:
        if seg_info is None:
            continue

        entity_id = int(seg_info["id"])
        category_id = int(seg_info["category_id"])

        mask = seg_id_map == entity_id
        area = int(mask.sum())
        if area == 0:
            continue

        color = id_color_gen.get_color(category_id)  # RGB
        panoptic_rgb[mask] = color

        ys, xs = np.where(mask)
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

    # If visualization is solid, we're done
    if visualization == "solid":
        pil_image = Image.fromarray(panoptic_rgb)

    # Otherwise, build an overlay with contours
    elif visualization == "overlay":
        pil_image = render_panoptic_overlay(
            orig_bgr_frame=orig_bgr_frame,
            seg_id_map=seg_id_map,
            segments_info_list=segments_info_list,
            categories_by_id=categories_by_id,
            panoptic_rgb=panoptic_rgb,
            alpha=alpha,
            contour_thickness=contour_thickness,
        )

    else:
        raise ValueError(f"Unknown visualization mode: {visualization!r}")

    png_filename = f"frame_{frame_idx:04d}"
    return (png_filename, pil_image), frame_segments


def render_panoptic_overlay(
    *,
    orig_bgr_frame: np.ndarray,
    seg_id_map: np.ndarray,  # [H, W], int ids, 0 = background
    segments_info_list: list[dict],
    categories_by_id: Dict[int, Dict[str, Any]],
    panoptic_rgb: np.ndarray,  # [H, W, 3], RGB solid render
    alpha: float = 0.5,
    contour_thickness: int = 2,
) -> Image.Image:
    """
    Render an overlay by blending `panoptic_rgb` over the original frame and drawing contours.

    Args:
        orig_bgr_frame: Original frame in BGR, shape (H, W, 3).
        seg_id_map: Per-pixel entity ids, 0 = background.
        segments_info_list: List of segment dicts with keys like {"id", "category_id", ...}.
        categories_by_id: Mapping category_id -> {"id", "isthing", "color": [R,G,B]}.
        panoptic_rgb: Solid color panoptic render (RGB) of shape (H, W, 3).
        alpha: Blend factor applied only on entity pixels.
        contour_thickness: Thickness for the entity contours.

    Returns:
        A PIL RGB image with overlay + contours.
    """

    height, width = seg_id_map.shape

    # Sanity checks
    if orig_bgr_frame is None:
        raise ValueError(
            "orig_bgr_frame must be provided when visualization='overlay'."
        )
    if orig_bgr_frame.shape[:2] != (height, width):
        raise ValueError("Original frame and panoptic mask must have matching shapes.")

    rgb = cv2.cvtColor(orig_bgr_frame, cv2.COLOR_BGR2RGB)

    entity_mask = seg_id_map > 0  # Discard background
    overlay = rgb.copy()
    overlay[entity_mask] = (
        (alpha * panoptic_rgb[entity_mask].astype(np.float32))
        + ((1.0 - alpha) * rgb[entity_mask].astype(np.float32))
    ).astype(np.uint8)

    # Draw contours (one quick pass using seg_id_map)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    for seg in segments_info_list:
        if seg is None:
            continue
        eid = int(seg["id"])
        cat_id = int(seg["category_id"])
        mask = seg_id_map == eid
        if not mask.any():
            continue

        # Find contours on binary mask. Do some erosion to avoid overlapping lines
        mask_u8 = (mask.astype(np.uint8)) * 255
        inset = max(1, contour_thickness // 2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)
        )
        mask_inset = cv2.erode(mask_u8, kernel, iterations=1)
        contours, _ = cv2.findContours(
            mask_inset, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        r, g, b = categories_by_id[cat_id]["color"]
        color_bgr = (b, g, r)
        cv2.polylines(
            overlay_bgr,
            contours,
            True,
            color_bgr,
            thickness=contour_thickness,
            lineType=cv2.LINE_AA,
        )

    return Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
