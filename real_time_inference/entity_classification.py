import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation


def select_frame_with_most_entities(
    panoptic_outputs: Dict[str, Any],
    pred_ious: torch.Tensor,  # [T, N] over all original candidates
):
    """
    Find the 0-indexed frame with the most visible entities. If there are multiple such frames,
    select the one with the highest average IoU over the surviving candidates.

    Args:
        panoptic_outputs: output of `post_process_results_for_vps`, with keys, among others:
            - "pred_masks": int tensor [T,H,W], 0=bg, >0=entity IDs after filtering
            - "pred_ids": List[int] original candidate indices that survived filtering
        pred_ious: Float tensor [T,N] with per-frame IoU scores for all original candidates.

    Returns:
        best_frame_idx: int, index of the selected frame (0 <= best_frame_idx < T)
        highest_avg_score: float,
    """
    pan_seg = panoptic_outputs["pred_masks"].to(torch.int64)  # [T,H,W]
    device = pred_ious.device

    if pred_ious.shape[0] != pan_seg.shape[0]:
        raise ValueError(
            "Temporal size mismatch between panoptic outputs and pred_ious."
        )

    # Count unique nonzero IDs per frame (exclude background)
    per_frame_counts = torch.tensor(
        [
            torch.unique(pan_seg[t][pan_seg[t] > 0]).numel()
            for t in range(pan_seg.shape[0])
        ]
    ).to(device)

    # Identify frames with the max count
    max_count = int(per_frame_counts.max().item())
    candidate_frames_idx = (
        torch.nonzero(per_frame_counts == max_count, as_tuple=False)
        .flatten()
        .to(device)
    )  # This gives a list of length M, where M is the number of frames with max count

    # pred_ious is [T,N] over all original candidates. We need to select only the
    # columns corresponding to the surviving candidates [M, N_kept]
    keep = torch.as_tensor(
        panoptic_outputs.get("pred_ids"), dtype=torch.long, device=device
    )
    scores = pred_ious.index_select(0, candidate_frames_idx).index_select(
        1, keep
    )  # [M, N_kept]

    # Per-frame average score over the kept candidates
    avg_per_frame_scores = scores.mean(dim=1)  # [M]
    # Which of the M candidate frames has the highest average score?
    candidate_idx = int(avg_per_frame_scores.argmax().item())  # scalar in [0, M-1]
    # Now map it back to the original frame index
    best_frame_idx = int(
        candidate_frames_idx[candidate_idx].item()
    )  # scalar in [0, T-1]
    highest_avg_score = float(avg_per_frame_scores[candidate_idx].item())

    if candidate_frames_idx.numel() == 1:
        print(
            f"Selected the only frame (idx {best_frame_idx}, 0-indexed) with entity count {max_count} and mean IoU {highest_avg_score:.4f}."
        )
    else:
        print(
            f"Multiple frames ({len(candidate_frames_idx)}) with max entity count {max_count}."
        )
        print(
            f"Selected frame with idx {best_frame_idx} (0-indexed) with highest mean IoU:"
        )
        for i, f_idx in enumerate(candidate_frames_idx):
            print(
                f" - Frame {int(f_idx)}: mean IoU {float(avg_per_frame_scores[i]):.4f}"
            )

    return best_frame_idx, highest_avg_score


def generate_entity_visual_prompts(
    panoptic_seg: torch.IntTensor,  # [T,H,W]
    segments_infos: list[dict],  # len N
    frame_dir: str,
    frame_names: list[str],
    crop_padding: int = 10,
    bbox_thickness: int = 2,
) -> Tuple[List[Tuple[str, Image.Image]], Dict[int, int]]:
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
        A list of (entity_id, selected_frame) tuples, one per entity
        A list of (filename, PIL Image) tuples, one per entity
    """

    # Some checks
    assert len(frame_names) == panoptic_seg.shape[0], (
        f"Number of frame names ({len(frame_names)}) does not match number of frames "
        f"in panoptic_seg ({panoptic_seg.shape[0]})."
    )

    pan_np = panoptic_seg.cpu().numpy()  # [T,H,W]

    entity_frame_assignments = {}  # entity_id: selected_frame_idx
    imgs_with_filenames = []  # (filename, PIL Image) tuples
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

        entity_frame_assignments[seg_id] = t
        imgs_with_filenames.append((save_name, im_cropped))

    return imgs_with_filenames, entity_frame_assignments
