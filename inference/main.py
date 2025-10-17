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
from PIL import Image

from inference.entity_classification import (
    generate_entity_visual_prompts,
    select_frame_with_most_entities,
)
from inference.utils import save_generated_images
from inference.vps import (
    build_panoptic_frames_and_annotations,
    post_process_results_for_vps,
)
from sam2.build_sam import build_sam2_video_query_iou_predictor
from sam2.sam2_video_query_iou_predictor import SAM2VideoQueryIoUPredictor

# VIPSeg uses 124 categories
NUM_CATEGORIES = 124

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
    predictor: SAM2VideoQueryIoUPredictor = build_sam2_video_query_iou_predictor(
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

    # Post-process the results into panoptic maps
    result_i = post_process_results_for_vps(
        pred_ious=pred_eious,
        pred_masks=pred_masks,
        out_size=out_size,
        pred_stability_scores=pred_stability_scores,
        num_categories=NUM_CATEGORIES,
    )

    # Generate entity visual prompt images
    visual_prompt_images, entity_frame_assignments = generate_entity_visual_prompts(
        panoptic_seg=result_i["pred_masks"],
        segments_infos=result_i["segments_infos"],
        frame_dir=video_dir,
        frame_names=frame_names,
    )

    # Identify the frame where the most entities are visible (for later VLM call)
    frame_with_most_entities_idx, avg_confidence_score = (
        select_frame_with_most_entities(result_i, pred_eious)
    )

    # Build and panoptic images and annotations
    panoptic_images, predictions = build_panoptic_frames_and_annotations(
        video_id, frame_names, result_i, categories_dict
    )

    del pred_masks, pred_eious, pred_masks_list

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

    # Save JSON with summary
    summary = {
        "entity_crops": {
            "frame_with_most_entities_idx": frame_with_most_entities_idx,
            "entity_to_frame_assignments": entity_frame_assignments,
        },
        "annotations": predictions,
    }
    file_path = os.path.join(output_dir, "panoptic_preds.json")
    with open(file_path, "w") as f:
        json.dump(summary, f, indent=2)
