# Copyright (c) Adobe EntitySAM team.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import torch

from real_time_inference.utils import save_generated_images
from real_time_inference.vps import (
    build_panoptic_frames_and_annotations,
    post_process_results_for_vps,
)
from sam2.build_sam import build_sam2_camera_query_iou_predictor
from sam2.sam2_camera_query_iou_predictor import SAM2CameraQueryIoUPredictor

# VIPSeg uses 124 categories
NUM_CATEGORIES = 124

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model with a specified checkpoint directory on a specified video."
    )

    # === Model-specific arguments ===
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

    # === Whether to visualize results on the fly ===
    viz_results_group = parser.add_mutually_exclusive_group()
    viz_results_group.add_argument(
        "--viz_results",
        dest="viz_results",
        action="store_true",
        help="Visualize results on the fly.",
    )
    viz_results_group.add_argument(
        "--no_viz_results",
        dest="viz_results",
        action="store_false",
        help="Do not visualize results on the fly.",
    )
    parser.set_defaults(viz_results=False)

    # === Saving options ===
    save_images_group = parser.add_mutually_exclusive_group()
    save_images_group.add_argument(
        "--save_images",
        dest="save_images",
        action="store_true",
        help="Save generated images.",
    )
    save_images_group.add_argument(
        "--no_save_images",
        dest="save_images",
        action="store_false",
        help="Do not save generated images.",
    )
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default="output_real_time",
        help="Root output directory",
    )
    parser.set_defaults(save_images=True)
    args = parser.parse_args()

    # Log args
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("\n")

    # Use datetime as video ID
    video_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If we need to save results, create output dir
    output_dir = os.path.join(
        args.output_root_dir,
        video_id,
    )
    if args.save_images:
        os.makedirs(output_dir, exist_ok=True)
    # Build (fake) categories dict
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

    # Initialize model
    torch.cuda.reset_peak_memory_stats()
    sam2_checkpoint = os.path.join(args.ckpt_dir, "model_0009999.pth")
    predictor: SAM2CameraQueryIoUPredictor = build_sam2_camera_query_iou_predictor(
        args.model_cfg, sam2_checkpoint, mask_decoder_depth=args.mask_decoder_depth
    )
    inference_state = predictor.init_state()
    predictor.reset_state()

    # Read video frames as a stream
    video_path = "/home/amiuser/repos/entitysam/data/robot_logger_device_2025_09_25_12_16_57_realsense_rgb_30fps.mp4"
    cap = cv2.VideoCapture(video_path)

    total_frames = 0
    peak_memory = 0
    is_first_frame_initialized = False
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            width, height = frame.shape[:2][::-1]
            out_size = (height, width)

            # First frame initialization
            if not is_first_frame_initialized:
                predictor.load_first_frame(frame)
                is_first_frame_initialized = True
            else:
                out_frame_idx, _, out_mask_logits, pred_eiou = predictor.track(frame)

                # pred_masks = torch.cat(pred_masks_list, dim=1)
                pred_masks = out_mask_logits
                # pred_eious = torch.stack(pred_eious)
                pred_eious = pred_eiou.unsqueeze(0)

                pred_stability_scores = None

                # Post-process the results into panoptic maps
                result_i = post_process_results_for_vps(
                    pred_ious=pred_eious,
                    pred_masks=pred_masks,
                    out_size=out_size,
                    pred_stability_scores=pred_stability_scores,
                    num_categories=NUM_CATEGORIES,
                )

                # Build and panoptic images and annotations
                panoptic_images, predictions = build_panoptic_frames_and_annotations(
                    video_id, [f"frame_{out_frame_idx:04d}"], result_i, categories_dict
                )

                # panoptic_images is a ist of (png_filename, PIL Image) tuples for each frame. In thi case there is only one frame.
                if args.viz_results:
                    cv2.imshow(
                        "Panoptic Segmentation",
                        np.array(panoptic_images[0][1])[:, :, ::-1],
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if args.save_images:
                    save_generated_images(
                        panoptic_images, output_dir=output_dir, subdir="panoptic_images"
                    )

                # TODO: summary dict (predictions) needs to grow over frames
                # TODO: implement visual prompt extraction for real-time.
                #       e.g. save best frame index and confidence from the last N frames?

            # Update running statistics
            total_frames += 1
            current_peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            peak_memory = max(peak_memory, current_peak_memory)

    cap.release()
