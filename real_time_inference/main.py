# Copyright (c) Adobe EntitySAM team.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

from real_time_inference.utils import save_generated_images, save_video
from real_time_inference.vps import (
    build_panoptic_frame_and_annotations,
    post_process_results_for_vps,
)
from sam2.build_sam import build_sam2_camera_query_iou_predictor
from sam2.sam2_camera_query_iou_predictor import SAM2CameraQueryIoUPredictor

NUM_QUERIES = 50  # That's what the model uses
NUM_CATEGORIES = 124  # OG code used 124 as in VIPSeg

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
    save_video_group = parser.add_mutually_exclusive_group()
    save_video_group.add_argument(
        "--save_video",
        dest="save_video",
        action="store_true",
        help="Save output video.",
    )
    save_video_group.add_argument(
        "--no_save_video",
        dest="save_video",
        action="store_false",
        help="Do not save output video.",
    )

    parser.add_argument(
        "--output_root_dir",
        type=str,
        default="output_real_time",
        help="Root output directory",
    )
    parser.set_defaults(save_images=False, save_video=True)
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

    # Build (fake) categories dict, used later for visualization
    categories_dict = {
        cat_id: {
            "id": cat_id,
            "isthing": 1,
            "color": np.random.randint(0, 255, size=3).tolist(),
        }
        for cat_id in range(1, NUM_CATEGORIES + 1)
    }
    # Map query index → unique category ID
    chosen_cats = np.random.choice(
        np.arange(1, NUM_CATEGORIES + 1), size=NUM_QUERIES, replace=False
    )
    query_to_category_map = {i: int(chosen_cats[i]) for i in range(NUM_QUERIES)}

    # Initialize per-entity best scores
    best_entity_scores = {i: {"score": -1, "frame_idx": None} for i in range(50)}

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
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Should be around 30
    frame_stride = 15

    total_frames = 0
    peak_memory = 0
    is_first_frame_initialized = False
    panoptic_images = []

    break_at_iteration = 60  # For faster testing

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for frame_idx in tqdm(
            range(0, frame_count, frame_stride),
            total=(
                break_at_iteration
                if break_at_iteration
                else frame_count // frame_stride
            ),
        ):
            if total_frames == break_at_iteration:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
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
                    query_to_category_map=query_to_category_map,
                )

                # Keep track of best scoring frames for each entity
                for entity in result_i["segments_infos"]:
                    query_id = entity["id"]
                    score = entity["score"]
                    if score > best_entity_scores[query_id]["score"]:
                        best_entity_scores[query_id] = {
                            "score": score,
                            "frame_idx": out_frame_idx,
                        }

                # Build and panoptic images and annotations
                panoptic_img_with_filename, predictions = (
                    build_panoptic_frame_and_annotations(
                        video_id,
                        f"frame_{out_frame_idx:04d}",
                        result_i,
                        categories_dict,
                    )
                )
                panoptic_images.append(panoptic_img_with_filename)

                if args.viz_results:
                    pano_bgr = np.array(panoptic_img_with_filename[0])[
                        :, :, ::-1
                    ]  # PIL → BGR
                    side_by_side = np.hstack((frame, pano_bgr))
                    cv2.imshow("Panoptic Segmentation", side_by_side)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # Update running statistics
            total_frames += 1
            current_peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            peak_memory = max(peak_memory, current_peak_memory)
            print(f"Processed frame {frame_idx}")
            print(
                f"Current peak memory: {current_peak_memory:.2f} GB, "
                f"Overall peak memory: {peak_memory:.2f} GB."
            )

    if args.save_images:
        save_generated_images(
            panoptic_images,
            output_dir=output_dir,
            subdir="panoptic_images",
        )

    # TODO: summary dict (predictions) needs to grow over frames
    if args.save_video:
        save_video(
            [panoptic_images[i][1] for i in range(len(panoptic_images))],
            output_name=f"{video_id}_panoptic_video",
            output_dir=output_dir,
            fps=fps / frame_stride,
        )
    cap.release()
    if args.viz_results:
        cv2.destroyAllWindows()

    print(f"Processed {total_frames} frames.")
    print(f"Peak (GPU memory usage: {peak_memory:.2f} GB.")
