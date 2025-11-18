# Copyright (c) Adobe EntitySAM team.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import ctypes
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import yarp
from PIL import Image

from real_time_inference.utils import save_generated_images, save_video
from real_time_inference.vps import (
    build_panoptic_frame_and_annotations,
    post_process_results_for_vps,
)
from sam2.build_sam import build_sam2_camera_query_iou_predictor
from sam2.sam2_camera_query_iou_predictor import SAM2CameraQueryIoUPredictor

NUM_QUERIES = 50  # That's what the model uses
NUM_CATEGORIES = 124  # OG code used 124 as in VIPSeg
MAX_STORED_MASKS = 50  # For temporal stability checks
BETA = 0.95
BIAS_CORRECTION = True  # Adam-style bias correction for EMA
YARP_IMAGE_PORT = "/depthCamera/rgbImage:i"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model with a specified checkpoint directory on a specified video."
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Name of the output directory and video file (without extension). "
        "If not specified, uses datetime.",
    )

    parser.add_argument(
        "--output_root_dir",
        type=str,
        default="output_real_time_webcam",
        help="Root output directory",
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
    )  # 8 for ViT-L. 4 for ViT-S

    parser.add_argument(
        "--avg_type",
        type=str,
        choices=["arithmetic", "ema", "none"],
        default="ema",
        help=(
            "Type of averaging for predicted IoU scores over time. "
            "'none' means no averaging."
        ),
    )

    # === Visualization options ===
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
    parser.set_defaults(viz_results=True)

    parser.add_argument(
        "--viz_type",
        type=str,
        choices=["solid", "overlay"],
        default="overlay",
        help="Type of visualization for panoptic segmentation.",
    )

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
    parser.set_defaults(save_images=False)

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
    parser.set_defaults(save_video=False)

    save_json_group = parser.add_mutually_exclusive_group()
    save_json_group.add_argument(
        "--save_json",
        dest="save_json",
        action="store_true",
        help="Save output JSON annotations.",
    )
    save_json_group.add_argument(
        "--no_save_json",
        dest="save_json",
        action="store_false",
        help="Do not save output JSON annotations.",
    )
    parser.set_defaults(save_json=False)

    args = parser.parse_args()

    # Log args
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("\n")

    # Use datetime as video ID if output_name not specified
    if args.output_name is not None:
        video_id = args.output_name
    else:
        video_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If we need to save results, create output dir
    output_dir = os.path.join(
        args.output_root_dir,
        video_id,
    )
    if args.save_images or args.save_video or args.save_json:
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

    # Initialize YARP network
    yarp.Network.init()

    port = yarp.BufferedPortImageRgb()
    print(
        f"Opening YARP image port at {YARP_IMAGE_PORT}. Connect your image source to it."
    )
    port.open(YARP_IMAGE_PORT)

    peak_memory = 0
    frame_counter = 0
    is_first_frame_initialized = False
    panoptic_images = []
    side_by_side_images = []
    all_pred_masks = []  # will become [N, T, H, W]
    segments_annotations = {}  # Frame index -> list of segment annotations
    frame_timestamps = []  # To compute output fps

    # Running statistics for IoU averaging
    ema_iou = None  # uncorrected EMA state [N]
    avg_iou = None  # arithmetic running mean state [N]

    stop_processing = False
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while stop_processing is not True:
            # Image frame from YARP
            img_rgb = port.read(False)
            if img_rgb:
                width = img_rgb.width()
                height = img_rgb.height()
                out_size = (height, width)

                char_array_ptr = ctypes.cast(
                    int(img_rgb.getRawImage()), ctypes.POINTER(ctypes.c_char)
                )
                bytes_data = ctypes.string_at(char_array_ptr, img_rgb.getRawImageSize())
                image_array = np.frombuffer(bytes_data, dtype=np.uint8)
                frame_rgb = image_array.reshape((height, width, 3))
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            else:
                continue

            print(f"Input shape: {out_size}")

            # First frame initialization
            if not is_first_frame_initialized:
                predictor.load_first_frame(frame_rgb)
                is_first_frame_initialized = True
            else:
                out_frame_idx, _, out_mask_logits, pred_eiou = predictor.track(
                    frame_rgb
                )

                # --- IoU aggregation: EMA or arithmetic ---
                if args.avg_type == "ema":
                    if ema_iou is None:
                        # start from zeros for exact Adam-style correction
                        ema_iou = torch.zeros_like(pred_eiou)
                    ema_iou = BETA * ema_iou + (1 - BETA) * pred_eiou
                    # First few frames are biased towards zero. Correct for that
                    # with Adam-style bias correction
                    if BIAS_CORRECTION:
                        denom = 1.0 - (BETA ** (frame_counter + 1))
                        iou_for_vps = ema_iou / max(denom, 1e-6)
                    else:
                        iou_for_vps = ema_iou
                elif args.avg_type == "arithmetic":
                    # numerically stable arithmetic running mean (Welford)
                    if avg_iou is None:
                        avg_iou = pred_eiou.clone()
                    else:
                        avg_iou = avg_iou + (pred_eiou - avg_iou) / (frame_counter + 1)
                    iou_for_vps = avg_iou
                else:  # args.avg_type == "none"
                    iou_for_vps = pred_eiou

                pred_eious = iou_for_vps.unsqueeze(
                    0
                )  # Now unsqueeze to have batch dim [1, N]
                pred_masks = out_mask_logits  # [N, 1, H, W]

                # Store pred_mask for temporal stability checks
                if len(all_pred_masks) >= MAX_STORED_MASKS:
                    all_pred_masks.pop(0)
                all_pred_masks.append(pred_masks.cpu())

                # Post-process the results into panoptic maps
                result_i = post_process_results_for_vps(
                    pred_ious=pred_eious,
                    pred_masks=pred_masks,
                    out_size=out_size,
                    query_to_category_map=query_to_category_map,
                    prev_raw_pred_masks=all_pred_masks,
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
                panoptic_img_with_filename, frame_segments_annotations = (
                    build_panoptic_frame_and_annotations(
                        frame_idx=frame_counter,
                        panoptic_outputs=result_i,
                        categories_by_id=categories_dict,
                        visualization=args.viz_type,
                        orig_bgr_frame=frame_bgr,
                    )
                )
                segments_annotations[frame_counter] = frame_segments_annotations

                # Raw frame | Panoptic image side by side
                pano_bgr = np.array(panoptic_img_with_filename[1])[
                    :, :, ::-1
                ]  # PIL -> BGR
                side_by_side_bgr = np.hstack((frame_bgr, pano_bgr))  # For cv2 viz
                # Now make it PIL
                side_by_side_rgb = cv2.cvtColor(side_by_side_bgr, cv2.COLOR_BGR2RGB)
                side_by_side = Image.fromarray(side_by_side_rgb)

                # Append to loggers
                panoptic_images.append(panoptic_img_with_filename)
                side_by_side_images.append(
                    (panoptic_img_with_filename[0], side_by_side)
                )
                frame_timestamps.append(time.perf_counter())

                if args.viz_results:
                    cv2.imshow("Panoptic Segmentation", side_by_side_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_processing = True

            # Update running statistics
            frame_counter += 1
            current_peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            peak_memory = max(peak_memory, current_peak_memory)
            print(f"Processed frame {frame_counter}")
            print(
                f"Current peak memory: {current_peak_memory:.2f} GB, "
                f"Overall peak memory: {peak_memory:.2f} GB."
            )

    # yarp cleanup
    port.close()
    yarp.Network.fini()

    # Save results
    if args.save_json:
        annotations_output_path = os.path.join(
            output_dir, f"{video_id}_panoptic_annotations.json"
        )
        with open(annotations_output_path, "w") as f:
            json.dump(segments_annotations, f, indent=2)
        print(f"Saved panoptic annotations to {annotations_output_path}")

    if args.save_images:
        save_generated_images(
            panoptic_images,
            output_dir=output_dir,
            subdir="panoptic_images",
        )
        print(f"Saved panoptic images to {output_dir}/panoptic_images")

    if args.save_video:
        if len(frame_timestamps) >= 2:
            elapsed = frame_timestamps[-1] - frame_timestamps[0]
            # Use average FPS over the whole run
            effective_fps = (len(frame_timestamps) - 1) / elapsed
        else:
            effective_fps = 30.0

        save_video(
            [panoptic_images[i][1] for i in range(len(panoptic_images))],
            output_name=f"{video_id}_panoptic_video",
            output_dir=output_dir,
            fps=effective_fps,
        )
        print(
            f"Saved panoptic video to {output_dir}/{video_id}_panoptic_video.mp4 with {effective_fps} FPS."
        )

    if args.viz_results:
        cv2.destroyAllWindows()

    print(f"Processed {frame_counter} frames.")
    print(f"Peak GPU memory usage: {peak_memory:.2f} GB.")
