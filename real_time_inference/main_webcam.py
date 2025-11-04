# Copyright (c) Adobe EntitySAM team.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
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

import threading
import time
from collections import deque

class LatestFrameCapture:
    """
    Read frames on a background thread and only keep the latest frame.
    `read_latest()` returns the newest frame and drops anything older.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

        # Try to minimize internal buffering when supported by the backend
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize=1)
        except Exception:
            pass  # Not all backends support this

        self.queue = deque(maxlen=1)
        self.lock = threading.Lock()
        self.stopped = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                # End of stream or camera error
                self.stop()
                break
            with self.lock:
                # Keep only the newest frame
                self.queue.append(frame)

    def read_latest(self, wait=True, sleep_secs=0.001):
        """
        Returns (ret, frame) like cv2.VideoCapture.read().
        If wait=True, block briefly until a frame arrives (useful for warm-up).
        Always returns the latest available frame and clears the buffer.
        """
        if wait:
            # Wait until at least one frame is available or stopped
            while not self.queue and not self.stopped:
                time.sleep(sleep_secs)
        with self.lock:
            if not self.queue:
                return False, None
            frame = self.queue[-1]
            self.queue.clear()
        return True, frame

    def stop(self):
        self.stopped = True

    def release(self):
        self.stop()
        try:
            self._thread.join(timeout=1.0)
        except RuntimeError:
            pass
        self.cap.release()


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
    # float to tune the strength of the temporal bias
    parser.add_argument(
        "--temporal_bias_strength",
        type=float,
        default=0.0,
        help="Strength of the temporal bias for query selection",
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

    parser.add_argument(
        "--output_root_dir",
        type=str,
        default="output_real_time_webcam",
        help="Root output directory",
    )
    args = parser.parse_args()

    # Log args
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("\n")

    # Use datetime as video ID
    video_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.temporal_bias_strength != 0.0:
        video_id += f"_{str(args.temporal_bias_strength).replace('.', 'p')}"

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

    # Read video frames as a stream
    cap = LatestFrameCapture(0)

    decoded_frame_idx = 0
    total_frames = 0
    peak_memory = 0
    is_first_frame_initialized = False
    panoptic_images = []
    segments_annotations = {}  # Frame index -> list of segment annotations
    prev_owner_query_map = None  # For temporal consistency
    frame_timestamps = [] # To compute output fps

    stop_processing = False
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while stop_processing is not True:

            ret, frame = cap.read_latest(wait=True)
            if not ret:
                break

            width, height = frame.shape[:2][::-1]
            out_size = (height, width)
            print(f"Input shape: {out_size}")

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
                    prev_owner_query_map=prev_owner_query_map,
                    bias_strength=args.temporal_bias_strength,
                )

                prev_owner_query_map = result_i["owner_query_map"]

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
                        frame_idx=decoded_frame_idx,
                        panoptic_outputs=result_i,
                        categories_by_id=categories_dict,
                        visualization=args.viz_type,
                        orig_bgr_frame=frame,
                    )
                )
                panoptic_images.append(panoptic_img_with_filename)
                segments_annotations[decoded_frame_idx] = frame_segments_annotations
                frame_timestamps.append(time.perf_counter())

                if args.viz_results:
                    pano_bgr = np.array(panoptic_img_with_filename[1])[
                        :, :, ::-1
                    ]  # PIL → BGR
                    side_by_side = np.hstack((frame, pano_bgr))
                    cv2.imshow("Panoptic Segmentation", side_by_side)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_processing = True

            # Update running statistics
            decoded_frame_idx += 1
            total_frames += 1
            current_peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            peak_memory = max(peak_memory, current_peak_memory)
            print(f"Processed frame {decoded_frame_idx}")
            print(
                f"Current peak memory: {current_peak_memory:.2f} GB, "
                f"Overall peak memory: {peak_memory:.2f} GB."
            )

    cap.release()

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
        print(f"Saved panoptic video to {output_dir}/{video_id}_panoptic_video.mp4 with {effective_fps} FPS.")

    cap.release()
    if args.viz_results:
        cv2.destroyAllWindows()

    print(f"Processed {total_frames} frames.")
    print(f"Peak GPU memory usage: {peak_memory:.2f} GB.")
