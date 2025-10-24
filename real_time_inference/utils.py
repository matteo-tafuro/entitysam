import os
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def save_generated_images(
    images: Iterable[Tuple[str, Image.Image]],
    output_dir: str,
    extension: str = "png",
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

    assert extension in {None, "jpg", "jpeg", "png"}, "Unsupported image extension."

    for filename, img in images:
        # Use suffix to choose format (e.g., .jpg -> JPEG, .png -> PNG)
        fname = os.path.basename(filename) + f".{extension}"
        fmt = (
            "PNG"
            if extension == "png"
            else "JPEG"
            if extension in {"jpg", "jpeg"}
            else None
        )
        img.save(os.path.join(out_dir, fname), format=fmt)


def save_video(
    frames: Iterable[Image.Image],
    output_name: str,
    output_dir: str,
    fps: float = 30.0,
) -> None:
    """
    Save a collection of PIL images as a video to disk.

    Args:
        frames: List of PIL Images.
        output_name: Name of the output video file (without extension).
        output_dir: Base output directory.
        fps: Frames per second for the output video.
    """

    os.makedirs(output_dir, exist_ok=True)

    video_filename = os.path.join(output_dir, f"{output_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    first_frame = True
    video_writer = None

    for img in frames:
        frame = np.array(img.convert("RGB"))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if first_frame:
            height, width, _ = frame.shape
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            first_frame = False

        video_writer.write(frame)

    if video_writer is not None:
        video_writer.release()


## uSE X264
