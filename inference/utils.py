import os
from typing import Iterable, Optional, Tuple

from PIL import Image


def save_generated_images(
    images: Iterable[Tuple[str, Image.Image]],
    output_dir: str,
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

    for filename, img in images:
        # Use suffix to choose format (e.g., .jpg -> JPEG, .png -> PNG)
        extension = os.path.splitext(filename)[-1].lower()
        fname = os.path.basename(filename)
        fmt = (
            "PNG"
            if extension == ".png"
            else "JPEG"
            if extension in {".jpg", ".jpeg"}
            else None
        )
        if fmt is None:
            raise ValueError(
                f"Unsupported image extension for '{fname}'. Use .png/.jpg/.jpeg."
            )
        img.save(os.path.join(out_dir, fname), format=fmt)
