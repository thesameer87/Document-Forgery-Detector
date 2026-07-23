# src/ela.py
"""
Error Level Analysis (ELA) computation pipeline.

ELA highlights compression-level inconsistencies in images by:
1. Re-compressing the image at a known JPEG quality level (in-memory).
2. Computing the absolute pixel difference between original and re-compressed.
3. Amplifying the difference by a configurable scale factor.

Tampered regions typically show higher ELA residuals because their
compression history differs from the rest of the image.
"""

import io
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from src.utils import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core ELA computation
# ---------------------------------------------------------------------------

def compute_ela(
    image: Union[str, Path, np.ndarray, Image.Image],
    quality: int = 90,
    scale: int = 15,
) -> np.ndarray:
    """Compute an Error Level Analysis map for a single image.

    Uses an in-memory JPEG buffer (BytesIO) — no temporary files are written.

    Args:
        image: File path, NumPy array (BGR or RGB), or PIL Image.
        quality: JPEG re-compression quality (1-100).
        scale: Amplification factor for the pixel difference.

    Returns:
        ELA map as a uint8 NumPy array with shape (H, W, 3).

    Raises:
        FileNotFoundError: If a path is given and the file does not exist.
        ValueError: If *quality* is outside the 1-100 range.
    """
    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality must be 1-100, got {quality}")

    # --- Normalise input to PIL RGB -----------------------------------------
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        original = Image.open(path).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            original = Image.fromarray(image).convert("RGB")
        else:
            # Assume BGR from OpenCV
            original = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(image, Image.Image):
        original = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # --- Re-compress in memory ----------------------------------------------
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    # --- Compute difference & amplify ---------------------------------------
    orig_arr = np.array(original, dtype=np.float64)
    recomp_arr = np.array(recompressed, dtype=np.float64)

    ela_map = np.abs(orig_arr - recomp_arr) * scale
    ela_map = np.clip(ela_map, 0, 255).astype(np.uint8)

    return ela_map


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def batch_process_ela(
    config: Optional[dict] = None,
    *,
    config_path: str = "configs/config.yaml",
    force: bool = False,
) -> dict:
    """Batch-convert all raw images to ELA maps, preserving directory structure.

    Reads parameters from *config* (or loads from *config_path* if not given).
    Already-processed images are skipped unless *force* is ``True``.

    Directory mapping example::

        data/raw/Au/img001.jpg  →  data/ela/Au/img001.png
        data/raw/Tp/img002.tif  →  data/ela/Tp/img002.png

    Args:
        config: Pre-loaded config dict.  Falls back to *config_path*.
        config_path: Path to YAML config (used only if *config* is None).
        force: Re-process images even if the output already exists.

    Returns:
        Dict with keys ``"processed"``, ``"skipped"``, ``"errors"``.
    """
    if config is None:
        config = load_config(config_path)

    raw_dir = Path(config["dataset"]["root_dir"])
    ela_dir = Path(config["dataset"]["ela_dir"])
    quality = config["ela"]["quality"]
    scale = config["ela"]["scale"]
    extensions = {ext.lower() for ext in config["dataset"]["image_extensions"]}

    stats = {"processed": 0, "skipped": 0, "errors": 0}

    if not raw_dir.exists():
        logger.warning("Raw data directory does not exist: %s", raw_dir.resolve())
        return stats

    # Collect all image paths first for progress reporting
    image_paths = [
        p for p in sorted(raw_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in extensions
    ]
    total = len(image_paths)
    logger.info("Found %d images in %s", total, raw_dir)

    for idx, src_path in enumerate(image_paths, 1):
        # Mirror the sub-directory structure, always save as PNG
        relative = src_path.relative_to(raw_dir)
        dst_path = ela_dir / relative.with_suffix(".png")

        if dst_path.exists() and not force:
            stats["skipped"] += 1
            continue

        try:
            ela_map = compute_ela(src_path, quality=quality, scale=scale)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(ela_map).save(str(dst_path))
            stats["processed"] += 1
        except Exception:
            stats["errors"] += 1
            logger.exception("Failed to process %s", src_path)

        # Log progress every 500 images
        if idx % 500 == 0 or idx == total:
            logger.info(
                "Progress: %d / %d  (processed=%d, skipped=%d, errors=%d)",
                idx, total,
                stats["processed"], stats["skipped"], stats["errors"],
            )

    logger.info(
        "Batch ELA complete — processed=%d, skipped=%d, errors=%d",
        stats["processed"], stats["skipped"], stats["errors"],
    )
    return stats


# ---------------------------------------------------------------------------
# Visual comparison helper
# ---------------------------------------------------------------------------

def save_ela_comparison(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    quality: int = 90,
    scale: int = 15,
) -> None:
    """Generate a side-by-side comparison of original image and its ELA map.

    Saves a single figure with two panels: [Original | ELA Map].

    Args:
        image_path: Path to the source image.
        output_path: Where to save the comparison figure (e.g. PNG).
        quality: JPEG re-compression quality for ELA.
        scale: ELA amplification factor.
    """
    # Import matplotlib only when needed (not required for batch processing)
    import matplotlib.pyplot as plt

    image_path = Path(image_path)
    original = Image.open(image_path).convert("RGB")
    ela_map = compute_ela(image_path, quality=quality, scale=scale)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(np.array(original))
    axes[0].set_title("Original", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(ela_map)
    axes[1].set_title(f"ELA  (Q={quality}, scale={scale})", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(image_path.name, fontsize=11, color="grey")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison to %s", output_path)


def generate_comparisons(
    config: Optional[dict] = None,
    *,
    config_path: str = "configs/config.yaml",
    samples_per_class: int = 3,
    seed: int = 42,
) -> list:
    """Generate ELA comparison figures for a random sample from each class.

    Saves figures to ``results/ela_comparisons/``.

    Args:
        config: Pre-loaded config dict.
        config_path: Fallback config path.
        samples_per_class: Number of samples per class folder.
        seed: Random seed for reproducible sampling.

    Returns:
        List of output file paths created.
    """
    import random as _random

    if config is None:
        config = load_config(config_path)

    raw_dir = Path(config["dataset"]["root_dir"])
    quality = config["ela"]["quality"]
    scale = config["ela"]["scale"]
    extensions = {ext.lower() for ext in config["dataset"]["image_extensions"]}
    output_dir = Path(config["paths"]["results"]) / "ela_comparisons"

    _random.seed(seed)
    created = []

    # Iterate over class sub-folders (Au, Tp, etc.)
    for class_folder in sorted(raw_dir.iterdir()):
        if not class_folder.is_dir():
            continue

        images = [
            p for p in sorted(class_folder.iterdir())
            if p.is_file() and p.suffix.lower() in extensions
        ]
        if not images:
            logger.warning("No images found in %s", class_folder)
            continue

        n = min(samples_per_class, len(images))
        selected = _random.sample(images, n)

        for img_path in selected:
            out_name = f"{class_folder.name}_{img_path.stem}_ela.png"
            out_path = output_dir / out_name
            save_ela_comparison(img_path, out_path, quality=quality, scale=scale)
            created.append(str(out_path))

    logger.info("Generated %d comparison figures in %s", len(created), output_dir)
    return created


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ELA preprocessing pipeline")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-process existing ELA maps"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Generate visual comparisons after batch processing"
    )
    parser.add_argument(
        "--samples", type=int, default=3,
        help="Samples per class for visual comparisons (default: 3)"
    )
    args = parser.parse_args()

    from src.utils import setup_logging
    setup_logging()

    cfg = load_config(args.config)
    stats = batch_process_ela(cfg, force=args.force)
    print(f"\nBatch ELA results: {stats}")

    if args.compare:
        paths = generate_comparisons(cfg, samples_per_class=args.samples)
        print(f"Saved {len(paths)} comparison figures.")
