# scripts/generate_ela_demo.py
"""
Generate ELA visual comparison figures using synthetic images.

This script creates:
  - A pristine synthetic image
  - A tampered version (with a spliced region)
  - Side-by-side ELA comparisons for both

Output is saved to results/ela_comparisons/.
Run from the project root:
    python scripts/generate_ela_demo.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ela import compute_ela, save_ela_comparison


def create_pristine_image(path: Path, size: tuple = (400, 300)) -> None:
    """Create a synthetic pristine image with smooth gradients."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)

    # Smooth colour gradient background
    for y in range(h):
        for x in range(w):
            arr[y, x] = [
                int(180 * x / w),       # R: left-to-right gradient
                int(140 * y / h) + 50,  # G: top-to-bottom gradient
                150,                     # B: constant
            ]

    img = Image.fromarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), format="JPEG", quality=95)
    print(f"  Created pristine image: {path}")


def create_tampered_image(pristine_path: Path, tampered_path: Path) -> None:
    """Create a tampered version by splicing a foreign block into the image.

    The spliced region is saved at a *different* JPEG quality to simulate
    a real-world copy-paste forgery where compression histories differ.
    """
    # Load the pristine image
    img = Image.open(pristine_path).convert("RGB")
    w, h = img.size

    # Create a foreign patch (bright red block saved at lower quality)
    patch_w, patch_h = w // 4, h // 4
    patch = Image.new("RGB", (patch_w, patch_h), color=(220, 50, 50))
    draw = ImageDraw.Draw(patch)
    draw.rectangle([5, 5, patch_w - 5, patch_h - 5], fill=(255, 80, 30))

    # Re-compress the patch at a very different quality (simulates foreign source)
    import io
    buf = io.BytesIO()
    patch.save(buf, format="JPEG", quality=40)
    buf.seek(0)
    patch_recompressed = Image.open(buf).convert("RGB")

    # Paste into the pristine image
    paste_x = w // 3
    paste_y = h // 3
    img.paste(patch_recompressed, (paste_x, paste_y))

    tampered_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(tampered_path), format="JPEG", quality=95)
    print(f"  Created tampered image: {tampered_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    demo_dir = project_root / "results" / "ela_comparisons"
    tmp_dir = project_root / "results" / "demo_inputs"

    print("=" * 60)
    print("ELA Visual Comparison Demo")
    print("=" * 60)

    # --- Create synthetic images ---
    pristine_path = tmp_dir / "pristine.jpg"
    tampered_path = tmp_dir / "tampered.jpg"

    print("\n1. Creating synthetic images...")
    create_pristine_image(pristine_path)
    create_tampered_image(pristine_path, tampered_path)

    # --- Generate ELA comparisons ---
    print("\n2. Generating ELA comparisons (Q=90, scale=15)...")
    save_ela_comparison(
        pristine_path,
        demo_dir / "pristine_ela_comparison.png",
        quality=90,
        scale=15,
    )
    save_ela_comparison(
        tampered_path,
        demo_dir / "tampered_ela_comparison.png",
        quality=90,
        scale=15,
    )

    # --- Generate multi-quality comparison for the tampered image ---
    print("\n3. Generating multi-quality ELA for tampered image...")
    import matplotlib.pyplot as plt

    qualities = [95, 90, 75, 50]
    fig, axes = plt.subplots(1, len(qualities) + 1, figsize=(5 * (len(qualities) + 1), 5))

    # Show original tampered image
    tampered_img = Image.open(tampered_path).convert("RGB")
    axes[0].imshow(np.array(tampered_img))
    axes[0].set_title("Tampered Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for i, q in enumerate(qualities):
        ela = compute_ela(tampered_path, quality=q, scale=15)
        axes[i + 1].imshow(ela)
        axes[i + 1].set_title(f"ELA Q={q}", fontsize=12, fontweight="bold")
        axes[i + 1].axis("off")

    plt.suptitle("Effect of JPEG Quality on ELA Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    multi_q_path = demo_dir / "multi_quality_ela.png"
    plt.savefig(str(multi_q_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {multi_q_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Demo complete. Output files:")
    for f in sorted(demo_dir.glob("*.png")):
        print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
