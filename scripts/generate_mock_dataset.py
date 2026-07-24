# scripts/generate_mock_dataset.py
"""
Generate a synthetic dataset to test the forgery detection pipeline end-to-end.
Creates 50 Authentic and 50 Tampered images in `data/raw/`.
"""

import os
from pathlib import Path

import cv2
import numpy as np

def create_synthetic_image(is_tampered: bool, size: int = 512) -> np.ndarray:
    """Create a synthetic image (authentic or tampered)."""
    # Create a smooth gradient background
    x = np.linspace(0, 255, size)
    y = np.linspace(0, 255, size)
    X, Y = np.meshgrid(x, y)
    
    # Base image: smooth blue-ish gradient
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = (X * 0.5 + Y * 0.5).astype(np.uint8) # B
    img[:, :, 1] = (X * 0.8).astype(np.uint8)          # G
    img[:, :, 2] = (Y * 0.8).astype(np.uint8)          # R
    
    # Add some random shapes to make it look like content
    for _ in range(5):
        cx, cy = np.random.randint(50, size-50, 2)
        r = np.random.randint(20, 80)
        color = np.random.randint(50, 200, 3).tolist()
        cv2.circle(img, (cx, cy), r, color, -1)
        
    if is_tampered:
        # Simulate a splice forgery: 
        # Copy a region, compress it heavily, then paste it back
        patch_size = 100
        px, py = np.random.randint(50, size-patch_size-50, 2)
        
        patch = img[py:py+patch_size, px:px+patch_size].copy()
        
        # Add artificial noise to the patch to simulate different origin
        noise = np.random.normal(0, 20, patch.shape).astype(np.int16)
        patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Paste it elsewhere
        tx, ty = np.random.randint(50, size-patch_size-50, 2)
        img[ty:ty+patch_size, tx:tx+patch_size] = patch
        
    return img

def main():
    raw_dir = Path("data/raw")
    au_dir = raw_dir / "Au"
    tp_dir = raw_dir / "Tp"
    
    au_dir.mkdir(parents=True, exist_ok=True)
    tp_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = 50
    print(f"Generating {num_samples} Authentic and {num_samples} Tampered images...")
    
    for i in range(num_samples):
        # Generate Authentic
        au_img = create_synthetic_image(is_tampered=False)
        cv2.imwrite(str(au_dir / f"Au_mock_{i:03d}.jpg"), au_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Generate Tampered
        tp_img = create_synthetic_image(is_tampered=True)
        cv2.imwrite(str(tp_dir / f"Tp_mock_{i:03d}.jpg"), tp_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1} / {num_samples} pairs")
            
    print("Dataset generation complete. Files are in `data/raw/Au` and `data/raw/Tp`.")

if __name__ == "__main__":
    main()
