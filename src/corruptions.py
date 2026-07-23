# src/corruptions.py
"""
Adversarial robustness corruption definitions.

Defines the severity sweeps (1-5) for Gaussian Blur, JPEG Compression, and Glare.
Also provides a static noisy condition for cross-model benchmarking.
"""

import albumentations as A

def get_corruption_sweep(corruption_type: str, severity: int) -> A.Compose:
    """Retrieve an Albumentations transform for a specific severity level (1-5).
    
    Args:
        corruption_type: "blur", "jpeg", or "glare".
        severity: Integer from 1 to 5.
        
    Returns:
        Albumentations Compose object.
    """
    assert 1 <= severity <= 5, "Severity must be between 1 and 5"
    
    if corruption_type == "blur":
        # Severity 1-5 maps to kernel sizes: 3, 5, 7, 11, 15
        kernels = {1: 3, 2: 5, 3: 7, 4: 11, 5: 15}
        k = kernels[severity]
        return A.Compose([
            A.GaussianBlur(blur_limit=(k, k), p=1.0)
        ])
        
    elif corruption_type == "jpeg":
        # Severity 1-5 maps to JPEG qualities: 90, 70, 50, 30, 10 (lower is worse)
        qualities = {1: 90, 2: 70, 3: 50, 4: 30, 5: 10}
        q = qualities[severity]
        return A.Compose([
            A.ImageCompression(quality_lower=q, quality_upper=q, p=1.0)
        ])
        
    elif corruption_type == "glare":
        # Severity 1-5 maps to flare_roi and src_radius intensities
        # (flare_roi controls bounding box, src_radius controls flare circle size)
        params = {
            1: (0.1, 50),
            2: (0.2, 100),
            3: (0.3, 150),
            4: (0.4, 200),
            5: (0.5, 300),
        }
        roi, radius = params[severity]
        return A.Compose([
            A.RandomSunFlare(
                flare_roi=(0, 0, roi, roi),
                src_radius=radius,
                p=1.0
            )
        ])
        
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")


def get_static_noisy_condition() -> A.Compose:
    """Deterministic noisy condition for cross-model benchmarking.
    
    Combines GaussianBlur (kernel=7) + ImageCompression (quality=50).
    """
    return A.Compose([
        A.GaussianBlur(blur_limit=(7, 7), p=1.0),
        A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0)
    ])
