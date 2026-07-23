# src/model.py
"""
Model architecture: configurable backbone via timm for transfer learning.

Supported backbones (from config.yaml):
    - resnet18
    - efficientnet_b0

The classifier head is replaced with a linear layer matching ``num_classes``.
"""

import logging
from typing import Optional

import timm
import torch.nn as nn

from src.utils import load_config

logger = logging.getLogger(__name__)


def create_model(
    config: Optional[dict] = None,
    *,
    config_path: str = "configs/config.yaml",
    backbone: Optional[str] = None,
) -> nn.Module:
    """Create a pretrained backbone with a fresh classification head.

    Args:
        config: Pre-loaded config dict (falls back to *config_path*).
        config_path: Path to YAML config.
        backbone: Override the backbone name from config (useful for
                  running experiments with different backbones).

    Returns:
        A ``torch.nn.Module`` ready for training.
    """
    if config is None:
        config = load_config(config_path)

    backbone_name = backbone or config["model"]["backbone"]
    pretrained = config["model"]["pretrained"]
    num_classes = config["model"]["num_classes"]

    model = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    # Log model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Created model '%s'  (pretrained=%s, classes=%d, "
        "total_params=%s, trainable_params=%s)",
        backbone_name, pretrained, num_classes,
        f"{total_params:,}", f"{trainable_params:,}",
    )

    return model
