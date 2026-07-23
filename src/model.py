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
    use_lora: bool = False,
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
    
    if use_lora:
        model = apply_lora(model, config["lora"])

    # Log model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Created model '%s'  (pretrained=%s, classes=%d, use_lora=%s, "
        "total_params=%s, trainable_params=%s)",
        backbone_name, pretrained, num_classes, use_lora,
        f"{total_params:,}", f"{trainable_params:,}",
    )

    return model


def apply_lora(model: nn.Module, lora_cfg: dict) -> nn.Module:
    """Dynamically apply LoRA (PEFT) to convolutional layers without hardcoding names.
    
    Enrolls the model in PEFT by discovering actual Conv2d leaf module names
    used by this specific backbone instance.
    """
    from peft import LoraConfig, get_peft_model
    
    # 1. Enumerate all modules to find Conv2d exact paths in topological order
    conv_paths = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_paths.append(name)
            
    # 2. Select only the deepest 30% of convolutions for true parameter efficiency
    # Early layers capture generic features (edges), deep layers capture semantic forgery artifacts.
    num_to_adapt = max(1, int(len(conv_paths) * 0.3)) if conv_paths else 0
    target_modules = conv_paths[-num_to_adapt:] if num_to_adapt > 0 else []
    
    # Optionally append explicitly configured targets (like 'classifier')
    configured_targets = lora_cfg.get("target_modules", [])
    for ct in configured_targets:
        if ct not in target_modules:
            target_modules.append(ct)
            
    logger.info(
        "LoRA dynamic target strategy: adapting %d deepest Conv2d layers + %d configured targets.", 
        num_to_adapt, len(configured_targets)
    )
    logger.debug("Exact LoRA targets: %s", target_modules)

    peft_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=target_modules,
        lora_dropout=lora_cfg["dropout"],
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    return model

