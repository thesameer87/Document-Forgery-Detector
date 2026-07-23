# src/utils.py
"""
Shared utilities: configuration loading, random seeding, device selection, logging.
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration from disk.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Select the compute device based on config preference.

    Args:
        preference: One of "auto", "cuda", or "cpu".

    Returns:
        torch.device instance.
    """
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def setup_logging(log_dir: str = "results/logs", level: int = logging.INFO) -> None:
    """Configure root logger with console and file handlers.

    Args:
        log_dir: Directory for log files.
        level: Logging level.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path / "project.log"),
        ],
    )
