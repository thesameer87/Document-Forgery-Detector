# src/train.py
"""
Training loop with validation logging and checkpoint saving.

Simple design per SSOT:
    - No scheduler
    - No unnecessary callbacks
    - No extra abstractions

Usage:
    python -m src.train --config configs/config.yaml --backbone resnet18
    python -m src.train --config configs/config.yaml --backbone efficientnet_b0
"""

import csv
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.dataset import create_data_loaders
from src.evaluate import evaluate_model, format_metrics
from src.model import create_model
from src.utils import get_device, load_config, seed_everything, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for a single epoch.

    Args:
        model: The network.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Compute device.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    backbone: str,
    checkpoint_dir: str,
) -> str:
    """Save model checkpoint.

    Args:
        model: Trained model.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        metrics: Validation metrics dict.
        backbone: Backbone name (used in filename).
        checkpoint_dir: Directory to save checkpoints.

    Returns:
        Path to saved checkpoint file.
    """
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    filename = ckpt_path / f"{backbone}_best.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "backbone": backbone,
        },
        str(filename),
    )
    logger.info("Checkpoint saved: %s", filename)
    return str(filename)


def train(
    config: Optional[dict] = None,
    *,
    config_path: str = "configs/config.yaml",
    backbone: Optional[str] = None,
) -> dict:
    """Full training run for a single backbone.

    Args:
        config: Pre-loaded config dict.
        config_path: Fallback config path.
        backbone: Override backbone name from config.

    Returns:
        Dict with final validation metrics and checkpoint path.
    """
    if config is None:
        config = load_config(config_path)

    backbone_name = backbone or config["model"]["backbone"]
    seed = config["dataset"]["random_seed"]
    num_epochs = config["training"]["num_epochs"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    optimizer_name = config["training"]["optimizer"]
    checkpoint_dir = config["paths"]["checkpoints"]
    results_dir = config["paths"]["results"]

    seed_everything(seed)
    device = get_device(config["training"]["device"])
    logger.info("Device: %s", device)

    # ── Data ─────────────────────────────────────────────────────────────
    loaders = create_data_loaders(config)
    logger.info(
        "Data loaded — train=%d, val=%d, test=%d",
        loaders["num_train"], loaders["num_val"], loaders["num_test"],
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = create_model(config, backbone=backbone_name)
    model = model.to(device)

    # ── Loss & Optimizer ─────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # ── Training ─────────────────────────────────────────────────────────
    best_f1 = 0.0
    best_metrics = {}
    best_epoch = 0
    checkpoint_path = ""

    # CSV log
    log_csv_path = Path(results_dir) / f"training_log_{backbone_name}.csv"
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(log_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "train_loss", "val_accuracy", "val_precision",
        "val_recall", "val_f1", "val_auc_roc", "elapsed_sec",
    ])

    logger.info("=" * 60)
    logger.info("Training '%s' for %d epochs", backbone_name, num_epochs)
    logger.info("=" * 60)

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
        )

        # Validate
        val_metrics = evaluate_model(model, loaders["val"], device)
        elapsed = time.time() - epoch_start

        # Log
        logger.info(
            "Epoch %d/%d  loss=%.4f  %s  (%.1fs)",
            epoch, num_epochs, train_loss,
            format_metrics(val_metrics), elapsed,
        )

        csv_writer.writerow([
            epoch, f"{train_loss:.6f}",
            f"{val_metrics['accuracy']:.4f}",
            f"{val_metrics['precision']:.4f}",
            f"{val_metrics['recall']:.4f}",
            f"{val_metrics['f1_score']:.4f}",
            f"{val_metrics['auc_roc']:.4f}",
            f"{elapsed:.1f}",
        ])
        csv_file.flush()

        # Save best checkpoint (by F1)
        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            best_metrics = val_metrics
            best_epoch = epoch
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, val_metrics,
                backbone_name, checkpoint_dir,
            )

    csv_file.close()
    total_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info(
        "Training complete — best epoch=%d  %s  (total %.1fs)",
        best_epoch, format_metrics(best_metrics), total_time,
    )
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Training log: %s", log_csv_path)
    logger.info("=" * 60)

    return {
        "backbone": backbone_name,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "checkpoint_path": checkpoint_path,
        "training_log": str(log_csv_path),
        "total_time_sec": total_time,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train forgery detection model")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config file",
    )
    parser.add_argument(
        "--backbone", type=str, default=None,
        help="Override backbone (e.g. resnet18, efficientnet_b0)",
    )
    args = parser.parse_args()

    setup_logging()
    result = train(config_path=args.config, backbone=args.backbone)

    print(f"\n{'='*60}")
    print(f"Backbone      : {result['backbone']}")
    print(f"Best epoch    : {result['best_epoch']}")
    print(f"Best F1       : {result['best_metrics']['f1_score']:.4f}")
    print(f"Best Accuracy : {result['best_metrics']['accuracy']:.4f}")
    print(f"Best AUC-ROC  : {result['best_metrics']['auc_roc']:.4f}")
    print(f"Checkpoint    : {result['checkpoint_path']}")
    print(f"Total time    : {result['total_time_sec']:.1f}s")
    print(f"{'='*60}")
