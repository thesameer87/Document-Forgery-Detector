# src/evaluate.py
"""
Evaluation metrics, confusion matrix, and classification report generation.

Computes: accuracy, precision, recall, F1 score, ROC-AUC.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true: list, y_pred: list, y_prob: list) -> dict:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground-truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_prob: Predicted probability for the positive class (tampered).

    Returns:
        Dict with keys: accuracy, precision, recall, f1_score, auc_roc.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    # AUC-ROC requires both classes to be present
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc_roc"] = 0.0
        logger.warning("AUC-ROC undefined (only one class present in y_true).")

    return metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Run a full evaluation pass on a dataloader.

    Args:
        model: Trained model (set to eval mode internally).
        dataloader: Validation or test DataLoader.
        device: Compute device.

    Returns:
        Dict with metric values and raw predictions.
    """
    model.eval()

    all_true, all_pred, all_prob = [], [], []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.numpy().tolist()

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        pred_labels = probs.argmax(dim=1).cpu().numpy().tolist()
        pos_probs = probs[:, 1].cpu().numpy().tolist()

        all_true.extend(labels)
        all_pred.extend(pred_labels)
        all_prob.extend(pos_probs)

    metrics = compute_metrics(all_true, all_pred, all_prob)

    return {
        **metrics,
        "y_true": all_true,
        "y_pred": all_pred,
        "y_prob": all_prob,
    }


def format_metrics(metrics: dict) -> str:
    """Return a formatted single-line summary of metrics."""
    return (
        f"Acc={metrics['accuracy']:.4f}  "
        f"P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}  "
        f"F1={metrics['f1_score']:.4f}  "
        f"AUC={metrics['auc_roc']:.4f}"
    )
