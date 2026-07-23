# src/failure_analysis.py
"""
Failure Analysis & Heuristic Categorization.

Evaluates the best checkpoint on the clean test set, isolates the top confident failures,
and generates visual contact sheets (Raw vs ELA) alongside a heuristically inferred
failure mode. Results are logged to a CSV and a Markdown summary table.
"""

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import create_data_loaders
from src.model import create_model
from src.utils import get_device, load_config, setup_logging
from src.run_robustness import load_checkpoint_and_model

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "Authentic", 1: "Tampered"}


def map_ela_to_raw(ela_path: Path, raw_dir: Path, image_extensions: list) -> Path:
    """Find the corresponding raw image for an ELA map."""
    class_name = ela_path.parent.name
    stem = ela_path.stem
    
    for ext in image_extensions:
        possible = raw_dir / class_name / f"{stem}{ext}"
        if possible.exists():
            return possible
            
    raise FileNotFoundError(f"Could not find raw image for {ela_path}")


def heuristic_failure_mode(raw_img: np.ndarray, ela_map: np.ndarray) -> str:
    """Analyze the raw and ELA images to heuristically guess the failure mode."""
    gray_raw = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
    gray_ela = cv2.cvtColor(ela_map, cv2.COLOR_RGB2GRAY)
    
    # 1. Glare / Overexposure: High percentage of saturated pixels
    if np.mean(gray_raw > 240) > 0.05:
        return "Strong Glare / Overexposure"
        
    # 2. Heavy JPEG Compression: Very weak overall ELA signal
    if np.mean(gray_ela) < 10.0:
        return "Heavy JPEG Compression (Weak ELA Signal)"
        
    # 3. Small Manipulated Area: Highly concentrated ELA signal in a small region
    threshold = np.percentile(gray_ela, 95)
    if threshold > 20 and np.mean(gray_ela > threshold) < 0.01:
        return "Small Manipulated Area"
        
    # 4. Complex Texture: High edge density
    edges = cv2.Canny(gray_raw, 100, 200)
    if np.mean(edges > 0) > 0.15:
        return "Complex Textured Background"
        
    return "Ambiguous / Indeterminate"


def collect_failures(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> list:
    """Run inference and collect all incorrect predictions."""
    model.eval()
    failures = []
    
    # We need the paths to fetch the exact images later
    image_paths = loader.dataset.image_paths
    current_idx = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            
            for i in range(len(labels)):
                gt = labels[i]
                pred = preds[i]
                
                if gt != pred:
                    conf = probs[i, pred]
                    failures.append({
                        "path": image_paths[current_idx],
                        "gt": gt,
                        "pred": pred,
                        "conf": float(conf)
                    })
                current_idx += 1
                
    # Sort by confidence descending (most confident mistakes first)
    failures.sort(key=lambda x: x["conf"], reverse=True)
    return failures


def generate_failure_gallery(
    failures: list, 
    config: dict, 
    output_dir: Path, 
    top_n: int = 20
) -> list:
    """Generate side-by-side visualization for the top N failures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(config["dataset"]["root_dir"])
    exts = config["dataset"]["image_extensions"]
    
    analyzed_data = []
    
    for idx, f in enumerate(failures[:top_n]):
        ela_path = Path(f["path"])
        raw_path = map_ela_to_raw(ela_path, raw_dir, exts)
        
        # Load images
        ela_img = cv2.cvtColor(cv2.imread(str(ela_path)), cv2.COLOR_BGR2RGB)
        raw_img = cv2.cvtColor(cv2.imread(str(raw_path)), cv2.COLOR_BGR2RGB)
        
        # Heuristic Categorization
        likely_mode = heuristic_failure_mode(raw_img, ela_img)
        
        f["likely_mode"] = likely_mode
        f["raw_path"] = str(raw_path)
        analyzed_data.append(f)
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(raw_img)
        axes[0].set_title("Raw Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        
        axes[1].imshow(ela_img)
        axes[1].set_title("ELA Map", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        
        gt_str = LABEL_MAP[f["gt"]]
        pred_str = LABEL_MAP[f["pred"]]
        conf_pct = f["conf"] * 100
        
        title = (
            f"Ground Truth: {gt_str}  |  Prediction: {pred_str} ({conf_pct:.1f}%)\n"
            f"Likely Failure Mode: {likely_mode} (Heuristic)"
        )
        fig.suptitle(title, fontsize=14, color="darkred" if f["gt"] == 1 else "darkorange")
        
        # Save false positive vs false negative
        type_str = "false_positive" if f["pred"] == 1 else "false_negative"
        save_path = output_dir / f"{type_str}_{idx:02d}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info("Saved failure analysis %d/%d to %s", idx + 1, top_n, save_path.name)
        
    return analyzed_data


def write_summary(analyzed_data: list, summary_path: Path):
    """Write the markdown summary table aggregating heuristic counts."""
    counts = Counter(d["likely_mode"] for d in analyzed_data)
    
    with open(summary_path, "w") as f:
        f.write("# Failure Analysis Summary\n\n")
        f.write("> **Note**: Failure modes are heuristically inferred based on image processing rules and serve as likely candidates rather than definitive ground truth.\n\n")
        f.write("| Heuristically Inferred Failure Mode | Count | Likely Cause |\n")
        f.write("|---|---|---|\n")
        
        for mode, count in counts.most_common():
            cause = "Various"
            if "JPEG" in mode:
                cause = "ELA residual signal is heavily suppressed"
            elif "Glare" in mode:
                cause = "Bright spots obscure manipulation artifacts"
            elif "Small" in mode:
                cause = "Discriminative signal is too localized"
            elif "Complex" in mode:
                cause = "False texture cues disrupt the classifier"
                
            f.write(f"| {mode} | {count} | {cause} |\n")


def write_csv(analyzed_data: list, csv_path: Path):
    """Save the metadata for the analyzed failures."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "ground_truth", "prediction", 
            "confidence", "likely_failure_mode"
        ])
        for d in analyzed_data:
            writer.writerow([
                Path(d["raw_path"]).name,
                LABEL_MAP[d["gt"]],
                LABEL_MAP[d["pred"]],
                f"{d['conf']:.4f}",
                d["likely_mode"]
            ])


def run_failure_analysis(config_path: str = "configs/config.yaml", run_id: str = "exp_04"):
    setup_logging()
    config = load_config(config_path)
    device = get_device("auto")
    
    out_dir = Path(config["paths"]["results"]) / "failure_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Day 6: Failure Analysis for %s", run_id)
    logger.info("=" * 60)
    
    model, _ = load_checkpoint_and_model(run_id, config["paths"]["checkpoints"], device)
    
    # We analyze the clean test set
    loaders = create_data_loaders(config)
    test_loader = loaders["test"]
    
    logger.info("Running inference on test set to isolate failures...")
    failures = collect_failures(model, test_loader, device)
    
    logger.info("Found %d total incorrect predictions.", len(failures))
    if not failures:
        logger.info("No failures found! Perfect model.")
        return
        
    top_n = min(20, len(failures))
    logger.info("Generating visual analysis for top %d most confident failures...", top_n)
    
    analyzed_data = generate_failure_gallery(failures, config, out_dir, top_n)
    
    write_summary(analyzed_data, Path(config["paths"]["results"]) / "failure_summary.md")
    write_csv(analyzed_data, out_dir / "failures.csv")
    
    logger.info("=" * 60)
    logger.info("Failure analysis complete. Artifacts saved in %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run_id", default="exp_04")
    args = parser.parse_args()
    
    run_failure_analysis(args.config, args.run_id)
