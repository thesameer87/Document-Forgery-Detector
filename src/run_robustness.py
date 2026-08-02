# src/run_robustness.py
"""
Adversarial Robustness Evaluation — Day 5.

Performs:
1. Severity Sweeps (1-5) on the best model to generate `results/degradation_curve.png`.
2. Cross-checkpoint benchmarking on a static noisy condition -> `results/robustness_results.csv`.
"""

import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.corruptions import get_corruption_sweep, get_static_noisy_condition
from src.dataset import RobustnessDataset, get_test_split_raw_paths
from src.evaluate import evaluate_model
from src.model import create_model
from src.utils import get_device, load_config, setup_logging

logger = logging.getLogger(__name__)


def evaluate_with_transform(
    model: torch.nn.Module,
    raw_paths: list,
    labels: list,
    config: dict,
    transform,
    device: torch.device,
) -> dict:
    """Helper to evaluate the model on the robustness dataset with a specific transform."""
    ds = RobustnessDataset(
        image_paths=raw_paths,
        labels=labels,
        transform=transform,
        input_size=config["model"]["input_size"],
        ela_quality=config["ela"]["quality"],
        ela_scale=config["ela"]["scale"],
        mean=config["augmentation"]["val"]["normalize"]["mean"],
        std=config["augmentation"]["val"]["normalize"]["std"],
    )
    
    loader = DataLoader(
        ds, 
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )
    
    return evaluate_model(model, loader, device)


def load_checkpoint_and_model(run_id: str, checkpoint_dir: str, device: torch.device):
    """Loads a specific run's config, initializes the model, and loads weights."""
    ckpt_dir = Path(checkpoint_dir)
    config_path = ckpt_dir / f"{run_id}_best_config.yaml"
    ckpt_path = ckpt_dir / f"{run_id}_best.pt"
    
    if not config_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint or config for {run_id} in {ckpt_dir}")
        
    config = load_config(str(config_path))
    model = create_model(config, use_lora=True)
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, config


def generate_degradation_curve(
    best_run_id: str,
    raw_paths: list,
    labels: list,
    device: torch.device,
    results_dir: Path,
    checkpoint_dir: str,
):
    """Phase 1: Sweep severities (1-5) and plot accuracy degradation."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Generating Degradation Curve for %s", best_run_id)
    logger.info("=" * 60)
    
    model, config = load_checkpoint_and_model(best_run_id, checkpoint_dir, device)
    
    corruptions = ["blur", "jpeg", "glare"]
    severities = [1, 2, 3, 4, 5]
    
    results = {c: [] for c in corruptions}
    
    # Pre-evaluate Clean accuracy (Severity 0 baseline for the plot, though we plot 1-5)
    # Actually, let's just plot 1-5 to keep X-axis uniform as requested.
    
    for c_type in corruptions:
        for s in severities:
            transform = get_corruption_sweep(c_type, s)
            metrics = evaluate_with_transform(model, raw_paths, labels, config, transform, device)
            acc = metrics["accuracy"]
            f1 = metrics["f1_score"]
            logger.info("  %s | Severity %d -> Acc: %.4f | F1: %.4f", c_type, s, acc, f1)
            results[c_type].append(acc)
            
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(severities, results["blur"], marker="o", label="Blur", linewidth=2)
    plt.plot(severities, results["jpeg"], marker="s", label="JPEG", linewidth=2)
    plt.plot(severities, results["glare"], marker="^", label="Glare", linewidth=2)
    
    plt.title("Robustness Degradation Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Severity Level (1-5)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(severities)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    
    # Caption mapping
    caption = (
        "Severity mappings:\n"
        "Blur (kernel): 1=3, 2=5, 3=7, 4=11, 5=15\n"
        "JPEG (quality): 1=90, 2=70, 3=50, 4=30, 5=10\n"
        "Glare (intensity): 1=Low, 5=High"
    )
    plt.figtext(0.15, -0.05, caption, wrap=True, horizontalalignment='left', fontsize=10, color="gray")
    
    out_path = results_dir / "degradation_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info("Saved degradation curve to %s", out_path)


def evaluate_all_checkpoints(
    run_ids: list,
    raw_paths: list,
    labels: list,
    device: torch.device,
    results_dir: Path,
    checkpoint_dir: str,
):
    """Phase 2: Benchmark all runs on Clean vs Noisy and generate a CSV table."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Consolidated Robustness Table")
    logger.info("=" * 60)
    
    noisy_transform = get_static_noisy_condition()
    out_path = results_dir / "robustness_results.csv"
    
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Exp ID", "Backbone", "Loss", "Optimizer", 
            "Acc (Clean)", "F1 (Clean)", 
            "Acc (Noisy)", "F1 (Noisy)"
        ])
        
        for run_id in run_ids:
            logger.info("Evaluating %s...", run_id)
            model, config = load_checkpoint_and_model(run_id, checkpoint_dir, device)
            
            backbone = config["model"]["backbone"]
            loss_fn = config["training"].get("loss", "cross_entropy")
            opt = config["training"]["optimizer"]
            
            # Clean evaluation (no transform)
            clean_metrics = evaluate_with_transform(
                model, raw_paths, labels, config, None, device
            )
            
            # Noisy evaluation
            noisy_metrics = evaluate_with_transform(
                model, raw_paths, labels, config, noisy_transform, device
            )
            
            writer.writerow([
                run_id, backbone, loss_fn, opt,
                f"{clean_metrics['accuracy']:.4f}",
                f"{clean_metrics['f1_score']:.4f}",
                f"{noisy_metrics['accuracy']:.4f}",
                f"{noisy_metrics['f1_score']:.4f}"
            ])
            
            logger.info(
                "  Clean Acc: %.4f | Noisy Acc: %.4f",
                clean_metrics['accuracy'], noisy_metrics['accuracy']
            )
            
    logger.info("Saved consolidated results to %s", out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Day 5 Robustness Evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--best_run", default="exp_04", help="Run ID for the degradation curve")
    parser.add_argument("--runs", nargs="+", default=["exp_01", "exp_02", "exp_03", "exp_04"])
    args = parser.parse_args()
    
    setup_logging()
    
    # We use the base config just to get the dataset paths and splits
    base_config = load_config(args.config)
    device = get_device("auto")
    results_dir = Path(base_config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = base_config["paths"]["checkpoints"]
    
    # 1. Get identical test split mapped to raw image paths
    raw_paths, labels = get_test_split_raw_paths(base_config)
    logger.info("Loaded identical test split mapped to raw files: %d images", len(raw_paths))
    
    # 2. Generate Degradation Curve for the best model
    generate_degradation_curve(args.best_run, raw_paths, labels, device, results_dir, ckpt_dir)
    
    # 3. Consolidate Benchmarks for all models
    evaluate_all_checkpoints(args.runs, raw_paths, labels, device, results_dir, ckpt_dir)
