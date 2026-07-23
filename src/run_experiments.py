# src/run_experiments.py
"""
Experiment runner for Day 4 — LoRA + Hyperparameter Log.

Executes the requested 4-run matrix to evaluate backbone, optimizer, and loss function
combinations on the ELA forgery detection dataset.

Matrix:
| Run | Backbone          | Optimizer | Loss         |
|-----|-------------------|-----------|--------------|
| 1   | resnet18          | adam      | cross_entropy|
| 2   | resnet18          | sgd       | cross_entropy|
| 3   | efficientnet_b0   | adam      | cross_entropy|
| 4   | efficientnet_b0   | adam      | focal        |

All experiments apply LoRA.
Results are summarized in `results/hparam_log.csv`.
"""

import csv
import logging
from copy import deepcopy
from pathlib import Path

from src.train import train
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

# The specific 4 runs requested
EXPERIMENTS = [
    {
        "run_id": "exp_01",
        "backbone": "resnet18",
        "optimizer": "adam",
        "loss": "cross_entropy",
    },
    {
        "run_id": "exp_02",
        "backbone": "resnet18",
        "optimizer": "sgd",
        "loss": "cross_entropy",
    },
    {
        "run_id": "exp_03",
        "backbone": "efficientnet_b0",
        "optimizer": "adam",
        "loss": "cross_entropy",
    },
    {
        "run_id": "exp_04",
        "backbone": "efficientnet_b0",
        "optimizer": "adam",
        "loss": "focal",
    },
]


def run_sweep(config_path: str = "configs/config.yaml"):
    setup_logging()
    base_config = load_config(config_path)
    
    results_dir = Path(base_config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    hparam_log_path = results_dir / "hparam_log.csv"
    
    # Initialize the aggregate hparam log
    with open(hparam_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "backbone", "optimizer", "loss_fn", "use_lora",
            "val_acc_clean", "val_f1_clean", "best_epoch"
        ])
    
    logger.info("=" * 70)
    logger.info("Starting Experiment Sweep: %d runs", len(EXPERIMENTS))
    logger.info("=" * 70)
    
    for exp in EXPERIMENTS:
        run_id = exp["run_id"]
        backbone = exp["backbone"]
        optimizer = exp["optimizer"]
        loss_fn = exp["loss"]
        
        logger.info(">>> Launching %s: %s | %s | %s", run_id, backbone, optimizer, loss_fn)
        
        # Deepcopy config to safely mutate it per-run
        cfg = deepcopy(base_config)
        cfg["model"]["backbone"] = backbone
        cfg["training"]["optimizer"] = optimizer
        cfg["training"]["loss"] = loss_fn
        
        # Run training loop for this experiment with LoRA enabled
        # We pass use_lora=True to actually invoke PEFT.
        result = train(
            config=cfg,
            backbone=backbone,
            use_lora=True,
            loss_fn_override=loss_fn,
            run_id=run_id,
        )
        
        # Append summary to the log
        with open(hparam_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                run_id,
                backbone,
                optimizer,
                loss_fn,
                True, # use_lora
                f"{result['best_metrics']['accuracy']:.4f}",
                f"{result['best_metrics']['f1_score']:.4f}",
                result["best_epoch"],
            ])
            
        logger.info("<<< Finished %s", run_id)
        
    logger.info("=" * 70)
    logger.info("Sweep Complete! Aggregate results saved to: %s", hparam_log_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full Day 4 Experiment Sweep")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    run_sweep(args.config)
