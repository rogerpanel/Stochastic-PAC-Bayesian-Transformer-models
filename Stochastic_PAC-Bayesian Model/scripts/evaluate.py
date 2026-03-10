#!/usr/bin/env python3
"""
Evaluation script: calibration, robustness, and uncertainty analysis.

Usage:
    python scripts/evaluate.py \
        --checkpoint outputs/cic_iot_2023/fold_0/best_model.pt \
        --domain network \
        --dataset cic_iot_2023 \
        --data_path /path/to/data.csv \
        --config configs/default_config.yaml

Produces:
  - Calibration metrics (ECE, MCE, Brier, AUROC)
  - Adversarial robustness (FGSM, PGD, C&W retention)
  - Uncertainty decomposition (epistemic vs aleatoric)
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.stochastic_transformer import StochasticPACBayesianTransformer, ModelConfig
from src.evaluation.calibration import CalibrationMetrics
from src.evaluation.robustness import RobustnessEvaluator
from src.evaluation.uncertainty import UncertaintyQuantifier
from src.data.network_datasets import NetworkDatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stochastic PAC-Bayesian Transformer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--domain", choices=["network", "toxic", "fakenews"], required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model
    from scripts.train import build_model
    model = build_model(config, args.domain, args.dataset)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    # Load data
    if args.domain == "network":
        loader = NetworkDatasetLoader(
            data_dir=os.path.dirname(args.data_path),
            dataset_name=args.dataset, seed=42)
        X, y = loader.load_and_preprocess(args.data_path)
        _, val_loader = loader.get_fold_loaders(X, y, fold=args.fold)
    else:
        logger.error("Text domain evaluation requires tokenized data; "
                      "see demo notebook for full example.")
        return

    # --- Calibration ---
    logger.info("Computing calibration metrics...")
    cal = CalibrationMetrics(n_bins=15)

    all_probs, all_targets = [], []
    for data, targets in val_loader:
        data = data.to(args.device)
        with torch.no_grad():
            logits = model(data)
            if isinstance(logits, tuple):
                logits = logits[0]
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_targets.append(targets.numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    cal_metrics = cal.compute_all(probs, targets)
    logger.info(f"Calibration: {cal_metrics}")

    # --- Robustness ---
    logger.info("Evaluating adversarial robustness...")
    eps = config.get("datasets", {}).get(args.dataset, {}).get("epsilon", 0.1)
    rob_eval = RobustnessEvaluator(model, device=args.device, epsilon=eps)
    rob_metrics = rob_eval.evaluate_all(val_loader)
    logger.info(f"Robustness: {rob_metrics}")

    # --- Uncertainty ---
    logger.info("Computing uncertainty decomposition...")
    uq = UncertaintyQuantifier(model, num_samples=50, device=args.device)
    sample_batch = next(iter(val_loader))
    data_sample = sample_batch[0][:32].to(args.device)
    unc = uq.predict_with_uncertainty(data_sample)
    unc_summary = {
        "mean_epistemic": unc["epistemic"].mean().item(),
        "mean_aleatoric": unc["aleatoric"].mean().item(),
        "mean_total": unc["total"].mean().item(),
    }
    logger.info(f"Uncertainty: {unc_summary}")

    # --- Save results ---
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "dataset": args.dataset,
        "fold": args.fold,
        "calibration": cal_metrics,
        "robustness": rob_metrics,
        "uncertainty": unc_summary,
    }
    out_path = os.path.join(args.output_dir, f"{args.dataset}_fold{args.fold}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
