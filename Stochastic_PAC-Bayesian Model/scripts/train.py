#!/usr/bin/env python3
"""
Training script for Stochastic PAC-Bayesian Transformer.

Usage:
    # Network intrusion detection (tabular)
    python scripts/train.py \
        --domain network \
        --dataset cic_iot_2023 \
        --data_path /path/to/cic_iot_2023.csv \
        --config configs/default_config.yaml

    # Toxic content detection (text)
    python scripts/train.py \
        --domain toxic \
        --dataset metahate \
        --data_path /path/to/metahate.csv \
        --config configs/default_config.yaml

    # Fake news detection (text)
    python scripts/train.py \
        --domain fakenews \
        --dataset liar \
        --data_path /path/to/liar.csv \
        --config configs/default_config.yaml

Reference: Section 4 and Appendix D of the paper.
"""

import argparse
import logging
import os
import sys
import random

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.stochastic_transformer import StochasticPACBayesianTransformer, ModelConfig
from src.training.trainer import StochasticTrainer
from src.data.network_datasets import NetworkDatasetLoader
from src.data.toxic_datasets import ToxicDatasetLoader
from src.data.fake_news_datasets import FakeNewsDatasetLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility (Appendix J)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config: dict, domain: str, dataset: str) -> StochasticPACBayesianTransformer:
    mc = config.get("model", {})
    vc = config.get("variational", {})
    mcs = config.get("mc_sampling", {})

    ds_cfg = config.get("datasets", {}).get(dataset, {})
    input_type = ds_cfg.get("input_type", "tabular" if domain == "network" else "text")
    num_classes = ds_cfg.get("num_classes", 2)
    input_dim = ds_cfg.get("num_features", None) if input_type == "tabular" else None

    model_config = ModelConfig(
        num_layers=mc.get("num_layers", 12),
        num_heads=mc.get("num_heads", 8),
        d_model=mc.get("d_model", 256),
        d_ff=mc.get("d_ff", 1024),
        dropout=mc.get("dropout", 0.1),
        activation=mc.get("activation", "gelu"),
        bayesian_layers=mc.get("bayesian_layers", [3, 6, 9, 12]),
        prior_sigma=vc.get("prior_variance", 0.01) ** 0.5,
        posterior_log_var_init=vc.get("posterior_log_var_init", -6.0),
        input_dim=input_dim,
        num_classes=num_classes,
        input_type=input_type,
        pretrained_backbone=mc.get("pretrained_backbone", "bert-base-uncased"),
        mc_train_samples=mcs.get("train_samples", 30),
        mc_inference_samples=mcs.get("inference_samples", 50),
    )
    return StochasticPACBayesianTransformer(model_config)


def main():
    parser = argparse.ArgumentParser(description="Train Stochastic PAC-Bayesian Transformer")
    parser.add_argument("--domain", choices=["network", "toxic", "fakenews"], required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)

    # --- Load data ---
    logger.info(f"Loading {args.dataset} from {args.data_path}")
    if args.domain == "network":
        loader = NetworkDatasetLoader(
            data_dir=os.path.dirname(args.data_path),
            dataset_name=args.dataset,
            batch_size=config.get("training", {}).get("batch_size", 32),
            seed=args.seed,
        )
        X, y = loader.load_and_preprocess(args.data_path)
        train_loader, val_loader = loader.get_fold_loaders(X, y, fold=args.fold)
    elif args.domain == "toxic":
        loader = ToxicDatasetLoader(
            data_dir=os.path.dirname(args.data_path),
            dataset_name=args.dataset,
            max_length=config.get("datasets", {}).get(args.dataset, {}).get("max_length", 512),
            batch_size=config.get("training", {}).get("batch_size", 32),
            seed=args.seed,
        )
        token_data, labels = loader.load_and_tokenize(args.data_path)
        train_loader, val_loader = loader.get_fold_loaders(token_data, labels, fold=args.fold)
    else:
        loader = FakeNewsDatasetLoader(
            data_dir=os.path.dirname(args.data_path),
            dataset_name=args.dataset,
            max_length=config.get("datasets", {}).get(args.dataset, {}).get("max_length", 256),
            batch_size=config.get("training", {}).get("batch_size", 32),
            seed=args.seed,
        )
        token_data, labels = loader.load_and_tokenize(args.data_path)
        train_loader, val_loader = loader.get_fold_loaders(token_data, labels, fold=args.fold)

    # --- Build model ---
    logger.info("Building model")
    model = build_model(config, args.domain, args.dataset)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}  Trainable: {trainable_params:,}")

    # --- Train ---
    checkpoint_dir = os.path.join(args.output_dir, args.dataset, f"fold_{args.fold}")
    trainer = StochasticTrainer(model, config, device=args.device)
    history = trainer.fit(train_loader, val_loader, checkpoint_dir=checkpoint_dir)

    logger.info("Training complete!")
    logger.info(f"Best val F1: {max(history['val_f1']):.4f}")
    logger.info(f"Best val ECE: {min(history['val_ece']):.4f}")


if __name__ == "__main__":
    main()
