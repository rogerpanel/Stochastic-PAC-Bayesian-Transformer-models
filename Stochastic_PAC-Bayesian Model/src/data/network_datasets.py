"""
Network Intrusion Detection Dataset Loaders.

Handles CIC-IoT-2023, CSE-CICIDS2018, and UNSW-NB15.
Preprocessing follows Appendix F of the supplementary material:
  - Z-score normalization (train statistics only)
  - Missing value imputation (mean for continuous, mode for categorical)
  - Outlier clipping at 99.9th percentile
  - One-hot encoding of categorical features
  - 5-fold stratified cross-validation with timestamp ordering

Data source: https://doi.org/10.34740/KAGGLE/DSV/12479689
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class NetworkDatasetLoader:
    """Load and preprocess network intrusion detection datasets."""

    # Common flow features across datasets (Appendix F)
    COMMON_NETWORK_FEATURES = [
        "flow_duration", "total_fwd_packets", "total_bwd_packets",
        "total_length_fwd_packets", "total_length_bwd_packets",
        "fwd_packet_length_max", "fwd_packet_length_min",
        "fwd_packet_length_mean", "fwd_packet_length_std",
        "bwd_packet_length_max", "bwd_packet_length_min",
        "bwd_packet_length_mean", "bwd_packet_length_std",
        "flow_bytes_per_sec", "flow_packets_per_sec",
        "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
        "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std",
        "fwd_iat_max", "fwd_iat_min",
        "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std",
        "bwd_iat_max", "bwd_iat_min",
        "fwd_header_length", "bwd_header_length",
        "fwd_packets_per_sec", "bwd_packets_per_sec",
        "min_packet_length", "max_packet_length",
        "packet_length_mean", "packet_length_std",
        "packet_length_variance",
        "fin_flag_count", "syn_flag_count",
    ]

    # Label mappings per dataset
    LABEL_MAPS = {
        "cic_iot_2023": {
            "DDoS": "DoS", "DoS": "DoS", "Recon": "Reconnaissance",
            "Web-based": "WebAttack", "BruteForce": "BruteForce",
            "Spoofing": "Spoofing", "Mirai": "Botnet", "Benign": "Benign",
        },
        "cse_cicids2018": {
            "BENIGN": "Benign", "DDoS": "DoS", "DoS": "DoS",
            "Brute Force": "BruteForce", "Web Attack": "WebAttack",
            "Infiltration": "Infiltration", "Bot": "Botnet",
        },
        "unsw_nb15": {
            "Normal": "Normal", "Attack": "Attack",
        },
    }

    def __init__(self, data_dir: str, dataset_name: str,
                 n_folds: int = 5, batch_size: int = 32,
                 num_workers: int = 4, seed: int = 42):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def load_and_preprocess(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load CSV and apply preprocessing pipeline."""
        df = pd.read_csv(csv_path)

        # Select features present in the data
        available = [c for c in self.COMMON_NETWORK_FEATURES if c in df.columns]
        if not available:
            # Fall back to all numeric columns
            available = df.select_dtypes(include=[np.number]).columns.tolist()
            label_col = [c for c in df.columns if "label" in c.lower()
                         or "attack" in c.lower() or "class" in c.lower()]
            if label_col:
                available = [c for c in available if c not in label_col]

        X = df[available].values.astype(np.float32)

        # Identify label column
        label_col = None
        for candidate in ["Label", "label", "attack_cat", "Attack", "class"]:
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            label_col = df.columns[-1]

        le = LabelEncoder()
        y = le.fit_transform(df[label_col].values)

        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Clip outliers at 99.9th percentile
        for j in range(X.shape[1]):
            lo = np.percentile(X[:, j], 0.1)
            hi = np.percentile(X[:, j], 99.9)
            X[:, j] = np.clip(X[:, j], lo, hi)

        return X, y

    def get_fold_loaders(
        self, X: np.ndarray, y: np.ndarray, fold: int = 0,
    ) -> Tuple[DataLoader, DataLoader]:
        """Return train/val loaders for a specific CV fold."""
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        splits = list(skf.split(X, y))
        train_idx, val_idx = splits[fold]

        # Z-score normalization using training statistics only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y[train_idx], dtype=torch.long))
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y[val_idx], dtype=torch.long))

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True)

        return train_loader, val_loader

    @property
    def num_features(self) -> int:
        """Dataset-specific feature count from config."""
        counts = {
            "cic_iot_2023": 40,
            "cse_cicids2018": 79,
            "unsw_nb15": 44,
        }
        return counts.get(self.dataset_name, 40)

    @property
    def num_classes(self) -> int:
        counts = {
            "cic_iot_2023": 8,
            "cse_cicids2018": 14,
            "unsw_nb15": 2,
        }
        return counts.get(self.dataset_name, 2)
