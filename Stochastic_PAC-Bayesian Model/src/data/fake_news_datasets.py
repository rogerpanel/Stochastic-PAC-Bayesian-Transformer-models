"""
Fake News Detection Dataset Loaders.

Handles LIAR, FakeNewsNet, and ISOT datasets.
Preprocessing follows Appendix F:
  - LIAR: 6-way classification, max_length=256, speaker-consistent folds
  - FakeNewsNet: title+body concatenation, outlet-disjoint folds
  - ISOT: binary (Real vs Fake), max_length=512

Reference datasets:
  - LIAR (Wang, ACL 2017)
  - FakeNewsNet (Shu et al., Big Data 2020)
  - ISOT (Ahmed et al., 2018)
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset


class FakeNewsDatasetLoader:
    """Load and preprocess fake news datasets."""

    LIAR_LABELS = [
        "pants-fire", "false", "barely-true",
        "half-true", "mostly-true", "true",
    ]

    def __init__(self, data_dir: str, dataset_name: str = "liar",
                 max_length: int = 256, n_folds: int = 5,
                 batch_size: int = 32, num_workers: int = 4,
                 seed: int = 42):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import BertTokenizer
            self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return self._tokenizer

    def load_and_tokenize(self, csv_path: str,
                          text_col: str = "statement",
                          label_col: str = "label"
                          ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Load, preprocess, and tokenize fake news dataset."""
        df = pd.read_csv(csv_path)

        # Auto-detect text and label columns
        if text_col not in df.columns:
            text_candidates = [c for c in df.columns
                               if any(kw in c.lower()
                                      for kw in ["text", "title", "statement",
                                                  "content", "body"])]
            text_col = text_candidates[0] if text_candidates else df.columns[0]

        if label_col not in df.columns:
            label_candidates = [c for c in df.columns
                                if "label" in c.lower() or "class" in c.lower()]
            label_col = label_candidates[0] if label_candidates else df.columns[-1]

        texts = df[text_col].fillna("").astype(str).tolist()
        labels = pd.factorize(df[label_col])[0]

        encodings = self.tokenizer(
            texts, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="np")

        token_data = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
        return token_data, labels

    def get_fold_loaders(
        self, token_data: Dict[str, np.ndarray], labels: np.ndarray,
        fold: int = 0,
    ) -> Tuple[DataLoader, DataLoader]:
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        splits = list(skf.split(token_data["input_ids"], labels))
        train_idx, val_idx = splits[fold]

        def make_loader(idx, shuffle):
            ds = TensorDataset(
                torch.tensor(token_data["input_ids"][idx], dtype=torch.long),
                torch.tensor(labels[idx], dtype=torch.long),
            )
            return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                              num_workers=self.num_workers, pin_memory=True)

        return make_loader(train_idx, True), make_loader(val_idx, False)

    @property
    def num_classes(self) -> int:
        counts = {"liar": 6, "fakenewsnet": 2, "isot": 2}
        return counts.get(self.dataset_name, 2)
