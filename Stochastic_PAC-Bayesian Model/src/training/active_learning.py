"""
Uncertainty-Guided Active Learning (Algorithm 2, lines 46-58 of Appendix D).

Acquisition score:
    A(x) = alpha * U_epi(x) + beta * H[p(y|x)] + gamma * d(x, D_train)

Reduces labeling requirements by 68% (35% labels -> 95% of full performance).
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np


class UncertaintyGuidedAL:
    """Active learning with uncertainty-guided acquisition."""

    def __init__(self, model: nn.Module, device: str = "cuda",
                 alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2,
                 query_fraction: float = 0.05, mc_samples: int = 50):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.query_fraction = query_fraction
        self.mc_samples = mc_samples

    @torch.no_grad()
    def compute_acquisition_scores(
        self,
        pool_loader: DataLoader,
        train_features: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Compute acquisition scores for all samples in the unlabeled pool."""
        self.model.train()  # enable stochastic sampling

        all_scores = []
        for data, _ in pool_loader:
            data = data.to(self.device)
            B = data.size(0)

            # MC forward passes
            preds = []
            for _ in range(self.mc_samples):
                logits = self.model(data)
                if isinstance(logits, tuple):
                    logits = logits[0]
                preds.append(F.softmax(logits, dim=-1))
            preds = torch.stack(preds)  # (S, B, C)

            # Epistemic uncertainty: prediction variance
            u_epi = preds.var(dim=0).sum(dim=-1)  # (B,)

            # Predictive entropy
            mean_pred = preds.mean(dim=0)
            entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1)

            # Distance to training set (optional)
            if train_features is not None:
                # Use L2 distance to nearest training point
                flat = data.view(B, -1)
                dists = torch.cdist(flat.cpu(), train_features.view(
                    train_features.size(0), -1))
                min_dist = dists.min(dim=1).values.to(self.device)
            else:
                min_dist = torch.zeros(B, device=self.device)

            score = (self.alpha * u_epi
                     + self.beta * entropy
                     + self.gamma * min_dist)
            all_scores.append(score.cpu().numpy())

        return np.concatenate(all_scores)

    def select_queries(
        self,
        pool_loader: DataLoader,
        pool_size: int,
        train_features: torch.Tensor | None = None,
    ) -> List[int]:
        """Select top-k samples from the pool by acquisition score."""
        scores = self.compute_acquisition_scores(pool_loader, train_features)
        k = max(1, int(pool_size * self.query_fraction))
        top_indices = np.argsort(scores)[-k:]
        return top_indices.tolist()
