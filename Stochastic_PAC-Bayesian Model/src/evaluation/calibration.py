"""
Calibration Metrics: ECE, MCE, Brier Score, AUROC.

Reference: Table 3 and Appendix G of the paper.
  - ECE (Expected Calibration Error): weighted average |acc - conf| in bins
  - MCE (Maximum Calibration Error): max |acc - conf| across bins
  - Brier Score: mean squared error between probabilities and one-hot labels
  - AUROC: area under ROC for detecting mispredictions via entropy
"""

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class CalibrationMetrics:
    """Compute calibration metrics from predicted probabilities and targets."""

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    def compute_all(
        self,
        probs: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> Dict[str, float]:
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        confidences = probs.max(axis=-1)
        predictions = probs.argmax(axis=-1)
        correct = (predictions == targets).astype(float)

        ece = self._ece(confidences, correct)
        mce = self._mce(confidences, correct)
        brier = self._brier(probs, targets)
        auroc = self._auroc_misprediction(probs, targets)

        return {"ece": ece, "mce": mce, "brier": brier, "auroc": auroc}

    def _ece(self, confidences: np.ndarray,
             correct: np.ndarray) -> float:
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() > 0:
                prop = mask.mean()
                acc = correct[mask].mean()
                conf = confidences[mask].mean()
                ece += abs(acc - conf) * prop
        return float(ece)

    def _mce(self, confidences: np.ndarray,
             correct: np.ndarray) -> float:
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        mce = 0.0
        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() > 0:
                acc = correct[mask].mean()
                conf = confidences[mask].mean()
                mce = max(mce, abs(acc - conf))
        return float(mce)

    @staticmethod
    def _brier(probs: np.ndarray, targets: np.ndarray) -> float:
        n_classes = probs.shape[-1]
        one_hot = np.eye(n_classes)[targets]
        return float(((probs - one_hot) ** 2).sum(axis=-1).mean())

    @staticmethod
    def _auroc_misprediction(probs: np.ndarray,
                             targets: np.ndarray) -> float:
        """AUROC for detecting mispredictions using predictive entropy."""
        predictions = probs.argmax(axis=-1)
        correct = (predictions == targets).astype(int)
        entropy = -(probs * np.log(probs + 1e-10)).sum(axis=-1)
        try:
            return float(roc_auc_score(1 - correct, entropy))
        except ValueError:
            return 0.5  # only one class present
