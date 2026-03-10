"""
Uncertainty Quantification via Monte Carlo Sampling.

Decomposes total predictive uncertainty into:
  - Epistemic (model) uncertainty: variance across MC samples
  - Aleatoric (data) uncertainty: mean entropy of individual predictions

Reference: Section 4, Cell 7 of original notebook; Appendix H Table H.2.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyQuantifier:
    """MC-based uncertainty decomposition."""

    def __init__(self, model: nn.Module, num_samples: int = 50,
                 device: str = "cuda"):
        self.model = model
        self.num_samples = num_samples
        self.device = device

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Run MC forward passes and decompose uncertainty.

        Returns:
            mean_probs: (B, C) averaged predictive distribution
            epistemic:  (B,)  variance-based epistemic uncertainty
            aleatoric:  (B,)  entropy-based aleatoric uncertainty
            total:      (B,)  predictive entropy (total uncertainty)
            predictions:(B,)  argmax of mean prediction
        """
        self.model.train()  # enable dropout + weight sampling

        all_probs = []
        for _ in range(self.num_samples):
            logits = self.model(x, attention_mask=attention_mask)
            if isinstance(logits, tuple):
                logits = logits[0]
            all_probs.append(F.softmax(logits, dim=-1))

        all_probs = torch.stack(all_probs)          # (S, B, C)
        mean_probs = all_probs.mean(dim=0)           # (B, C)
        predictions = mean_probs.argmax(dim=-1)      # (B,)

        # Epistemic uncertainty (model uncertainty)
        epistemic = all_probs.var(dim=0).sum(dim=-1)  # (B,)

        # Aleatoric uncertainty (data uncertainty)
        per_sample_entropy = -(
            all_probs * torch.log(all_probs + 1e-10)
        ).sum(dim=-1)  # (S, B)
        aleatoric = per_sample_entropy.mean(dim=0)    # (B,)

        # Total predictive entropy
        total = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        self.model.eval()

        return {
            "mean_probs": mean_probs,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total": total,
            "predictions": predictions,
            "all_probs": all_probs,
        }
