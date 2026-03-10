"""
Multi-Objective Loss Function (Eq. 9 from the main paper).

L_tot = L_cls + lambda_1 * L_KL + lambda_2 * L_cal + lambda_3 * L_adv + lambda_4 * L_reg

Components:
  L_cls  : cross-entropy classification loss
  L_KL   : PAC-Bayesian KL divergence penalty  sum_l KL(q_phi(theta_l) || p(theta_l))
  L_cal  : Brier score calibration surrogate
  L_adv  : EOT adversarial loss (cross-entropy on adversarial examples)
  L_reg  : L2 weight decay
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiObjectiveLoss(nn.Module):
    """Five-component multi-objective loss as described in Eq. (9)."""

    def __init__(self, lambda_kl: float = 0.01, lambda_cal: float = 0.05,
                 lambda_adv: float = 0.2, lambda_reg: float = 0.001):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_cal = lambda_cal
        self.lambda_adv = lambda_adv
        self.lambda_reg = lambda_reg

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
        adv_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = logits.device

        # 1. Classification loss
        loss_cls = F.cross_entropy(logits, targets)

        # 2. KL divergence (PAC-Bayesian complexity penalty)
        if hasattr(model, "compute_kl_loss"):
            loss_kl = model.compute_kl_loss()
        else:
            loss_kl = torch.tensor(0.0, device=device)

        # 3. Calibration loss (Brier score surrogate)
        probs = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        loss_cal = ((probs - one_hot) ** 2).sum(dim=-1).mean()

        # 4. Adversarial loss
        if adv_logits is not None:
            loss_adv = F.cross_entropy(adv_logits, targets)
        else:
            loss_adv = torch.tensor(0.0, device=device)

        # 5. L2 regularization
        loss_reg = torch.tensor(0.0, device=device)
        for p in model.parameters():
            if p.requires_grad:
                loss_reg = loss_reg + p.pow(2).sum()

        # Total
        loss_total = (
            loss_cls
            + self.lambda_kl * loss_kl
            + self.lambda_cal * loss_cal
            + self.lambda_adv * loss_adv
            + self.lambda_reg * loss_reg
        )

        return {
            "total": loss_total,
            "cls": loss_cls,
            "kl": loss_kl,
            "cal": loss_cal,
            "adv": loss_adv,
            "reg": loss_reg,
        }
