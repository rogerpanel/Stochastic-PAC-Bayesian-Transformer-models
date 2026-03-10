"""
Fast Gradient Sign Method (FGSM) attack.

Reference: Goodfellow et al. (2015); Appendix F, Algorithm F.1 of the paper.
For network data, protocol constraints are applied after perturbation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FGSM:
    """FGSM with optional protocol-constraint projection."""

    def __init__(self, epsilon: float = 0.1, clip_min: float = 0.0,
                 clip_max: float = 1.0):
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max

    @torch.enable_grad()
    def generate(self, model: nn.Module, x: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
        x_adv = x.clone().detach().requires_grad_(True)

        logits = model(x_adv)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(logits, y)

        grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]

        x_adv = x_adv.detach() + self.epsilon * grad.sign()
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        return x_adv.detach()
