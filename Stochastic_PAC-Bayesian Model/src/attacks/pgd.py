"""
Projected Gradient Descent (PGD) attack.

Reference: Madry et al. (2018); Appendix F, Algorithm F.2 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PGD:
    """PGD with L-inf constraint and random initialization."""

    def __init__(self, epsilon: float = 0.1, steps: int = 10,
                 step_size: float | None = None,
                 clip_min: float = 0.0, clip_max: float = 1.0,
                 random_start: bool = True):
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size if step_size else epsilon / steps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.random_start = random_start

    @torch.enable_grad()
    def generate(self, model: nn.Module, x: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
        x_orig = x.clone().detach()

        if self.random_start:
            x_adv = x_orig + torch.empty_like(x_orig).uniform_(
                -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        else:
            x_adv = x_orig.clone()

        for _ in range(self.steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)

            logits = model(x_adv)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(logits, y)

            grad = torch.autograd.grad(loss, x_adv)[0]

            x_adv = x_adv.detach() + self.step_size * grad.sign()
            # Project back to epsilon-ball around original
            delta = torch.clamp(x_adv - x_orig, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_orig + delta, self.clip_min, self.clip_max)

        return x_adv.detach()
