"""
Carlini-Wagner (C&W) L2 attack.

Reference: Carlini & Wagner (2017); Appendix F of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CarliniWagner:
    """Carlini-Wagner L2 attack with protocol constraint projection."""

    def __init__(self, confidence: float = 1.0, lr: float = 0.01,
                 iterations: int = 100, clip_min: float = 0.0,
                 clip_max: float = 1.0, c: float = 1.0):
        self.confidence = confidence
        self.lr = lr
        self.iterations = iterations
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.c = c

    @torch.enable_grad()
    def generate(self, model: nn.Module, x: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
        x_orig = x.clone().detach()

        # Use tanh-space parameterization for box constraints
        w = torch.zeros_like(x, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=self.lr)

        best_adv = x_orig.clone()
        best_l2 = torch.full((x.size(0),), float("inf"), device=x.device)

        for _ in range(self.iterations):
            x_adv = 0.5 * (torch.tanh(w) + 1.0) * (self.clip_max - self.clip_min) + self.clip_min

            logits = model(x_adv)
            if isinstance(logits, tuple):
                logits = logits[0]

            # f(x') = max(Z(x')_y - max_{i != y} Z(x')_i, -kappa)
            real = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            other_mask = torch.ones_like(logits).scatter_(1, y.unsqueeze(1), 0)
            other = (logits * other_mask - 1e9 * (1 - other_mask)).max(dim=1).values

            f_loss = torch.clamp(real - other + self.confidence, min=0.0)
            l2_dist = ((x_adv - x_orig) ** 2).view(x.size(0), -1).sum(dim=1)

            loss = l2_dist.sum() + self.c * f_loss.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track best adversarial examples
            with torch.no_grad():
                pred = model(x_adv)
                if isinstance(pred, tuple):
                    pred = pred[0]
                misclassified = pred.argmax(dim=1) != y
                improved = misclassified & (l2_dist < best_l2)
                best_l2[improved] = l2_dist[improved]
                best_adv[improved] = x_adv[improved].clone()

        return best_adv.detach()
