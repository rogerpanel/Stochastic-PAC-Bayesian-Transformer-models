"""
Expectation-over-Transformations (EOT) wrapper for adversarial attacks.

EOT generates adversarial examples by solving (Eq. 8):
    x_adv = argmax_{x' in A_eps(x)} E_{theta ~ q_phi}[L(f_theta(x'), y)]

The expectation over stochastic model instantiations is approximated
with M gradient samples (default M = 5, Appendix D).
"""

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F


@runtime_checkable
class Attack(Protocol):
    """Minimal interface that FGSM / PGD / CW implement."""
    def generate(self, model: nn.Module, x: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor: ...


class EOTWrapper:
    """Wraps any gradient-based attack to use EOT against stochastic models.

    Instead of computing the gradient of a single forward pass, EOT averages
    gradients over *M* stochastic samples of the model parameters.
    """

    def __init__(self, base_attack: Attack, eot_samples: int = 5):
        self.base_attack = base_attack
        self.eot_samples = eot_samples

    @torch.enable_grad()
    def generate(self, model: nn.Module, x: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
        """Generate EOT adversarial examples.

        For PGD-style iterative attacks, we average gradients at each step.
        For single-step attacks (FGSM), we average the gradient directly.
        """
        model.train()  # ensure stochastic sampling

        if hasattr(self.base_attack, "steps"):
            return self._iterative_eot(model, x, y)
        return self._single_step_eot(model, x, y)

    def _single_step_eot(self, model: nn.Module, x: torch.Tensor,
                         y: torch.Tensor) -> torch.Tensor:
        x_in = x.clone().detach().requires_grad_(True)

        avg_grad = torch.zeros_like(x)
        for _ in range(self.eot_samples):
            if x_in.grad is not None:
                x_in.grad.zero_()
            logits = model(x_in)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(logits, y)
            g = torch.autograd.grad(loss, x_in, retain_graph=False)[0]
            avg_grad += g

        avg_grad /= self.eot_samples

        epsilon = self.base_attack.epsilon
        x_adv = x.detach() + epsilon * avg_grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv.detach()

    def _iterative_eot(self, model: nn.Module, x: torch.Tensor,
                       y: torch.Tensor) -> torch.Tensor:
        epsilon = self.base_attack.epsilon
        step_size = self.base_attack.step_size
        steps = self.base_attack.steps
        x_orig = x.clone().detach()

        # Random start
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(steps):
            x_adv_var = x_adv.clone().detach().requires_grad_(True)

            avg_grad = torch.zeros_like(x)
            for _ in range(self.eot_samples):
                if x_adv_var.grad is not None:
                    x_adv_var.grad.zero_()
                logits = model(x_adv_var)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = F.cross_entropy(logits, y)
                g = torch.autograd.grad(loss, x_adv_var, retain_graph=False)[0]
                avg_grad += g
            avg_grad /= self.eot_samples

            x_adv = x_adv.detach() + step_size * avg_grad.sign()
            delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

        return x_adv.detach()
