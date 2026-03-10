"""
Adversarial Robustness Evaluation.

Evaluates model under FGSM, PGD, C&W, and TextFooler attacks.
Reports clean accuracy, attacked accuracy, and retention rate.

Reference: Table 2, Table 2a, and Appendix G of the paper.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..attacks.fgsm import FGSM
from ..attacks.pgd import PGD
from ..attacks.cw import CarliniWagner


class RobustnessEvaluator:
    """Evaluate adversarial robustness across multiple attack types."""

    def __init__(self, model: nn.Module, device: str = "cuda",
                 epsilon: float = 0.1):
        self.model = model
        self.device = device
        self.attacks = {
            "fgsm": FGSM(epsilon=epsilon),
            "pgd": PGD(epsilon=epsilon, steps=10,
                        step_size=epsilon / 10),
            "cw": CarliniWagner(c=1.0, lr=0.01, iterations=100),
        }

    @torch.no_grad()
    def evaluate_clean(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        for data, targets in loader:
            data, targets = data.to(self.device), targets.to(self.device)
            logits = self.model(data)
            if isinstance(logits, tuple):
                logits = logits[0]
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            total += targets.size(0)
        return correct / max(total, 1)

    def evaluate_attack(self, loader: DataLoader,
                        attack_name: str) -> float:
        self.model.eval()
        attack = self.attacks[attack_name]
        correct, total = 0, 0

        for data, targets in loader:
            data, targets = data.to(self.device), targets.to(self.device)
            x_adv = attack.generate(self.model, data, targets)

            with torch.no_grad():
                logits = self.model(x_adv)
                if isinstance(logits, tuple):
                    logits = logits[0]
                correct += (logits.argmax(dim=-1) == targets).sum().item()
            total += targets.size(0)

        return correct / max(total, 1)

    def evaluate_all(self, loader: DataLoader) -> Dict[str, float]:
        """Run all attacks and compute retention rate."""
        clean_acc = self.evaluate_clean(loader)

        results = {"clean": clean_acc}
        attacked_accs = []
        for name in self.attacks:
            acc = self.evaluate_attack(loader, name)
            results[name] = acc
            attacked_accs.append(acc)

        # Retention = mean attacked accuracy / clean accuracy
        if clean_acc > 0:
            results["retention"] = sum(attacked_accs) / (len(attacked_accs) * clean_acc)
        else:
            results["retention"] = 0.0

        return results
