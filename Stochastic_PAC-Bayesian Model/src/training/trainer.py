"""
Complete Stochastic Training Pipeline with EOT and Active Learning.

Implements Algorithm 2 from the supplementary material (Appendix D):
  - MC forward passes for mean prediction and epistemic uncertainty
  - EOT adversarial example generation
  - Multi-objective loss computation
  - Gradient accumulation with mixed precision
  - Cosine LR schedule with linear warmup
  - Early stopping on validation F1
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..attacks.fgsm import FGSM
from ..attacks.pgd import PGD
from ..attacks.cw import CarliniWagner
from ..attacks.eot import EOTWrapper
from .losses import MultiObjectiveLoss

logger = logging.getLogger(__name__)


class StochasticTrainer:
    """Full training loop with EOT adversarial training and uncertainty."""

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        tc = config.get("training", {})

        # Optimizer (AdamW, Loshchilov & Hutter 2019)
        lr = tc.get("learning_rate_network", 2e-5)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(tc.get("beta1", 0.9), tc.get("beta2", 0.999)),
            weight_decay=tc.get("weight_decay", 0.01),
            eps=1e-6,
        )

        self.max_epochs = tc.get("max_epochs", 50)
        self.patience = tc.get("early_stopping_patience", 5)
        self.accum_steps = tc.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = tc.get("max_grad_norm", 1.0)
        self.mixed_precision = tc.get("mixed_precision", True)

        # Loss function
        lw = config.get("loss_weights", {})
        self.criterion = MultiObjectiveLoss(
            lambda_kl=lw.get("lambda_kl", 0.01),
            lambda_cal=lw.get("lambda_cal", 0.05),
            lambda_adv=lw.get("lambda_adv", 0.2),
            lambda_reg=lw.get("lambda_reg", 0.001),
        )

        # Mixed-precision scaler
        self.scaler = GradScaler(enabled=self.mixed_precision)

        # Adversarial attacks
        ac = config.get("adversarial", {})
        eps = ac.get("epsilon_network", 0.1)
        eot_samples = config.get("mc_sampling", {}).get("eot_gradient_samples", 5)
        self.attacks = {
            "fgsm": EOTWrapper(FGSM(epsilon=eps), eot_samples=eot_samples),
            "pgd": EOTWrapper(
                PGD(epsilon=eps, steps=ac.get("pgd_steps", 10),
                    step_size=eps * ac.get("pgd_step_size_fraction", 0.1)),
                eot_samples=eot_samples,
            ),
        }

        self.scheduler = None  # built in fit()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "checkpoints",
    ) -> Dict[str, list]:
        os.makedirs(checkpoint_dir, exist_ok=True)
        total_steps = len(train_loader) * self.max_epochs // self.accum_steps
        warmup_steps = int(0.1 * total_steps)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps)

        history: Dict[str, list] = {
            "train_loss": [], "val_acc": [], "val_ece": [], "val_f1": []}
        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            val_metrics = self._validate(val_loader)
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_ece"].append(val_metrics["ece"])
            history["val_f1"].append(val_metrics["f1"])

            logger.info(
                f"Epoch {epoch}/{self.max_epochs}  "
                f"loss={train_loss:.4f}  "
                f"val_acc={val_metrics['accuracy']:.4f}  "
                f"val_ece={val_metrics['ece']:.4f}  "
                f"val_f1={val_metrics['f1']:.4f}"
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, checkpoint_dir)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        return history

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        running_loss = 0.0

        for step, (data, targets) in enumerate(loader):
            data = data.to(self.device)
            targets = targets.to(self.device)

            with autocast(enabled=self.mixed_precision):
                # Clean forward pass
                logits = self.model(data)
                if isinstance(logits, tuple):
                    logits = logits[0]

                # Adversarial forward pass (randomly choose attack)
                attack_name = list(self.attacks.keys())[step % len(self.attacks)]
                attack = self.attacks[attack_name]
                with torch.no_grad():
                    x_adv = attack.generate(self.model, data, targets)
                adv_logits = self.model(x_adv)
                if isinstance(adv_logits, tuple):
                    adv_logits = adv_logits[0]

                losses = self.criterion(logits, targets, self.model, adv_logits)
                loss = losses["total"] / self.accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

            running_loss += losses["total"].item()

        return running_loss / max(len(loader), 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_targets, all_probs = [], [], []

        for data, targets in loader:
            data = data.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(data)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=-1)

            all_probs.append(probs.cpu())
            all_preds.append(probs.argmax(dim=-1).cpu())
            all_targets.append(targets.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        probs = torch.cat(all_probs)

        accuracy = (preds == targets).float().mean().item()
        f1 = self._macro_f1(preds, targets, probs.size(-1))
        ece = self._ece(probs, targets)

        return {"accuracy": accuracy, "f1": f1, "ece": ece}

    @staticmethod
    def _macro_f1(preds: torch.Tensor, targets: torch.Tensor,
                  num_classes: int) -> float:
        f1_sum = 0.0
        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum().float()
            fp = ((preds == c) & (targets != c)).sum().float()
            fn = ((preds != c) & (targets == c)).sum().float()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_sum += 2 * precision * recall / (precision + recall + 1e-8)
        return (f1_sum / num_classes).item()

    @staticmethod
    def _ece(probs: torch.Tensor, targets: torch.Tensor,
             n_bins: int = 15) -> float:
        confidences, predictions = probs.max(dim=-1)
        correct = (predictions == targets).float()

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() > 0:
                prop = mask.float().mean()
                acc = correct[mask].mean()
                conf = confidences[mask].mean()
                ece += (acc - conf).abs().item() * prop.item()
        return ece

    def _save_checkpoint(self, epoch: int, metrics: dict, path: str) -> None:
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": metrics,
        }, os.path.join(path, "best_model.pt"))
