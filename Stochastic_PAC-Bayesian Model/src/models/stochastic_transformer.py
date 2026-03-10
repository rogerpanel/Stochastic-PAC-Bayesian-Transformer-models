"""
Stochastic PAC-Bayesian Transformer.

Complete architecture implementing:
  - 12-layer transformer with Bayesian attention at layers {3, 6, 9, 12}
  - Variational embeddings for tabular and text inputs
  - MC sampling for uncertainty propagation
  - Classification + uncertainty heads

Reference: Section 4, Figure 1 of the main paper.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bayesian_attention import BayesianMultiHeadAttention
from .positional_encoding import PositionalEncoding
from .variational_embedding import TabularEmbedding, TextEmbedding


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model hyper-parameters matching the paper's Table F.1."""
    num_layers: int = 12
    num_heads: int = 8
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    bayesian_layers: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    prior_sigma: float = 0.1
    posterior_log_var_init: float = -6.0
    max_seq_len: int = 512
    # Set at runtime based on domain
    input_dim: Optional[int] = None        # tabular feature count
    num_classes: int = 2
    input_type: str = "tabular"            # "tabular" or "text"
    pretrained_backbone: str = "bert-base-uncased"
    mc_train_samples: int = 30
    mc_inference_samples: int = 50


# ---------------------------------------------------------------------------
# Transformer layers
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """Single transformer encoder layer (pre-norm variant).

    When *bayesian=True* the multi-head attention uses variational weights;
    otherwise standard deterministic attention is used.
    """

    def __init__(self, config: ModelConfig, bayesian: bool = False):
        super().__init__()
        self.bayesian = bayesian

        # Attention
        if bayesian:
            self.attn = BayesianMultiHeadAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout,
                prior_sigma=config.prior_sigma,
                posterior_log_var_init=config.posterior_log_var_init,
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )

        # Feed-forward
        act = nn.GELU() if config.activation == "gelu" else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            act,
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm attention
        h = self.norm1(x)
        if self.bayesian:
            h = self.attn(h, mask=mask)
        else:
            h, _ = self.attn(h, h, h, key_padding_mask=mask)
        x = x + self.dropout(h)

        # Pre-norm feed-forward
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.dropout(h)
        return x

    def kl_divergence(self) -> torch.Tensor:
        if self.bayesian:
            return self.attn.kl_divergence()
        return torch.tensor(0.0)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class StochasticPACBayesianTransformer(nn.Module):
    """Stochastic PAC-Bayesian Transformer (main model).

    Supports both tabular (network IDS) and text (toxic / fake news) inputs.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # --- Domain-specific embedding ---
        if config.input_type == "tabular":
            assert config.input_dim is not None
            self.embedding = TabularEmbedding(
                num_features=config.input_dim,
                d_model=config.d_model,
                log_var_init=config.posterior_log_var_init,
            )
        else:
            self.embedding = TextEmbedding(
                d_model=config.d_model,
                pretrained_name=config.pretrained_backbone,
                freeze_bert=True,
                log_var_init=config.posterior_log_var_init,
            )

        # --- Positional encoding ---
        self.pos_enc = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # --- Transformer layers ---
        self.layers = nn.ModuleList()
        for i in range(1, config.num_layers + 1):
            bayesian = i in config.bayesian_layers
            self.layers.append(TransformerLayer(config, bayesian=bayesian))

        # --- Output heads ---
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.uncertainty_head = nn.Linear(config.d_model, 1)

    # ----- forward helpers -----

    def _embed(self, x: torch.Tensor,
               attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.config.input_type == "text":
            return self.embedding(input_ids=x, attention_mask=attention_mask)
        return self.embedding(x)

    def _encode(self, h: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.pos_enc(h)
        for layer in self.layers:
            h = layer(h, mask=mask)
        h = self.final_norm(h)
        return h.mean(dim=1)  # global average pooling

    # ----- public API -----

    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                return_uncertainty: bool = False):
        h = self._embed(x, attention_mask)
        pooled = self._encode(h, mask=None)

        logits = self.classifier(pooled)
        if return_uncertainty:
            unc = self.uncertainty_head(pooled)
            return logits, unc
        return logits

    def mc_forward(self, x: torch.Tensor,
                   attention_mask: torch.Tensor | None = None,
                   num_samples: int | None = None) -> Dict[str, torch.Tensor]:
        """Monte Carlo forward pass for uncertainty estimation.

        Returns dict with keys:
            mean_logits, mean_probs, epistemic, aleatoric, predictions
        """
        S = num_samples or (
            self.config.mc_train_samples if self.training
            else self.config.mc_inference_samples
        )

        was_training = self.training
        self.train()  # enable dropout / sampling

        all_probs = []
        with torch.no_grad() if not was_training else torch.enable_grad():
            for _ in range(S):
                logits = self.forward(x, attention_mask)
                all_probs.append(F.softmax(logits, dim=-1))

        all_probs = torch.stack(all_probs, dim=0)  # (S, B, C)

        mean_probs = all_probs.mean(dim=0)          # (B, C)
        predictions = mean_probs.argmax(dim=-1)     # (B,)

        # Epistemic uncertainty: variance of predictions across samples
        epistemic = all_probs.var(dim=0).sum(dim=-1)  # (B,)

        # Aleatoric uncertainty: mean entropy of individual predictions
        per_sample_entropy = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=-1)
        aleatoric = per_sample_entropy.mean(dim=0)    # (B,)

        if not was_training:
            self.eval()

        return {
            "mean_logits": torch.log(mean_probs + 1e-10),
            "mean_probs": mean_probs,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "predictions": predictions,
            "all_probs": all_probs,
        }

    def compute_kl_loss(self) -> torch.Tensor:
        """Total KL divergence across all Bayesian layers + embedding."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            kl = kl + layer.kl_divergence()
        kl = kl + self.embedding.kl_divergence()
        return kl

    def get_bayesian_params(self):
        """Yield only the variational (mu, log_var) parameters."""
        for layer in self.layers:
            if layer.bayesian:
                yield from layer.attn.parameters()
        yield from self.embedding.variational.parameters()
