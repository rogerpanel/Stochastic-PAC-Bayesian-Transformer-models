"""
Variational Embedding Layers for Multi-Domain Inputs.

Represent inputs as Gaussian random variables:
    z = mu_emb(x) + sigma_emb(x) * eps,   eps ~ N(0, I)
propagating input ambiguity through network layers (Section 4.1).

Domain-specific embedding functions:
  - TabularEmbedding:  phi_net(x)  = W_net * normalize(x) + b_net
  - TextEmbedding:     phi_text(x) = BERT_frozen(x)  +  variational noise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalEmbedding(nn.Module):
    """Base variational embedding: adds learned Gaussian noise to embeddings."""

    def __init__(self, d_model: int, log_var_init: float = -6.0):
        super().__init__()
        self.mu_proj = nn.Linear(d_model, d_model)
        self.log_var_proj = nn.Linear(d_model, d_model)
        nn.init.constant_(self.log_var_proj.bias, log_var_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mu_proj(x)
        log_var = self.log_var_proj(x)
        if self.training:
            sigma = torch.exp(0.5 * log_var)
            eps = torch.randn_like(sigma)
            return mu + sigma * eps
        return mu

    def kl_divergence(self) -> torch.Tensor:
        """KL(q(z|x) || N(0, I)) averaged over last forward pass."""
        # Approximate: use weight norms as proxy (exact KL needs stored activations)
        mu_norm = self.mu_proj.weight.pow(2).sum()
        log_var = self.log_var_proj.bias
        var = torch.exp(log_var)
        kl = 0.5 * (var + mu_norm / log_var.numel() - 1.0 - log_var).sum()
        return kl


class TabularEmbedding(nn.Module):
    """Embedding for tabular network intrusion detection data.

    phi_net(x) = W * normalize(x) + b   (Appendix A, Eq. A.1)
    """

    def __init__(self, num_features: int, d_model: int,
                 log_var_init: float = -6.0):
        super().__init__()
        self.linear = nn.Linear(num_features, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.variational = VariationalEmbedding(d_model, log_var_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (B, num_features) or (B, T, num_features)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, F) — single-step sequence
        h = self.linear(x)
        h = self.layer_norm(h)
        return self.variational(h)

    def kl_divergence(self) -> torch.Tensor:
        return self.variational.kl_divergence()


class TextEmbedding(nn.Module):
    """Embedding for text (toxic content / fake news) using frozen BERT.

    phi_text(x) = BERT_frozen(x)  -->  variational projection
    """

    def __init__(self, d_model: int, pretrained_name: str = "bert-base-uncased",
                 freeze_bert: bool = True, log_var_init: float = -6.0):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(pretrained_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        bert_hidden = self.bert.config.hidden_size  # 768
        self.projection = nn.Linear(bert_hidden, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.variational = VariationalEmbedding(d_model, log_var_init)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        h = bert_out.last_hidden_state           # (B, T, 768)
        h = self.projection(h)                   # (B, T, d_model)
        h = self.layer_norm(h)
        return self.variational(h)

    def kl_divergence(self) -> torch.Tensor:
        return self.variational.kl_divergence()
