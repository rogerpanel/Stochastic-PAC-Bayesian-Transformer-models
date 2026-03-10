"""
Bayesian Multi-Head Attention with Variational Inference.

Replaces fixed projection matrices W_Q, W_K, W_V with learned Gaussian
distributions q_phi(W) = N(mu, diag(sigma^2)).  Each forward pass samples
weights via the reparameterization trick, enabling unbiased gradient
estimation through the variational parameters.

Reference: Section 4.1, Algorithm 1 of the main paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    """Linear layer with variational weight distributions.

    Parameters are stored as (mu, log_var) pairs so that
    W ~ N(mu, exp(log_var)).  KL(q || p) is computed analytically
    against a Gaussian prior N(prior_mu, prior_sigma^2 I).
    """

    def __init__(self, in_features: int, out_features: int,
                 prior_mu: float = 0.0, prior_sigma: float = 0.1,
                 posterior_log_var_init: float = -6.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational posterior parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_log_var = nn.Parameter(torch.empty(out_features))

        # Prior (fixed, not learned)
        self.register_buffer("prior_mu", torch.tensor(prior_mu))
        self.register_buffer("prior_log_var",
                             torch.tensor(2.0 * math.log(prior_sigma)))

        # Initialization
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_in")
        nn.init.constant_(self.weight_log_var, posterior_log_var_init)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_log_var, posterior_log_var_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self._sample(self.weight_mu, self.weight_log_var)
            bias = self._sample(self.bias_mu, self.bias_log_var)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    @staticmethod
    def _sample(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def kl_divergence(self) -> torch.Tensor:
        """Analytic KL(q(W) || p(W)) for Gaussian posterior vs Gaussian prior."""
        kl_w = self._kl_gaussian(self.weight_mu, self.weight_log_var)
        kl_b = self._kl_gaussian(self.bias_mu, self.bias_log_var)
        return kl_w + kl_b

    def _kl_gaussian(self, mu: torch.Tensor,
                     log_var: torch.Tensor) -> torch.Tensor:
        # KL(N(mu, sigma^2) || N(prior_mu, prior_sigma^2))
        prior_var = torch.exp(self.prior_log_var)
        var = torch.exp(log_var)
        kl = 0.5 * (
            (var + (mu - self.prior_mu) ** 2) / prior_var
            - 1.0
            + self.prior_log_var
            - log_var
        )
        return kl.sum()

    def init_from_deterministic(self, weight: torch.Tensor,
                                bias: torch.Tensor) -> None:
        """Initialize posterior mean from pre-trained deterministic weights."""
        self.weight_mu.data.copy_(weight)
        self.bias_mu.data.copy_(bias)


class BayesianMultiHeadAttention(nn.Module):
    """Multi-head attention with Bayesian (variational) Q, K, V projections.

    Implements Eq. (5) from the paper:
        BayesAttn(Q,K,V) = E_{W_Q,W_K,W_V ~ q_phi}[softmax(QK^T/sqrt(d_k)) V]

    Head-wise sampling: each attention head draws independent weight samples
    for computational tractability (Section 4.1).
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8,
                 dropout: float = 0.1, prior_sigma: float = 0.1,
                 posterior_log_var_init: float = -6.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Bayesian projections for Q, K, V
        self.q_proj = BayesianLinear(d_model, d_model,
                                     prior_sigma=prior_sigma,
                                     posterior_log_var_init=posterior_log_var_init)
        self.k_proj = BayesianLinear(d_model, d_model,
                                     prior_sigma=prior_sigma,
                                     posterior_log_var_init=posterior_log_var_init)
        self.v_proj = BayesianLinear(d_model, d_model,
                                     prior_sigma=prior_sigma,
                                     posterior_log_var_init=posterior_log_var_init)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        # Project through Bayesian layers (weight sampling happens inside)
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)                   # (B, H, T, d_k)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.out_proj(attn_out)

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence from Q, K, V projections (Eq. 11 in Alg 1)."""
        return (self.q_proj.kl_divergence()
                + self.k_proj.kl_divergence()
                + self.v_proj.kl_divergence())
