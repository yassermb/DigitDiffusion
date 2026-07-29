"""
Noise schedules and forward-process utilities for DDPM.

Implements:
  q(x_t | x_0) = N(x_t ; √ᾱ_t · x_0 , (1-ᾱ_t) · I)

where:
  β_t   = variance schedule
  α_t   = 1 - β_t
  ᾱ_t   = ∏_{s=1}^{t} α_s
"""

import torch
import numpy as np


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Linear schedule from β_start to β_end over T steps."""
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)


def cosine_beta_schedule(T: int, s: float = 0.008):
    """
    Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
    (Nichol & Dhariwal, 2021).
    """
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    """
    Pre-computes and stores all the diffusion constants needed for the
    forward process q(x_t | x_0) and the reverse process p_θ(x_{t-1} | x_t).
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, schedule: str = "linear",
                 device: str = "cpu"):
        self.T = T
        self.device = device

        # ── Compute betas ───────────────────────────────────────────────────
        if schedule == "linear":
            betas = linear_beta_schedule(T, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # ── Derived quantities (all in float64 for numerical precision) ─────
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64),
                                         alphas_cumprod[:-1]])

        # Store everything as float32 tensors on the target device
        self.betas = betas.float().to(device)                       # β_t
        self.alphas = alphas.float().to(device)                     # α_t
        self.alphas_cumprod = alphas_cumprod.float().to(device)     # ᾱ_t
        self.alphas_cumprod_prev = alphas_cumprod_prev.float().to(device)

        # Quantities needed for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float().to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - alphas_cumprod).float().to(device)

        # Quantities needed for the reverse process posterior
        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float().to(device)
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        ).to(device)
        self.posterior_mean_coeff1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float().to(device)
        self.posterior_mean_coeff2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).float().to(device)

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape):
        """Index into tensor `a` at positions `t`, reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None):
        """
        Forward diffusion: sample x_t from q(x_t | x_0).

        x_t = √ᾱ_t · x_0  +  √(1-ᾱ_t) · ε ,   ε ~ N(0, I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute mean and variance of the posterior q(x_{t-1} | x_t, x_0).
        """
        mean = (
            self._extract(self.posterior_mean_coeff1, t, x_t.shape) * x_0
            + self._extract(self.posterior_mean_coeff2, t, x_t.shape) * x_t
        )
        var = self._extract(self.posterior_variance, t, x_t.shape)
        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var
