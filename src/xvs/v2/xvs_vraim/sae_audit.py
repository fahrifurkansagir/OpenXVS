"""Pure PyTorch Top-K Sparse Autoencoder (SAE) content-vs-style shortcut audit (DR-08 / R-LM-09).

Decomposes penultimate visual representations into sparse interpretable monosemantic features
and quantifies reliance on geometric/semantic content vs spurious visual style (fog/lighting/texture).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSparseAutoencoder(nn.Module):
    """Pure PyTorch Top-K Sparse Autoencoder with unit-norm decoder dictionary columns."""

    def __init__(self, in_dim: int, dict_dim: int, top_k: int) -> None:
        super().__init__()
        if in_dim <= 0 or dict_dim <= 0 or not (1 <= top_k <= dict_dim):
            raise ValueError(
                f"Invalid TopKSparseAutoencoder dims: in_dim={in_dim}, dict_dim={dict_dim}, top_k={top_k}"
            )
        self.in_dim = int(in_dim)
        self.dict_dim = int(dict_dim)
        self.top_k = int(top_k)

        self.pre_bias = nn.Parameter(torch.zeros(in_dim))
        self.encoder = nn.Linear(in_dim, dict_dim, bias=True)
        self.decoder_weight = nn.Parameter(torch.randn(in_dim, dict_dim) * 0.05)
        self.normalize_decoder_()

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Keep every dictionary atom on the unit L2 sphere."""
        norms = self.decoder_weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder_weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode `[N, in_dim]` activations into strictly Top-K non-negative sparse latents `[N, dict_dim]`."""
        pre = F.relu(self.encoder(x - self.pre_bias))
        vals, idx = torch.topk(pre, k=self.top_k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct `[N, in_dim]` activations from sparse codes `z`."""
        w_unit = F.normalize(self.decoder_weight, p=2, dim=0, eps=1e-8)
        return F.linear(z, w_unit) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        recon_mse = F.mse_loss(x_hat, x)
        return x_hat, z, recon_mse


@dataclass(frozen=True)
class SaeAuditResult:
    """Quantitative content-vs-style attribution summary for EASA R-LM-09 audit."""

    reconstruction_mse: float
    mean_active_latents: float
    content_energy: float
    style_energy: float
    content_reliance_ratio: float
    style_reliance_ratio: float
    content_selective_count: int
    style_selective_count: int
    shortcut_warning: bool


def audit_content_vs_style_reliance(
    sae: TopKSparseAutoencoder,
    content_paired_a: torch.Tensor,
    content_paired_b: torch.Tensor,
    style_paired_a: torch.Tensor,
    style_paired_b: torch.Tensor,
    *,
    train_steps: int = 80,
    lr: float = 1e-2,
    threshold_min_content_ratio: float = 0.60,
) -> SaeAuditResult:
    """Audit whether SAE latents encode true geometric content vs spurious style perturbations.

    Inputs:
      - `(content_paired_a, content_paired_b)`: pairs differing in true scene geometry/content
        (same style/weather).
      - `(style_paired_a, style_paired_b)`: pairs sharing the SAME scene geometry/content under
        different visual style (fog/glare/season/lighting).
    """
    all_x = torch.cat(
        [content_paired_a, content_paired_b, style_paired_a, style_paired_b], dim=0
    ).detach().float()

    if train_steps > 0:
        sae.train()
        opt = torch.optim.Adam(sae.parameters(), lr=lr)
        for _ in range(train_steps):
            opt.zero_grad()
            _, _, loss = sae(all_x)
            loss.backward()
            opt.step()
            sae.normalize_decoder_()

    sae.eval()
    with torch.no_grad():
        _, z_all, recon_mse_t = sae(all_x)
        z_ca = sae.encode(content_paired_a.float())
        z_cb = sae.encode(content_paired_b.float())
        z_sa = sae.encode(style_paired_a.float())
        z_sb = sae.encode(style_paired_b.float())

        # Per-latent sensitivity (mean squared change) under content shifts vs style shifts
        content_sens = (z_ca - z_cb).pow(2).mean(dim=0)  # [dict_dim]
        style_sens = (z_sa - z_sb).pow(2).mean(dim=0)  # [dict_dim]

        content_energy = float(content_sens.sum().item())
        style_energy = float(style_sens.sum().item())
        denom = content_energy + style_energy + 1e-12
        content_ratio = content_energy / denom
        style_ratio = style_energy / denom

        active_mask = (content_sens + style_sens) > 1e-8
        content_sel = int(((content_sens > style_sens) & active_mask).sum().item())
        style_sel = int(((style_sens >= content_sens) & active_mask).sum().item())
        mean_active = float((z_all > 0.0).float().sum(dim=-1).mean().item())
        recon_mse = float(recon_mse_t.item())

    return SaeAuditResult(
        reconstruction_mse=recon_mse,
        mean_active_latents=mean_active,
        content_energy=content_energy,
        style_energy=style_energy,
        content_reliance_ratio=content_ratio,
        style_reliance_ratio=style_ratio,
        content_selective_count=content_sel,
        style_selective_count=style_sel,
        shortcut_warning=bool(content_ratio < threshold_min_content_ratio),
    )
