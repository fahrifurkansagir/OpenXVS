"""Asymmetric EO + 2-ch LWIR fusion network with WP-07 deterministic health hard-gating & Dirichlet OOD.

Implements DR-04 §4.2 & SP2_M2_MODELS_DESIGN.md §3.3 (including BC-M2-m2 dual-blind guarantee):
  - Asymmetric encoders:
      * EO stream (`[B, 3, H_eo, W_eo]`): stride-4 feature pyramid preserving tiny distant targets.
      * LWIR stream (`[B, 2, H_lwir, W_lwir]` = `[norm_radiance, local_contrast]`): stride-2 pyramid
        aligning naturally with asymmetric EO/LWIR resolutions without 10x FLOP upsampling bloat.
  - Mandatory auxiliary uni-modal EO head & auxiliary uni-modal LWIR head (prevents modality laziness).
  - WP-07 deterministic physical health hard-gating:
      denom = q_eo * w_eo + q_lwir * w_lwir
      * q_eo == 0.0 -> w_eo_eff == 0.0 exact (zero activation leakage from blind EO, even on garbage/NaN)
      * q_lwir == 0.0 -> w_lwir_eff == 0.0 exact (zero activation leakage from blind LWIR)
      * q_eo == 0.0 AND q_lwir == 0.0 -> fused feature contribution == 0.0 identically AND
        evidential Dirichlet head forces evidence == 0.0 -> alpha == 1.0 -> vacuity == 1.0 exact!
  - ASTM F3269-21 Simplex RTA mode selector (`0..3`).
"""

from __future__ import annotations

from enum import IntEnum
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermalSimplexMode(IntEnum):
    """ASTM F3269-21 Simplex RTA modes for XVS-ThermalNet (DR-04 §4.2)."""

    NOMINAL_FUSION = 0
    EO_DEGRADED_LWIR_PRIMARY = 1
    LWIR_DEGRADED_EO_PRIMARY = 2
    DUAL_DVE_SVS_FALLBACK = 3


def _broadcast_health(q: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast scalar/1D `[B]`/`[B,1,1,1]` health tensor to `[B, 1, H, W]` and fail-close NaN/Inf -> 0."""
    q_t = q.to(device=ref.device, dtype=ref.dtype)
    if q_t.ndim == 0:
        q_t = q_t.view(1, 1, 1, 1).expand(ref.shape[0], 1, 1, 1)
    elif q_t.ndim == 1:
        q_t = q_t.view(-1, 1, 1, 1)
    elif q_t.ndim == 2:
        q_t = q_t.view(q_t.shape[0], 1, 1, 1)
    elif q_t.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported health tensor ndim={q_t.ndim}, shape={tuple(q_t.shape)}")
    valid = torch.isfinite(q_t) & (q_t >= 0.0) & (q_t <= 1.0)
    return torch.where(valid, q_t, q_t.new_zeros(()))


class _ConvBnSilu(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _AsymmetricEoEncoder(nn.Module):
    """High-resolution EO branch (stride-4 aligned output `[B, C, H_eo/4, W_eo/4]`)."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32, out_ch: int = 64) -> None:
        super().__init__()
        self.stem = _ConvBnSilu(in_ch, base_ch, kernel_size=3, stride=2)
        self.stage1 = _ConvBnSilu(base_ch, out_ch, kernel_size=3, stride=2)
        self.stage2 = _ConvBnSilu(out_ch, out_ch, kernel_size=3, stride=1)

    def forward(self, x_eo: torch.Tensor) -> torch.Tensor:
        h = self.stem(x_eo)
        h = self.stage1(h)
        return self.stage2(h)


class _AsymmetricLwirEncoder(nn.Module):
    """Native-resolution 2-ch LWIR branch (`[norm_radiance, local_contrast]` -> stride-2 output)."""

    def __init__(self, in_ch: int = 2, base_ch: int = 32, out_ch: int = 64) -> None:
        super().__init__()
        self.stem = _ConvBnSilu(in_ch, base_ch, kernel_size=3, stride=2)
        self.stage1 = _ConvBnSilu(base_ch, out_ch, kernel_size=3, stride=1)
        self.stage2 = _ConvBnSilu(out_ch, out_ch, kernel_size=3, stride=1)

    def forward(self, x_lwir: torch.Tensor) -> torch.Tensor:
        h = self.stem(x_lwir)
        h = self.stage1(h)
        return self.stage2(h)


class DeterministicHealthGate(nn.Module):
    """WP-07 deterministic physical health hard-gating + learned spatial reliability fusion (DR-04 §4.2).

    Enforces:
      - `q_eo == 0.0`  -> `w_eo_eff == 0.0` exact (zero leakage from blind EO stream)
      - `q_lwir == 0.0` -> `w_lwir_eff == 0.0` exact (zero leakage from blind LWIR stream)
      - `q_eo == 0.0 && q_lwir == 0.0` -> `f_fused == 0.0` identically (`BC-M2-m2`)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.eo_rel = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
        )
        self.lwir_rel = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
        )
        # Differential complementary cue active ONLY when both modalities are simultaneously healthy
        self.diff_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=False),
        )

    def forward(
        self,
        f_eo: torch.Tensor,
        f_lwir: torch.Tensor,
        q_eo: torch.Tensor,
        q_lwir: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        q_e = _broadcast_health(q_eo, f_eo)
        q_l = _broadcast_health(q_lwir, f_lwir)

        # Sanitize blind streams before computing learned reliability / diff so NaN/Inf in a blind
        # sensor can NEVER contaminate gradients or fused activations.
        eo_active = q_e > 0.0
        lwir_active = q_l > 0.0
        f_eo_safe = torch.where(eo_active, f_eo, f_eo.new_zeros(()))
        f_lwir_safe = torch.where(lwir_active, f_lwir, f_lwir.new_zeros(()))

        # Strictly positive learned weights in (0.05, 1.0) so a healthy lone sensor never divides by 0
        w_eo = 0.05 + 0.95 * torch.sigmoid(self.eo_rel(f_eo_safe))
        w_lwir = 0.05 + 0.95 * torch.sigmoid(self.lwir_rel(f_lwir_safe))

        num_eo = q_e * w_eo
        num_lwir = q_l * w_lwir
        denom = num_eo + num_lwir
        any_active = denom > 0.0
        safe_denom = torch.where(any_active, denom, denom.new_ones(()))

        w_eo_eff = torch.where(eo_active & any_active, num_eo / safe_denom, num_eo.new_zeros(()))
        w_lwir_eff = torch.where(
            lwir_active & any_active, num_lwir / safe_denom, num_lwir.new_zeros(())
        )

        # Differential interaction term multiplies by (w_eo_eff * w_lwir_eff), identically 0 if either blind
        both_active = eo_active & lwir_active
        diff_raw = self.diff_proj(f_eo_safe - f_lwir_safe)
        diff_term = torch.where(both_active, (w_eo_eff * w_lwir_eff) * diff_raw, diff_raw.new_zeros(()))

        eo_term = torch.where(eo_active, w_eo_eff * f_eo_safe, f_eo_safe.new_zeros(()))
        lwir_term = torch.where(lwir_active, w_lwir_eff * f_lwir_safe, f_lwir_safe.new_zeros(()))
        f_fused = eo_term + lwir_term + diff_term

        return {
            "f_fused": f_fused,
            "f_eo_safe": f_eo_safe,
            "f_lwir_safe": f_lwir_safe,
            "w_eo_raw": w_eo,
            "w_lwir_raw": w_lwir,
            "w_eo_eff": w_eo_eff,
            "w_lwir_eff": w_lwir_eff,
            "q_eo": q_e,
            "q_lwir": q_l,
            "both_active": both_active,
            "dual_blind": ~(eo_active | lwir_active),
        }


class EvidentialDirichletHead(nn.Module):
    """Evidential Deep Learning Dirichlet head emitting `(evidence, alpha, vacuity)` (DR-04 §4.2).

    `evidence_c = softplus(z_c)`, `alpha_c = evidence_c + 1`, `vacuity = K / sum_c alpha_c in (0, 1]`.
    Dual-blind hard lock (`BC-M2-m2`): whenever `dual_blind` is True (`q_eo == 0 && q_lwir == 0`),
    `evidence` is forced identically to `0.0`, guaranteeing `alpha == 1.0` and `vacuity == 1.0`.
    """

    def __init__(self, in_ch: int, num_classes: int = 3) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2 for Dirichlet head, got {num_classes}")
        self.num_classes = int(num_classes)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=False),
            nn.Conv2d(in_ch, self.num_classes, kernel_size=1),
        )

    def forward(
        self,
        feat: torch.Tensor,
        dual_blind: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = self.proj(feat)
        evidence = F.softplus(logits)
        if dual_blind is not None:
            mask = dual_blind.to(device=evidence.device, dtype=torch.bool)
            evidence = torch.where(mask, evidence.new_zeros(()), evidence)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=1, keepdim=True)
        vacuity = float(self.num_classes) / strength
        probs = alpha / strength
        return {
            "logits": logits,
            "evidence": evidence,
            "alpha": alpha,
            "vacuity": vacuity,
            "probs": probs,
        }


class _UnimodalAuxHead(nn.Module):
    """Mandatory independent uni-modal detection head (heatmap logit + 4D box delta)."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _ConvBnSilu(in_ch, in_ch, kernel_size=3, stride=1),
            nn.Conv2d(in_ch, 5, kernel_size=1),  # [hm_logit, dx, dy, log_w, log_h]
        )

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(feat)
        return {
            "hm_logits": out[:, :1],
            "box_reg": out[:, 1:5],
        }


class XvsThermalNet(nn.Module):
    """Asymmetric EO + 2-ch LWIR multispectral detection & OOD fusion network (DR-04 / SP2 §3.3).

    Additive extended flag `enable_disagreement_vacuity=True` (default `False` for v5.2 compat)
    concatenates `[|p_eo - p_lwir| * both_active, (1 - cos(f_eo, f_lwir)) * both_active]`
    into `EvidentialDirichletHead` so EO<->LWIR parallax conflicts raise vacuity >= 0.80
    while strictly preserving exact `dual_blind vacuity == 1.0000` and zero blind-sensor leakage.
    """

    def __init__(
        self,
        eo_in_ch: int = 3,
        lwir_in_ch: int = 2,
        feat_ch: int = 64,
        num_evidential_classes: int = 3,
        *,
        enable_disagreement_vacuity: bool = False,
    ) -> None:
        super().__init__()
        self.eo_in_ch = int(eo_in_ch)
        self.lwir_in_ch = int(lwir_in_ch)
        self.feat_ch = int(feat_ch)
        self.num_evidential_classes = int(num_evidential_classes)
        self.enable_disagreement_vacuity = bool(enable_disagreement_vacuity)

        self.eo_encoder = _AsymmetricEoEncoder(in_ch=eo_in_ch, base_ch=feat_ch // 2, out_ch=feat_ch)
        self.lwir_encoder = _AsymmetricLwirEncoder(
            in_ch=lwir_in_ch, base_ch=feat_ch // 2, out_ch=feat_ch
        )
        self.health_gate = DeterministicHealthGate(channels=feat_ch)
        self.eo_aux_head = _UnimodalAuxHead(in_ch=feat_ch)
        self.lwir_aux_head = _UnimodalAuxHead(in_ch=feat_ch)
        self.fused_head = _UnimodalAuxHead(in_ch=feat_ch)
        ev_in_ch = feat_ch + 4 if self.enable_disagreement_vacuity else feat_ch
        self.evidential_head = EvidentialDirichletHead(
            in_ch=ev_in_ch, num_classes=num_evidential_classes
        )

    def forward(
        self,
        x_eo: torch.Tensor,
        x_lwir: torch.Tensor,
        q_eo: torch.Tensor,
        q_lwir: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x_lwir.shape[1] != 2:
            raise ValueError(
                f"Expected 2-channel LWIR input [norm_radiance, local_contrast], got shape {tuple(x_lwir.shape)}"
            )
        f_eo = self.eo_encoder(x_eo)
        f_lwir = self.lwir_encoder(x_lwir)

        # Align LWIR pyramid spatially to EO stride-4 feature grid without upsampling raw LWIR pixels
        if f_lwir.shape[-2:] != f_eo.shape[-2:]:
            f_lwir_aligned = F.interpolate(
                f_lwir, size=f_eo.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            f_lwir_aligned = f_lwir

        # Mandatory independent uni-modal auxiliary predictions
        eo_aux = self.eo_aux_head(f_eo)
        lwir_aux = self.lwir_aux_head(f_lwir_aligned)

        # WP-07 deterministic physical health hard-gating
        gated = self.health_gate(f_eo, f_lwir_aligned, q_eo=q_eo, q_lwir=q_lwir)
        f_fused = gated["f_fused"]
        dual_blind = gated["dual_blind"]
        both_active = gated["both_active"]

        fused_det = self.fused_head(f_fused)
        # On dual-blind, suppress fused heatmap logits cleanly to large negative (zero detections)
        fused_hm = torch.where(
            dual_blind,
            fused_det["hm_logits"].new_full((), -20.0),
            fused_det["hm_logits"],
        )
        fused_box = torch.where(
            dual_blind,
            fused_det["box_reg"].new_zeros(()),
            fused_det["box_reg"],
        )

        d_hm: torch.Tensor | None = None
        d_cos: torch.Tensor | None = None
        if self.enable_disagreement_vacuity:
            # Cross-modal spatial peak, cosine & global/salient pooled disagreement gated strictly by `both_active`
            p_eo = torch.where(both_active, torch.sigmoid(eo_aux["hm_logits"]), eo_aux["hm_logits"].new_zeros(()))
            p_lwir = torch.where(both_active, torch.sigmoid(lwir_aux["hm_logits"]), lwir_aux["hm_logits"].new_zeros(()))
            d_hm = torch.where(both_active, torch.abs(p_eo - p_lwir), p_eo.new_zeros(()))

            fe_s = gated["f_eo_safe"]
            fl_s = gated["f_lwir_safe"]
            dot_el = (fe_s * fl_s).sum(dim=1, keepdim=True)
            norm_e = torch.sqrt((fe_s * fe_s).sum(dim=1, keepdim=True) + 1.0e-6)
            norm_l = torch.sqrt((fl_s * fl_s).sum(dim=1, keepdim=True) + 1.0e-6)
            cos_el = (dot_el / (norm_e * norm_l)).clamp(min=-1.0, max=1.0)
            d_cos = torch.where(both_active, 1.0 - cos_el, cos_el.new_zeros(()))

            # Global peak & soft top-k salient disagreement broadcast so localized (6,6) aircraft
            # parallax genuinely elevates frame-level Dirichlet vacuity without border overfitting.
            d_hm_global = torch.where(
                both_active,
                d_hm.amax(dim=(2, 3), keepdim=True),
                d_hm.new_zeros(()),
            ).expand_as(d_hm)
            sal_logit = (torch.maximum(p_eo, p_lwir) * 12.0).flatten(2)
            sal_w = F.softmax(sal_logit, dim=-1).view_as(p_eo)
            d_cos_salient = torch.where(
                both_active,
                (sal_w * d_cos).sum(dim=(2, 3), keepdim=True),
                d_cos.new_zeros(()),
            ).expand_as(d_cos)

            ev_in = torch.cat([f_fused, d_hm, d_cos, d_hm_global, d_cos_salient], dim=1)
        else:
            ev_in = f_fused
        ev = self.evidential_head(ev_in, dual_blind=dual_blind)

        out: dict[str, torch.Tensor] = {
            "f_eo": f_eo,
            "f_lwir": f_lwir_aligned,
            "f_fused": f_fused,
            "w_eo_eff": gated["w_eo_eff"],
            "w_lwir_eff": gated["w_lwir_eff"],
            "dual_blind": dual_blind,
            "eo_aux_hm_logits": eo_aux["hm_logits"],
            "eo_aux_box_reg": eo_aux["box_reg"],
            "lwir_aux_hm_logits": lwir_aux["hm_logits"],
            "lwir_aux_box_reg": lwir_aux["box_reg"],
            "fused_hm_logits": fused_hm,
            "fused_box_reg": fused_box,
            "evidence": ev["evidence"],
            "alpha": ev["alpha"],
            "vacuity": ev["vacuity"],
            "probs": ev["probs"],
        }
        if d_hm is not None and d_cos is not None:
            out["disagreement_hm"] = d_hm
            out["disagreement_cos"] = d_cos
        return out


def select_thermal_simplex_mode(
    q_eo: float,
    q_lwir: float,
    vacuity: float,
    tau_ok: float = 0.5,
    tau_vac: float = 0.7,
) -> ThermalSimplexMode:
    """Deterministic ASTM F3269-21 Simplex RTA mode switch (DR-04 §4.2, strictly fail-closed)."""
    if not (
        math.isfinite(tau_ok)
        and (0.0 < tau_ok <= 1.0)
        and math.isfinite(tau_vac)
        and (0.0 < tau_vac <= 1.0)
    ):
        return ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK

    vac_ok = math.isfinite(vacuity) and (0.0 <= vacuity < tau_vac)
    if not vac_ok:
        return ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK

    eo_ok = math.isfinite(q_eo) and (tau_ok <= q_eo <= 1.0)
    lwir_ok = math.isfinite(q_lwir) and (tau_ok <= q_lwir <= 1.0)

    if not eo_ok and not lwir_ok:
        return ThermalSimplexMode.DUAL_DVE_SVS_FALLBACK
    if eo_ok and lwir_ok:
        return ThermalSimplexMode.NOMINAL_FUSION
    if lwir_ok and not eo_ok:
        return ThermalSimplexMode.EO_DEGRADED_LWIR_PRIMARY
    return ThermalSimplexMode.LWIR_DEGRADED_EO_PRIMARY


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """Closed-form `KL(Dir(alpha) || Dir(1, ..., 1))` in FP32 (`[B, H, W]`), strictly >= 0."""
    a = alpha.float().clamp(min=1.0)
    k = float(a.shape[1])
    s = a.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(s).squeeze(1)
        - math.lgamma(k)
        - torch.lgamma(a).sum(dim=1)
        + ((a - 1.0) * (torch.digamma(a) - torch.digamma(s))).sum(dim=1)
    )
    return kl.clamp(min=0.0)


def compute_parallax_vacuity_loss(
    outputs: dict[str, torch.Tensor],
    parallax_conflict_mask: torch.Tensor,
    *,
    target_vacuity: float = 0.85,
    lambda_kl: float = 1.0,
    lambda_hinge: float = 2.0,
) -> dict[str, torch.Tensor]:
    """KL-to-Uniform Dirichlet vacuity loss + explicit hinge `relu(0.85 - vacuity_mean)` on parallax conflict."""
    mask = parallax_conflict_mask.float().reshape(-1)
    alpha = outputs["alpha"].float()
    vac = outputs["vacuity"].float()
    kl_ps = dirichlet_kl_to_uniform(alpha).mean(dim=(-2, -1))  # [B]
    vac_ps = vac.mean(dim=(1, 2, 3))  # [B]
    hinge_ps = F.relu(float(target_vacuity) - vac_ps)

    denom = mask.sum()
    if float(denom.detach().item()) <= 0.0:
        zero = (kl_ps * 0.0).sum()
        return {"total": zero, "kl_uniform": zero, "vacuity_hinge": zero}
    kl_term = (kl_ps * mask).sum() / denom.clamp(min=1.0)
    hinge_term = (hinge_ps * mask).sum() / denom.clamp(min=1.0)
    total = float(lambda_kl) * kl_term + float(lambda_hinge) * hinge_term
    return {"total": total, "kl_uniform": kl_term, "vacuity_hinge": hinge_term}


def _sparse_focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: float = 16.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Positive-weighted generalized focal BCE for extreme foreground sparsity (tiny IR targets)."""
    l_f = logits.float()
    t_f = target.float().clamp(0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(l_f, t_f, reduction="none")
    prob = torch.sigmoid(l_f)
    focal_w = torch.abs(t_f - prob).clamp(min=1.0e-4) ** float(gamma)
    pw = 1.0 + (float(pos_weight) - 1.0) * t_f
    return (pw * focal_w * bce).mean()


def compute_thermal_multitask_loss(
    outputs: dict[str, torch.Tensor],
    target_hm: torch.Tensor,
    target_cls: torch.Tensor,
    lambda_aux: float = 0.4,
    lambda_ev: float = 0.2,
    *,
    eo_target_hm: torch.Tensor | None = None,
    lwir_target_hm: torch.Tensor | None = None,
    parallax_conflict_mask: torch.Tensor | None = None,
    lambda_parallax_kl: float = 0.5,
    lambda_parallax_hinge: float = 1.5,
    parallax_vacuity_target: float = 0.85,
    hm_pos_weight: float = 16.0,
    hm_focal_gamma: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Multi-task fusion + auxiliary uni-modal + evidential Dirichlet loss (DR-04:203 + extended parallax vacuity)."""
    eo_tgt = target_hm if eo_target_hm is None else eo_target_hm
    lwir_tgt = target_hm if lwir_target_hm is None else lwir_target_hm

    if parallax_conflict_mask is not None:
        conflict_m = parallax_conflict_mask.float().reshape(-1)
        aligned_m = (1.0 - conflict_m).clamp(min=0.0, max=1.0)
        fused_tgt = target_hm.float() * aligned_m.view(-1, 1, 1, 1)
    else:
        conflict_m = None
        aligned_m = None
        fused_tgt = target_hm.float()

    l_fused = _sparse_focal_bce_with_logits(
        outputs["fused_hm_logits"], fused_tgt, pos_weight=hm_pos_weight, gamma=hm_focal_gamma
    )
    l_eo_aux = _sparse_focal_bce_with_logits(
        outputs["eo_aux_hm_logits"], eo_tgt, pos_weight=hm_pos_weight, gamma=hm_focal_gamma
    )
    l_lwir_aux = _sparse_focal_bce_with_logits(
        outputs["lwir_aux_hm_logits"], lwir_tgt, pos_weight=hm_pos_weight, gamma=hm_focal_gamma
    )

    # Evidential expected cross-entropy (Sensoy et al. Dirichlet MSE/log-risk form)
    alpha = outputs["alpha"].float()
    strength = alpha.sum(dim=1, keepdim=True)
    y_oh = F.one_hot(target_cls.long(), num_classes=alpha.shape[1]).permute(0, 3, 1, 2).float()
    edl_map = (y_oh * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=1)  # [B, H, W]
    if aligned_m is not None:
        edl_ps = edl_map.mean(dim=(-2, -1))
        if float(aligned_m.sum().detach().item()) > 0.0:
            l_edl = (edl_ps * aligned_m).sum() / aligned_m.sum().clamp(min=1.0)
        else:
            l_edl = (edl_ps * 0.0).sum()
    else:
        l_edl = edl_map.mean()

    total = l_fused + lambda_aux * l_eo_aux + lambda_aux * l_lwir_aux + lambda_ev * l_edl
    res: dict[str, torch.Tensor] = {
        "total": total,
        "l_fused": l_fused,
        "l_eo_aux": l_eo_aux,
        "l_lwir_aux": l_lwir_aux,
        "l_edl": l_edl,
    }
    if conflict_m is not None:
        p_loss = compute_parallax_vacuity_loss(
            outputs,
            conflict_m,
            target_vacuity=parallax_vacuity_target,
            lambda_kl=lambda_parallax_kl,
            lambda_hinge=lambda_parallax_hinge,
        )
        res["l_parallax_kl"] = p_loss["kl_uniform"]
        res["l_parallax_hinge"] = p_loss["vacuity_hinge"]
        res["total"] = total + p_loss["total"]
    return res
