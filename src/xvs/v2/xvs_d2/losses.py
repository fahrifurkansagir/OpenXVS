"""BF16-Safe Logit-Domain Focal Loss, Masked NWD, Faithful Heteroscedastic & KD Losses (`DR-01`).

Implements `SP2_M2_MODELS_DESIGN.md` §3.2 (`V2-M2L-2`):
- Logit-domain FP32 `softplus` focal loss inside `torch.amp.autocast(..., enabled=False)` on
  `.float()` logits — guaranteed zero `NaN`/`Inf` even at `logits = ±50` under `bfloat16`!
- Normalized Wasserstein Distance (`L_NWD`, `C = 12.0 px`) strictly masked by `m_box = 0` on
  AOT fixed `6x6` / `16x16` circle-approximation labels.
- Faithful Heteroscedastic `beta-NLL` (`L_mean`, `L_var`) with stop-gradient decoupling.
- `CSCWD` (Cross-Scale Channel-Wise Distillation) + `CrossKD` (student `F_P2` through frozen
  teacher heads) distillation objectives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def logit_domain_focal_loss(
    hm_logits: torch.Tensor,
    gt_heatmap: torch.Tensor,
    *,
    neg_mask: torch.Tensor | None = None,
    quality_weights: torch.Tensor | None = None,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Logit-domain quality-aware penalty-reduced focal loss strictly in FP32 (`DR-01:88-93`).

    Uses `ln(sigmoid(z)) = -softplus(-z)` and `ln(1 - sigmoid(z)) = -softplus(z)` so no
    explicit `log(p)` or `log(1 - p)` ever executes, remaining 100% finite for `|z| <= 80`
    even when invoked under `torch.amp.autocast(..., dtype=torch.bfloat16)`.
    """
    dev_type = hm_logits.device.type if hm_logits.device.type in ("cpu", "cuda") else "cpu"
    with torch.amp.autocast(device_type=dev_type, enabled=False):
        z = hm_logits.to(dtype=torch.float32)
        y = gt_heatmap.to(dtype=torch.float32).clamp(min=0.0, max=1.0)

        pos_mask = y.eq(1.0)
        neg_region = ~pos_mask
        if neg_mask is not None:
            neg_weight_mask = neg_mask.to(dtype=torch.float32)
        else:
            neg_weight_mask = torch.ones_like(z)

        p = torch.sigmoid(z)
        one_minus_p = torch.sigmoid(-z)
        log_p = -F.softplus(-z)
        log_1mp = -F.softplus(z)

        if quality_weights is not None:
            q = quality_weights.to(dtype=torch.float32).clamp(min=1e-4, max=1.0).detach()
        else:
            q = torch.ones_like(z)

        pos_mod = torch.pow((q - p).abs(), float(alpha))
        pos_bce = -(q * log_p + (1.0 - q) * log_1mp)
        pos_loss = torch.where(pos_mask, pos_mod * pos_bce, torch.zeros_like(z)).sum()

        neg_mod = torch.pow(1.0 - y, float(beta)) * torch.pow(p, float(alpha))
        neg_loss = torch.where(
            neg_region,
            -neg_weight_mask * neg_mod * log_1mp,
            torch.zeros_like(z),
        ).sum()

        # Keep unused `one_minus_p` cleanly referenced if needed; prevent division by zero
        _ = one_minus_p
        n_pos = pos_mask.to(dtype=torch.float32).sum().clamp(min=1.0)
        return (pos_loss + neg_loss) / n_pos


def normalized_wasserstein_loss(
    pred_boxes_cxcywh: torch.Tensor,
    gt_boxes_cxcywh: torch.Tensor,
    *,
    box_mask: torch.Tensor | None = None,
    c_const: float = 12.0,
) -> torch.Tensor:
    """Normalized Wasserstein Distance loss `1 - exp(-W2 / C)` strictly masked by `m_box` (`DR-01:94-96`).

    When `box_mask` (`m_box`) is `0.0` (e.g., AOT fixed `6x6` or `16x16` circle approximations),
    contributes identically `0.0` loss and `0.0` gradient.
    """
    dev_type = pred_boxes_cxcywh.device.type if pred_boxes_cxcywh.device.type in ("cpu", "cuda") else "cpu"
    with torch.amp.autocast(device_type=dev_type, enabled=False):
        p = pred_boxes_cxcywh.to(dtype=torch.float32)
        g = gt_boxes_cxcywh.to(dtype=torch.float32)

        dx = p[..., 0] - g[..., 0]
        dy = p[..., 1] - g[..., 1]
        dw = p[..., 2] - g[..., 2]
        dh = p[..., 3] - g[..., 3]

        w2_sq = dx * dx + dy * dy + 0.25 * (dw * dw + dh * dh)
        w2 = torch.sqrt(w2_sq + 1e-12)
        nwd = torch.exp(-w2 / float(c_const))
        loss_elem = 1.0 - nwd

        if box_mask is not None:
            m = box_mask.to(dtype=torch.float32)
            masked = loss_elem * m
            denom = m.sum().clamp(min=1.0)
            return masked.sum() / denom
        return loss_elem.mean()


def faithful_heteroscedastic_loss(
    pred_mean: torch.Tensor,
    gt_target: torch.Tensor,
    pred_log_var: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Faithful Heteroscedastic `beta-NLL` decoupling `L_mean` and `L_var` (`DR-01:97-99`).

    `pred_log_var` is clamped to `[-6.0, +4.0]` and evaluated against `(gt_target - pred_mean.detach())^2`
    so variance estimation never perturbs mean localization gradients.
    """
    dev_type = pred_mean.device.type if pred_mean.device.type in ("cpu", "cuda") else "cpu"
    with torch.amp.autocast(device_type=dev_type, enabled=False):
        mu = pred_mean.to(dtype=torch.float32)
        gt = gt_target.to(dtype=torch.float32)
        lv = pred_log_var.to(dtype=torch.float32).clamp(min=-6.0, max=4.0)

        l_mean_elem = F.smooth_l1_loss(mu, gt, reduction="none")
        sq_err_detached = (gt - mu.detach()) ** 2
        l_var_elem = 0.5 * (torch.exp(-lv) * sq_err_detached + lv)

        if mask is not None:
            m = mask.to(dtype=torch.float32)
            while m.ndim < mu.ndim:
                m = m.unsqueeze(-1)
            m_broad = m.expand_as(mu)
            denom = m_broad.sum().clamp(min=1.0)
            return (l_mean_elem * m_broad).sum() / denom, (l_var_elem * m_broad).sum() / denom
        return l_mean_elem.mean(), l_var_elem.mean()


def cscwd_distillation_loss(
    student_pyramid: dict[str, torch.Tensor],
    teacher_pyramid: dict[str, torch.Tensor],
    *,
    adapters: nn.ModuleDict | None = None,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Cross-Scale Channel-Wise Distillation (`CSCWD`) over `(p2, p3, p4)` (`DR-01:135`)."""
    keys = [k for k in ("p2", "p3", "p4") if k in student_pyramid and k in teacher_pyramid]
    if not keys:
        raise ValueError("student_pyramid and teacher_pyramid must share at least one of p2, p3, p4")

    first = student_pyramid[keys[0]]
    dev_type = first.device.type if first.device.type in ("cpu", "cuda") else "cpu"
    with torch.amp.autocast(device_type=dev_type, enabled=False):
        total = torch.zeros((), device=first.device, dtype=torch.float32)
        tau = float(temperature)
        for k in keys:
            s_feat = student_pyramid[k].to(dtype=torch.float32)
            t_feat = teacher_pyramid[k].detach().to(dtype=torch.float32)
            if adapters is not None and k in adapters:
                s_feat = adapters[k](s_feat)
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode="nearest")
            if s_feat.shape[1] != t_feat.shape[1]:
                # Deterministic channel-group average to match teacher width when no adapter is provided
                c_min = min(s_feat.shape[1], t_feat.shape[1])
                s_feat = s_feat[:, :c_min]
                t_feat = t_feat[:, :c_min]

            b_sz, c_sz, _, _ = s_feat.shape
            s_flat = s_feat.reshape(b_sz, c_sz, -1) / tau
            t_flat = t_feat.reshape(b_sz, c_sz, -1) / tau
            log_s = F.log_softmax(s_flat, dim=-1)
            prob_t = F.softmax(t_flat, dim=-1)
            kl = F.kl_div(log_s, prob_t, reduction="batchmean") * (tau * tau) / float(c_sz)
            total = total + kl
        return total / float(len(keys))


def cross_kd_loss(
    student_fp2: torch.Tensor,
    teacher_heads: nn.Module,
    teacher_outputs: dict[str, torch.Tensor],
    *,
    fp2_adapter: nn.Module | None = None,
    temperature: float = 2.0,
) -> torch.Tensor:
    """CrossKD loss passing student `F_P2` through frozen Teacher heads (`DR-01:135`)."""
    dev_type = student_fp2.device.type if student_fp2.device.type in ("cpu", "cuda") else "cpu"
    with torch.amp.autocast(device_type=dev_type, enabled=False):
        s_in = student_fp2.to(dtype=torch.float32)
        if fp2_adapter is not None:
            s_in = fp2_adapter(s_in)
        s_hm, s_off, s_scale, _, _ = teacher_heads(s_in)
        t_hm = teacher_outputs["hm_logits"].detach().to(dtype=torch.float32)
        t_off = teacher_outputs["offset"].detach().to(dtype=torch.float32)
        t_scale = teacher_outputs["log_scale"].detach().to(dtype=torch.float32)

        tau = float(temperature)
        # Logit-domain binary KL via softplus/sigmoid in FP32
        p_t = torch.sigmoid(t_hm / tau)
        log_p_s = -F.softplus(-(s_hm / tau))
        log_1mp_s = -F.softplus(s_hm / tau)
        hm_kd = (-(p_t * log_p_s + (1.0 - p_t) * log_1mp_s)).mean() * (tau * tau)
        reg_kd = F.smooth_l1_loss(s_off, t_off) + F.smooth_l1_loss(s_scale, t_scale)
        return hm_kd + reg_kd


def compute_xvs_d2_composite_loss(
    *,
    hm_logits: torch.Tensor,
    gt_heatmap: torch.Tensor,
    pred_boxes_cxcywh: torch.Tensor,
    gt_boxes_cxcywh: torch.Tensor,
    pred_mean: torch.Tensor,
    gt_mean: torch.Tensor,
    pred_log_var: torch.Tensor,
    neg_mask: torch.Tensor | None = None,
    box_mask: torch.Tensor | None = None,
    mean_mask: torch.Tensor | None = None,
    kd_loss: torch.Tensor | None = None,
    lambda_kd: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Composite `DR-01:132` objective `L_hm + 2.0 L_NWD + 1.5 L_mean + 0.25 L_var + lambda_kd L_kd`."""
    l_hm = logit_domain_focal_loss(hm_logits, gt_heatmap, neg_mask=neg_mask)
    l_nwd = normalized_wasserstein_loss(pred_boxes_cxcywh, gt_boxes_cxcywh, box_mask=box_mask)
    l_mean, l_var = faithful_heteroscedastic_loss(pred_mean, gt_mean, pred_log_var, mask=mean_mask)
    total = l_hm + 2.0 * l_nwd + 1.5 * l_mean + 0.25 * l_var
    if kd_loss is not None and float(lambda_kd) > 0.0:
        total = total + float(lambda_kd) * kd_loss.to(dtype=torch.float32)
    return {
        "total": total,
        "l_hm": l_hm,
        "l_nwd": l_nwd,
        "l_mean": l_mean,
        "l_var": l_var,
    }
