"""Pure-PyTorch XVS-Land two-stage architecture (DR-03 & SP2_M2_MODELS_DESIGN.md §3.4).

Zero `timm` / `torchvision` imports:
  - Stage-1: D-FINE-M COCO-only ROI detector wrapper + `expand_roi_conformal(box, q_alpha)`
  - Stage-2A: ConvNeXt-V1 MIT backbone + 12-keypoint SimCC + 5 Continuous Hesse Keylines (OKLS)
              with epsilon-guarded unit normalization `(c, s) / sqrt(c^2 + s^2 + 1e-12)` (`BC-M2-m5`)
  - Stage-2B: Frozen-mean Cholesky 2x2 bivariate GNLL head (`feat.detach()`, strictly SPD
              `L = [[exp(clamp(l0, -6, 4)), 0], [l1, exp(clamp(l2, -6, 4))]]`, `Sigma = L @ L^T`)
              + Student-t variance calibration.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_KEYPOINTS: int = 12
NUM_KEYLINES: int = 5
CHOLESKY_LOG_DIAG_MIN: float = -6.0
CHOLESKY_LOG_DIAG_MAX: float = 4.0
OKLS_NORM_EPS: float = 1.0e-12
KEYLINE_SIGMA_MIN_PX: float = 0.5
KEYLINE_SIGMA_MAX_PX: float = 50.0

# Canonical 5 runway keyline endpoint pairs indexing into the 12 runway keypoints:
#   0: left edge (0 -> 9), 1: right edge (2 -> 11), 2: centerline (1 -> 10),
#   3: threshold bar (0 -> 2), 4: aiming-point bar (3 -> 5)
KEYLINE_ENDPOINT_INDICES: tuple[tuple[int, int], ...] = (
    (0, 9),
    (2, 11),
    (1, 10),
    (0, 2),
    (3, 5),
)

# All keypoints collinear with each canonical runway keyline (for geometric consistency loss)
KEYLINE_COLLINEAR_KEYPOINTS: tuple[tuple[int, ...], ...] = (
    (0, 3, 6, 9),
    (2, 5, 8, 11),
    (1, 4, 7, 10),
    (0, 1, 2),
    (3, 4, 5),
)


def expand_roi_conformal(
    box_xyxy: tuple[float, float, float, float] | list[float],
    q_alpha: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Expand Stage-1 ROI `[x1, y1, x2, y2]` symmetrically by split-conformal margin `q_alpha`."""
    if not math.isfinite(q_alpha) or q_alpha < 0.0:
        raise ValueError(f"Conformal quantile margin q_alpha must be finite and >= 0, got {q_alpha}")
    if img_w < 2 or img_h < 2:
        raise ValueError(f"Invalid image dimensions ({img_w}, {img_h})")
    x1, y1, x2, y2 = (float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3]))
    if x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid box_xyxy {box_xyxy}: require x2 >= x1 and y2 >= y1")
    ex1 = max(0.0, x1 - q_alpha)
    ey1 = max(0.0, y1 - q_alpha)
    ex2 = min(float(img_w - 1), x2 + q_alpha)
    ey2 = min(float(img_h - 1), y2 + q_alpha)
    return (ex1, ey1, ex2, ey2)


class DFineMRoiDetector(nn.Module):
    """Lightweight pure-PyTorch Stage-1 D-FINE-M COCO-only runway ROI detector wrapper."""

    weight_id: str = "dfine_m_coco"
    license_id: str = "Apache-2.0"

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.box_head = nn.Linear(hidden_dim, 4)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return normalized `[cx, cy, w, h]` in `(0, 1)` and confidence logits `[B, 1]`."""
        feat = self.pool(self.stem(x)).flatten(1)
        box_cxcywh = torch.sigmoid(self.box_head(feat))
        score_logit = self.score_head(feat)
        return {"box_cxcywh": box_cxcywh, "score_logit": score_logit}

    @torch.no_grad()
    def predict_expanded_roi(
        self,
        x: torch.Tensor,
        q_alpha: float,
    ) -> list[tuple[float, float, float, float]]:
        """Predict pixel-space `[x1, y1, x2, y2]` ROIs expanded by conformal margin `q_alpha`."""
        _, _, h_img, w_img = x.shape
        out = self.forward(x)
        boxes = out["box_cxcywh"].cpu().tolist()
        expanded: list[tuple[float, float, float, float]] = []
        for cx_n, cy_n, bw_n, bh_n in boxes:
            cx = cx_n * float(w_img - 1)
            cy = cy_n * float(h_img - 1)
            hw = max(4.0, 0.5 * bw_n * float(w_img))
            hh = max(4.0, 0.5 * bh_n * float(h_img))
            expanded.append(
                expand_roi_conformal((cx - hw, cy - hh, cx + hw, cy + hh), q_alpha, w_img, h_img)
            )
        return expanded


class ConvNeXtV1Block(nn.Module):
    """MIT ConvNeXt-V1 block (7x7 depthwise -> LayerNorm -> 4x MLP -> LayerScale + residual)."""

    def __init__(self, dim: int, layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones((dim,), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.dwconv(x)
        y = y.permute(0, 2, 3, 1)  # NCHW -> NHWC
        y = self.norm(y)
        y = self.pwconv2(self.act(self.pwconv1(y)))
        y = y * self.gamma
        y = y.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return residual + y


class ConvNeXtV1TinyBackbone(nn.Module):
    """Pure-PyTorch ConvNeXt-V1-style MIT backbone (zero ConvNeXt-V2 GRN/NC weights)."""

    weight_id: str = "convnext_v1_small"
    license_id: str = "MIT"

    def __init__(
        self,
        in_channels: int = 3,
        dims: tuple[int, int, int, int] = (48, 96, 192, 256),
        depths: tuple[int, int, int, int] = (1, 1, 2, 1),
    ) -> None:
        super().__init__()
        self.dims = dims
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                nn.GroupNorm(1, dims[0]),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.GroupNorm(1, dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )
        self.stages = nn.ModuleList(
            [nn.Sequential(*[ConvNeXtV1Block(dims[i]) for _ in range(depths[i])]) for i in range(4)]
        )
        self.out_channels = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x
        for ds, st in zip(self.downsample_layers, self.stages):
            feat = st(ds(feat))
        return feat


class SimCCKeypointHead(nn.Module):
    """12-keypoint SimCC 1D horizontal & vertical coordinate classification head."""

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int = NUM_KEYPOINTS,
        roi_width: int = 256,
        roi_height: int = 256,
        split_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.split_ratio = split_ratio
        self.bins_x = int(round(roi_width * split_ratio))
        self.bins_y = int(round(roi_height * split_ratio))

        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, num_keypoints * 8, kernel_size=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flat_dim = num_keypoints * 8 * 16
        self.fc_x = nn.Linear(flat_dim, num_keypoints * self.bins_x)
        self.fc_y = nn.Linear(flat_dim, num_keypoints * self.bins_y)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = feat.shape[0]
        h = self.pool(self.conv_reduce(feat)).flatten(1)
        logits_x = self.fc_x(h).view(b, self.num_keypoints, self.bins_x)
        logits_y = self.fc_y(h).view(b, self.num_keypoints, self.bins_y)
        return logits_x, logits_y


def decode_simcc(
    logits_x: torch.Tensor,
    logits_y: torch.Tensor,
    split_ratio: float = 2.0,
) -> torch.Tensor:
    """Decode SimCC 1D logits `[B, K, W_s]`, `[B, K, H_s]` via soft-argmax into `[B, K, 2]` px."""
    px = F.softmax(logits_x.float(), dim=-1)
    py = F.softmax(logits_y.float(), dim=-1)
    grid_x = torch.arange(logits_x.shape[-1], device=logits_x.device, dtype=torch.float32) / float(split_ratio)
    grid_y = torch.arange(logits_y.shape[-1], device=logits_y.device, dtype=torch.float32) / float(split_ratio)
    u = torch.sum(px * grid_x.view(1, 1, -1), dim=-1)
    v = torch.sum(py * grid_y.view(1, 1, -1), dim=-1)
    return torch.stack([u, v], dim=-1)


class HesseKeylineHead(nn.Module):
    """5 Continuous Hesse Keylines (OKLS): `(rho, cos_theta, sin_theta)` with epsilon-guarded unit norm."""

    def __init__(self, in_channels: int, num_keylines: int = NUM_KEYLINES) -> None:
        super().__init__()
        self.num_keylines = num_keylines
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, num_keylines * 3),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b = feat.shape[0]
        raw = self.mlp(self.pool(feat).flatten(1)).view(b, self.num_keylines, 3)
        rho = raw[..., 0:1]
        c_raw = raw[..., 1:2]
        s_raw = raw[..., 2:3]
        # BC-M2-m5: epsilon-guarded unit normalization `(c, s) / sqrt(c^2 + s^2 + 1e-12)`
        # Guard exact (0, 0) input by offsetting c_raw by tiny tie-breaker when c^2+s^2 == 0
        sq_norm = c_raw * c_raw + s_raw * s_raw
        c_safe = torch.where(sq_norm < OKLS_NORM_EPS, torch.ones_like(c_raw), c_raw)
        s_safe = torch.where(sq_norm < OKLS_NORM_EPS, torch.zeros_like(s_raw), s_raw)
        denom = torch.sqrt(c_safe * c_safe + s_safe * s_safe + OKLS_NORM_EPS)
        cos_u = c_safe / denom
        sin_u = s_safe / denom
        return torch.cat([rho, cos_u, sin_u], dim=-1)


def okls_keyline_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Continuous Hesse keyline loss (rho SmoothL1 + unit-normal cosine distance)."""
    rho_elem = F.smooth_l1_loss(pred[..., 0], target[..., 0], reduction="none")
    cos_sim = pred[..., 1] * target[..., 1] + pred[..., 2] * target[..., 2]
    ang_elem = 1.0 - cos_sim
    elem = rho_elem + ang_elem
    if sample_mask is None:
        return elem.mean()
    m = sample_mask.float().reshape(-1, 1)
    per_sample = elem.mean(dim=-1)
    return (per_sample * m.squeeze(-1)).sum() / m.sum().clamp(min=1.0)


def compute_simcc_uncertainty_cues(
    logits_x: torch.Tensor,
    logits_y: torch.Tensor,
    *,
    bins_x: int | None = None,
    bins_y: int | None = None,
) -> torch.Tensor:
    """Compute detached ONNX-friendly SimCC marginal entropy & peak-to-secondary ratio `[B, K, 4]`.

    Uses strictly Opset-17 ops (zero `TopK`): `[H_norm(p_x), H_norm(p_y), PSR_norm_x, PSR_norm_y]`.
    """
    lx = logits_x.detach().float()
    ly = logits_y.detach().float()
    px = F.softmax(lx, dim=-1)
    py = F.softmax(ly, dim=-1)

    nx = int(bins_x) if bins_x is not None else int(lx.shape[-1])
    ny = int(bins_y) if bins_y is not None else int(ly.shape[-1])
    log_nx = max(1.0, math.log(float(nx)))
    log_ny = max(1.0, math.log(float(ny)))
    hx = -(px * torch.log(px.clamp(min=1.0e-8))).sum(dim=-1) / log_nx
    hy = -(py * torch.log(py.clamp(min=1.0e-8))).sum(dim=-1) / log_ny

    # Peak-to-secondary ratio via ArgMax-centered lobe suppression (zero TopK; exact 0.0 on uniform logits)
    idx_x = px.argmax(dim=-1, keepdim=True)
    bins_x_grid = torch.arange(nx, device=px.device, dtype=idx_x.dtype).view(1, 1, nx)
    lobe_x = ((bins_x_grid - idx_x).abs() <= 4).to(dtype=torch.float32)
    p1_x = px.amax(dim=-1)
    p2_x = (px * (1.0 - lobe_x)).amax(dim=-1)
    psr_x = torch.log((p1_x + 1.0e-6) / (p2_x + 1.0e-6)).clamp(min=0.0, max=10.0) * 0.1

    idx_y = py.argmax(dim=-1, keepdim=True)
    bins_y_grid = torch.arange(ny, device=py.device, dtype=idx_y.dtype).view(1, 1, ny)
    lobe_y = ((bins_y_grid - idx_y).abs() <= 4).to(dtype=torch.float32)
    p1_y = py.amax(dim=-1)
    p2_y = (py * (1.0 - lobe_y)).amax(dim=-1)
    psr_y = torch.log((p1_y + 1.0e-6) / (p2_y + 1.0e-6)).clamp(min=0.0, max=10.0) * 0.1

    return torch.stack([hx, hy, psr_x, psr_y], dim=-1)


def focal_bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal binary cross-entropy loss in FP32 for `runway_present_logit`."""
    l_f = logits.float().reshape(-1)
    t_f = targets.float().reshape(-1)
    bce = F.binary_cross_entropy_with_logits(l_f, t_f, reduction="none")
    prob = torch.sigmoid(l_f)
    pt = torch.where(t_f > 0.5, prob, 1.0 - prob)
    alpha_t = torch.where(t_f > 0.5, torch.full_like(t_f, float(alpha)), torch.full_like(t_f, 1.0 - float(alpha)))
    focal = alpha_t * ((1.0 - pt).clamp(min=1.0e-6) ** float(gamma)) * bce
    return focal.mean()


def monotonic_degradation_ranking_loss(
    log_sigma_clean: torch.Tensor,
    log_sigma_corrupt: torch.Tensor,
    err_clean: torch.Tensor,
    err_corrupt: torch.Tensor,
    *,
    margin: float = 0.35,
    err_ratio_thr: float = 1.25,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Monotonic degradation ranking loss `relu(margin - (log_sigma_corrupt - log_sigma_clean))`.

    Activates whenever `err_corrupt > err_ratio_thr * err_clean` (plus optional `sample_mask`),
    preventing Stage-2B covariance anti-calibration under glare, occlusion, rain, and motion blur.
    """
    ls_c = log_sigma_clean.float()
    ls_d = log_sigma_corrupt.float()
    ec = err_clean.detach().float()
    ed = err_corrupt.detach().float()
    degraded = (ed > (float(err_ratio_thr) * ec.clamp(min=1.0e-3))).float()
    if sample_mask is not None:
        m = sample_mask.float()
        while m.ndim < degraded.ndim:
            m = m.unsqueeze(-1)
        degraded = degraded * m
    hinge = F.relu(float(margin) - (ls_d - ls_c))
    denom = degraded.sum()
    if float(denom.detach().item()) <= 0.0:
        return (hinge * 0.0).sum()
    return (hinge * degraded).sum() / denom.clamp(min=1.0)


class CholeskyBivariateGnllHead(nn.Module):
    """Stage-2B frozen-mean 2x2 Cholesky covariance head (`BC-M2-m5`).

    Strictly detaches backbone features (`feat.detach()`) so variance calibration cannot corrupt
    Stage-2A point/line localization, and constructs strictly SPD `Sigma = L @ L^T` with:
      `L = [[exp(clamp(l0, -6, 4)), 0], [l1, exp(clamp(l2, -6, 4))]]`.
    When `extra_cue_dim > 0` (extended), consumes per-keypoint detached SimCC entropy + PSR cues.
    """

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int = NUM_KEYPOINTS,
        *,
        extra_cue_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.extra_cue_dim = int(extra_cue_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if self.extra_cue_dim > 0:
            hidden = max(32, in_channels // 2)
            self.mlp = nn.Sequential(
                nn.Linear(in_channels + self.extra_cue_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels // 2),
                nn.GELU(),
                nn.Linear(in_channels // 2, num_keypoints * 3),
            )

    @staticmethod
    def build_cholesky_and_cov(raw_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build strictly SPD lower-triangular `L [..., 2, 2]` and `Sigma = L @ L^T [..., 2, 2]`."""
        l0 = torch.clamp(raw_params[..., 0], min=CHOLESKY_LOG_DIAG_MIN, max=CHOLESKY_LOG_DIAG_MAX)
        l2 = torch.clamp(raw_params[..., 2], min=CHOLESKY_LOG_DIAG_MIN, max=CHOLESKY_LOG_DIAG_MAX)

        d0 = torch.exp(l0)
        d1 = torch.exp(l2)
        # Bound off-diagonal l1 relative to d1 so correlation stays |rho| <= 0.97 and Sigma stays strictly SPD in FP32
        max_off = 4.0 * d1
        l1 = torch.clamp(raw_params[..., 1], min=-max_off, max=max_off)
        zeros = torch.zeros_like(d0)

        row0 = torch.stack([d0, zeros], dim=-1)
        row1 = torch.stack([l1, d1], dim=-1)
        chol_l = torch.stack([row0, row1], dim=-2)
        sigma = torch.matmul(chol_l, chol_l.transpose(-1, -2))
        return chol_l, sigma

    def forward(
        self,
        feat: torch.Tensor,
        extra_cues: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Strict frozen-mean Stage-2B gradient barrier
        detached = feat.detach()
        b = detached.shape[0]
        pooled = self.pool(detached).flatten(1)
        if self.extra_cue_dim > 0:
            pooled_f = pooled.float().unsqueeze(1).expand(b, self.num_keypoints, pooled.shape[1])
            if extra_cues is not None:
                cues_f = extra_cues.detach().float()
            else:
                cues_f = torch.zeros(
                    (b, self.num_keypoints, self.extra_cue_dim),
                    device=pooled.device,
                    dtype=torch.float32,
                )
            raw = self.mlp(torch.cat([pooled_f, cues_f], dim=-1))
        else:
            raw = self.mlp(pooled).view(b, self.num_keypoints, 3)
        return self.build_cholesky_and_cov(raw)


def bivariate_gnll_loss(
    residual_xy: torch.Tensor,
    chol_l: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact closed-form 2D bivariate Gaussian NLL using Cholesky factor `L` (FP32 stable)."""
    r = residual_xy.float()
    l_mat = chol_l.float()
    l00 = l_mat[..., 0, 0]
    l10 = l_mat[..., 1, 0]
    l11 = l_mat[..., 1, 1]

    # Forward substitution solve L z = r
    z0 = r[..., 0] / l00
    z1 = (r[..., 1] - l10 * z0) / l11
    quad = z0 * z0 + z1 * z1
    log_det = 2.0 * (torch.log(l00) + torch.log(l11))
    nll = 0.5 * (quad + log_det + 2.0 * math.log(2.0 * math.pi))
    if sample_mask is None:
        return nll.mean()
    m = sample_mask.float().reshape(-1)
    per_sample = nll.mean(dim=-1)
    return (per_sample * m).sum() / m.sum().clamp(min=1.0)


class StudentTCalibrator:
    """Heavy-tailed Student-t post-hoc covariance calibrator (`nu > 2`)."""

    def __init__(self, nu: float = 6.0, scale: float = 1.0) -> None:
        if not (math.isfinite(nu) and nu > 2.0):
            raise ValueError(f"Student-t degrees of freedom nu must be > 2, got {nu}")
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(f"Student-t scale must be > 0, got {scale}")
        self.nu = float(nu)
        self.scale = float(scale)

    def fit_scale_from_mahalanobis(
        self,
        mahalanobis_sq: torch.Tensor,
        target_dof: float = 2.0,
    ) -> float:
        """Calibrate scale factor so empirical normalized Mahalanobis squared matches Student-t mean."""
        m2 = mahalanobis_sq.detach().float()
        finite_m2 = m2[torch.isfinite(m2)]
        if finite_m2.numel() == 0:
            raise ValueError("Cannot calibrate Student-t scale on empty/non-finite Mahalanobis samples")
        emp_mean = float(finite_m2.mean().item())
        theo_factor = self.nu / (self.nu - 2.0)
        self.scale = math.sqrt(max(1e-4, emp_mean / (target_dof * theo_factor)))
        return self.scale

    def calibrate_covariance(self, sigma: torch.Tensor) -> torch.Tensor:
        """Return calibrated strictly SPD covariance `(scale^2 * nu / (nu - 2)) * Sigma`."""
        factor = (self.scale * self.scale) * (self.nu / (self.nu - 2.0))
        return sigma * factor


class XvsLandModel(nn.Module):
    """Unified XVS-Land Stage-2A (SimCC + OKLS) + Stage-2B (Frozen-Mean Cholesky GNLL) model."""

    def __init__(
        self,
        in_channels: int = 3,
        roi_width: int = 256,
        roi_height: int = 256,
        split_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.split_ratio = split_ratio
        self.backbone = ConvNeXtV1TinyBackbone(in_channels=in_channels)
        c_out = self.backbone.out_channels
        self.simcc_head = SimCCKeypointHead(
            in_channels=c_out,
            num_keypoints=NUM_KEYPOINTS,
            roi_width=roi_width,
            roi_height=roi_height,
            split_ratio=split_ratio,
        )
        self.keyline_head = HesseKeylineHead(in_channels=c_out, num_keylines=NUM_KEYLINES)
        self.cov_head = CholeskyBivariateGnllHead(in_channels=c_out, num_keypoints=NUM_KEYPOINTS)

    def forward(self, roi_crop: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(roi_crop)
        logits_x, logits_y = self.simcc_head(feat)
        keypoints_xy = decode_simcc(logits_x, logits_y, split_ratio=self.split_ratio)
        keylines_hesse = self.keyline_head(feat)
        chol_l, cov_2x2 = self.cov_head(feat)
        return {
            "logits_x": logits_x,
            "logits_y": logits_y,
            "keypoints_xy": keypoints_xy,
            "keylines_hesse": keylines_hesse,
            "cholesky_l": chol_l,
            "cov_2x2": cov_2x2,
        }


# =============================================================================
# XVS-Land v5.2 Architecture (<= 2.5M params, <= 6.0 MiB FP16 ONNX, <= 12 ms ORT)
# =============================================================================


# Canonical normalized runway keyline endpoints in [-1, 1] for well-conditioned init
# (left edge, right edge, centerline, threshold bar, aiming-point bar)
_CANONICAL_KEYLINE_ENDPOINTS_NORM: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (
    ((-0.30, 0.55), (-0.06, -0.15)),
    ((0.30, 0.55), (0.06, -0.15)),
    ((0.00, 0.55), (0.00, -0.15)),
    ((-0.30, 0.55), (0.30, 0.55)),
    ((-0.20, 0.25), (0.20, 0.25)),
)


def _append_coord_channels_1d(
    feat: torch.Tensor,
    x_1d: torch.Tensor,
    y_1d: torch.Tensor,
    *,
    coord_dropout_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Append `(x_norm, y_norm)` CoordConv channels from tiny 1D buffers (3 KB vs 1.13 MiB).

    When `training=True` and `coord_dropout_prob > 0.0` (extended), randomly zeroes `(x_norm, y_norm)`
    per sample so the backbone cannot shortcut-memorize absolute pixel coordinates.
    """
    b, _, h, w = feat.shape
    if not torch.onnx.is_in_onnx_export() and (x_1d.shape[3] != w or y_1d.shape[2] != h):
        xs = torch.linspace(-1.0, 1.0, steps=w, device=feat.device, dtype=feat.dtype).view(1, 1, 1, w)
        ys = torch.linspace(-1.0, 1.0, steps=h, device=feat.device, dtype=feat.dtype).view(1, 1, h, 1)
    else:
        xs = x_1d.to(device=feat.device, dtype=feat.dtype)
        ys = y_1d.to(device=feat.device, dtype=feat.dtype)
    xx = xs.expand(b, 1, h, w)
    yy = ys.expand(b, 1, h, w)
    if training and float(coord_dropout_prob) > 0.0 and not torch.onnx.is_in_onnx_export():
        keep = (torch.rand((b, 1, 1, 1), device=feat.device) >= float(coord_dropout_prob)).to(dtype=feat.dtype)
        xx = xx * keep
        yy = yy * keep
    return torch.cat([feat, xx, yy], dim=1)


class ConvNeXtV1SpatialBlock(nn.Module):
    """NCHW-native MIT ConvNeXt-V1 block (7x7 dwconv -> BN -> 1x1 MLP -> LayerScale + residual).

    `BatchNorm2d` folds cleanly into `dwconv` during ONNX export and `ReLU` fuses inside
    ORT NCHWc `FusedConv` kernels (zero `Reshape`/`InstanceNormalization` layout switches).
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 3.0,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden = max(dim, int(round(dim * mlp_ratio)))
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.ReLU(inplace=False)
        self.pwconv2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones((1, dim, 1, 1), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwconv(x)
        y = self.norm(y)
        y = self.pwconv2(self.act(self.pwconv1(y)))
        return x + y * self.gamma


class ConvNeXtV1CoordFpnBackbone(nn.Module):
    """Pure-PyTorch MIT ConvNeXt-V1 multi-scale FPN encoder with CoordConv `(x_norm, y_norm)`.

    Extracts 4 hierarchical stages (`C2: H/4`, `C3: H/8`, `C4: H/16`, `C5: H/32`) with CoordConv
    injected cleanly at `H/4` (zero full-res concat bloat) and fuses them via static
    `ConvTranspose2d` top-down FPN (zero `Resize` ops in ONNX) into `fpn_feat [B, fpn_dim, H/8, W/8]`
    (or `H/4`) plus semantic context `c5 [B, dims[-1], H/32, W/32]`.
    """

    weight_id: str = "convnext_v1_small"
    license_id: str = "MIT"

    def __init__(
        self,
        in_channels: int = 3,
        roi_width: int = 384,
        roi_height: int = 384,
        dims: tuple[int, int, int, int] = (32, 64, 128, 192),
        depths: tuple[int, int, int, int] = (1, 2, 3, 2),
        fpn_dim: int = 64,
        mlp_ratio: float = 3.0,
        fpn_stride: int = 8,
        coord_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if fpn_stride not in (4, 8):
            raise ValueError(f"fpn_stride must be 4 or 8, got {fpn_stride}")
        self.dims = dims
        self.fpn_dim = int(fpn_dim)
        self.fpn_stride = int(fpn_stride)
        self.coord_dropout_prob = float(coord_dropout_prob)
        self.out_channels = int(fpn_dim)
        self.c5_channels = int(dims[-1])

        c2_w = max(1, int(roi_width) // 4)
        c2_h = max(1, int(roi_height) // 4)
        self.register_buffer(
            "c2_x_1d",
            torch.linspace(-1.0, 1.0, steps=c2_w, dtype=torch.float32).view(1, 1, 1, c2_w),
        )
        self.register_buffer(
            "c2_y_1d",
            torch.linspace(-1.0, 1.0, steps=c2_h, dtype=torch.float32).view(1, 1, c2_h, 1),
        )

        self.downsample_layers = nn.ModuleList()
        # Stem: RGB -> dims[0] - 2 at H/4 x W/4, then append exact (x_norm, y_norm) CoordConv -> dims[0]
        stem_out = max(4, dims[0] - 2)
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, stem_out, kernel_size=4, stride=4, bias=False),
                nn.BatchNorm2d(stem_out),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(dims[i + 1]),
                )
            )

        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[ConvNeXtV1SpatialBlock(dims[i], mlp_ratio=mlp_ratio) for _ in range(depths[i])]
                )
                for i in range(4)
            ]
        )

        # Lateral 1x1 projections for FPN
        self.lat5 = nn.Conv2d(dims[3], fpn_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(dims[2], fpn_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(dims[1], fpn_dim, kernel_size=1)
        self.lat2 = (
            nn.Conv2d(dims[0], fpn_dim, kernel_size=1)
            if fpn_stride == 4
            else nn.Conv2d(dims[0], fpn_dim, kernel_size=2, stride=2)
        )

        # Static depthwise 2x ConvTranspose upsamplers (zero ONNX Resize ops)
        self.up5_to_4 = nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=2, stride=2, groups=fpn_dim)
        self.up4_to_3 = nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=2, stride=2, groups=fpn_dim)
        self.up3_to_2 = (
            nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=2, stride=2, groups=fpn_dim)
            if fpn_stride == 4
            else None
        )

        self.fpn_refine = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, groups=fpn_dim, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s0 = self.downsample_layers[0](x)
        s0_coord = _append_coord_channels_1d(
            s0,
            self.c2_x_1d,
            self.c2_y_1d,
            coord_dropout_prob=self.coord_dropout_prob,
            training=self.training,
        )
        c2 = self.stages[0](s0_coord)
        c3 = self.stages[1](self.downsample_layers[1](c2))
        c4 = self.stages[2](self.downsample_layers[2](c3))
        c5 = self.stages[3](self.downsample_layers[3](c4))

        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + self.up5_to_4(p5)
        p3 = self.lat3(c3) + self.up4_to_3(p4)
        if self.up3_to_2 is not None:
            p_out = self.lat2(c2) + self.up3_to_2(p3)
        else:
            p_out = p3 + self.lat2(c2)
        return self.fpn_refine(p_out), c5


class SpatialSimCCHead(nn.Module):
    """High-resolution spatial SimCC keypoint head (zero 4x4 spatial crush, <150K params, <0.4ms ORT).

    Operates on FPN spatial feature map `[B, C, H_f, W_f]` (`H/8 x W/8` or `H/4 x W/4`),
    projects to `K=12` spatial keypoint maps via lightweight depthwise-separable CoordConv,
    pools along Y (mean, max, softmax-weighted response + Y-elevation expectation) for X-1D
    logits (`bins_x = roi_w * split_ratio`) and symmetrically along X for Y-1D logits via
    cross-keypoint 1D Conv mixers + lightweight axis projections + 1D local refinement.
    """

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int = NUM_KEYPOINTS,
        roi_width: int = 384,
        roi_height: int = 384,
        split_ratio: float = 2.0,
        feat_stride: int = 8,
        coord_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.roi_width = int(roi_width)
        self.roi_height = int(roi_height)
        self.split_ratio = float(split_ratio)
        self.feat_stride = int(feat_stride)
        self.coord_dropout_prob = float(coord_dropout_prob)
        self.feat_w = max(1, self.roi_width // self.feat_stride)
        self.feat_h = max(1, self.roi_height // self.feat_stride)
        self.bins_x = int(round(self.roi_width * self.split_ratio))
        self.bins_y = int(round(self.roi_height * self.split_ratio))

        self.register_buffer(
            "y_coord_1d",
            torch.linspace(-1.0, 1.0, steps=self.feat_h, dtype=torch.float32).view(1, 1, self.feat_h, 1),
        )
        self.register_buffer(
            "x_coord_1d",
            torch.linspace(-1.0, 1.0, steps=self.feat_w, dtype=torch.float32).view(1, 1, 1, self.feat_w),
        )

        # Lightweight depthwise-separable CoordConv projection -> K=12 spatial keypoint maps
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(in_channels + 2, in_channels + 2, kernel_size=3, padding=1, groups=in_channels + 2, bias=False),
            nn.BatchNorm2d(in_channels + 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels + 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, self.num_keypoints, kernel_size=1),
        )

        k = self.num_keypoints
        # Cross-keypoint 1D mixers operate at compact W_f / H_f resolution (zero high-res ConvTranspose bloat)
        self.x_mixer = nn.Sequential(
            nn.Conv1d(k * 4, k * 4, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.Conv1d(k * 4, k * 2, kernel_size=3, padding=1, groups=k),
            nn.ReLU(inplace=False),
            nn.Conv1d(k * 2, k, kernel_size=1, groups=k),
        )
        self.y_mixer = nn.Sequential(
            nn.Conv1d(k * 4, k * 4, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.Conv1d(k * 4, k * 2, kernel_size=3, padding=1, groups=k),
            nn.ReLU(inplace=False),
            nn.Conv1d(k * 2, k, kernel_size=1, groups=k),
        )

        # Shared 1D coordinate expansion + lightweight per-keypoint 1D local refinement
        self.x_proj = nn.Linear(self.feat_w, self.bins_x)
        self.y_proj = nn.Linear(self.feat_h, self.bins_y)
        self.x_refine = nn.Conv1d(k, k, kernel_size=5, padding=2, groups=k)
        self.y_refine = nn.Conv1d(k, k, kernel_size=5, padding=2, groups=k)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_c = _append_coord_channels_1d(
            feat,
            self.x_coord_1d,
            self.y_coord_1d,
            coord_dropout_prob=self.coord_dropout_prob,
            training=self.training,
        )
        kp_maps = self.spatial_proj(feat_c)  # [B, K, H_f, W_f]

        # Keep marginal pooling & softmax in FP32 for numerical stability
        kp_f = kp_maps.float()
        y_grid = self.y_coord_1d
        x_grid = self.x_coord_1d
        if not torch.onnx.is_in_onnx_export():
            if y_grid.shape[2] != kp_f.shape[2]:
                y_grid = torch.linspace(-1.0, 1.0, steps=kp_f.shape[2], device=kp_f.device, dtype=torch.float32).view(
                    1, 1, kp_f.shape[2], 1
                )
            if x_grid.shape[3] != kp_f.shape[3]:
                x_grid = torch.linspace(-1.0, 1.0, steps=kp_f.shape[3], device=kp_f.device, dtype=torch.float32).view(
                    1, 1, 1, kp_f.shape[3]
                )

        # Pool along Y (dim=2) -> X-1D sequence [B, K*4, W_f]
        w_y = F.softmax(kp_f, dim=2)
        x_mean = kp_f.mean(dim=2)
        x_max = kp_f.amax(dim=2)
        x_attn = (kp_f * w_y).sum(dim=2)
        x_ypos = (y_grid * w_y).sum(dim=2)
        x_seq = torch.cat([x_mean, x_max, x_attn, x_ypos], dim=1)

        # Pool along X (dim=3) -> Y-1D sequence [B, K*4, H_f]
        w_x = F.softmax(kp_f, dim=3)
        y_mean = kp_f.mean(dim=3)
        y_max = kp_f.amax(dim=3)
        y_attn = (kp_f * w_x).sum(dim=3)
        y_xpos = (x_grid * w_x).sum(dim=3)
        y_seq = torch.cat([y_mean, y_max, y_attn, y_xpos], dim=1)

        x_feat = self.x_mixer(x_seq) + x_attn  # [B, K, W_f]
        y_feat = self.y_mixer(y_seq) + y_attn  # [B, K, H_f]

        logits_x = self.x_proj(x_feat)
        logits_y = self.y_proj(y_feat)
        logits_x = logits_x + self.x_refine(logits_x)
        logits_y = logits_y + self.y_refine(logits_y)
        return logits_x, logits_y


class CoordConvHesseKeylineHead(nn.Module):
    """CoordConv spatial Hesse Keyline head with endpoint-consistent geometry & per-line sigma.

    Keeps full `(u, v)` CoordConv spatial maps (zero `AdaptiveAvgPool2d((1, 1))` geometry loss!),
    predicts differentiable 2D endpoint pairs `(p0, p1)` via spatial soft-argmax + sub-pixel
    refinement, derives canonical epsilon-guarded unit-norm Hesse parameters `(rho, cos_theta, sin_theta)`
    matching `compute_hesse_line_from_points`, AND predicts strictly positive detached per-keyline
    uncertainty `keyline_sigma_px [B, 5]` clamped strictly to `[0.5, 50.0]`.
    """

    def __init__(
        self,
        in_channels: int,
        num_keylines: int = NUM_KEYLINES,
        roi_width: int = 384,
        roi_height: int = 384,
        feat_stride: int = 8,
        context_channels: int = 0,
        *,
        coord_dropout_prob: float = 0.0,
        use_endpoint_prior: bool = True,
        extra_cue_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_keylines = int(num_keylines)
        self.roi_width = int(roi_width)
        self.roi_height = int(roi_height)
        self.feat_stride = int(feat_stride)
        self.feat_w = max(1, self.roi_width // self.feat_stride)
        self.feat_h = max(1, self.roi_height // self.feat_stride)
        self.context_channels = int(context_channels)
        self.coord_dropout_prob = float(coord_dropout_prob)
        self.use_endpoint_prior = bool(use_endpoint_prior)
        self.extra_cue_dim = int(extra_cue_dim)

        self.register_buffer(
            "x_coord_1d",
            torch.linspace(-1.0, 1.0, steps=self.feat_w, dtype=torch.float32).view(1, 1, 1, self.feat_w),
        )
        self.register_buffer(
            "y_coord_1d",
            torch.linspace(-1.0, 1.0, steps=self.feat_h, dtype=torch.float32).view(1, 1, self.feat_h, 1),
        )
        # Exact pixel-space cell-center coordinate grids spanning [0, roi - 1]
        u_px = (torch.arange(self.feat_w, dtype=torch.float32) + 0.5) * (
            float(self.roi_width) / float(self.feat_w)
        ) - 0.5
        v_px = (torch.arange(self.feat_h, dtype=torch.float32) + 0.5) * (
            float(self.roi_height) / float(self.feat_h)
        ) - 0.5
        vv, uu = torch.meshgrid(v_px, u_px, indexing="ij")
        self.register_buffer("u_flat_px", uu.reshape(1, 1, self.feat_h * self.feat_w))
        self.register_buffer("v_flat_px", vv.reshape(1, 1, self.feat_h * self.feat_w))

        # Lightweight depthwise-separable CoordConv tower preserving exact (u, v) line geometry
        self.line_tower = nn.Sequential(
            nn.Conv2d(in_channels + 2, in_channels + 2, kernel_size=5, padding=2, groups=in_channels + 2, bias=False),
            nn.BatchNorm2d(in_channels + 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels + 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
        )
        # 2 spatial endpoint attention maps per keyline (p0, p1)
        self.endpoint_attn = nn.Conv2d(in_channels, self.num_keylines * 2, kernel_size=1)
        # Learned canonical endpoint offset prior keeps ||p1 - p0|| well-conditioned even at init
        prior_list = [
            _CANONICAL_KEYLINE_ENDPOINTS_NORM[k % len(_CANONICAL_KEYLINE_ENDPOINTS_NORM)]
            for k in range(self.num_keylines)
        ]
        prior_t = torch.tensor(prior_list, dtype=torch.float32).reshape(1, self.num_keylines * 2, 2)
        if not self.use_endpoint_prior:
            prior_t = torch.zeros_like(prior_t)
        self.endpoint_prior_norm = nn.Parameter(prior_t, requires_grad=self.use_endpoint_prior)

        # Sub-pixel endpoint offset head removes soft-argmax boundary shrinkage
        self.endpoint_offset = nn.Sequential(
            nn.Linear(in_channels, max(16, in_channels // 2)),
            nn.GELU(),
            nn.Linear(max(16, in_channels // 2), 2),
        )
        if self.use_endpoint_prior:
            nn.init.zeros_(self.endpoint_offset[-1].weight)
            nn.init.zeros_(self.endpoint_offset[-1].bias)
        else:
            # Data-dependent non-zero init prevents p0==p1 0/0 Hessian tangent singularity at untrained init
            # without introducing any static (cx, cy) coordinate shortcut
            nn.init.orthogonal_(self.endpoint_attn.weight, gain=2.0)
            nn.init.normal_(self.endpoint_offset[-1].weight, mean=0.0, std=0.08)
            nn.init.zeros_(self.endpoint_offset[-1].bias)

        # Stage-2B frozen-mean detached per-keyline uncertainty head: sigma_px in [0.5, 50.0]
        sigma_in = in_channels * 2 + self.context_channels + self.extra_cue_dim
        self.sigma_head = nn.Sequential(
            nn.Linear(sigma_in, max(32, in_channels)),
            nn.GELU(),
            nn.Linear(max(32, in_channels), 1),
        )
        nn.init.constant_(self.sigma_head[-1].bias, math.log(2.0))

    def geometry_parameters(self) -> list[nn.Parameter]:
        """Stage-2A geometry parameters (strictly excludes Stage-2B `sigma_head`)."""
        params: list[nn.Parameter] = [self.endpoint_prior_norm] if self.use_endpoint_prior else []
        for mod in (self.line_tower, self.endpoint_attn, self.endpoint_offset):
            params.extend(mod.parameters())
        return params

    def sigma_parameters(self) -> list[nn.Parameter]:
        """Stage-2B uncertainty parameters (`sigma_head` only)."""
        return list(self.sigma_head.parameters())

    def forward_with_endpoints(
        self,
        feat: torch.Tensor,
        context_feat: torch.Tensor | None = None,
        simcc_cues: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(keylines_hesse [B, L, 3], keyline_sigma_px [B, L], endpoints_uv [B, L, 2, 2])`."""
        b, _, h, w = feat.shape
        n_sp = h * w
        feat_c = _append_coord_channels_1d(
            feat,
            self.x_coord_1d,
            self.y_coord_1d,
            coord_dropout_prob=self.coord_dropout_prob,
            training=self.training,
        )
        line_feat = self.line_tower(feat_c)  # [B, C, H, W]
        attn_logits = self.endpoint_attn(line_feat).float()  # [B, 2L, H, W]

        attn_flat = F.softmax(attn_logits.reshape(b, self.num_keylines * 2, n_sp), dim=-1)  # [B, 2L, HW]

        u_flat = self.u_flat_px
        v_flat = self.v_flat_px
        if not torch.onnx.is_in_onnx_export() and u_flat.shape[-1] != n_sp:
            u_px = (torch.arange(w, device=feat.device, dtype=torch.float32) + 0.5) * (
                float(self.roi_width) / float(w)
            ) - 0.5
            v_px = (torch.arange(h, device=feat.device, dtype=torch.float32) + 0.5) * (
                float(self.roi_height) / float(h)
            ) - 0.5
            vv, uu = torch.meshgrid(v_px, u_px, indexing="ij")
            u_flat = uu.reshape(1, 1, n_sp)
            v_flat = vv.reshape(1, 1, n_sp)

        # Differentiable 2D soft-argmax endpoints [B, 2L]
        u_base = (attn_flat * u_flat).sum(dim=-1)
        v_base = (attn_flat * v_flat).sum(dim=-1)

        # Attention-pooled local visual descriptor per endpoint: [B, 2L, HW] @ [B, HW, C] -> [B, 2L, C]
        feat_hw_c = line_feat.float().reshape(b, line_feat.shape[1], n_sp).transpose(1, 2)
        ep_desc = torch.matmul(attn_flat, feat_hw_c)

        half_w = 0.5 * float(self.roi_width)
        half_h = 0.5 * float(self.roi_height)
        raw_off = torch.tanh(self.endpoint_offset(ep_desc))
        if self.use_endpoint_prior:
            u_ep = u_base + self.endpoint_prior_norm[..., 0] * half_w + raw_off[..., 0] * (0.20 * float(self.roi_width))
            v_ep = v_base + self.endpoint_prior_norm[..., 1] * half_h + raw_off[..., 1] * (0.20 * float(self.roi_height))
        else:
            u_ep = u_base + raw_off[..., 0] * (0.20 * float(self.roi_width))
            v_ep = v_base + raw_off[..., 1] * (0.20 * float(self.roi_height))

        ep_uv = torch.stack([u_ep, v_ep], dim=-1).reshape(b, self.num_keylines, 2, 2)
        p0 = ep_uv[:, :, 0, :]  # [B, L, 2]
        p1 = ep_uv[:, :, 1, :]  # [B, L, 2]

        # Endpoint-consistent analytical Hesse line with BC-M2-m5 epsilon-guarded unit norm (1e-12)
        du = p1[..., 0:1] - p0[..., 0:1]
        dv = p1[..., 1:2] - p0[..., 1:2]
        c_raw = -dv
        s_raw = du
        sq_norm = c_raw * c_raw + s_raw * s_raw
        c_safe = torch.where(sq_norm < OKLS_NORM_EPS, torch.ones_like(c_raw), c_raw)
        s_safe = torch.where(sq_norm < OKLS_NORM_EPS, torch.zeros_like(s_raw), s_raw)
        denom = torch.sqrt(c_safe * c_safe + s_safe * s_safe + OKLS_NORM_EPS)
        cos_u = c_safe / denom
        sin_u = s_safe / denom

        mid_u = 0.5 * (p0[..., 0:1] + p1[..., 0:1])
        mid_v = 0.5 * (p0[..., 1:2] + p1[..., 1:2])
        rho_raw = cos_u * mid_u + sin_u * mid_v
        neg_mask = rho_raw < 0.0
        rho = torch.where(neg_mask, -rho_raw, rho_raw)
        cos_u = torch.where(neg_mask, -cos_u, cos_u)
        sin_u = torch.where(neg_mask, -sin_u, sin_u)
        keylines_hesse = torch.cat([rho, cos_u, sin_u], dim=-1)  # [B, L, 3]

        # Stage-2B frozen-mean detached per-keyline sigma in [0.5, 50.0] px
        ep_desc_det = ep_desc.detach().reshape(b, self.num_keylines, 2 * line_feat.shape[1])
        sig_parts: list[torch.Tensor] = [ep_desc_det]
        if self.context_channels > 0 and context_feat is not None:
            ctx_vec = context_feat.detach().float().mean(dim=(2, 3))  # [B, C_ctx]
            ctx_exp = ctx_vec.unsqueeze(1).expand(b, self.num_keylines, self.context_channels)
            sig_parts.append(ctx_exp)
        if self.extra_cue_dim > 0:
            if simcc_cues is not None:
                cues_det = simcc_cues.detach().float()
                line_cues = torch.stack(
                    [
                        0.5 * (cues_det[:, i0, :] + cues_det[:, i1, :])
                        for i0, i1 in KEYLINE_ENDPOINT_INDICES[: self.num_keylines]
                    ],
                    dim=1,
                )
            else:
                line_cues = torch.zeros(
                    (b, self.num_keylines, self.extra_cue_dim),
                    device=ep_desc_det.device,
                    dtype=torch.float32,
                )
            sig_parts.append(line_cues)
        sig_in = torch.cat(sig_parts, dim=-1) if len(sig_parts) > 1 else sig_parts[0]

        log_sig_raw = self.sigma_head(sig_in).squeeze(-1)  # [B, L]
        log_sig = torch.clamp(
            log_sig_raw,
            min=math.log(KEYLINE_SIGMA_MIN_PX),
            max=math.log(KEYLINE_SIGMA_MAX_PX),
        )
        keyline_sigma_px = torch.clamp(
            torch.exp(log_sig),
            min=KEYLINE_SIGMA_MIN_PX,
            max=KEYLINE_SIGMA_MAX_PX,
        )
        return keylines_hesse, keyline_sigma_px, ep_uv

    def forward(
        self,
        feat: torch.Tensor,
        context_feat: torch.Tensor | None = None,
        simcc_cues: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keylines_hesse, keyline_sigma_px, _ = self.forward_with_endpoints(
            feat, context_feat=context_feat, simcc_cues=simcc_cues
        )
        return keylines_hesse, keyline_sigma_px


def keyline_endpoint_loss(
    pred_endpoints_uv: torch.Tensor,
    gt_keypoints_xy: torch.Tensor,
    endpoint_indices: tuple[tuple[int, int], ...] = KEYLINE_ENDPOINT_INDICES,
    *,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise predicted 5 keyline endpoint pairs `[B, 5, 2, 2]` against canonical GT keypoints."""
    gt_kp = gt_keypoints_xy.float()
    ep_targets = torch.stack(
        [torch.stack([gt_kp[:, i0, :], gt_kp[:, i1, :]], dim=1) for i0, i1 in endpoint_indices],
        dim=1,
    )  # [B, 5, 2, 2]
    elem = F.smooth_l1_loss(pred_endpoints_uv.float(), ep_targets, reduction="none")
    if sample_mask is None:
        return elem.mean()
    m = sample_mask.float().reshape(-1)
    per_sample = elem.mean(dim=(1, 2, 3))
    return (per_sample * m).sum() / m.sum().clamp(min=1.0)


def keyline_keypoint_consistency_loss(
    pred_keylines_hesse: torch.Tensor,
    pred_keypoints_xy: torch.Tensor,
    collinear_groups: tuple[tuple[int, ...], ...] = KEYLINE_COLLINEAR_KEYPOINTS,
    *,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Geometric consistency loss enforcing projective collinearity of 5 keylines & 12 SimCC keypoints.

    Computes sign-invariant perpendicular pixel distance `|c_k * u_j + s_k * v_j - rho_k|`
    for every canonical keypoint `j` lying on runway keyline `k`.
    """
    kl = pred_keylines_hesse.float()  # [B, 5, 3]
    kp = pred_keypoints_xy.float()    # [B, 12, 2]
    dists: list[torch.Tensor] = []
    for k_idx, kp_ids in enumerate(collinear_groups):
        rho_k = kl[:, k_idx : k_idx + 1, 0]  # [B, 1]
        c_k = kl[:, k_idx : k_idx + 1, 1]    # [B, 1]
        s_k = kl[:, k_idx : k_idx + 1, 2]    # [B, 1]
        pts_k = kp[:, list(kp_ids), :]       # [B, M, 2]
        d_perp = c_k * pts_k[..., 0] + s_k * pts_k[..., 1] - rho_k  # [B, M]
        dists.append(d_perp)
    all_d = torch.cat(dists, dim=1)
    elem = F.smooth_l1_loss(all_d, torch.zeros_like(all_d), reduction="none")
    if sample_mask is None:
        return elem.mean()
    m = sample_mask.float().reshape(-1)
    per_sample = elem.mean(dim=-1)
    return (per_sample * m).sum() / m.sum().clamp(min=1.0)


def keyline_sigma_nll_loss(
    pred_keylines_hesse: torch.Tensor,
    pred_keyline_sigma_px: torch.Tensor,
    gt_keypoints_xy: torch.Tensor,
    *,
    endpoint_indices: tuple[tuple[int, int], ...] = KEYLINE_ENDPOINT_INDICES,
    loss_type: str = "gaussian",
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stage-2B line-residual NLL training strictly positive `keyline_sigma_px [B, 5]`.

    Uses detached `pred_keylines_hesse` against ground-truth line endpoints so the predicted
    `keyline_sigma_px` matches empirical perpendicular pixel standard deviation (Gaussian NLL)
    or scale (Laplacian NLL) directly usable by 4-DoF MAP-PnPL.
    """
    kl = pred_keylines_hesse.detach().float()
    sig = pred_keyline_sigma_px.float().clamp(min=KEYLINE_SIGMA_MIN_PX, max=KEYLINE_SIGMA_MAX_PX)
    gt_kp = gt_keypoints_xy.float()

    res_list: list[torch.Tensor] = []
    for k_idx, (i0, i1) in enumerate(endpoint_indices):
        rho_k = kl[:, k_idx, 0]
        c_k = kl[:, k_idx, 1]
        s_k = kl[:, k_idx, 2]
        d0 = c_k * gt_kp[:, i0, 0] + s_k * gt_kp[:, i0, 1] - rho_k
        d1 = c_k * gt_kp[:, i1, 0] + s_k * gt_kp[:, i1, 1] - rho_k
        res_list.append(torch.stack([d0, d1], dim=-1))  # [B, 2]

    res = torch.stack(res_list, dim=1)  # [B, 5, 2]
    sig_exp = sig.unsqueeze(-1)         # [B, 5, 1]
    if loss_type == "laplacian":
        nll = torch.abs(res) / sig_exp + torch.log(2.0 * sig_exp)
    else:
        nll = 0.5 * ((res / sig_exp) ** 2 + 2.0 * torch.log(sig_exp) + math.log(2.0 * math.pi))
    if sample_mask is None:
        return nll.mean()
    m = sample_mask.float().reshape(-1)
    per_sample = nll.mean(dim=(1, 2))
    return (per_sample * m).sum() / m.sum().clamp(min=1.0)


class XvsLandV52Model(nn.Module):
    """XVS-Land v1.1.1 architecture: ConvNeXt-V1 CoordConv FPN + SpatialSimCC + CoordConvHesse + SPD GNLL.

    Fixes v4.5 parameter bloat (`20.56M -> ~1.3M` params, `42.5 MiB -> <= 6.0 MiB` FP16 ONNX),
    removes `4x4` spatial crush in SimCC and `1x1` pooling in OKLS, and adds calibrated
    per-keyline uncertainty `keyline_sigma_px [B, 5]` clamped to `[0.5, 50.0]`.

    Additive extended flags (all defaulting to exact v5.2 behavior):
      - `enable_runway_present_head=True`: adds lightweight `<15K`-param `runway_present_logit [B, 1]` head.
      - `coord_dropout_prob=0.5`: randomly drops CoordConv `(x, y)` channels during training.
      - `use_endpoint_prior=False`: disables static `endpoint_prior_norm` center shortcut.
      - `enable_simcc_uncertainty_cues=True`: feeds detached SimCC entropy + PSR cues into `cov_head` & `sigma_head`.
    """

    arch_id: str = "v52"

    def __init__(
        self,
        in_channels: int = 3,
        roi_width: int = 384,
        roi_height: int = 384,
        split_ratio: float = 2.0,
        dims: tuple[int, int, int, int] = (32, 64, 128, 192),
        depths: tuple[int, int, int, int] = (1, 2, 3, 2),
        fpn_dim: int = 64,
        fpn_stride: int = 8,
        *,
        enable_runway_present_head: bool = False,
        coord_dropout_prob: float = 0.0,
        use_endpoint_prior: bool = True,
        enable_simcc_uncertainty_cues: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.roi_width = int(roi_width)
        self.roi_height = int(roi_height)
        self.split_ratio = float(split_ratio)
        self.dims = dims
        self.depths = depths
        self.fpn_dim = int(fpn_dim)
        self.fpn_stride = int(fpn_stride)
        self.enable_runway_present_head = bool(enable_runway_present_head)
        self.coord_dropout_prob = float(coord_dropout_prob)
        self.use_endpoint_prior = bool(use_endpoint_prior)
        self.enable_simcc_uncertainty_cues = bool(enable_simcc_uncertainty_cues)
        self.arch_id = (
            "v53"
            if (
                self.enable_runway_present_head
                or self.enable_simcc_uncertainty_cues
                or not self.use_endpoint_prior
                or self.coord_dropout_prob > 0.0
            )
            else "v52"
        )

        self.backbone = ConvNeXtV1CoordFpnBackbone(
            in_channels=self.in_channels,
            roi_width=self.roi_width,
            roi_height=self.roi_height,
            dims=dims,
            depths=depths,
            fpn_dim=fpn_dim,
            fpn_stride=fpn_stride,
            coord_dropout_prob=self.coord_dropout_prob,
        )
        self.simcc_head = SpatialSimCCHead(
            in_channels=self.backbone.out_channels,
            num_keypoints=NUM_KEYPOINTS,
            roi_width=self.roi_width,
            roi_height=self.roi_height,
            split_ratio=self.split_ratio,
            feat_stride=self.fpn_stride,
            coord_dropout_prob=self.coord_dropout_prob,
        )
        cue_dim = 4 if self.enable_simcc_uncertainty_cues else 0
        self.keyline_head = CoordConvHesseKeylineHead(
            in_channels=self.backbone.out_channels,
            num_keylines=NUM_KEYLINES,
            roi_width=self.roi_width,
            roi_height=self.roi_height,
            feat_stride=self.fpn_stride,
            context_channels=self.backbone.c5_channels,
            coord_dropout_prob=self.coord_dropout_prob,
            use_endpoint_prior=self.use_endpoint_prior,
            extra_cue_dim=cue_dim,
        )
        self.cov_head = CholeskyBivariateGnllHead(
            in_channels=self.backbone.c5_channels,
            num_keypoints=NUM_KEYPOINTS,
            extra_cue_dim=cue_dim,
        )
        self.runway_present_head: nn.Module | None = None
        if self.enable_runway_present_head:
            pres_in = self.backbone.out_channels + self.backbone.c5_channels
            self.runway_present_head = nn.Sequential(
                nn.Linear(pres_in, 48),
                nn.ReLU(inplace=False),
                nn.Linear(48, 1),
            )

    def stage2a_parameters(self) -> list[nn.Parameter]:
        """Parameters trained in Stage-2A (backbone, FPN, SimCC, keyline geometry, optional runway_present_head)."""
        params = (
            list(self.backbone.parameters())
            + list(self.simcc_head.parameters())
            + self.keyline_head.geometry_parameters()
        )
        if self.runway_present_head is not None:
            params.extend(self.runway_present_head.parameters())
        return params

    def stage2b_parameters(self) -> list[nn.Parameter]:
        """Frozen-mean uncertainty parameters trained in Stage-2B (cov_head + keyline sigma_head)."""
        return list(self.cov_head.parameters()) + self.keyline_head.sigma_parameters()

    def forward(self, roi_crop: torch.Tensor) -> dict[str, torch.Tensor]:
        fpn_feat, c5_feat = self.backbone(roi_crop)
        logits_x, logits_y = self.simcc_head(fpn_feat)
        keypoints_xy = decode_simcc(logits_x, logits_y, split_ratio=self.split_ratio)
        simcc_cues = (
            compute_simcc_uncertainty_cues(
                logits_x,
                logits_y,
                bins_x=self.simcc_head.bins_x,
                bins_y=self.simcc_head.bins_y,
            )
            if self.enable_simcc_uncertainty_cues
            else None
        )
        keylines_hesse, keyline_sigma_px, keyline_endpoints = self.keyline_head.forward_with_endpoints(
            fpn_feat, context_feat=c5_feat, simcc_cues=simcc_cues
        )
        chol_l, cov_2x2 = self.cov_head(c5_feat, extra_cues=simcc_cues)
        out: dict[str, torch.Tensor] = {
            "logits_x": logits_x,
            "logits_y": logits_y,
            "keypoints_xy": keypoints_xy,
            "keylines_hesse": keylines_hesse,
            "keyline_sigma_px": keyline_sigma_px,
            "keyline_endpoints": keyline_endpoints,
            "cholesky_l": chol_l,
            "cov_2x2": cov_2x2,
        }
        if self.runway_present_head is not None:
            fpn_gap = fpn_feat.float().mean(dim=(2, 3))
            c5_gap = c5_feat.float().mean(dim=(2, 3))
            out["runway_present_logit"] = self.runway_present_head(torch.cat([fpn_gap, c5_gap], dim=1))
        return out


class XvsLandV52OnnxWrapper(nn.Module):
    """Static Opset-17 flat-tuple ONNX export wrapper for `XvsLandV52Model` (6 outputs v5.2; 7 outputs extended)."""

    OUTPUT_NAMES: tuple[str, ...] = (
        "keypoints_xy",
        "keylines_hesse",
        "cov_2x2",
        "logits_x",
        "logits_y",
        "keyline_sigma_px",
    )
    OUTPUT_NAMES_V53: tuple[str, ...] = OUTPUT_NAMES + ("runway_present_logit",)

    def __init__(self, core: XvsLandV52Model) -> None:
        super().__init__()
        self.core = core

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.OUTPUT_NAMES_V53 if self.core.enable_runway_present_head else self.OUTPUT_NAMES

    def forward(
        self, roi_crop: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        out = self.core(roi_crop)
        base = (
            out["keypoints_xy"],
            out["keylines_hesse"],
            out["cov_2x2"],
            out["logits_x"],
            out["logits_y"],
            out["keyline_sigma_px"],
        )
        if self.core.enable_runway_present_head:
            return base + (out["runway_present_logit"],)
        return base
