"""XVS-VRAIM Offline Certification Audits: Mahalanobis OOD, Top-K SAE & EASA AST (DR-08 / SP2 §3.5)."""

from .ast_sampler import (
    AdaptiveStressTester,
    StressParameters,
    StressTrajectoryResult,
    build_easa_rlm_07_11_report,
)
from .mahalanobis_ood import (
    PenultimateMahalanobisOOD,
    compute_auroc,
)
from .sae_audit import (
    SaeAuditResult,
    TopKSparseAutoencoder,
    audit_content_vs_style_reliance,
)

__all__ = [
    "AdaptiveStressTester",
    "PenultimateMahalanobisOOD",
    "SaeAuditResult",
    "StressParameters",
    "StressTrajectoryResult",
    "TopKSparseAutoencoder",
    "audit_content_vs_style_reliance",
    "build_easa_rlm_07_11_report",
    "compute_auroc",
]
