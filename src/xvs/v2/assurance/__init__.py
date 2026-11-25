"""OpenXVS v5.0 additive assurance traceability & SPDX 2.3 SBOM utilities.

Strictly additive alongside frozen `assurance/openxvs_v2.gsn.yaml` and `scripts/xvs/xvs_certgen.py`.
"""

from .sora25_f3442_matrix import (
    EASA_SORA_25_MITIGATIONS,
    EASA_SORA_25_OSOS,
    ASTM_F3442_26_REQUIREMENTS,
    ED324_OBJECTIVES,
    build_sora25_f3442_ed324_matrix,
    build_spdx_2_3_sbom,
    evaluate_spdx_expression_for_copyleft,
)

__all__ = [
    "ASTM_F3442_26_REQUIREMENTS",
    "EASA_SORA_25_MITIGATIONS",
    "EASA_SORA_25_OSOS",
    "ED324_OBJECTIVES",
    "build_sora25_f3442_ed324_matrix",
    "build_spdx_2_3_sbom",
    "evaluate_spdx_expression_for_copyleft",
]
