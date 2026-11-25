"""XVS-ThermalNet: Fault-tolerant asymmetric EO + 2-channel LWIR fusion (DR-04 / SP2 §3.3)."""

from .curriculum import (
    CURRICULUM_PROBS,
    CurriculumSample,
    FaultCurriculumSampler,
    summarize_curriculum_distribution,
)
from .datasets import (
    ALLOWED_THERMAL_TRAIN_DATASETS,
    BANNED_THERMAL_DATASETS,
    EVAL_ONLY_THERMAL_DATASETS,
    HitUavDataset,
    Irstd1kDataset,
    ThermalSample,
    assert_thermal_dataset_allowed,
)
from .model import (
    DeterministicHealthGate,
    EvidentialDirichletHead,
    ThermalSimplexMode,
    XvsThermalNet,
    compute_parallax_vacuity_loss,
    compute_thermal_multitask_loss,
    dirichlet_kl_to_uniform,
    select_thermal_simplex_mode,
)

__all__ = [
    "ALLOWED_THERMAL_TRAIN_DATASETS",
    "BANNED_THERMAL_DATASETS",
    "CURRICULUM_PROBS",
    "CurriculumSample",
    "DeterministicHealthGate",
    "EVAL_ONLY_THERMAL_DATASETS",
    "EvidentialDirichletHead",
    "FaultCurriculumSampler",
    "HitUavDataset",
    "Irstd1kDataset",
    "ThermalSample",
    "ThermalSimplexMode",
    "XvsThermalNet",
    "assert_thermal_dataset_allowed",
    "compute_parallax_vacuity_loss",
    "compute_thermal_multitask_loss",
    "dirichlet_kl_to_uniform",
    "select_thermal_simplex_mode",
    "summarize_curriculum_distribution",
]
