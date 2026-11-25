"""OpenXVS v2.5/v3.0 Clean-Room Apache-2.0 Adverse-ODD Falsifier (`XVS-AST v2`, `DR-13`)."""

from xvs.v2.xvs_ast.falsifier import (
    CROISSANT_1_1_CONFORMS_TO,
    CROISSANT_RAI_CONFORMS_TO,
    QUARANTINED_DATASET_TOKENS,
    AstFalsificationReport,
    AstRolloutResult,
    DatasetQuarantineError,
    MetarOddScenario,
    XvsAstFalsifier,
    assert_dataset_not_quarantined,
    build_croissant_1_1_rai_metadata,
    compute_metar_log_prior,
    evaluate_odd_scenario,
    synthesize_odd_frame_input,
)

__all__ = [
    "AstFalsificationReport",
    "AstRolloutResult",
    "CROISSANT_1_1_CONFORMS_TO",
    "CROISSANT_RAI_CONFORMS_TO",
    "DatasetQuarantineError",
    "MetarOddScenario",
    "QUARANTINED_DATASET_TOKENS",
    "XvsAstFalsifier",
    "assert_dataset_not_quarantined",
    "build_croissant_1_1_rai_metadata",
    "compute_metar_log_prior",
    "evaluate_odd_scenario",
    "synthesize_odd_frame_input",
]
