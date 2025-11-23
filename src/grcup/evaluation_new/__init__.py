"""Evaluation utilities for GR Cup strategy validation."""
from .walkforward import walkforward_validate, save_walkforward_results
from .conformal import apply_conformal_adjustment
from .calibration import (
    compute_quantile_coverage,
    check_quantile_calibration,
    compute_brier_score,
)
from .ablations import (
    run_ablations,
    save_ablation_report,
    compare_to_baseline,
)

__all__ = [
    "walkforward_validate",
    "save_walkforward_results",
    "apply_conformal_adjustment",
    "compute_quantile_coverage",
    "check_quantile_calibration",
    "compute_brier_score",
    "run_ablations",
    "save_ablation_report",
    "compare_to_baseline",
]

