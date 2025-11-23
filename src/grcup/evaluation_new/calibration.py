"""Calibration metrics for quantile predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_quantile_coverage(
    predictions: pd.DataFrame,
    actuals: pd.Series | np.ndarray,
    quantile: float = 0.9,
) -> float:
    """
    Compute quantile coverage (fraction of actuals within predicted quantile band).
    
    Args:
        predictions: DataFrame with columns 'q10', 'q50', 'q90' (or similar)
        actuals: Actual values
        quantile: Target quantile (0.9 = 90% coverage)
    
    Returns:
        Coverage fraction (0-1)
    """
    actuals = np.asarray(actuals)
    
    # Determine quantile columns
    if quantile == 0.9:
        q_low_col = "q10"
        q_high_col = "q90"
    elif quantile == 0.8:
        q_low_col = "q20"
        q_high_col = "q80"
    else:
        # Fallback: use q10/q90 for any quantile
        q_low_col = "q10"
        q_high_col = "q90"
    
    if q_low_col not in predictions.columns or q_high_col not in predictions.columns:
        # Fallback: use q50 ± some margin
        q50 = predictions.get("q50", predictions.iloc[:, 0] if len(predictions.columns) > 0 else pd.Series([0]))
        margin = q50 * 0.1  # 10% margin
        q_low = q50 - margin
        q_high = q50 + margin
    else:
        q_low = predictions[q_low_col].values
        q_high = predictions[q_high_col].values
    
    within_band = (actuals >= q_low) & (actuals <= q_high)
    return float(np.mean(within_band))


def check_quantile_calibration(
    coverage: float,
    target: float = 0.9,
    tolerance: float = 0.03,
) -> tuple[bool, str]:
    """
    Check if quantile predictions are calibrated.
    
    Args:
        coverage: Actual coverage fraction
        target: Target coverage (default 0.9 = 90%)
        tolerance: Acceptable deviation from target
    
    Returns:
        (is_calibrated, message)
    """
    deviation = abs(coverage - target)
    is_calibrated = deviation <= tolerance
    
    if is_calibrated:
        msg = f"Calibrated ({coverage:.1%} coverage, target: {target:.1%})"
    else:
        direction = "over" if coverage > target else "under"
        msg = f"Not calibrated ({coverage:.1%} coverage, target: {target:.1%}, {direction}-confident)"
    
    return is_calibrated, msg


def compute_brier_score(
    predictions: pd.DataFrame,
    actuals: pd.Series | np.ndarray,
    quantile: float = 0.9,
) -> float:
    """
    Compute Brier score for quantile predictions.
    
    Lower is better. Measures calibration quality.
    """
    actuals = np.asarray(actuals)
    coverage = compute_quantile_coverage(predictions, actuals, quantile)
    target = quantile
    
    # Brier score: squared difference between predicted and actual coverage
    brier = (coverage - target) ** 2
    return float(brier)


