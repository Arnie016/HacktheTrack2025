"""Calibration metrics: quantile coverage, Brier score."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def compute_quantile_coverage(
    predictions: pd.DataFrame,  # Columns: q10, q50, q90
    actuals: pd.Series,
    quantile: float = 0.9,
) -> float:
    """
    Compute quantile coverage.
    
    For 90% quantile, check if 90% of actuals fall within [q10, q90] band.
    
    Args:
        predictions: DataFrame with quantile columns
        actuals: Actual values (must align with predictions index)
        quantile: Target quantile (0.9 = 90%)
    
    Returns:
        Coverage fraction (should be close to quantile)
    """
    if "q10" not in predictions.columns or "q90" not in predictions.columns:
        raise ValueError("Need q10 and q90 columns")
    
    # Ensure indices align - reset both to RangeIndex for safe comparison
    predictions_aligned = predictions.reset_index(drop=True).copy()
    actuals_aligned = actuals.reset_index(drop=True).copy()
    
    # Ensure same length
    min_len = min(len(predictions_aligned), len(actuals_aligned))
    if min_len == 0:
        raise ValueError("Empty predictions or actuals")
    
    predictions_aligned = predictions_aligned.iloc[:min_len]
    actuals_aligned = actuals_aligned.iloc[:min_len]
    
    # Ensure quantile ordering (q10 <= q50 <= q90) - no crossing
    q_values = predictions_aligned[["q10", "q50", "q90"]].values
    q_values.sort(axis=1)  # Sort each row
    predictions_aligned[["q10", "q50", "q90"]] = q_values
    
    # Check if actuals fall within prediction band
    within_band = (
        (actuals_aligned >= predictions_aligned["q10"]) &
        (actuals_aligned <= predictions_aligned["q90"])
    )
    
    coverage = within_band.mean()
    
    return float(coverage)


def compute_brier_score(
    predicted_probs: pd.Series,  # Predicted probabilities
    actuals: pd.Series,  # Binary outcomes (0/1)
) -> float:
    """
    Compute Brier score for probability predictions.
    
    Lower is better (0 = perfect, 1 = worst).
    """
    return float(brier_score_loss(actuals, predicted_probs))


def compute_calibration_curve_data(
    predicted_probs: pd.Series,
    actuals: pd.Series,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration curve data.
    
    Returns:
        Dict with fraction_of_positives, mean_predicted_value
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        actuals,
        predicted_probs,
        n_bins=n_bins,
    )
    
    return {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
    }


def check_quantile_calibration(
    coverage: float,
    target_quantile: float = 0.9,
    tolerance: float = 0.05,
) -> tuple[bool, str]:
    """
    Check if quantile coverage is calibrated.
    
    Args:
        coverage: Actual coverage (e.g., 0.88)
        target_quantile: Target quantile (e.g., 0.9)
        tolerance: Allowed deviation (e.g., 0.05)
    
    Returns:
        (is_calibrated, message)
    """
    deviation = abs(coverage - target_quantile)
    is_calibrated = deviation <= tolerance
    
    status = "✓" if is_calibrated else "✗"
    message = f"{status} Coverage: {coverage:.2%} (target: {target_quantile:.0%}, deviation: {deviation:.2%})"
    
    return is_calibrated, message


