"""Conformalized Quantile Regression (CQR) for guaranteed coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd


def conformalize_quantiles(
    q10_pred: np.ndarray | pd.Series,
    q90_pred: np.ndarray | pd.Series,
    y_actual: np.ndarray | pd.Series,
    alpha: float = 0.1,
) -> tuple[float, float]:
    """
    Compute conformal adjustment factors from calibration data.
    
    Args:
        q10_pred: Predicted 10th percentile
        q90_pred: Predicted 90th percentile
        y_actual: Actual values
        alpha: Target coverage (0.1 = 90% coverage)
    
    Returns:
        (adjustment_low, adjustment_high) - amounts to add/subtract from quantiles
    """
    q10_pred = np.asarray(q10_pred)
    q90_pred = np.asarray(q90_pred)
    y_actual = np.asarray(y_actual)
    
    # Nonconformity scores
    e_low = np.maximum(q10_pred - y_actual, 0.0)  # How much too low
    e_high = np.maximum(y_actual - q90_pred, 0.0)  # How much too high
    
    # Compute quantile of nonconformity scores
    n = len(y_actual)
    k = int(np.ceil((1 - alpha) * (n + 1))) - 1
    k = max(0, min(k, n - 1))  # Ensure valid index
    
    # Get k-th largest errors
    q_low = np.partition(e_low, k)[k] if len(e_low) > 0 else 0.0
    q_high = np.partition(e_high, k)[k] if len(e_high) > 0 else 0.0
    
    return float(q_low), float(q_high)


def apply_conformal_adjustment(
    q10_pred: np.ndarray | pd.Series,
    q90_pred: np.ndarray | pd.Series,
    adjustment_low: float,
    adjustment_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply conformal adjustments to quantile predictions.
    
    Args:
        q10_pred: Predicted 10th percentile
        q90_pred: Predicted 90th percentile
        adjustment_low: Adjustment to subtract from q10
        adjustment_high: Adjustment to add to q90
    
    Returns:
        (q10_adjusted, q90_adjusted)
    """
    q10_pred = np.asarray(q10_pred)
    q90_pred = np.asarray(q90_pred)
    
    q10_adj = q10_pred - adjustment_low
    q90_adj = q90_pred + adjustment_high
    
    # Guard against crossing
    mask_cross = q10_adj > q90_adj
    if np.any(mask_cross):
        midpoint = 0.5 * (q10_adj + q90_adj)
        q10_adj[mask_cross] = midpoint[mask_cross] - 1e-3
        q90_adj[mask_cross] = midpoint[mask_cross] + 1e-3
    
    return q10_adj, q90_adj


def compute_conformal_coverage(
    q10_pred: np.ndarray | pd.Series,
    q90_pred: np.ndarray | pd.Series,
    y_actual: np.ndarray | pd.Series,
) -> float:
    """
    Compute actual coverage of quantile band.
    
    Returns:
        Coverage fraction (0-1)
    """
    q10_pred = np.asarray(q10_pred)
    q90_pred = np.asarray(q90_pred)
    y_actual = np.asarray(y_actual)
    
    within_band = (y_actual >= q10_pred) & (y_actual <= q90_pred)
    return float(np.mean(within_band))

