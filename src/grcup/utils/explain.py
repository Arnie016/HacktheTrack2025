"""Explainability utilities: SHAP, feature importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor


def compute_shap_values(
    model: XGBRegressor,
    X: np.ndarray,
    feature_names: list[str],
    n_samples: int = 100,
) -> pd.DataFrame:
    """
    Compute SHAP values for feature importance.
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix
        feature_names: Names of features
        n_samples: Number of samples for SHAP (can downsample for speed)
    
    Returns:
        DataFrame with SHAP values per feature
    """
    # Downsample if needed
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Average absolute SHAP values per feature
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(shap_values).mean(axis=0),
    }).sort_values("importance", ascending=False)
    
    return importance


def get_top_reasons(
    shap_importance: pd.DataFrame,
    n: int = 3,
) -> list[str]:
    """
    Get top N features as "reasons" for recommendation.
    
    Args:
        shap_importance: DataFrame from compute_shap_values
        n: Number of top features
    
    Returns:
        List of reason strings
    """
    top_features = shap_importance.head(n)
    
    reasons = []
    for _, row in top_features.iterrows():
        feature = row["feature"]
        importance = row["importance"]
        reasons.append(f"{feature}: {importance:.3f}")
    
    return reasons


