"""XGBoost quantile regression for tire degradation with monotonic constraints."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from ..data.schemas import Stint


def train_wear_quantile_model(
    features_df: pd.DataFrame,
    quantiles: list[float] = [0.1, 0.5, 0.9],
    monotone_tire_age: int = 1,  # +1 = increasing constraint
) -> dict:
    """
    Train XGBoost quantile regression model for tire wear.
    
    Args:
        features_df: DataFrame with features and pace_delta target
        quantiles: Quantiles to predict [q10, q50, q90]
        monotone_tire_age: Monotonic constraint on tire_age (+1 = increasing)
    
    Returns:
        Dict with models, scaler, feature_names
    """
    # Feature columns (tire_age must be first for monotonic constraint)
    # Include new interaction features for better model performance
    base_features = [
        "tire_age",  # First = index 0 for monotonic constraint
        "track_temp",
        "temp_anomaly",
        "stint_len",
        "sector_S3_coeff",
        "clean_air",
        "traffic_density",
    ]
    
    # Add interaction features if available
    interaction_features = [
        "tire_temp_interaction",
        "tire_clean_interaction",
        "traffic_temp_interaction",
    ]
    
    feature_cols = base_features + [f for f in interaction_features if f in features_df.columns]
    
    # Add driver_TE if available
    if "driver_TE" in features_df.columns:
        feature_cols.append("driver_TE")
    
    # Check all features exist
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    X = features_df[feature_cols].values
    y = features_df["pace_delta"].values
    
    # Remove NaNs
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        raise ValueError("No valid training data after removing NaNs")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Build monotonic constraints string
    # tire_age (index 0) = +1 (increasing)
    # others = 0 (no constraint)
    monotone_constraints = tuple([monotone_tire_age] + [0] * (len(feature_cols) - 1))
    
    # Train separate model for each quantile
    models = {}
    
    for q in quantiles:
        model = xgb.XGBRegressor(
            objective=f"reg:quantileerror",
            quantile_alpha=q,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            tree_method='hist',
            max_bin=256,
            monotone_constraints=monotone_constraints,
            random_state=42,
        )
        
        model.fit(X_scaled, y)
        models[f"q{int(q*100)}"] = model
    
    return {
        "models": models,
        "scaler": scaler,
        "feature_names": feature_cols,
        "quantiles": quantiles,
    }


def predict_quantiles(
    model_data: dict,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict quantiles for wear model.
    
    Returns:
        DataFrame with columns q10, q50, q90
    """
    feature_cols = model_data["feature_names"]
    scaler = model_data["scaler"]
    models = model_data["models"]
    
    X = features_df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    predictions = {}
    for q_name, model in models.items():
        pred = model.predict(X_scaled)
        predictions[q_name] = pred
    
    pred_df = pd.DataFrame(predictions)
    
    # Ensure quantile ordering (q10 <= q50 <= q90) - no crossing
    # Sort each row to enforce ordering
    q_values = pred_df[["q10", "q50", "q90"]].values
    q_values.sort(axis=1)  # Sort each row ascending
    pred_df[["q10", "q50", "q90"]] = q_values
    
    return pred_df


def save_model(model_data: dict, path: Path | str):
    """Save trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(model_data, f)


def load_model(path: Path | str) -> dict:
    """Load trained model from disk."""
    path = Path(path)
    
    with open(path, "rb") as f:
        return pickle.load(f)


