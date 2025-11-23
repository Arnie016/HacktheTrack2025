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
    hyperparams: dict | None = None,
) -> dict:
    """
    Train XGBoost quantile regression model for tire wear.
    
    Args:
        features_df: DataFrame with features and pace_delta target
        quantiles: Quantiles to predict [q10, q50, q90]
        monotone_tire_age: Monotonic constraint on tire_age (+1 = increasing)
        hyperparams: Optional dict of XGBoost hyperparameters to override defaults
    
    Returns:
        Dict with models, scaler, feature_names
    """
    # Feature columns (tire_age must be first for monotonic constraint)
    required_base = [
        "tire_age",
        "track_temp",
        "temp_anomaly",
        "stint_len",
        "sector_S3_coeff",
        "clean_air",
        "traffic_density",
    ]
    missing_required = [c for c in required_base if c not in features_df.columns]
    if missing_required:
        raise ValueError(f"Missing required features: {missing_required}")
    
    base_features = [c for c in required_base]
    
    # Add interaction features if available
    interaction_features = [
        "tire_temp_interaction",
        "tire_clean_interaction",
        "traffic_temp_interaction",
    ]
    
    # Weather + race-context features (optional, only if present)
    weather_context = [
        "air_temp_c",
        "humidity_pct",
        "wind_speed_kph",
        "rain_intensity",
        "pressure_hpa",
        "wind_direction_deg",
    ]
    race_context = [
        "gap_to_leader_s",
        "gap_to_car_ahead_s",
        "lap_position",
        "position_fraction",
        "flag_state_code",
        "damage_indicator",
        "top_speed",
        "pit_time_s",
        "lap_time_s",
        "s1_seconds",
        "s2_seconds",
        "s3_seconds",
        "final_position",
        "laps_completed",
        "status_classified",
        "class_code",
        "division_code",
    ]
    
    feature_cols = (
        base_features
        + [f for f in interaction_features if f in features_df.columns]
        + [f for f in weather_context if f in features_df.columns]
        + [f for f in race_context if f in features_df.columns]
    )
    
    # Add driver_TE if available
    if "driver_TE" in features_df.columns:
        feature_cols.append("driver_TE")
    
    # Add physics features (telemetry-based wear indicators)
    physics_features = [
        "accel_mag_mean", "accel_mag_max", "accel_mag_std", "accel_mag_p95",
        "jerk_mean", "jerk_max", "jerk_std", "jerk_p95", "jerk_rms",
        "accx_can_mean_abs", "accx_can_max", "accx_can_std",
        "accy_can_mean_abs", "accy_can_max", "accy_can_std",
        "throttle_mean", "throttle_std", "throttle_p95",
        "combined_aggression",
        "avg_long_accel", "avg_lat_accel", "avg_jerk"  # Alternative names
    ]
    feature_cols += [f for f in physics_features if f in features_df.columns]
    
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
    
    # Default params - use GPU if available (XGBoost 2.0+ API)
    import os
    use_gpu = os.getenv("USE_GPU", "0") == "1"
    
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "tree_method": "hist",  # Always use hist, set device separately
        "device": "cuda" if use_gpu else "cpu",  # New XGBoost 2.0+ API
        "max_bin": 256,
        "monotone_constraints": monotone_constraints,
        "random_state": 42,
    }
    
    # Override with provided hyperparams
    if hyperparams:
        params.update(hyperparams)
    
    for q in quantiles:
        model = xgb.XGBRegressor(
            objective=f"reg:quantileerror",
            quantile_alpha=q,
            **params
        )
        
        model.fit(X_scaled, y)
        models[f"q{int(q*100)}"] = model
    
    feature_defaults = {}
    for col in feature_cols:
        if col in features_df.columns:
            median_val = float(features_df[col].median())
            if np.isnan(median_val):
                median_val = 0.0
        else:
            median_val = 0.0
        feature_defaults[col] = median_val
    
    return {
        "models": models,
        "scaler": scaler,
        "feature_names": feature_cols,
        "quantiles": quantiles,
        "feature_defaults": feature_defaults,
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


def build_feature_vector(model_data: dict, overrides: dict[str, float]) -> dict[str, float]:
    """
    Assemble a feature vector that matches the model's training schema.
    
    Args:
        model_data: Wear model dictionary with feature_names/defaults.
        overrides: Dict of feature_name -> value for known runtime context.
    
    Returns:
        Dict ready to be wrapped in a DataFrame for predict_quantiles.
    """
    feature_names = model_data.get("feature_names", [])
    defaults = model_data.get("feature_defaults", {})
    
    feature_row = {}
    for name in feature_names:
        if name in overrides and overrides[name] is not None:
            feature_row[name] = overrides[name]
        elif name in defaults:
            feature_row[name] = defaults[name]
        else:
            feature_row[name] = 0.0
    return feature_row


