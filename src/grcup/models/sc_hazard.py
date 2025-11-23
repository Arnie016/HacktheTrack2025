"""Cox Proportional Hazards model for safety car probability."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def prepare_hazard_data(
    sectors_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare data for Cox PH model.
    
    Features:
    - green_run_len (laps since last SC)
    - pack_density (cars within proximity)
    - rain (0/1)
    - wind_speed
    - incidents_proxy (from flag data)
    
    Duration: time to next SC (right-censored if race ends)
    Event: SC occurred (1) or not (0)
    """
    # Extract green-flag run lengths from FLAG_AT_FL
    if "FLAG_AT_FL" not in sectors_df.columns:
        raise ValueError("Need FLAG_AT_FL column for SC detection")
    
    hazard_data = []
    
    # Group by lap to compute pack density
    for lap_num in sectors_df["LAP_NUMBER"].unique():
        lap_data = sectors_df[sectors_df["LAP_NUMBER"] == lap_num]
        
        # Detect SC flag
        sc_flag = (lap_data["FLAG_AT_FL"] == "FCY").any()
        
        # Green run length (laps since last SC)
        prev_laps = sectors_df[sectors_df["LAP_NUMBER"] < lap_num]
        last_sc_lap = prev_laps[prev_laps["FLAG_AT_FL"] == "FCY"]["LAP_NUMBER"]
        green_run_len = lap_num - last_sc_lap.max() if len(last_sc_lap) > 0 else lap_num
        
        # Pack density (simplified: number of cars in this lap)
        pack_density = len(lap_data) / 20.0  # Normalize
        
        # Weather (use latest available)
        if len(weather_df) > 0:
            rain = weather_df.iloc[-1]["RAIN"] if "RAIN" in weather_df.columns else 0
            wind_speed = weather_df.iloc[-1]["WIND_SPEED"] if "WIND_SPEED" in weather_df.columns else 0
        else:
            rain = 0
            wind_speed = 0
        
        hazard_data.append({
            "lap": lap_num,
            "green_run_len": green_run_len,
            "pack_density": pack_density,
            "rain": rain,
            "wind_speed": wind_speed,
            "event": 1 if sc_flag else 0,
        })
    
    return pd.DataFrame(hazard_data)


def train_cox_hazard(
    hazard_df: pd.DataFrame,
    penalizer: float = 0.1,
) -> CoxPHFitter:
    """
    Train Cox Proportional Hazards model.
    
    Args:
        hazard_df: DataFrame with features + event column
        penalizer: Ridge regularization penalty (default 0.1)
    
    Returns:
        Fitted CoxPHFitter model
    """
    if len(hazard_df) == 0:
        raise ValueError("No hazard data")
    
    # CoxPHFitter expects duration and event columns
    # Duration = green_run_len (time to event)
    # Event = 1 if SC occurred
    
    # Features for model
    features = ["green_run_len", "pack_density", "rain", "wind_speed"]
    
    # Prepare data
    cox_data = hazard_df[features + ["event"]].copy()
    cox_data["duration"] = hazard_df["green_run_len"]
    
    # Remove NaNs
    cox_data = cox_data.dropna()
    
    if len(cox_data) == 0:
        raise ValueError("No valid data after removing NaNs")
    
    # Check for collinearity and drop redundant columns
    corr = cox_data[features].corr().abs()
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] > 0.9:
                high_corr_pairs.append((corr.columns[i], corr.columns[j]))
    
    # Drop columns with high correlation (keep the first one)
    cols_to_drop = set()
    for col1, col2 in high_corr_pairs:
        # Prefer keeping green_run_len and pack_density (most informative)
        if col1 not in ["green_run_len", "pack_density"]:
            cols_to_drop.add(col1)
        elif col2 not in ["green_run_len", "pack_density"]:
            cols_to_drop.add(col2)
        else:
            cols_to_drop.add(col2)  # Drop the second if both are important
    
    features_clean = [f for f in features if f not in cols_to_drop]
    
    # Drop low variance columns (can cause convergence issues)
    for feat in features_clean:
        if cox_data[feat].var() < 1e-6:
            cols_to_drop.add(feat)
    
    features_final = [f for f in features if f not in cols_to_drop]
    
    if len(features_final) == 0:
        # Fallback to just green_run_len if everything is dropped
        features_final = ["green_run_len"]
    
    # Create model with regularization
    model = CoxPHFitter(penalizer=penalizer)
    
    # Fit model with cleaned features
    cox_fit_data = cox_data[features_final + ["duration", "event"]].copy()
    
    model.fit(cox_fit_data, duration_col="duration", event_col="event")
    
    return model


def predict_sc_probability(
    model: CoxPHFitter,
    green_run_len: float,
    pack_density: float,
    rain: int,
    wind_speed: float,
    k_laps: int = 3,
) -> float:
    """
    Predict probability of SC in next k_laps.
    
    Args:
        model: Fitted Cox model
        green_run_len: Current green-flag run length
        pack_density: Current pack density
        rain: Rain flag (0/1)
        wind_speed: Wind speed
        k_laps: Number of laps ahead to predict
    
    Returns:
        Probability of SC in next k_laps (0-1)
    """
    # Cache identical queries because this function is called millions of
    # times inside Monte Carlo loops.
    cache = getattr(model, "_prob_cache", None)
    if cache is None:
        cache = {}
        setattr(model, "_prob_cache", cache)
    cache_key = (
        round(float(green_run_len), 3),
        round(float(pack_density), 3),
        int(rain),
        round(float(wind_speed), 3),
        int(k_laps),
    )
    if cache_key in cache:
        return cache[cache_key]

    # Prepare input
    input_df = pd.DataFrame([{
        "green_run_len": green_run_len,
        "pack_density": pack_density,
        "rain": rain,
        "wind_speed": wind_speed,
    }])
    
    # Predict survival probability
    survival_prob = model.predict_survival_function(input_df, times=[k_laps])
    
    if len(survival_prob) == 0:
        return 0.1  # Default low probability
    
    # SC probability = 1 - survival probability
    sc_prob = 1.0 - float(survival_prob.iloc[0, 0])
    
    sc_prob = max(0.0, min(1.0, sc_prob))  # Clamp to [0, 1]
    cache[cache_key] = sc_prob
    
    return sc_prob


def save_hazard_model(model: CoxPHFitter, path: Path | str):
    """Save Cox model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_hazard_model(path: Path | str) -> CoxPHFitter:
    """Load Cox model from disk."""
    path = Path(path)
    
    with open(path, "rb") as f:
        return pickle.load(f)

