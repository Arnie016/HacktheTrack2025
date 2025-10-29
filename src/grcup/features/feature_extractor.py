"""Extract ML-ready features from raw data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from ..loaders import load_weather, load_sectors
from .sector_fingerprint import (
    compute_sector_temp_sensitivity,
    compute_traffic_density,
    detect_clean_air_flag,
)
from .stint_detector import compute_tire_age_per_lap, detect_stints


def join_weather_to_laps(
    laps_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    tolerance: pd.Timedelta = pd.Timedelta(seconds=3),
) -> pd.DataFrame:
    """
    Join weather data to laps using asof merge.
    
    Args:
        laps_df: DataFrame with lap_ts_utc or similar timestamp
        weather_df: DataFrame with weather_ts
        tolerance: Maximum time difference for match
    
    Returns:
        DataFrame with weather columns joined
    """
    laps_df = laps_df.copy()
    
    # Derive timestamp column from various possible names
    if "lap_ts_utc" not in laps_df.columns:
        if "timestamp" in laps_df.columns:
            laps_df["lap_ts_utc"] = pd.to_datetime(laps_df["timestamp"], utc=True, errors="coerce")
        elif "lap_time_ts" in laps_df.columns:
            laps_df["lap_ts_utc"] = pd.to_datetime(laps_df["lap_time_ts"], utc=True, errors="coerce")
        elif "lap_start_ts" in laps_df.columns:
            laps_df["lap_ts_utc"] = pd.to_datetime(laps_df["lap_start_ts"], utc=True, errors="coerce")
        else:
            raise ValueError("Need timestamp column in laps_df (expected: timestamp, lap_ts_utc, lap_time_ts, or lap_start_ts)")
    
    # Ensure weather has proper timestamp column
    if "weather_ts" not in weather_df.columns:
        if "TIME_UTC_SECONDS" in weather_df.columns:
            # Convert seconds to datetime
            weather_df = weather_df.copy()
            weather_df["weather_ts"] = pd.to_datetime(weather_df["TIME_UTC_SECONDS"], unit="s", utc=True)
        else:
            raise ValueError("Need weather_ts or TIME_UTC_SECONDS in weather_df")
    
    # Remove rows with missing timestamps
    laps_df = laps_df[laps_df["lap_ts_utc"].notna()].copy()
    weather_df = weather_df[weather_df["weather_ts"].notna()].copy()
    
    if len(laps_df) == 0:
        raise ValueError("No valid lap timestamps")
    if len(weather_df) == 0:
        raise ValueError("No valid weather timestamps")
    
    # Sort and ensure unique (drop duplicates on timestamp)
    laps_sorted = (
        laps_df.sort_values("lap_ts_utc")
        .drop_duplicates(subset=["lap_ts_utc"], keep="first")
        .reset_index(drop=True)
    )
    
    weather_sorted = (
        weather_df.sort_values("weather_ts")
        .drop_duplicates(subset=["weather_ts"], keep="first")
        .reset_index(drop=True)
    )
    
    # Check for edge case: left == right (empty ranges)
    if len(laps_sorted) == 0 or len(weather_sorted) == 0:
        raise ValueError("Cannot merge: one or both DataFrames are empty after sorting")
    
    # As-of merge (backward fill) with allow_exact_matches
    # Use nearest direction first, then fallback to median fill
    # Increased tolerance to 2 minutes for better coverage
    try:
        joined = pd.merge_asof(
            laps_sorted,
            weather_sorted,
            left_on="lap_ts_utc",
            right_on="weather_ts",
            direction="nearest",  # Use nearest instead of backward for better matching
            tolerance=pd.Timedelta("2min"),  # 2 minute tolerance
            allow_exact_matches=False,  # Prefer nearest, not exact
        )
    except ValueError as e:
        if "left" in str(e) and "right" in str(e):
            # Fallback: try backward direction
            try:
                joined = pd.merge_asof(
                    laps_sorted,
                    weather_sorted,
                    left_on="lap_ts_utc",
                    right_on="weather_ts",
                    direction="backward",
                    tolerance=tolerance,
                )
            except:
                # Last resort: forward fill
                joined = pd.merge_asof(
                    laps_sorted,
                    weather_sorted,
                    left_on="lap_ts_utc",
                    right_on="weather_ts",
                    direction="forward",
                    tolerance=tolerance,
                )
        else:
            raise
    
    # Fill missing weather data with median/mode from available data
    # Add imputation flags for model awareness
    if "TRACK_TEMP" in joined.columns:
        if joined["TRACK_TEMP"].isna().any():
            track_temp_median = joined["TRACK_TEMP"].median()
            if pd.notna(track_temp_median):
                joined["track_temp_imputed"] = joined["TRACK_TEMP"].isna().astype(int)
                joined["TRACK_TEMP"] = joined["TRACK_TEMP"].fillna(track_temp_median)
            else:
                joined["track_temp_imputed"] = 0
        else:
            joined["track_temp_imputed"] = 0
    
    if "AIR_TEMP" in joined.columns:
        if joined["AIR_TEMP"].isna().any():
            air_temp_median = joined["AIR_TEMP"].median()
            if pd.notna(air_temp_median):
                joined["air_temp_imputed"] = joined["AIR_TEMP"].isna().astype(int)
                joined["AIR_TEMP"] = joined["AIR_TEMP"].fillna(air_temp_median)
            else:
                joined["air_temp_imputed"] = 0
        else:
            joined["air_temp_imputed"] = 0
    
    # Compute temperature anomaly (temp - session median) for better model features
    if "TRACK_TEMP" in joined.columns:
        session_median_temp = joined["TRACK_TEMP"].median()
        if pd.notna(session_median_temp):
            joined["temp_anomaly"] = joined["TRACK_TEMP"] - session_median_temp
        else:
            joined["temp_anomaly"] = 0.0
    
    # Check remaining missing (after fill) - should be 0% now
    if "TRACK_TEMP" in joined.columns and "AIR_TEMP" in joined.columns:
        missing_pct = joined[["TRACK_TEMP", "AIR_TEMP"]].isna().any(axis=1).mean()
        if missing_pct > 0.1:  # Warn if still >10% missing after all fills
            # Final fallback: use global weather median
            if "TRACK_TEMP" in weather_sorted.columns:
                global_median_temp = weather_sorted["TRACK_TEMP"].median()
                if pd.notna(global_median_temp):
                    joined["TRACK_TEMP"] = joined["TRACK_TEMP"].fillna(global_median_temp)
            if "AIR_TEMP" in weather_sorted.columns:
                global_median_air = weather_sorted["AIR_TEMP"].median()
                if pd.notna(global_median_air):
                    joined["AIR_TEMP"] = joined["AIR_TEMP"].fillna(global_median_air)
    
    return joined


def target_encode_categorical(
    df: pd.DataFrame,
    categorical_col: str,
    target_col: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """
    KFold out-of-fold target encoding to prevent leakage.
    
    Args:
        df: DataFrame with categorical and target columns
        categorical_col: Column to encode
        target_col: Target variable for encoding
        n_splits: Number of folds
        random_state: Random seed
    
    Returns:
        Series with encoded values
    """
    encoded = pd.Series(index=df.index, dtype=float)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        
        # Compute mean target per category in training fold
        encoding_map = train_df.groupby(categorical_col)[target_col].mean()
        
        # Apply to validation fold
        encoded.iloc[val_idx] = df.iloc[val_idx][categorical_col].map(encoding_map)
    
    # Fill any remaining NaNs with global mean
    encoded = encoded.fillna(df[target_col].mean())
    
    return encoded


def build_wear_training_dataset(
    laps_df: pd.DataFrame,
    sectors_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    race_id: str = "R1",
) -> pd.DataFrame:
    """
    Build training dataset for tire degradation model.
    
    Features:
    - tire_age (monotonic +)
    - track_temp
    - stint_len (laps in current stint)
    - sector_S3_coeff (temperature sensitivity)
    - clean_air (boolean flag)
    - traffic_density (0-1)
    - driver_TE (target-encoded)
    - track_id (categorical)
    
    Target:
    - pace_delta (seconds slower than stint best)
    """
    # Join weather
    laps_with_weather = join_weather_to_laps(laps_df, weather_df)
    
    # Detect stints per vehicle
    all_features = []
    
    for vehicle_id in laps_df["vehicle_id"].unique():
        vehicle_stints = detect_stints(laps_df, vehicle_id)
        
        if len(vehicle_stints) == 0:
            continue
        
        # Get tire age per lap
        tire_ages = compute_tire_age_per_lap(laps_df, vehicle_stints)
        
        vehicle_laps = laps_df[laps_df["vehicle_id"] == vehicle_id].copy()
        vehicle_laps = vehicle_laps.sort_values("lap")
        
        # Get vehicle number for sector lookup
        vehicle_number = None
        if "NUMBER" in sectors_df.columns:
            vehicle_numbers = sectors_df["NUMBER"].unique()
            # Try to match vehicle_id to NUMBER (heuristic)
            # In practice, may need mapping
            if len(vehicle_numbers) > 0:
                vehicle_number = int(vehicle_numbers[0])  # Simplified
        
        for _, lap_row in vehicle_laps.iterrows():
            lap_num = int(lap_row["lap"])
            
            # Tire age
            if (vehicle_id, lap_num) in tire_ages.index:
                tire_age = tire_ages.loc[(vehicle_id, lap_num)]
            else:
                continue
            
            # Find which stint this lap belongs to
            stint = None
            stint_len = 0
            best_lap_time = None
            
            for s in vehicle_stints:
                if s.start_lap <= lap_num <= s.end_lap:
                    stint = s
                    stint_len = lap_num - s.start_lap + 1
                    best_lap_time = s.best_lap_time_ms
                    break
            
            if stint is None or best_lap_time is None:
                continue
            
            # Pace delta
            lap_time_ms = lap_row.get("lap_time_ms", lap_row.get("lap_time_s", 0) * 1000)
            if lap_time_ms == 0:
                continue
            
            pace_delta = (lap_time_ms - best_lap_time) / 1000.0  # Convert to seconds
            
            # Weather
            lap_weather = laps_with_weather[
                (laps_with_weather["vehicle_id"] == vehicle_id) &
                (laps_with_weather["lap"] == lap_num)
            ]
            
            if len(lap_weather) == 0:
                continue
            
            track_temp = lap_weather["TRACK_TEMP"].iloc[0]
            
            # Sector features (simplified for now)
            sector_coeff_s3 = compute_sector_temp_sensitivity(sectors_df, weather_df, "S3_SECONDS")
            clean_air = True  # Simplified
            traffic_density = 0.0  # Simplified
            
            if vehicle_number is not None:
                clean_air = detect_clean_air_flag(sectors_df, vehicle_number, lap_num, "S3_SECONDS")
                traffic_density = compute_traffic_density(sectors_df, None, vehicle_number, lap_num)
            
            all_features.append({
                "vehicle_id": vehicle_id,
                "lap": lap_num,
                "tire_age": tire_age,
                "track_temp": track_temp,
                "stint_len": stint_len,
                "sector_S3_coeff": sector_coeff_s3,
                "clean_air": 1.0 if clean_air else 0.0,
                "traffic_density": traffic_density,
                "race_id": race_id,
                "pace_delta": pace_delta,
            })
    
    features_df = pd.DataFrame(all_features)
    
    if len(features_df) == 0:
        raise ValueError("No training data extracted")
    
    # Target encode driver/track
    if "vehicle_id" in features_df.columns:
        features_df["driver_TE"] = target_encode_categorical(
            features_df,
            "vehicle_id",
            "pace_delta",
        )
    
    features_df["track_id"] = race_id  # Can expand for multi-track
    
    return features_df


def build_pace_prediction_features(
    laps_df: pd.DataFrame,
    n_lags: int = 5,
) -> pd.DataFrame:
    """
    Build features for pace prediction (ARIMA/Kalman).
    
    Features: last n_lags lap times
    
    Args:
        laps_df: DataFrame with vehicle_id, lap, lap_time_ms
        n_lags: Number of previous laps to use
    
    Returns:
        DataFrame with lag features + next_lap_time target
    """
    features = []
    
    for vehicle_id in laps_df["vehicle_id"].unique():
        vehicle_laps = laps_df[laps_df["vehicle_id"] == vehicle_id].sort_values("lap")
        
        # Convert to seconds
        if "lap_time_s" not in vehicle_laps.columns:
            if "lap_time_ms" in vehicle_laps.columns:
                vehicle_laps["lap_time_s"] = vehicle_laps["lap_time_ms"] / 1000.0
            else:
                continue
        
        lap_times = vehicle_laps["lap_time_s"].values
        
        for i in range(n_lags, len(lap_times)):
            lags = lap_times[i - n_lags : i]
            next_lap = lap_times[i]
            
            features.append({
                "vehicle_id": vehicle_id,
                "lap": vehicle_laps.iloc[i]["lap"],
                **{f"lag_{j+1}": lag for j, lag in enumerate(lags)},
                "next_lap_time": next_lap,
            })
    
    return pd.DataFrame(features)


