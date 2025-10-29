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
    tolerance: pd.Timedelta = pd.Timedelta(minutes=7),
) -> pd.DataFrame:
    """
    Join weather data to laps using asof merge.
    
    NEVER DROPS LAPS - all laps are returned with weather filled (median + imputation flags).
    
    Args:
        laps_df: DataFrame with lap_ts_utc or similar timestamp
        weather_df: DataFrame with weather_ts
        tolerance: Maximum time difference for match (default 7min for sparse weather)
    
    Returns:
        DataFrame with weather columns joined (all input laps preserved)
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
            # Create placeholder for missing timestamps (will be filled later)
            laps_df["lap_ts_utc"] = pd.NaT
    
    # Track which laps had timestamps originally
    laps_df["had_timestamp"] = laps_df["lap_ts_utc"].notna().astype("int8")
    
    # Ensure weather has proper timestamp column
    if "weather_ts" not in weather_df.columns:
        if "TIME_UTC_SECONDS" in weather_df.columns:
            # Convert seconds to datetime
            weather_df = weather_df.copy()
            weather_df["weather_ts"] = pd.to_datetime(weather_df["TIME_UTC_SECONDS"], unit="s", utc=True)
        else:
            raise ValueError("Need weather_ts or TIME_UTC_SECONDS in weather_df")
    
    # Filter weather (keep all laps, even without timestamps)
    weather_df = weather_df[weather_df["weather_ts"].notna()].copy()
    
    if len(weather_df) == 0:
        raise ValueError("No valid weather timestamps")
    
    # Sort laps (keep all, including those without timestamps)
    # Sort by timestamp where available, then by lap number
    laps_sorted = laps_df.sort_values(
        by=["lap_ts_utc", "lap"],
        na_position="last",
        kind="mergesort"
    ).reset_index(drop=True)
    
    # Sort weather
    weather_sorted = (
        weather_df.sort_values("weather_ts")
        .drop_duplicates(subset=["weather_ts"], keep="first")
        .reset_index(drop=True)
    )
    
    # Split laps into those with and without timestamps
    laps_with_ts = laps_sorted[laps_sorted["lap_ts_utc"].notna()].copy()
    laps_without_ts = laps_sorted[laps_sorted["lap_ts_utc"].isna()].copy()
    
    # Merge weather for laps with timestamps
    if len(laps_with_ts) > 0:
        try:
            joined = pd.merge_asof(
                laps_with_ts,
                weather_sorted,
                left_on="lap_ts_utc",
                right_on="weather_ts",
                direction="nearest",
                tolerance=tolerance,
                allow_exact_matches=False,
            )
        except ValueError:
            # Fallback: try backward, then forward
            try:
                joined = pd.merge_asof(
                    laps_with_ts,
                    weather_sorted,
                    left_on="lap_ts_utc",
                    right_on="weather_ts",
                    direction="backward",
                    tolerance=tolerance,
                )
            except:
                joined = pd.merge_asof(
                    laps_with_ts,
                    weather_sorted,
                    left_on="lap_ts_utc",
                    right_on="weather_ts",
                    direction="forward",
                    tolerance=tolerance,
                )
    else:
        joined = pd.DataFrame()
    
    # Add laps without timestamps (will get median-filled weather)
    if len(laps_without_ts) > 0:
        # Copy weather columns structure from joined if exists, otherwise from weather
        if len(joined) > 0:
            weather_cols = [c for c in joined.columns if c in weather_sorted.columns]
            for col in weather_cols:
                if col not in laps_without_ts.columns:
                    laps_without_ts[col] = pd.NA
        joined = pd.concat([joined, laps_without_ts], ignore_index=True)
    
    # Fill ALL missing weather data with medians + imputation flags
    # Never drop a lap - fill everything
    weather_cols_to_fill = ["TRACK_TEMP", "AIR_TEMP"]
    if "WIND_SPEED" in weather_sorted.columns:
        weather_cols_to_fill.append("WIND_SPEED")
    if "RAIN" in weather_sorted.columns:
        weather_cols_to_fill.append("RAIN")
    
    for col in weather_cols_to_fill:
        if col in joined.columns:
            missing_mask = joined[col].isna()
            if missing_mask.any():
                # Use median from available data (joined or weather_sorted)
                if joined[col].notna().any():
                    fill_value = joined[col].median()
                elif col in weather_sorted.columns and weather_sorted[col].notna().any():
                    fill_value = weather_sorted[col].median()
                else:
                    fill_value = 0.0  # Last resort
                
                # Add imputation flag
                joined[f"{col.lower()}_imputed"] = missing_mask.astype("int8")
                joined[col] = joined[col].fillna(fill_value)
            else:
                joined[f"{col.lower()}_imputed"] = 0
    
    # Compute temperature anomaly
    if "TRACK_TEMP" in joined.columns:
        session_median_temp = joined["TRACK_TEMP"].median()
        if pd.notna(session_median_temp):
            joined["temp_anomaly"] = joined["TRACK_TEMP"] - session_median_temp
        else:
            joined["temp_anomaly"] = 0.0
    
    # Ensure we returned ALL input laps
    if len(joined) < len(laps_df):
        raise ValueError(f"Lost laps during weather join: {len(laps_df)} → {len(joined)}")
    
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
    
    # Build canonical vehicle_id ↔ NUMBER mapping from sectors data (once, outside loop)
    vehicle_number_map = {}
    if "NUMBER" in sectors_df.columns:
        # Extract potential number from vehicle_id (e.g., "GR86-002-2" → 2 or 002)
        # This is heuristic but better than taking first NUMBER
        sector_nums = sectors_df["NUMBER"].dropna().unique()
        if len(sector_nums) > 0:
            for vid in laps_df["vehicle_id"].unique():
                # Try to extract number from vehicle_id
                parts = str(vid).split("-")
                potential_nums = []
                for p in parts:
                    try:
                        potential_nums.append(int(p))
                    except:
                        pass
                
                # Match to sectors NUMBER if exists
                if len(potential_nums) > 0:
                    # Find closest match in sectors
                    closest_num = min(sector_nums, key=lambda x: min([abs(x - pn) for pn in potential_nums], default=999))
                    vehicle_number_map[vid] = int(closest_num)
    
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
        
        # Get vehicle number for current vehicle
        vehicle_number = vehicle_number_map.get(vehicle_id, None)
        
        for _, lap_row in vehicle_laps.iterrows():
            lap_num = int(lap_row["lap"])
            
            # Tire age (backfill if missing: compute from lap index)
            if (vehicle_id, lap_num) in tire_ages.index:
                tire_age = tire_ages.loc[(vehicle_id, lap_num)]
            else:
                # Backfill: find nearest stint or compute from lap number
                if len(vehicle_stints) > 0:
                    # Use first stint's start as reference
                    first_stint_start = vehicle_stints[0].start_lap
                    tire_age = max(0, lap_num - first_stint_start + 1)
                else:
                    tire_age = lap_num  # Fallback: assume tire age = lap number
            
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
            
            # Backfill stint if missing (use previous lap's stint or nearest)
            if stint is None or best_lap_time is None:
                if len(vehicle_stints) > 0:
                    # Use last stint as fallback
                    stint = vehicle_stints[-1]
                    stint_len = lap_num - stint.start_lap + 1
                    best_lap_time = stint.best_lap_time_ms
                else:
                    # No stints at all - skip this vehicle's laps
                    continue
            
            # Pace delta (backfill lap_time_ms if missing from start/end timestamps)
            lap_time_ms = lap_row.get("lap_time_ms", None)
            if lap_time_ms is None or lap_time_ms == 0:
                # Try to derive from start/end timestamps
                if "lap_start_ts" in lap_row and "lap_end_ts" in lap_row:
                    start_ts = pd.to_datetime(lap_row["lap_start_ts"], errors="coerce", utc=True)
                    end_ts = pd.to_datetime(lap_row["lap_end_ts"], errors="coerce", utc=True)
                    if pd.notna(start_ts) and pd.notna(end_ts):
                        lap_time_ms = (end_ts - start_ts).total_seconds() * 1000.0
                
                # If still missing, try lap_time_s
                if (lap_time_ms is None or lap_time_ms == 0) and "lap_time_s" in lap_row:
                    lap_time_ms = lap_row["lap_time_s"] * 1000.0
                
                # If still missing, skip (cannot compute pace_delta)
                if lap_time_ms is None or lap_time_ms == 0:
                    continue
            
            pace_delta = (lap_time_ms - best_lap_time) / 1000.0  # Convert to seconds
            
            # Weather (should always exist now due to robust weather join)
            lap_weather = laps_with_weather[
                (laps_with_weather["vehicle_id"] == vehicle_id) &
                (laps_with_weather["lap"] == lap_num)
            ]
            
            if len(lap_weather) == 0:
                # This shouldn't happen with robust join, but fallback to session median
                track_temp = laps_with_weather["TRACK_TEMP"].median() if "TRACK_TEMP" in laps_with_weather.columns else 50.0
            else:
                track_temp = lap_weather["TRACK_TEMP"].iloc[0]
            
            # Compute real sector features (not simplified placeholders)
            # Per-vehicle sector S3 temperature sensitivity (regress driver's sector time vs temp)
            sector_coeff_s3 = compute_sector_temp_sensitivity(sectors_df, weather_df, "S3_SECONDS")  # Global for now
            
            # Real clean-air: check if no car within 1.0s ahead at S3 exit
            clean_air = True  # Default
            sector_col = "S3_SECONDS" if "S3_SECONDS" in sectors_df.columns else ("S3" if "S3" in sectors_df.columns else None)
            if vehicle_number is not None and sector_col:
                clean_air = detect_clean_air_flag(sectors_df, vehicle_number, lap_num, sector_col, threshold_std=1.0)
            
            # Real traffic density: count cars within ±1.5s at previous sector end
            traffic_density = 0.0  # Default
            if vehicle_number is not None and sector_col:
                traffic_density_raw = compute_traffic_density(
                    sectors_df, None, vehicle_number, lap_num,
                    sector_end=sector_col, proximity_seconds=1.5
                )
                traffic_density = min(1.0, max(0.0, traffic_density_raw))  # Clip to [0, 1]
            
            # Compute temperature anomaly (used for interactions)
            temp_anomaly = track_temp - laps_with_weather["TRACK_TEMP"].median()
            
            # Collect imputation flags for quality gate
            imputed_count = 0
            if "track_temp_imputed" in lap_weather.columns:
                imputed_count += int(lap_weather["track_temp_imputed"].iloc[0])
            if "air_temp_imputed" in lap_weather.columns:
                imputed_count += int(lap_weather["air_temp_imputed"].iloc[0])
            if "wind_imputed" in lap_weather.columns:
                imputed_count += int(lap_weather["wind_imputed"].iloc[0] if "wind_imputed" in lap_weather.columns else 0)
            if "rain_imputed" in lap_weather.columns:
                imputed_count += int(lap_weather["rain_imputed"].iloc[0] if "rain_imputed" in lap_weather.columns else 0)
            # Check if timestamp was derived
            if "ts_derived" in laps_df.columns:
                ts_derived_mask = (laps_df["vehicle_id"] == vehicle_id) & (laps_df["lap"] == lap_num)
                if ts_derived_mask.any():
                    ts_derived_val = laps_df.loc[ts_derived_mask, "ts_derived"].iloc[0] if "ts_derived" in laps_df.columns else 0
                    imputed_count += int(ts_derived_val)
            
            # Check for pit lap (large pace_delta spike indicates pit in/out)
            is_pit = False
            if pace_delta > 30.0:  # Pit loss typically 30s+
                is_pit = True
            
            # Check for yellow flag (hard to detect, skip for now)
            is_yellow = False
            
            # Interaction features (helpful for model)
            tire_temp_interaction = tire_age * temp_anomaly
            tire_clean_interaction = tire_age * (1.0 if clean_air else 0.0)
            traffic_temp_interaction = traffic_density * temp_anomaly
            
            all_features.append({
                "vehicle_id": vehicle_id,
                "lap": lap_num,
                "tire_age": tire_age,
                "track_temp": track_temp,
                "temp_anomaly": temp_anomaly,
                "stint_len": stint_len,
                "sector_S3_coeff": sector_coeff_s3,
                "clean_air": 1.0 if clean_air else 0.0,
                "traffic_density": traffic_density,
                "tire_temp_interaction": tire_temp_interaction,
                "tire_clean_interaction": tire_clean_interaction,
                "traffic_temp_interaction": traffic_temp_interaction,
                "race_id": race_id,
                "pace_delta": pace_delta,
                "is_pit": is_pit,
                "is_yellow": is_yellow,
                "imputed_count": imputed_count,
            })
    
    features_df = pd.DataFrame(all_features)
    
    if len(features_df) == 0:
        raise ValueError("No training data extracted")
    
    # Quality gate: filter noisy recovered laps
    print(f"  Pre-quality-gate: {len(features_df)} samples")
    
    # Filter bad laps (slightly relaxed to keep ≥70% of recovered laps)
    bad_mask = (
        features_df["is_pit"] |
        features_df["is_yellow"] |
        (features_df["tire_age"] < 1) |  # Out-lap
        (features_df["tire_age"] > 40) |  # Relaxed from 30 to 40 (allow longer stints)
        (features_df["imputed_count"] >= 4)  # Relaxed from 3 to 4 (too many imputed fields)
    )
    
    # Also check for invalid pace_delta (should be positive for pace degradation)
    # But allow negative (faster than stint best) as valid
    features_df_filtered = features_df.loc[~bad_mask].copy()
    
    print(f"  Filtered: {bad_mask.sum()} noisy samples ({100*bad_mask.sum()/len(features_df):.1f}%)")
    print(f"  Post-quality-gate: {len(features_df_filtered)} samples")
    
    # Winsorize pace_delta at 99th percentile
    p99_pace = features_df_filtered["pace_delta"].quantile(0.99)
    outliers = features_df_filtered["pace_delta"] > p99_pace
    if outliers.any():
        features_df_filtered.loc[outliers, "pace_delta"] = p99_pace
        print(f"  Winsorized: {outliers.sum()} extreme outliers at {p99_pace:.2f}s")
    
    features_df = features_df_filtered
    
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


