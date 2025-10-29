"""Compute sector-level temperature sensitivity and clean-air flags."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_sector_temp_sensitivity(
    sectors_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    sector_name: str,  # "S1_SECONDS", "S2_SECONDS", "S3_SECONDS"
) -> float:
    """
    Compute temperature sensitivity coefficient for a sector.
    
    Regression: sector_time = base + coeff * track_temp + error
    
    Returns:
        Coefficient (seconds per degree C)
    """
    # Join sectors to weather by timestamp approximation
    # For now, use mean track temp (can refine with asof join later)
    mean_temp = weather_df["TRACK_TEMP"].mean()
    temp_std = weather_df["TRACK_TEMP"].std()
    
    if sector_name not in sectors_df.columns:
        return 0.0
    
    sector_times = sectors_df[sector_name].dropna()
    
    if len(sector_times) < 10:  # Need enough data
        return 0.0
    
    # Simplified: use track temp variation over race
    # More sophisticated: join by lap timestamp
    temps_normalized = (weather_df["TRACK_TEMP"].values - mean_temp) / temp_std
    temps_aligned = np.tile(temps_normalized, len(sector_times) // len(weather_df) + 1)[:len(sector_times)]
    
    if len(temps_aligned) != len(sector_times):
        return 0.0
    
    # Fit linear model
    try:
        model = LinearRegression()
        X = temps_aligned.reshape(-1, 1)
        y = sector_times.values
        model.fit(X, y)
        return float(model.coef_[0])
    except:
        return 0.0


def detect_clean_air_flag(
    sectors_df: pd.DataFrame,
    vehicle_number: int,
    lap_number: int,
    sector_name: str,  # "S3", "S3_SECONDS", etc.
    threshold_std: float = 1.0,
) -> bool:
    """
    Detect if driver is in clean air based on sector time residuals.
    
    Clean air = sector time is faster than class average (negative residual).
    
    Args:
        sectors_df: All sector data
        vehicle_number: Vehicle number
        lap_number: Lap number
        sector_name: Sector column name
        threshold_std: Standard deviations below mean to count as clean air
    
    Returns:
        True if clean air, False if traffic
    """
    if sector_name not in sectors_df.columns:
        return True  # Assume clean if no data
    
    # Get sector time for this lap
    vehicle_lap = sectors_df[
        (sectors_df["NUMBER"] == vehicle_number) &
        (sectors_df["LAP_NUMBER"] == lap_number)
    ]
    
    if len(vehicle_lap) == 0:
        return True
    
    sector_time = vehicle_lap[sector_name].iloc[0]
    
    # Compare to class average (same lap)
    class_lap_times = sectors_df[
        sectors_df["LAP_NUMBER"] == lap_number
    ][sector_name].dropna()
    
    if len(class_lap_times) < 3:
        return True
    
    mean_time = class_lap_times.mean()
    std_time = class_lap_times.std()
    
    if pd.isna(sector_time) or pd.isna(mean_time) or std_time == 0:
        return True
    
    # Clean air if faster than mean - threshold_std * std
    residual = sector_time - mean_time
    clean_threshold = -threshold_std * std_time
    
    return residual < clean_threshold


def compute_traffic_density(
    sectors_df: pd.DataFrame,
    results_df: pd.DataFrame,
    vehicle_number: int,
    lap_number: int,
    sector_end: str = "S3",  # Use sector 3 end for gap calculation ("S3" or "S3_SECONDS")
    proximity_seconds: float = 1.5,
) -> float:
    """
    Proxy for traffic density: number of cars within ±proximity_seconds at sector end.
    
    Args:
        sectors_df: Sector timing data
        results_df: Race results (for positions)
        vehicle_number: Vehicle to analyze
        lap_number: Lap number
        sector_end: Sector column to use for gap calculation
        proximity_seconds: Time window to count as "nearby"
    
    Returns:
        Traffic density (0.0 = no traffic, higher = more traffic)
    """
    # Try S3_SECONDS first, fallback to S3 if not found
    if sector_end not in sectors_df.columns:
        if sector_end == "S3" and "S3_SECONDS" in sectors_df.columns:
            sector_end = "S3_SECONDS"
        elif sector_end == "S3_SECONDS" and "S3" in sectors_df.columns:
            sector_end = "S3"
        else:
            return 0.0
    
    if sector_end not in sectors_df.columns:
        return 0.0
    
    # Get this car's sector time
    vehicle_lap = sectors_df[
        (sectors_df["NUMBER"] == vehicle_number) &
        (sectors_df["LAP_NUMBER"] == lap_number)
    ]
    
    if len(vehicle_lap) == 0:
        return 0.0
    
    vehicle_time = vehicle_lap[sector_end].iloc[0]
    
    if pd.isna(vehicle_time):
        return 0.0
    
    # Find cars within proximity
    all_lap_times = sectors_df[
        sectors_df["LAP_NUMBER"] == lap_number
    ][sector_end].dropna()
    
    if len(all_lap_times) < 2:
        return 0.0
    
    time_diffs = np.abs(all_lap_times - vehicle_time)
    nearby_cars = (time_diffs <= proximity_seconds).sum() - 1  # Subtract self
    
    return float(nearby_cars) / len(all_lap_times)  # Normalized density


