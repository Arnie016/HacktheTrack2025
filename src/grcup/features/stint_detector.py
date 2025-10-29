"""Detect stints (consecutive laps between pit stops) from lap timing data."""
from __future__ import annotations

import pandas as pd
from typing import Optional

from ..data.schemas import Stint


def detect_stints(
    laps_df: pd.DataFrame,
    vehicle_id: str,
    pit_threshold_ms: float = 30000.0,  # 30 seconds = likely pit stop
) -> list[Stint]:
    """
    Detect stints for a single vehicle.
    
    Args:
        laps_df: DataFrame with columns: vehicle_id, lap, lap_time_ms (or lap_time_s)
        vehicle_id: Vehicle to analyze
        pit_threshold_ms: Lap time threshold to detect pit stop (default 30s)
    
    Returns:
        List of Stint objects
    """
    vehicle_laps = laps_df[laps_df["vehicle_id"] == vehicle_id].copy()
    vehicle_laps = vehicle_laps.sort_values("lap").reset_index(drop=True)
    
    if len(vehicle_laps) == 0:
        return []
    
    # Convert lap_time_s to ms if needed
    if "lap_time_ms" not in vehicle_laps.columns and "lap_time_s" in vehicle_laps.columns:
        vehicle_laps["lap_time_ms"] = vehicle_laps["lap_time_s"] * 1000.0
    
    if "lap_time_ms" not in vehicle_laps.columns:
        raise ValueError("Need lap_time_ms or lap_time_s column")
    
    stints = []
    current_stint_start = vehicle_laps.iloc[0]["lap"]
    
    for i in range(1, len(vehicle_laps)):
        current_lap = vehicle_laps.iloc[i]
        prev_lap = vehicle_laps.iloc[i - 1]
        
        # Check for pit stop: large gap in lap time
        time_diff = current_lap["lap_time_ms"] - prev_lap["lap_time_ms"]
        
        if time_diff > pit_threshold_ms:
            # Pit detected - end previous stint
            end_lap = prev_lap["lap"]
            stint_laps = vehicle_laps[
                (vehicle_laps["lap"] >= current_stint_start) &
                (vehicle_laps["lap"] <= end_lap)
            ]
            
            if len(stint_laps) > 0:
                best_lap_time_ms = stint_laps["lap_time_ms"].min()
                # Skip stints with invalid lap times or lap numbers
                if best_lap_time_ms > 0 and current_stint_start >= 1 and end_lap >= 1:
                    stints.append(Stint(
                        car_id=vehicle_id,
                        start_lap=int(current_stint_start),
                        end_lap=int(end_lap),
                        best_lap_time_ms=float(best_lap_time_ms),
                        laps=stint_laps["lap"].tolist()
                    ))
            
            current_stint_start = current_lap["lap"]
    
    # Add final stint
    final_laps = vehicle_laps[vehicle_laps["lap"] >= current_stint_start]
    if len(final_laps) > 0:
        best_lap_time_ms = final_laps["lap_time_ms"].min()
        end_lap_final = int(vehicle_laps.iloc[-1]["lap"])
        # Skip stints with invalid lap times or lap numbers
        if best_lap_time_ms > 0 and current_stint_start >= 1 and end_lap_final >= 1:
            stints.append(Stint(
                car_id=vehicle_id,
                start_lap=int(current_stint_start),
                end_lap=end_lap_final,
                best_lap_time_ms=float(best_lap_time_ms),
                laps=final_laps["lap"].tolist()
            ))
    
    return stints


def compute_tire_age_per_lap(
    laps_df: pd.DataFrame,
    stints: list[Stint],
) -> pd.Series:
    """
    Compute tire age (laps since pit stop) for each lap.
    
    Args:
        laps_df: DataFrame with vehicle_id and lap columns
        stints: List of Stint objects for the vehicle
    
    Returns:
        Series with tire_age per lap (indexed by (vehicle_id, lap))
    """
    result = pd.Series(index=pd.MultiIndex.from_tuples([], names=["vehicle_id", "lap"]), dtype=float)
    
    for stint in stints:
        for lap_num in stint.laps:
            tire_age = lap_num - stint.start_lap
            result.loc[(stint.car_id, lap_num)] = tire_age
    
    return result


def estimate_pit_loss_empirical(
    sectors_df: pd.DataFrame,
    race_id: str,
) -> tuple[float, float]:
    """
    Estimate empirical pit stop loss from PIT_TIME data.
    
    Args:
        sectors_df: DataFrame with PIT_TIME column
        race_id: Race identifier for filtering
    
    Returns:
        (mean_pit_loss_seconds, std_pit_loss_seconds)
    """
    if "PIT_TIME" not in sectors_df.columns:
        return 30.0, 5.0  # Default fallback
    
    pit_times = sectors_df["PIT_TIME"].dropna()
    
    if len(pit_times) == 0:
        return 30.0, 5.0
    
    # Remove outliers (beyond 3 std devs)
    mean_pt = pit_times.mean()
    std_pt = pit_times.std()
    pit_times_clean = pit_times[
        (pit_times >= mean_pt - 3 * std_pt) &
        (pit_times <= mean_pt + 3 * std_pt)
    ]
    
    if len(pit_times_clean) == 0:
        pit_times_clean = pit_times
    
    mean_loss = float(pit_times_clean.mean())
    std_loss = float(pit_times_clean.std())
    
    return mean_loss, std_loss

