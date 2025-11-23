"""Data loaders for GR Cup race data.

This module provides functions to load and parse various CSV files
from GR Cup race data, including lap times, sectors, weather, and results.
"""
import pandas as pd
from pathlib import Path
from typing import Optional


def _read_csv(filepath: Path, sep: Optional[str] = None) -> pd.DataFrame:
    """Helper to read CSV with optional separator detection."""
    if sep:
        return pd.read_csv(filepath, sep=sep)
    return pd.read_csv(filepath)


def load_lap_times(filepath: Path) -> pd.DataFrame:
    """Load lap time data from CSV.
    
    Args:
        filepath: Path to vir_lap_time_R*.csv file
        
    Returns:
        DataFrame with columns: vehicle_id, lap, value (lap time in seconds)
    """
    df = _read_csv(filepath)
    # Standardize column names if needed
    if "value" in df.columns and "lap_time" not in df.columns:
        df = df.rename(columns={"value": "lap_time"})
    return df


def load_lap_starts(filepath: Path) -> pd.DataFrame:
    """Load lap start timestamp data from CSV.
    
    Args:
        filepath: Path to vir_lap_start_R*.csv file
        
    Returns:
        DataFrame with columns: vehicle_id, lap, value (start timestamp)
    """
    df = _read_csv(filepath)
    if "value" in df.columns and "lap_start" not in df.columns:
        df = df.rename(columns={"value": "lap_start"})
    return df


def load_lap_ends(filepath: Path) -> pd.DataFrame:
    """Load lap end timestamp data from CSV.
    
    Args:
        filepath: Path to vir_lap_end_R*.csv file
        
    Returns:
        DataFrame with columns: vehicle_id, lap, value (end timestamp)
    """
    df = _read_csv(filepath)
    if "value" in df.columns and "lap_end" not in df.columns:
        df = df.rename(columns={"value": "lap_end"})
    return df


def load_sectors(filepath: Path) -> pd.DataFrame:
    """Load sector timing data from AnalysisEnduranceWithSections CSV.
    
    Args:
        filepath: Path to 23_AnalysisEnduranceWithSections_Race *_Anonymized.CSV
        
    Returns:
        DataFrame with sector timing data
    """
    return _read_csv(filepath, sep=";")


def load_weather(filepath: Path) -> pd.DataFrame:
    """Load weather data from CSV.
    
    Args:
        filepath: Path to 26_Weather_Race *_Anonymized.CSV
        
    Returns:
        DataFrame with weather data (track_temp, air_temp, etc.)
    """
    return _read_csv(filepath, sep=";")


def load_results(filepath: Path) -> pd.DataFrame:
    """Load race results from CSV.
    
    Args:
        filepath: Path to 03_Provisional Results_Race *_Anonymized.CSV or
                 03_Results GR Cup Race *_Official_Anonymized.CSV
        
    Returns:
        DataFrame with race results (position, vehicle_id, total_time, etc.)
    """
    return _read_csv(filepath, sep=";")


def load_telemetry_features(filepath: Path) -> pd.DataFrame:
    """Load precomputed telemetry/physics features."""
    return _read_csv(filepath)


def build_lap_table(
    lap_times: pd.DataFrame,
    lap_starts: pd.DataFrame,
    lap_ends: pd.DataFrame,
) -> pd.DataFrame:
    """Build a unified lap table from lap timing data.
    
    Args:
        lap_times: DataFrame from load_lap_times()
        lap_starts: DataFrame from load_lap_starts()
        lap_ends: DataFrame from load_lap_ends()
        
    Returns:
        Merged DataFrame with vehicle_id, lap, lap_time, lap_start, lap_end
    """
    # Merge on vehicle_id and lap
    merged = lap_times.copy()
    
    if "lap_time" not in merged.columns and "value" in lap_times.columns:
        merged["lap_time"] = lap_times["value"]
    
    # Normalize numeric columns
    merged["lap_time_ms"] = pd.to_numeric(merged["lap_time"], errors="coerce")
    # Some feeds output seconds—detect by magnitude
    seconds_mask = merged["lap_time_ms"].between(-1e6, 1000)
    merged.loc[seconds_mask, "lap_time_ms"] = merged.loc[seconds_mask, "lap_time_ms"] * 1000.0
    merged["lap_time_s"] = merged["lap_time_ms"] / 1000.0
    
    # Merge starts
    lap_starts_renamed = lap_starts.rename(columns={"lap_start": "lap_start_ts"})
    if "value" in lap_starts_renamed.columns and "lap_start_ts" not in lap_starts_renamed.columns:
        lap_starts_renamed = lap_starts_renamed.rename(columns={"value": "lap_start_ts"})
    merged = merged.merge(
        lap_starts_renamed[["vehicle_id", "lap", "lap_start_ts"]],
        on=["vehicle_id", "lap"],
        how="left",
    )
    
    # Merge ends
    lap_ends_renamed = lap_ends.rename(columns={"lap_end": "lap_end_ts"})
    if "value" in lap_ends_renamed.columns and "lap_end_ts" not in lap_ends_renamed.columns:
        lap_ends_renamed = lap_ends_renamed.rename(columns={"value": "lap_end_ts"})
    merged = merged.merge(
        lap_ends_renamed[["vehicle_id", "lap", "lap_end_ts"]],
        on=["vehicle_id", "lap"],
        how="left",
    )
    
    # Convert timestamps to datetime
    for col in ["lap_start_ts", "lap_end_ts"]:
        if col in merged.columns:
            merged[col] = pd.to_datetime(merged[col], errors="coerce", utc=True)
    
    # Derive canonical timestamp (prefer lap_end_ts)
    merged["lap_ts_utc"] = merged["lap_end_ts"].combine_first(merged["lap_start_ts"])
    merged["ts_derived"] = merged["lap_ts_utc"].isna().astype("int8")
    
    return merged

