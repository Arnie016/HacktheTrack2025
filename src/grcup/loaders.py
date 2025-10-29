from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd


def _ensure_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    return p


def load_lap_times(csv_path: str | Path) -> pd.DataFrame:
    p = _ensure_path(csv_path)
    df = pd.read_csv(p)
    # Expected columns: expire_at, lap, meta_event, meta_session, meta_source, meta_time, outing, timestamp, value, vehicle_id
    # Parse times and coerce dtypes
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["meta_time"] = pd.to_datetime(df["meta_time"], utc=True, errors="coerce")
    for c in ("lap", "outing", "value"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.rename(columns={"value": "lap_time_ms"}, inplace=True)
    return df


def load_lap_starts(csv_path: str | Path) -> pd.DataFrame:
    p = _ensure_path(csv_path)
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["meta_time"] = pd.to_datetime(df["meta_time"], utc=True, errors="coerce")
    for c in ("lap", "outing"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.rename(columns={"timestamp": "lap_start_ts"}, inplace=True)
    return df


def load_lap_ends(csv_path: str | Path) -> pd.DataFrame:
    p = _ensure_path(csv_path)
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["meta_time"] = pd.to_datetime(df["meta_time"], utc=True, errors="coerce")
    for c in ("lap", "outing"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.rename(columns={"timestamp": "lap_end_ts"}, inplace=True)
    return df


def load_weather(csv_path: str | Path, delimiter: Optional[Literal[",",";"]] = ";") -> pd.DataFrame:
    p = _ensure_path(csv_path)
    df = pd.read_csv(p, delimiter=delimiter)
    # Expected columns: TIME_UTC_SECONDS;TIME_UTC_STR;AIR_TEMP;TRACK_TEMP;HUMIDITY;PRESSURE;WIND_SPEED;WIND_DIRECTION;RAIN
    # Normalize
    time_col = "TIME_UTC_SECONDS" if "TIME_UTC_SECONDS" in df.columns else "time_utc_seconds"
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df["weather_ts"] = pd.to_datetime(df[time_col], unit="s", utc=True, errors="coerce")
    return df


def load_sectors(csv_path: str | Path, delimiter: Optional[Literal[",",";"]] = ";") -> pd.DataFrame:
    """Load AnalysisEnduranceWithSections CSV with sector timing data."""
    p = _ensure_path(csv_path)
    df = pd.read_csv(p, delimiter=delimiter, low_memory=False)
    
    # Normalize column names (handle spaces)
    df.columns = df.columns.str.strip()
    
    # Parse lap time from MM:SS.mmm format
    if "LAP_TIME" in df.columns:
        def parse_lap_time(t_str):
            if pd.isna(t_str) or t_str == "":
                return None
            parts = str(t_str).split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return None
        
        df["lap_time_s"] = df["LAP_TIME"].apply(parse_lap_time)
    
    # Ensure numeric types
    numeric_cols = ["NUMBER", "LAP_NUMBER", "S1_SECONDS", "S2_SECONDS", "S3_SECONDS", "PIT_TIME"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def load_results(csv_path: str | Path, delimiter: Optional[Literal[",",";"]] = ";") -> pd.DataFrame:
    """Load Results CSV with race positions and times."""
    p = _ensure_path(csv_path)
    df = pd.read_csv(p, delimiter=delimiter, low_memory=False)
    
    # Normalize column names
    df.columns = df.columns.str.strip()
    
    # Ensure numeric types
    numeric_cols = ["POSITION", "NUMBER", "LAPS", "FL_LAPNUM"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Parse FL_TIME (fastest lap) if present
    if "FL_TIME" in df.columns:
        def parse_time(t_str):
            if pd.isna(t_str) or t_str == "":
                return None
            parts = str(t_str).split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return None
        
        df["fl_time_s"] = df["FL_TIME"].apply(parse_time)
    
    return df


def build_lap_table(
    lap_times: pd.DataFrame,
    lap_starts: pd.DataFrame,
    lap_ends: pd.DataFrame,
) -> pd.DataFrame:
    # Minimal canonical schema per (vehicle_id, lap)
    # Ensure timestamp columns exist
    lt_cols = ["vehicle_id", "lap", "lap_time_ms"]
    if "timestamp" in lap_times.columns:
        lt_cols.append("timestamp")
        lt = lap_times[lt_cols].rename(columns={"timestamp": "lap_time_ts"})
    else:
        lt = lap_times[lt_cols].copy()
        lt["lap_time_ts"] = None
    
    ls_cols = ["vehicle_id", "lap"]
    if "lap_start_ts" in lap_starts.columns:
        ls_cols.append("lap_start_ts")
    ls = lap_starts[ls_cols]
    
    le_cols = ["vehicle_id", "lap"]
    if "lap_end_ts" in lap_ends.columns:
        le_cols.append("lap_end_ts")
    le = lap_ends[le_cols]

    merged = (
        lt.merge(ls, on=["vehicle_id", "lap"], how="outer")
        .merge(le, on=["vehicle_id", "lap"], how="outer")
        .sort_values(["vehicle_id", "lap"], kind="mergesort")
        .reset_index(drop=True)
    )
    
    # Filter out invalid lap numbers (0, 32768 sentinel, negative, >1000)
    # 32768 is 2^15, likely int16 overflow or sentinel value
    if "lap" in merged.columns:
        merged = merged[
            (merged["lap"] > 0) & 
            (merged["lap"] < 1000) &
            (merged["lap"] != 32768)
        ].copy()
    
    # Derive fallbacks: if lap_time_ms missing but start/end present
    if "lap_time_ms" in merged.columns:
        missing = merged["lap_time_ms"].isna()
        if missing.any() and "lap_start_ts" in merged.columns and "lap_end_ts" in merged.columns:
            dt = (merged["lap_end_ts"] - merged["lap_start_ts"]).dt.total_seconds() * 1000.0
            merged.loc[missing, "lap_time_ms"] = dt[missing]
    
    # Ensure timestamp column exists (prefer lap_start_ts, fallback to lap_time_ts)
    if "timestamp" not in merged.columns:
        if "lap_start_ts" in merged.columns:
            merged["timestamp"] = pd.to_datetime(merged["lap_start_ts"], utc=True, errors="coerce")
        elif "lap_time_ts" in merged.columns and merged["lap_time_ts"].notna().any():
            merged["timestamp"] = pd.to_datetime(merged["lap_time_ts"], utc=True, errors="coerce")
        elif "lap_end_ts" in merged.columns:
            merged["timestamp"] = pd.to_datetime(merged["lap_end_ts"], utc=True, errors="coerce")
    
    return merged

