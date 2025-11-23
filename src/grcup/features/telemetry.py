"""Feature engineering for telemetry data (accel, jerk, physics metrics)."""
import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path

def load_telemetry(telemetry_path: Union[str, Path]) -> pd.DataFrame:
    """Load raw telemetry CSV (long format)."""
    return pd.read_csv(telemetry_path)

def pivot_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format telemetry to wide format.
    Extracts accx_can (longitudinal) and accy_can (lateral).
    """
    # Filter for relevant signals to reduce memory usage
    relevant_signals = ["accx_can", "accy_can", "ath"]  # ath = throttle if needed later
    df_filtered = df[df["telemetry_name"].isin(relevant_signals)].copy()
    
    # Convert timestamp to datetime for sorting
    df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"])
    
    # Pivot: index=(vehicle_id, lap, timestamp), columns=telemetry_name, values=telemetry_value
    # Using pivot_table handles duplicates by averaging (though duplicates shouldn't exist ideally)
    df_pivot = df_filtered.pivot_table(
        index=["vehicle_id", "lap", "timestamp"], 
        columns="telemetry_name", 
        values="telemetry_value",
        aggfunc="first"  # Assuming consistent timestamps
    ).reset_index()
    
    # Sort by vehicle and time
    df_pivot = df_pivot.sort_values(["vehicle_id", "timestamp"])
    
    return df_pivot

def compute_physics_metrics(df_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Compute acceleration magnitude and jerk.
    
    Args:
        df_pivot: Wide-format telemetry dataframe with accx_can, accy_can
        
    Returns:
        Dataframe aggregated by vehicle_id and lap with physics features.
    """
    # Ensure required columns exist
    for col in ["accx_can", "accy_can"]:
        if col not in df_pivot.columns:
            df_pivot[col] = 0.0
    if "ath" not in df_pivot.columns:
        df_pivot["ath"] = 0.0
            
    # 1. Acceleration Magnitude (G-force roughly)
    df_pivot["accel_mag"] = np.sqrt(
        df_pivot["accx_can"]**2 + df_pivot["accy_can"]**2
    )
    
    # 2. Jerk (Rate of change of acceleration)
    # Group by vehicle to ensure we don't diff across vehicles
    # Convert timestamp difference to seconds
    df_pivot["dt"] = df_pivot.groupby("vehicle_id")["timestamp"].diff().dt.total_seconds()
    df_pivot["d_accel"] = df_pivot.groupby("vehicle_id")["accel_mag"].diff().abs()
    
    # Avoid division by zero or tiny dt
    df_pivot["jerk"] = df_pivot["d_accel"] / df_pivot["dt"].replace(0, np.nan)
    
    # Filter out unrealistic spikes (sensor noise)
    df_pivot.loc[df_pivot["jerk"] > 50.0, "jerk"] = np.nan
    
    # Helper columns for richer aggregates
    df_pivot["accx_can_abs"] = df_pivot["accx_can"].abs()
    df_pivot["accy_can_abs"] = df_pivot["accy_can"].abs()
    df_pivot["jerk_sq"] = df_pivot["jerk"].fillna(0.0) ** 2
    
    # 3. Aggression Metrics (Aggregated per lap)
    metrics = df_pivot.groupby(["vehicle_id", "lap"]).agg(
        accel_mag_mean=("accel_mag", "mean"),
        accel_mag_max=("accel_mag", "max"),
        accel_mag_std=("accel_mag", "std"),
        accel_mag_p95=("accel_mag", lambda x: x.quantile(0.95)),
        jerk_mean=("jerk", "mean"),
        jerk_max=("jerk", "max"),
        jerk_std=("jerk", "std"),
        jerk_p95=("jerk", lambda x: x.quantile(0.95)),
        jerk_sq_mean=("jerk_sq", "mean"),
        accx_can_mean_abs=("accx_can_abs", "mean"),
        accx_can_max=("accx_can", "max"),
        accx_can_std=("accx_can", "std"),
        accy_can_mean_abs=("accy_can_abs", "mean"),
        accy_can_max=("accy_can", "max"),
        accy_can_std=("accy_can", "std"),
        throttle_mean=("ath", "mean"),
        throttle_std=("ath", "std"),
        throttle_p95=("ath", lambda x: x.quantile(0.95)),
    ).reset_index()
    
    # Derived features
    metrics["jerk_rms"] = np.sqrt(metrics["jerk_sq_mean"].clip(lower=0.0))
    metrics["combined_aggression"] = metrics["accel_mag_mean"] * metrics["jerk_mean"].fillna(0.0)
    
    # Drop helper columns
    metrics = metrics.drop(columns=["jerk_sq_mean"])
    
    # Fill NaNs (e.g. first lap might have missing jerk)
    metrics = metrics.fillna(0.0)
    
    return metrics

def build_telemetry_features(telemetry_path: Union[str, Path]) -> pd.DataFrame:
    """Pipeline to build lap-level telemetry features."""
    print(f"Loading telemetry from {telemetry_path}...")
    df = load_telemetry(telemetry_path)
    
    print("Pivoting data...")
    df_pivot = pivot_telemetry(df)
    
    print("Computing physics metrics (accel, jerk)...")
    metrics = compute_physics_metrics(df_pivot)
    
    return metrics

if __name__ == "__main__":
    # Test run
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        feats = build_telemetry_features(path)
        print("\nExtracted Features Head:")
        print(feats.head())
        print("\nFeature Columns:", feats.columns.tolist())

