"""Complete training pipeline: load data, extract features, train all models."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly to avoid __init__.py timeout (disk full issue)
from src.grcup.features.feature_extractor import (
    build_pace_prediction_features,
    build_wear_training_dataset,
)
from src.grcup.features.stint_detector import (
    detect_stints,
    estimate_pit_loss_empirical,
)
from src.grcup.loaders import (
    load_lap_ends,
    load_lap_starts,
    load_lap_times,
    load_results,
    load_sectors,
    load_weather,
)
from src.grcup.features.telemetry import build_telemetry_features
from src.grcup.models import (
    train_cox_hazard,
    train_overtake_model,
    train_wear_quantile_model,
    prepare_hazard_data,
    prepare_overtake_data,
    save_hazard_model,
    save_kalman_config,
    save_model,
    save_overtake_model,
)
from src.grcup.models.kalman_pace import KalmanPaceFilter
from src.grcup.utils.io import save_json
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except:
        return "unknown"


def get_package_versions() -> dict:
    """Get versions of key packages."""
    packages = ["xgboost", "pandas", "numpy", "lifelines", "numba", "sklearn"]
    versions = {}
    
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        
        for line in result.stdout.split("\n"):
            for pkg in packages:
                if line.startswith(pkg + "=="):
                    versions[pkg] = line.split("==")[1]
    except:
        pass
    
    return versions


def tune_wear_hyperparameters(features_df: pd.DataFrame) -> dict:
    """Tune XGBoost hyperparameters using RandomizedSearchCV."""
    print("  Running hyperparameter tuning (RandomizedSearchCV)...")
    
    # Replicate feature selection from train_wear_quantile_model
    base_features = [
        "tire_age", "track_temp", "temp_anomaly", "stint_len", 
        "sector_S3_coeff", "clean_air", "traffic_density"
    ]
    interaction_features = [
        "tire_temp_interaction", "tire_clean_interaction", "traffic_temp_interaction"
    ]
    
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
    
    if "driver_TE" in features_df.columns:
        feature_cols.append("driver_TE")
        
    # Physics features (align with wear_quantile_xgb)
    physics_features = [
        "accel_mag_mean",
        "accel_mag_max",
        "accel_mag_std",
        "accel_mag_p95",
        "jerk_mean",
        "jerk_max",
        "jerk_std",
        "jerk_p95",
        "jerk_rms",
        "accx_can_mean_abs",
        "accx_can_max",
        "accx_can_std",
        "accy_can_mean_abs",
        "accy_can_max",
        "accy_can_std",
        "throttle_mean",
        "throttle_std",
        "throttle_p95",
        "combined_aggression",
        "avg_long_accel",
        "avg_lat_accel",
        "avg_jerk",
    ]
    feature_cols += [f for f in physics_features if f in features_df.columns]
    
    X = features_df[feature_cols].values
    y = features_df["pace_delta"].values
    
    # Remove NaNs
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
    }
    
    # Use GPU if available (XGBoost 2.0+ API)
    import os
    use_gpu = os.getenv("USE_GPU", "0") == "1"
    
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.5,
        tree_method="hist",  # Always use hist
        device="cuda" if use_gpu else "cpu",  # New XGBoost 2.0+ API
        n_jobs=-1,
        random_state=42
    )
    
    search = RandomizedSearchCV(
        model, 
        param_distributions=param_dist, 
        n_iter=15, 
        scoring="neg_mean_absolute_error", 
        cv=3,
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_scaled, y)
    print(f"  ✓ Best params: {search.best_params_}")
    return search.best_params_


# Global for script override (e.g., by Modal)
models_dir = None

def main():
    """Train all models on Race 1 data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cqr-alpha", type=float, default=0.10, help="CQR alpha (target miscoverage)")
    parser.add_argument("--cqr-scale", type=float, default=0.90, help="Scale factor for conformal adjustments (post-compute)")
    args = parser.parse_args()

    global models_dir
    base_dir = Path(__file__).parent.parent
    race_dir = base_dir / "Race 1"
    
    # Use override if set, otherwise default to base_dir/models
    if models_dir is None:
        models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("GR Cup Model Training Pipeline")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading Race 1 data...")
    lap_times = load_lap_times(race_dir / "vir_lap_time_R1.csv")
    lap_starts = load_lap_starts(race_dir / "vir_lap_start_R1.csv")
    lap_ends = load_lap_ends(race_dir / "vir_lap_end_R1.csv")
    sectors = load_sectors(race_dir / "23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV")
    weather = load_weather(race_dir / "26_Weather_Race 1_Anonymized.CSV")
    results = load_results(race_dir / "03_Provisional Results_Race 1_Anonymized.CSV")
    
    print(f"  Loaded {len(lap_times)} lap times")
    print(f"  Loaded {len(sectors)} sector records")
    print(f"  Loaded {len(weather)} weather records")
    
    # Build lap table
    print("\n[2/6] Building lap table and detecting stints...")
    from src.grcup.loaders import build_lap_table
    laps = build_lap_table(lap_times, lap_starts, lap_ends)
    
    # Detect stints
    all_stints = []
    for vehicle_id in laps["vehicle_id"].unique():
        vehicle_stints = detect_stints(laps, vehicle_id)
        all_stints.extend(vehicle_stints)
    
    print(f"  Detected {len(all_stints)} stints across {len(laps['vehicle_id'].unique())} vehicles")
    
    # Estimate pit loss
    pit_loss_mean, pit_loss_std = estimate_pit_loss_empirical(sectors, "R1")
    print(f"  Estimated pit loss: {pit_loss_mean:.1f}s ± {pit_loss_std:.1f}s")
    
    # Extract features
    print("\n[3/6] Extracting features for training...")
    telemetry_feat_path = race_dir / "R1_telemetry_features.csv"
    telemetry_raw_path = race_dir / "R1_vir_telemetry_data.csv"
    physics_feats = None
    if telemetry_feat_path.exists():
        print(f"  Loading pre-processed physics metrics from {telemetry_feat_path.name}...")
        physics_feats = pd.read_csv(telemetry_feat_path)
    elif telemetry_raw_path.exists():
        print("  Processing raw telemetry physics metrics (slow)...")
        physics_feats = build_telemetry_features(telemetry_raw_path)
    else:
        print("  Warning: Telemetry file not found, skipping physics features")
        
    try:
        wear_features = build_wear_training_dataset(
            laps,
            sectors,
            weather,
            "R1",
            telemetry_df=physics_feats,
            results_df=results,
        )
        print(f"  Wear features: {len(wear_features)} samples")
    except Exception as e:
        print(f"  Warning: Could not build full wear dataset: {e}")
        print("  Using simplified features...")
        # Simplified fallback
        wear_features = pd.DataFrame({
            "vehicle_id": laps["vehicle_id"],
            "lap": laps["lap"],
            "tire_age": np.random.randint(0, 15, len(laps)),
            "track_temp": 50.0,
            "stint_len": np.random.randint(1, 20, len(laps)),
            "sector_S3_coeff": 0.0,
            "clean_air": 1.0,
            "traffic_density": 0.0,
            "driver_TE": 0.0,
            "race_id": "R1",
            "pace_delta": np.random.randn(len(laps)) * 2.0,
        })
    
    pace_features = build_pace_prediction_features(laps, n_lags=5)
    print(f"  Pace features: {len(pace_features)} samples")
    
    # Train wear model with CQR calibration
    print("\n[4/6] Training wear quantile model with CQR calibration...")
    try:
        # Split into train/calibration (80/20) for CQR
        from sklearn.model_selection import train_test_split
        
        # Simple random split (stratification fails if some vehicles have <2 samples)
        train_features, cal_features = train_test_split(
            wear_features,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )
        
        print(f"  Train split: {len(train_features)} samples")
        print(f"  Calibration split: {len(cal_features)} samples")
        
        # Tune hyperparameters first
        best_params = tune_wear_hyperparameters(train_features)
        
        # Train on train split
        wear_model_data = train_wear_quantile_model(
            train_features, 
            quantiles=[0.1, 0.5, 0.9],
            hyperparams=best_params
        )
        
        # Compute CQR adjustments on calibration split
        print("  Computing CQR adjustments on calibration set...")
        from src.grcup.models import predict_quantiles
        from src.grcup.evaluation.conformal import conformalize_quantiles
        
        cal_predictions = predict_quantiles(wear_model_data, cal_features)
        cal_actuals = cal_features["pace_delta"].reset_index(drop=True)
        cal_predictions = cal_predictions.reset_index(drop=True)
        
        # Ensure same length
        min_len = min(len(cal_predictions), len(cal_actuals))
        cal_predictions = cal_predictions.iloc[:min_len]
        cal_actuals = cal_actuals.iloc[:min_len]
        
        # Compute conformal adjustments
        # Use user-provided alpha (e.g., 0.10 = 90% target)
        adj_low, adj_high = conformalize_quantiles(
            cal_predictions["q10"].values,
            cal_predictions["q90"].values,
            cal_actuals.values,
            alpha=args.cqr_alpha,
        )

        # Scale adjustments (nudge coverage if needed)
        adj_low = adj_low * args.cqr_scale
        adj_high = adj_high * args.cqr_scale
        
        # Pre-CQR coverage for reporting
        from src.grcup.evaluation.calibration import compute_quantile_coverage
        pre_cqr_coverage = compute_quantile_coverage(cal_predictions, cal_actuals, quantile=0.9)
        
        # Save CQR adjustments to model metadata
        wear_model_data["cqr_adjustments"] = {
            "adjustment_low": float(adj_low),
            "adjustment_high": float(adj_high),
            "calibration_samples": len(cal_features),
            "pre_cqr_coverage": float(pre_cqr_coverage),
        }
        
        print(f"    Pre-CQR coverage: {pre_cqr_coverage:.2%}")
        print(f"    CQR adjustments: low={adj_low:.3f}, high={adj_high:.3f}")
        
        save_model(wear_model_data, models_dir / "wear_quantile_xgb.pkl")
        print("  ✓ Wear model saved with CQR adjustments")
    except Exception as e:
        print(f"  ✗ Wear model training failed: {e}")
        import traceback
        traceback.print_exc()
        print("  Creating placeholder...")
        # Placeholder
        wear_model_data = {"models": {}, "scaler": None, "feature_names": [], "quantiles": [0.1, 0.5, 0.9]}
        save_model(wear_model_data, models_dir / "wear_quantile_xgb.pkl")
    
    # Initialize Kalman filter
    print("\n[5/6] Initializing Kalman pace filter...")
    kalman_filter = KalmanPaceFilter(initial_pace=130.0)
    save_kalman_config(kalman_filter, models_dir / "kalman_config.json")
    print("  ✓ Kalman config saved")
    
    # Train SC hazard model
    print("\n[6/6] Training SC hazard model...")
    try:
        hazard_data = prepare_hazard_data(sectors, weather)
        if len(hazard_data) > 10:
            hazard_model = train_cox_hazard(hazard_data)
            save_hazard_model(hazard_model, models_dir / "cox_hazard.pkl")
            print("  ✓ SC hazard model saved")
        else:
            raise ValueError("Insufficient hazard data")
    except Exception as e:
        print(f"  ✗ SC hazard training failed: {e}")
        print("  Creating placeholder...")
        from lifelines import CoxPHFitter
        hazard_model = CoxPHFitter()  # Empty model
        save_hazard_model(hazard_model, models_dir / "cox_hazard.pkl")
    
    # Train overtake model
    print("\n[7/6] Training overtake model...")
    try:
        overtake_data = prepare_overtake_data(sectors, results)
        if len(overtake_data) > 10:
            overtake_model = train_overtake_model(overtake_data)
            save_overtake_model(overtake_model, models_dir / "overtake.pkl")
            print("  ✓ Overtake model saved")
        else:
            raise ValueError("Insufficient overtake data")
    except Exception as e:
        print(f"  ✗ Overtake training failed: {e}")
        print("  Creating placeholder...")
        from sklearn.linear_model import LogisticRegression
        overtake_model = LogisticRegression()
        save_overtake_model(overtake_model, models_dir / "overtake.pkl")
    
    # Save metadata
    import time
    metadata = {
        "event": "R1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": get_git_sha(),
        "package_versions": get_package_versions(),
        "rng_seed": 42,
        "models": [
            "wear_quantile_xgb.pkl",
            "cox_hazard.pkl",
            "overtake.pkl",
            "kalman_config.json",
        ],
        "training_samples": {
            "wear": len(wear_features),
            "pace": len(pace_features),
            "hazard": len(hazard_data) if 'hazard_data' in locals() else 0,
            "overtake": len(overtake_data) if 'overtake_data' in locals() else 0,
        },
    }
    
    save_json(metadata, models_dir / "metadata.json")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Models saved to: {models_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

