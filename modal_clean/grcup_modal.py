"""Modal functions for GR Cup strategy engine.

Usage:
    # Train models on Modal
    modal run modal.py::train_models

    # Validate on Race 2
    modal run modal.py::validate_walkforward --scenario base

    # Run all scenarios in parallel
    modal run modal.py::validate_all_scenarios
"""
import modal
import os
from pathlib import Path

# Define the image with all dependencies
# Mount only necessary directories, excluding .venv, .DS_Store, and other build artifacts
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    # Mount only what we need, excluding .venv and system files
    # .add_local_dir("src", remote_path="/workspace/src")
    # .add_local_dir("notebooks", remote_path="/workspace/notebooks")
    # Mount data directories (they seem clean, but let's be explicit to avoid hidden file issues)
    # Manually mount Race files to avoid uploading 1.4GB telemetry files
    # Race 1
    # .add_local_file("Race 1/03_Provisional Results_Race 1_Anonymized.CSV", remote_path="/workspace/Race 1/03_Provisional Results_Race 1_Anonymized.CSV")
    # .add_local_file("Race 1/23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV", remote_path="/workspace/Race 1/23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV")
    # .add_local_file("Race 1/26_Weather_Race 1_Anonymized.CSV", remote_path="/workspace/Race 1/26_Weather_Race 1_Anonymized.CSV")
    # .add_local_file("Race 1/vir_lap_end_R1.csv", remote_path="/workspace/Race 1/vir_lap_end_R1.csv")
    # .add_local_file("Race 1/vir_lap_start_R1.csv", remote_path="/workspace/Race 1/vir_lap_start_R1.csv")
    # .add_local_file("Race 1/vir_lap_time_R1.csv", remote_path="/workspace/Race 1/vir_lap_time_R1.csv")
    # .add_local_file("Race 1/R1_telemetry_features.csv", remote_path="/workspace/Race 1/R1_telemetry_features.csv")
    
    # Race 2
    # .add_local_file("Race 2/03_Results GR Cup Race 2 Official_Anonymized.CSV", remote_path="/workspace/Race 2/03_Results GR Cup Race 2 Official_Anonymized.CSV")
    # .add_local_file("Race 2/03_Provisional Results_Race 2_Anonymized.CSV", remote_path="/workspace/Race 2/03_Provisional Results_Race 2_Anonymized.CSV")
    # .add_local_file("Race 2/23_AnalysisEnduranceWithSections_Race 2_Anonymized.CSV", remote_path="/workspace/Race 2/23_AnalysisEnduranceWithSections_Race 2_Anonymized.CSV")
    # .add_local_file("Race 2/26_Weather_Race 2_Anonymized.CSV", remote_path="/workspace/Race 2/26_Weather_Race 2_Anonymized.CSV")
    # .add_local_file("Race 2/vir_lap_end_R2.csv", remote_path="/workspace/Race 2/vir_lap_end_R2.csv")
    # .add_local_file("Race 2/vir_lap_start_R2.csv", remote_path="/workspace/Race 2/vir_lap_start_R2.csv")
    # .add_local_file("Race 2/vir_lap_time_R2.csv", remote_path="/workspace/Race 2/vir_lap_time_R2.csv")
    # .add_local_file("Race 2/R2_telemetry_features.csv", remote_path="/workspace/Race 2/R2_telemetry_features.csv")
    # .add_local_dir("scripts", remote_path="/workspace/scripts")
    .add_local_file("requirements.txt", remote_path="/workspace/requirements.txt")
    # .add_local_file("app.py", remote_path="/workspace/app.py")
    # .add_local_dir("templates", remote_path="/workspace/templates")
)

# Create volumes to persist models and reports
models_volume = modal.Volume.from_name("grcup-models", create_if_missing=True)
reports_volume = modal.Volume.from_name("grcup-reports", create_if_missing=True)

app = modal.App("grcup-strategy")


@app.function(
    image=image,
    volumes={
        "/models": models_volume,
        "/reports": reports_volume,
    },
    timeout=3600,  # 1 hour timeout
    cpu=4.0,  # 4 CPU cores
    memory=8192,  # 8GB RAM
)
def train_models(event: str = "R1", cqr_alpha: float = 0.10, cqr_scale: float = 0.90):
    """Train all models on Race data."""
    import sys
    from pathlib import Path
    
    base_dir = Path("/workspace")
    os.chdir(base_dir)
    
    # Set environment variables
    os.environ["CQR_ALPHA"] = str(cqr_alpha)
    os.environ["CQR_SCALE"] = str(cqr_scale)
    
    # Import and run training
    sys.path.insert(0, str(base_dir / "notebooks"))
    from train_models import main as train_main
    
    # Override models_dir to use mounted volume
    import train_models as tm
    tm.models_dir = Path("/models")
    
    try:
        train_main()
        # Sync to volume
        models_volume.commit()
        return {"status": "success", "models_dir": "/models"}
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


@app.function(
    image=image,
    volumes={
        "/models": models_volume,
        "/reports": reports_volume,
    },
    timeout=7200,  # 2 hour timeout
    cpu=4.0,
    memory=8192,
)
def validate_walkforward(
    event: str = "R2",
    scenario: str = "base",
    cqr_scale: float = 2.2,
    cqr_band_scale: float = 1.5,
):
    """Run walk-forward validation on Race 2."""
    import sys
    from pathlib import Path
    
    base_dir = Path("/workspace")
    os.chdir(base_dir)
    
    # Set environment variables
    os.environ["CQR_SCALE"] = str(cqr_scale)
    os.environ["CQR_BAND_SCALE"] = str(cqr_band_scale)
    
    # Import and run validation
    sys.path.insert(0, str(base_dir / "notebooks"))
    from validate_walkforward import main as validate_main
    
    # Override reports_dir to use mounted volume
    import validate_walkforward as vw
    vw.reports_dir = Path(f"/reports/{scenario}")
    vw.reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Override models_dir to use mounted volume
    vw.models_dir = Path("/models")
    
    # Set scenario
    try:
        setattr(vw, "scenario", scenario)
    except:
        pass
    
    try:
        validate_main()
        # Sync to volume
        reports_volume.commit()
        return {"status": "success", "scenario": scenario, "reports_dir": f"/reports/{scenario}"}
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


@app.function(
    image=image,
    volumes={
        "/models": models_volume,
        "/reports": reports_volume,
    },
    timeout=14400,  # 4 hour timeout for parallel runs
    cpu=8.0,
    memory=16384,
)
def validate_all_scenarios(
    event: str = "R2",
    cqr_scale: float = 2.2,
    cqr_band_scale: float = 1.5,
):
    """Run all scenarios in parallel."""
    scenarios = [
        "base",
        "early_sc",
        "late_sc",
        "hot_track",
        "heavy_traffic",
        "undercut",
        "no_weather",
    ]
    
    # Run all scenarios in parallel using Modal's map
    results = {}
    for scenario in scenarios:
        result = validate_walkforward.spawn(
            event=event,
            scenario=scenario,
            cqr_scale=cqr_scale,
            cqr_band_scale=cqr_band_scale,
        )
        results[scenario] = result
    
    # Wait for all to complete
    for scenario in results:
        results[scenario] = results[scenario].get()
    
    return {"status": "complete", "scenarios": results}


@app.function(
    image=image,
    volumes={
        "/reports": reports_volume,
    },
    timeout=3600,
    cpu=2.0,
    memory=4096,
)
@modal.wsgi_app()
def serve_dashboard():
    """Serve the Flask dashboard on Modal."""
    import sys
    from pathlib import Path
    
    base_dir = Path("/workspace")
    os.chdir(base_dir)
    
    sys.path.insert(0, str(base_dir))
    from app import app as flask_app
    
    # Override reports_dir to use mounted volume
    flask_app.config["REPORTS_DIR"] = Path("/reports")
    
    return flask_app


@app.local_entrypoint()
def main(
    command: str = "train",
    event: str = "R1",
    scenario: str = "base",
    cqr_alpha: float = 0.10,
    cqr_scale: float = 2.2,
    cqr_band_scale: float = 1.5,
):
    """CLI entrypoint for Modal functions."""
    if command == "train":
        result = train_models.remote(
            event=event,
            cqr_alpha=cqr_alpha,
            cqr_scale=cqr_scale,
        )
        print(f"Training result: {result}")
    
    elif command == "validate":
        result = validate_walkforward.remote(
            event=event,
            scenario=scenario,
            cqr_scale=cqr_scale,
            cqr_band_scale=cqr_band_scale,
        )
        print(f"Validation result: {result}")
    
    elif command == "validate-all":
        result = validate_all_scenarios.remote(
            event=event,
            cqr_scale=cqr_scale,
            cqr_band_scale=cqr_band_scale,
        )
        print(f"All scenarios result: {result}")
    
    else:
        print(f"Unknown command: {command}")

