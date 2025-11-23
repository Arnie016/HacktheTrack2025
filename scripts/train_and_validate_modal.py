#!/usr/bin/env python3
"""One-shot training and validation pipeline using Modal.

Trains models on Race 1, then validates on Race 2 with walk-forward analysis.
Runs everything on Modal cloud infrastructure.

Usage:
    # Basic: train + validate base scenario
    python scripts/train_and_validate_modal.py

    # Train + validate all scenarios
    python scripts/train_and_validate_modal.py --all-scenarios

    # Custom CQR parameters
    python scripts/train_and_validate_modal.py --cqr-alpha 0.10 --cqr-scale 2.2

    # Download results locally after completion
    python scripts/train_and_validate_modal.py --download-results
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def print_section(title: str, width: int = 70):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_step(step_num: int, total: int, description: str):
    """Print a step indicator."""
    print(f"\n[{step_num}/{total}] {description}...")


def run_modal_function(function_path: str, **kwargs) -> tuple[bool, str]:
    """Run a Modal function and return success status and output."""
    # Build command: modal run <file>::<function> --arg1 value1 --arg2 value2
    cmd = ["run", function_path]
    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    try:
        result = subprocess.run(
            ["python3", "-m", "modal"] + cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr or e.stdout


def download_volume(volume_name: str, remote_path: str, local_path: str) -> bool:
    """Download files from Modal volume to local directory."""
    print(f"  Downloading {volume_name}:{remote_path} -> {local_path}")
    try:
        # Ensure parent directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        result = subprocess.run(
            ["python3", "-m", "modal", "volume", "get", volume_name, remote_path, local_path, "--force"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  ✓ Downloaded to {local_path}")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout
        # If it's just a "file exists" error, that's okay
        if "already exists" in error_msg.lower():
            print(f"  ✓ Already exists at {local_path}")
            return True
        print(f"  ✗ Download failed: {error_msg}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train models on Race 1 and validate on Race 2 using Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run on Modal
  python scripts/train_and_validate_modal.py

  # All scenarios on Modal
  python scripts/train_and_validate_modal.py --all-scenarios

  # Custom CQR calibration
  python scripts/train_and_validate_modal.py --cqr-alpha 0.10 --cqr-scale 2.2

  # Download results after completion
  python scripts/train_and_validate_modal.py --download-results
        """,
    )
    
    # Training arguments
    parser.add_argument(
        "--train-event",
        default="R1",
        help="Race event for training (default: R1)",
    )
    parser.add_argument(
        "--cqr-alpha",
        type=float,
        default=0.10,
        help="CQR alpha (target miscoverage, default: 0.10)",
    )
    parser.add_argument(
        "--cqr-scale",
        type=float,
        default=0.90,
        help="CQR scale factor for training (default: 0.90)",
    )
    
    # Validation arguments
    parser.add_argument(
        "--validate-event",
        default="R2",
        help="Race event for validation (default: R2)",
    )
    parser.add_argument(
        "--scenario",
        default="base",
        help="Scenario to validate (default: base)",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run validation for all scenarios (overrides --scenario)",
    )
    parser.add_argument(
        "--cqr-band-scale",
        type=float,
        default=1.5,
        help="CQR band scale for validation (default: 1.5)",
    )
    
    # Options
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing models in Modal volume)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation (only train)",
    )
    parser.add_argument(
        "--download-results",
        action="store_true",
        help="Download models and reports from Modal volumes after completion",
    )
    parser.add_argument(
        "--local-models-dir",
        default="models",
        help="Local directory to download models (default: models)",
    )
    parser.add_argument(
        "--local-reports-dir",
        default="reports",
        help="Local directory to download reports (default: reports)",
    )
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    print_section("GR Cup Training & Validation Pipeline (Modal)")
    print(f"Training event: {args.train_event}")
    print(f"Validation event: {args.validate_event}")
    print(f"Models: Modal volume 'grcup-models'")
    print(f"Reports: Modal volume 'grcup-reports'")
    
    # Step 1: Training on Modal
    if not args.skip_training:
        print_step(1, 3, "Training models on Modal")
        print(f"  Running: python3 -m modal run grcup_modal.py::train_models")
        print(f"    event={args.train_event}")
        print(f"    cqr_alpha={args.cqr_alpha}")
        print(f"    cqr_scale={args.cqr_scale}")
        
        success, output = run_modal_function(
            "grcup_modal.py::train_models",
            event=args.train_event,
            cqr_alpha=args.cqr_alpha,
            cqr_scale=args.cqr_scale,
        )
        
        if success:
            print("  ✓ Training completed successfully on Modal")
            print(f"  View logs: https://modal.com/apps")
        else:
            print(f"  ✗ Training failed: {output}")
            sys.exit(1)
    else:
        print_step(1, 3, "Training models")
        print("  Skipped (--skip-training)")
    
    # Step 2: Validation on Modal
    if not args.skip_validation:
        scenarios_to_run = []
        
        if args.all_scenarios:
            scenarios_to_run = [
                "base",
                "early_sc",
                "late_sc",
                "hot_track",
                "heavy_traffic",
                "undercut",
                "no_weather",
            ]
        else:
            scenarios_to_run = [args.scenario]
        
        print_step(2, 3, f"Validating scenarios on Modal: {', '.join(scenarios_to_run)}")
        
        validation_results = {}
        
        for scenario in scenarios_to_run:
            print(f"\n  Running scenario: {scenario}")
            print(f"    python3 -m modal run grcup_modal.py::validate_walkforward")
            print(f"      event={args.validate_event}")
            print(f"      scenario={scenario}")
            print(f"      cqr_scale={args.cqr_scale}")
            print(f"      cqr_band_scale={args.cqr_band_scale}")
            
            success, output = run_modal_function(
                "grcup_modal.py::validate_walkforward",
                event=args.validate_event,
                scenario=scenario,
                cqr_scale=args.cqr_scale,
                cqr_band_scale=args.cqr_band_scale,
            )
            
            if success:
                validation_results[scenario] = {"status": "success"}
                print(f"    ✓ {scenario}: Completed successfully")
                print(f"    View logs: https://modal.com/apps")
            else:
                validation_results[scenario] = {"status": "error", "error": output}
                print(f"    ✗ {scenario}: Failed - {output}")
                if len(scenarios_to_run) == 1:
                    # If only one scenario, exit on error
                    sys.exit(1)
        
        print("\n✓ Validation completed on Modal")
        
        # Summary
        print_section("Validation Summary")
        successful = sum(1 for r in validation_results.values() if r.get("status") == "success")
        print(f"Successful scenarios: {successful}/{len(scenarios_to_run)}")
        
        if successful < len(scenarios_to_run):
            failed = [s for s, r in validation_results.items() if r.get("status") == "error"]
            print(f"Failed scenarios: {', '.join(failed)}")
    else:
        print_step(2, 3, "Validation")
        print("  Skipped (--skip-validation)")
    
    # Step 3: Download results (optional)
    if args.download_results:
        print_step(3, 3, "Downloading results from Modal")
        
        # Download models
        if not args.skip_training:
            print("\n  Downloading models...")
            # Models are stored at root level in volume
            download_volume("grcup-models", "/", args.local_models_dir)
        
        # Download reports
        if not args.skip_validation:
            print("\n  Downloading reports...")
            for scenario in scenarios_to_run:
                # Scenarios are at root level, not under /reports/
                remote_path = scenario
                local_path = str(Path(args.local_reports_dir) / scenario)
                download_volume("grcup-reports", remote_path, local_path)
        
        print("\n✓ Download completed")
    else:
        print_step(3, 3, "Download results")
        print("  Skipped (use --download-results to download from Modal volumes)")
        print(f"  Models available in Modal volume: grcup-models")
        print(f"  Reports available in Modal volume: grcup-reports")
        print(f"  Download manually: python3 -m modal volume get <volume> <remote_path> <local_path>")
    
    # Summary
    elapsed_time = time.time() - start_time
    print_section("Pipeline Complete")
    print(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Models: Modal volume 'grcup-models'")
    print(f"Reports: Modal volume 'grcup-reports'")
    if args.download_results:
        print(f"Local models: {args.local_models_dir}")
        print(f"Local reports: {args.local_reports_dir}")
    print(f"\nView runs: https://modal.com/apps")
    
    # Exit with error if validation failed
    if not args.skip_validation and args.all_scenarios:
        failed = sum(1 for r in validation_results.values() if r.get("status") == "error")
        if failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()

