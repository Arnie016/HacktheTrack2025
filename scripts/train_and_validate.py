#!/usr/bin/env python3
"""One-shot training and validation pipeline.

Trains models on Race 1, then validates on Race 2 with walk-forward analysis.
Optionally runs all scenarios in parallel.

Usage:
    # Basic: train + validate base scenario
    python scripts/train_and_validate.py

    # Train + validate all scenarios
    python scripts/train_and_validate.py --all-scenarios

    # Custom CQR parameters
    python scripts/train_and_validate.py --cqr-alpha 0.10 --cqr-scale 2.2

    # Custom output directories
    python scripts/train_and_validate.py --models-dir models/ --reports-dir reports/
"""
import argparse
import sys
import time
from pathlib import Path

# Add notebooks to path
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))


def print_section(title: str, width: int = 70):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_step(step_num: int, total: int, description: str):
    """Print a step indicator."""
    print(f"\n[{step_num}/{total}] {description}...")


def main():
    parser = argparse.ArgumentParser(
        description="Train models on Race 1 and validate on Race 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python scripts/train_and_validate.py

  # All scenarios
  python scripts/train_and_validate.py --all-scenarios

  # Custom CQR calibration
  python scripts/train_and_validate.py --cqr-alpha 0.10 --cqr-scale 2.2

  # Custom directories
  python scripts/train_and_validate.py --models-dir custom_models/ --reports-dir custom_reports/
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
        help="CQR scale factor (default: 0.90)",
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
        help="Scenario to validate (default: base). Options: base, early_sc, late_sc, hot_track, heavy_traffic, undercut, no_weather",
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
    
    # Output directories
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory for trained models (default: models)",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory for validation reports (default: reports)",
    )
    
    # Options
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing models)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation (only train)",
    )
    parser.add_argument(
        "--parallel-scenarios",
        action="store_true",
        help="Run scenarios in parallel (requires Modal or multiprocessing)",
    )
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    print_section("GR Cup Training & Validation Pipeline")
    print(f"Training event: {args.train_event}")
    print(f"Validation event: {args.validate_event}")
    print(f"Models directory: {args.models_dir}")
    print(f"Reports directory: {args.reports_dir}")
    
    # Step 1: Training
    if not args.skip_training:
        print_step(1, 3, "Training models")
        try:
            from train_models import main as train_main
            import train_models
            
            # Override models directory
            train_models.models_dir = Path(args.models_dir)
            
            # Override CQR parameters via environment
            import os
            os.environ["CQR_ALPHA"] = str(args.cqr_alpha)
            os.environ["CQR_SCALE"] = str(args.cqr_scale)
            
            # Set argparse args for train_models by modifying sys.argv
            import sys as sys_module
            old_argv = sys_module.argv.copy()
            sys_module.argv = [
                "train_models.py",
                "--cqr-alpha", str(args.cqr_alpha),
                "--cqr-scale", str(args.cqr_scale),
            ]
            
            try:
                train_main()
            finally:
                # Always restore argv
                sys_module.argv = old_argv
            
            print("\n✓ Training completed successfully")
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print_step(1, 3, "Training models")
        print("  Skipped (--skip-training)")
    
    # Step 2: Validation
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
        
        print_step(2, 3, f"Validating scenarios: {', '.join(scenarios_to_run)}")
        
        # Set environment variables for validation
        import os
        os.environ["CQR_SCALE"] = str(args.cqr_scale)
        os.environ["CQR_BAND_SCALE"] = str(args.cqr_band_scale)
        
        validation_results = {}
        
        for scenario in scenarios_to_run:
            print(f"\n  Running scenario: {scenario}")
            try:
                from validate_walkforward import main as validate_main
                import validate_walkforward
                
                # Override reports directory
                scenario_reports_dir = Path(args.reports_dir) / scenario
                validate_walkforward.reports_dir = scenario_reports_dir
                scenario_reports_dir.mkdir(parents=True, exist_ok=True)
                
                # Set scenario
                setattr(validate_walkforward, "scenario", scenario)
                
                # Run validation
                validate_main()
                
                # Load and report results
                validation_report_file = scenario_reports_dir / "validation_report.json"
                if validation_report_file.exists():
                    import json
                    with open(validation_report_file) as f:
                        report = json.load(f)
                    
                    summary = report.get("summary", {})
                    time_saved = summary.get("time_saved_mean_s", 0.0)
                    coverage = summary.get("quantile_coverage_90", 0.0)
                    
                    validation_results[scenario] = {
                        "time_saved": time_saved,
                        "coverage": coverage,
                        "status": "success",
                    }
                    
                    print(f"    ✓ {scenario}: Time saved={time_saved:.1f}s, Coverage={coverage:.1%}")
                else:
                    validation_results[scenario] = {
                        "status": "warning",
                        "message": "Report file not found",
                    }
                    print(f"    ⚠ {scenario}: Report file not found")
                    
            except Exception as e:
                validation_results[scenario] = {
                    "status": "error",
                    "error": str(e),
                }
                print(f"    ✗ {scenario}: Failed - {e}")
                if len(scenarios_to_run) == 1:
                    # If only one scenario, show full traceback
                    import traceback
                    traceback.print_exc()
        
        print("\n✓ Validation completed")
        
        # Summary
        print_section("Validation Summary")
        successful = sum(1 for r in validation_results.values() if r.get("status") == "success")
        print(f"Successful scenarios: {successful}/{len(scenarios_to_run)}")
        
        if successful > 0:
            print("\nResults:")
            for scenario, result in validation_results.items():
                if result.get("status") == "success":
                    print(f"  {scenario:15s}: Time saved={result['time_saved']:6.1f}s, Coverage={result['coverage']:5.1%}")
    else:
        print_step(2, 3, "Validation")
        print("  Skipped (--skip-validation)")
    
    # Step 3: Summary
    print_step(3, 3, "Pipeline Summary")
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Models saved to: {args.models_dir}")
    print(f"Reports saved to: {args.reports_dir}")
    
    print_section("Pipeline Complete")
    
    # Exit with error if validation failed
    if not args.skip_validation and args.all_scenarios:
        failed = sum(1 for r in validation_results.values() if r.get("status") == "error")
        if failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()

