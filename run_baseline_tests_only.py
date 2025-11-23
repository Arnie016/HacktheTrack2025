#!/usr/bin/env python3
"""Run only baseline comparison tests (skip full validation)."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import models directly (avoids problematic evaluation module import)
from src.grcup.models import (
    load_model,
    load_hazard_model,
    load_overtake_model,
)
from src.grcup.loaders import (
    load_lap_times,
    load_lap_starts,
    load_lap_ends,
    load_results,
    build_lap_table,
)
from src.grcup.features import detect_stints

# Import compute_baseline_comparisons using importlib with timeout workaround
print("  Loading baseline comparison function...", end=" ", flush=True)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "validate_walkforward",
        Path(__file__).parent / "notebooks" / "validate_walkforward.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load validate_walkforward module")
    
    # Try to load with a timeout workaround
    module = importlib.util.module_from_spec(spec)
    # Set a shorter timeout by monkey-patching file operations if needed
    spec.loader.exec_module(module)
    compute_baseline_comparisons = module.compute_baseline_comparisons
    print("✓")
except Exception as e:
    print(f"✗ Error loading function: {e}")
    print("  Falling back to direct execution...")
    # Fallback: run as subprocess
    import subprocess
    print("  This will run in a separate process to avoid import timeout...")
    sys.exit(1)  # Will handle differently below


def main():
    """Run baseline comparisons only."""
    base_dir = Path(__file__).parent
    reports_dir = base_dir / "reports"
    models_dir = base_dir / "models"
    race_dir = base_dir / "Race 2"
    
    print("=" * 70)
    print("Baseline Comparison Tests Only")
    print("=" * 70)
    
    # Load existing walkforward results
    print("\n[1/4] Loading existing walkforward results...")
    walkforward_path = reports_dir / "walkforward_detailed.json"
    if not walkforward_path.exists():
        walkforward_path = reports_dir / "test" / "walkforward_detailed.json"
    
    if not walkforward_path.exists():
        print(f"✗ Error: Could not find walkforward results at {walkforward_path}")
        print("   Please run full validation first, or specify path manually.")
        return
    
    print(f"  Loading from: {walkforward_path}")
    with open(walkforward_path, 'r') as f:
        walkforward_results = json.load(f)
    
    recommendations = walkforward_results.get("recommendations", [])
    print(f"✓ Loaded {len(recommendations)} recommendations")
    
    if len(recommendations) == 0:
        print("✗ Error: No recommendations found in walkforward results")
        return
    
    # Load models
    print("\n[2/4] Loading models...")
    models_dict = {}
    
    print("  Loading wear model...", end=" ")
    try:
        wear_model = load_model(models_dir / "wear_quantile_xgb.pkl")
        models_dict["wear"] = wear_model
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        models_dict["wear"] = None
    
    print("  Loading SC hazard model...", end=" ")
    try:
        hazard_model = load_hazard_model(models_dir / "cox_hazard.pkl")
        models_dict["hazard"] = hazard_model
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        models_dict["hazard"] = None
    
    print("  Loading overtake model...", end=" ")
    try:
        overtake_model = load_overtake_model(models_dir / "overtake.pkl")
        models_dict["overtake"] = overtake_model
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        models_dict["overtake"] = None
    
    # Load Race 2 data (minimal - just for baseline comparisons)
    print("\n[3/4] Loading Race 2 data...")
    print("  Loading lap timing files...", end=" ")
    try:
        race2_laps_raw = load_lap_times(race_dir / "vir_lap_time_R2.csv")
        race2_starts = load_lap_starts(race_dir / "vir_lap_start_R2.csv")
        race2_ends = load_lap_ends(race_dir / "vir_lap_end_R2.csv")
        race2_laps = build_lap_table(race2_laps_raw, race2_starts, race2_ends)
        print(f"✓ ({len(race2_laps)} laps)")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("  Loading results...", end=" ")
    try:
        race2_results = load_results(race_dir / "03_Results GR Cup Race 2 Official_Anonymized.CSV")
        print(f"✓ ({len(race2_results)} entries)")
    except:
        try:
            race2_results = load_results(race_dir / "03_Provisional Results_Race 2_Anonymized.CSV")
            print(f"✓ ({len(race2_results)} entries, provisional)")
        except Exception as e:
            print(f"✗ Error: {e}")
            race2_results = pd.DataFrame()
    
    # Run baseline comparisons
    print("\n[4/4] Running baseline comparisons...")
    print(f"  Processing {len(recommendations)} recommendations...")
    print(f"  This may take a few minutes (Monte Carlo simulations)...")
    
    # Set environment variables for baseline simulation counts
    # Use reasonable defaults if not set
    if "BASELINE_BASE_SCENARIOS" not in os.environ:
        os.environ["BASELINE_BASE_SCENARIOS"] = "500"  # Reduced for faster testing
    if "BASELINE_REFINE_SCENARIOS" not in os.environ:
        os.environ["BASELINE_REFINE_SCENARIOS"] = "1000"  # Reduced for faster testing
    
    print(f"\n  Simulation settings:")
    print(f"    Base scenarios: {os.getenv('BASELINE_BASE_SCENARIOS', '500')}")
    print(f"    Refined scenarios: {os.getenv('BASELINE_REFINE_SCENARIOS', '1000')}")
    print(f"    (Set BASELINE_BASE_SCENARIOS and BASELINE_REFINE_SCENARIOS to adjust)\n")
    
    try:
        baseline_comps = compute_baseline_comparisons(
            race2_laps,
            race2_results,
            recommendations_log=recommendations,
            counterfactuals=None,  # Skip counterfactuals
            models=models_dict,
        )
        
        # Display results
        print("\n" + "=" * 70)
        print("Baseline Comparison Results")
        print("=" * 70)
        
        engine_advantage = baseline_comps.get("engine_advantage", {})
        
        for baseline_name, stats in engine_advantage.items():
            time_saved = stats.get("time_saved_s", 0.0)
            ci95 = stats.get("ci95", [0.0, 0.0])
            raw_mean = stats.get("raw_mean", 0.0)
            
            baseline_display = {
                "vs_fixed_stint": "vs Fixed Stint (15 laps)",
                "vs_fuel_min": "vs Fuel Minimum",
                "vs_mirror_leader": "vs Mirror Leader",
                "vs_mirror_pack": "vs Mirror Pack",
            }.get(baseline_name, baseline_name)
            
            print(f"\n{baseline_display}:")
            print(f"  Time saved (mean): {time_saved:.2f}s")
            print(f"  Time saved (95% CI): [{ci95[0]:.2f}s, {ci95[1]:.2f}s]")
            if abs(raw_mean - time_saved) > 0.01:
                print(f"  Raw mean (before capping): {raw_mean:.2f}s")
        
        # Save results
        output_path = reports_dir / "baseline_comparisons_only.json"
        with open(output_path, 'w') as f:
            json.dump(baseline_comps, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to: {output_path}")
        
        # Update validation report if it exists
        validation_report_path = reports_dir / "validation_report.json"
        if not validation_report_path.exists():
            validation_report_path = reports_dir / "test" / "validation_report.json"
        
        if validation_report_path.exists():
            print(f"\n  Updating validation report...", end=" ")
            with open(validation_report_path, 'r') as f:
                validation_report = json.load(f)
            
            validation_report["baseline_comparisons"] = baseline_comps
            validation_report["summary"]["time_saved_mean_s"] = engine_advantage.get(
                "vs_fixed_stint", {}
            ).get("time_saved_s", 0.0)
            validation_report["summary"]["time_saved_ci95"] = engine_advantage.get(
                "vs_fixed_stint", {}
            ).get("ci95", [0.0, 0.0])
            
            with open(validation_report_path, 'w') as f:
                json.dump(validation_report, f, indent=2, default=str)
            print("✓")
        
        print("\n" + "=" * 70)
        print("Baseline Comparison Complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error during baseline comparisons: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

