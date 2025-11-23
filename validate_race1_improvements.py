#!/usr/bin/env python3
"""Validate Race 1 improvements - test improved strategy on Race 1 data."""
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import models directly (avoid problematic evaluation module and lazy loading)
print("Loading models...", end=" ", flush=True)
try:
    # Import directly to avoid lazy loading timeout
    from src.grcup.models.wear_quantile_xgb import load_model
    from src.grcup.models.sc_hazard import load_hazard_model
    # Skip overtake_model for now (causes timeout)
    # from src.grcup.models.overtake_model import load_overtake_model
    load_overtake_model = None  # Will be set to None
    from src.grcup.loaders import (
        load_lap_times,
        load_lap_starts,
        load_lap_ends,
        load_results,
        load_sectors,
        load_weather,
        build_lap_table,
    )
    from src.grcup.features import detect_stints
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Import from new location (workaround for filesystem issue)
print("Loading validation functions...", end=" ", flush=True)
try:
    # Use new evaluation module location (workaround for corrupted files)
    import sys
    sys.path.insert(0, 'src/grcup/evaluation_new')
    from src.grcup.evaluation_new import walkforward_validate, save_walkforward_results
    print("✓")
except Exception as e:
    print(f"⚠ Warning: {e}")
    walkforward_validate = None
    save_walkforward_results = None

# Import strategy solver (use deterministic optimizer as fallback)
print("Loading strategy solver...", end=" ", flush=True)
try:
    from src.grcup.strategy.optimizer import solve_pit_strategy
    print("✓")
except Exception as e:
    # Fallback to deterministic optimizer (simpler, fewer dependencies)
    try:
        from src.grcup.strategy.deterministic_optimizer import solve_pit_strategy_deterministic
        # Wrap it to match the expected API
        def solve_pit_strategy(*args, **kwargs):
            return solve_pit_strategy_deterministic(*args, **kwargs)
        print("✓ (using deterministic optimizer)")
    except Exception as e2:
        print(f"✗ Error: {e2}")
        print("  Note: Filesystem timeout issue - files exist but can't be read")
        print("  Try: diskutil verifyVolume / or copy project to new location")
        sys.exit(1)


def main():
    """Validate Race 1 improvements."""
    base_dir = Path(__file__).parent
    race_dir = base_dir / "Race 1"
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports" / "race1_validation"
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("Race 1 Improvements Validation")
    print("=" * 70)
    
    # Load models
    print("\n[1/5] Loading trained models...")
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
    if load_overtake_model is None:
        print("⚠ Skipped (import timeout)")
        models_dict["overtake"] = None
    else:
        try:
            overtake_model = load_overtake_model(models_dir / "overtake.pkl")
            models_dict["overtake"] = overtake_model
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            models_dict["overtake"] = None
    
    # Load Race 1 data
    print("\n[2/5] Loading Race 1 data...")
    print("  Loading lap timing files...", end=" ")
    try:
        race1_laps_raw = load_lap_times(race_dir / "vir_lap_time_R1.csv")
        race1_starts = load_lap_starts(race_dir / "vir_lap_start_R1.csv")
        race1_ends = load_lap_ends(race_dir / "vir_lap_end_R1.csv")
        race1_laps = build_lap_table(race1_laps_raw, race1_starts, race1_ends)
        print(f"✓ ({len(race1_laps)} laps, {race1_laps['vehicle_id'].nunique()} vehicles)")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("  Loading sectors...", end=" ")
    try:
        race1_sectors = load_sectors(race_dir / "23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV")
        print(f"✓ ({len(race1_sectors)} records)")
    except Exception as e:
        print(f"✗ Error: {e}")
        race1_sectors = pd.DataFrame()
    
    print("  Loading weather...", end=" ")
    try:
        race1_weather = load_weather(race_dir / "26_Weather_Race 1_Anonymized.CSV")
        print(f"✓ ({len(race1_weather)} records)")
    except Exception as e:
        print(f"✗ Error: {e}")
        race1_weather = pd.DataFrame()
    
    print("  Loading results...", end=" ")
    try:
        race1_results = load_results(race_dir / "03_Provisional Results_Race 1_Anonymized.CSV")
        print(f"✓ ({len(race1_results)} entries)")
    except:
        try:
            race1_results = load_results(race_dir / "05_Results by Class GR Cup Race 1 Official_Anonymized.CSV")
            print(f"✓ ({len(race1_results)} entries, official)")
        except Exception as e:
            print(f"✗ Error: {e}")
            race1_results = pd.DataFrame()
    
    # Load telemetry features if available
    print("  Checking for telemetry features...", end=" ")
    race1_telemetry = None
    telemetry_path = race_dir / "R1_telemetry_features.csv"
    if telemetry_path.exists():
        try:
            race1_telemetry = pd.read_csv(telemetry_path)
            print(f"✓ ({len(race1_telemetry)} records)")
        except Exception as e:
            print(f"⚠ Warning: {e}")
    else:
        print("Not found (optional)")
    
    # Run walk-forward validation on Race 1
    print("\n[3/5] Running walk-forward validation on Race 1...")
    
    def strategy_solver(*args, **kwargs):
        """Wrapper for strategy optimizer."""
        return solve_pit_strategy(*args, **kwargs)
    
    if walkforward_validate is None:
        print("  ⚠ Warning: walkforward_validate not available (import timeout)")
        print("  Skipping full validation - will compute basic metrics only")
        walkforward_results = {"recommendations": [], "metrics": {}}
    else:
        print("  Processing recommendations...", end=" ", flush=True)
        try:
            walkforward_results = walkforward_validate(
                race1_laps,
                race1_sectors,
                race1_results,
                models_dict,
                strategy_solver,
                n_jobs=1,  # Single-threaded
            )
            n_recs = len(walkforward_results.get('recommendations', []))
            mean_conf = walkforward_results.get('metrics', {}).get('mean_confidence', 0.0)
            print(f"✓ ({n_recs} recommendations, avg confidence: {mean_conf:.2f})")
            
            # Save intermediate results
            save_walkforward_results(walkforward_results, reports_dir / "walkforward_detailed.json")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            walkforward_results = {"recommendations": [], "metrics": {}}
    
    # Compute basic statistics
    print("\n[4/5] Computing statistics...")
    recommendations = walkforward_results.get("recommendations", [])
    
    if len(recommendations) > 0:
        # Count pit recommendations
        pit_recommendations = sum(1 for r in recommendations if r.get("recommended_pit_lap") is not None)
        no_pit_recommendations = len(recommendations) - pit_recommendations
        
        # Average confidence
        confidences = [r.get("confidence", 0.0) for r in recommendations]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Expected time improvements
        expected_times = [r.get("expected_time", 0.0) for r in recommendations if r.get("expected_time") is not None]
        avg_expected_time = np.mean(expected_times) if expected_times else 0.0
        
        print(f"  Total recommendations: {len(recommendations)}")
        print(f"  Pit recommendations: {pit_recommendations}")
        print(f"  No-pit recommendations: {no_pit_recommendations}")
        print(f"  Average confidence: {avg_confidence:.2%}")
        print(f"  Average expected time: {avg_expected_time:.2f}s")
    else:
        print("  ⚠ No recommendations generated")
    
    # Compare to actual Race 1 outcomes
    print("\n[5/5] Comparing to actual Race 1 outcomes...")
    
    # Detect actual stints for each vehicle
    vehicle_actual_pits = {}
    for vehicle_id in race1_laps["vehicle_id"].unique():
        try:
            stints = detect_stints(race1_laps, vehicle_id)
            # Extract pit laps (start of each stint after first)
            pit_laps = [s["start_lap"] for s in stints[1:]] if len(stints) > 1 else []
            vehicle_actual_pits[vehicle_id] = pit_laps
        except Exception:
            vehicle_actual_pits[vehicle_id] = []
    
    # Compare AI recommendations to actual
    matches = 0
    total_comparisons = 0
    
    for rec in recommendations:
        vehicle_id = rec.get("vehicle_id")
        recommended_pit = rec.get("recommended_pit_lap")
        current_lap = rec.get("lap")
        
        if vehicle_id not in vehicle_actual_pits:
            continue
        
        actual_pits = vehicle_actual_pits[vehicle_id]
        # Check if recommendation matches actual (within 1 lap tolerance)
        if recommended_pit is None:
            # AI says no pit - check if actual had no pit near this lap
            nearby_pits = [p for p in actual_pits if abs(p - current_lap) <= 2]
            if len(nearby_pits) == 0:
                matches += 1
        else:
            # AI says pit - check if actual pit near recommended lap
            nearby_pits = [p for p in actual_pits if abs(p - recommended_pit) <= 1]
            if len(nearby_pits) > 0:
                matches += 1
        
        total_comparisons += 1
    
    if total_comparisons > 0:
        match_rate = matches / total_comparisons
        print(f"  Recommendations matching actual: {matches}/{total_comparisons} ({match_rate:.1%})")
    else:
        print("  ⚠ No comparisons possible")
    
    # Save validation report
    validation_report = {
        "event": "R1",
        "validation_type": "improvements_test",
        "walkforward": {
            "total_recommendations": len(recommendations),
            **walkforward_results.get("metrics", {}),
        },
        "statistics": {
            "pit_recommendations": pit_recommendations if len(recommendations) > 0 else 0,
            "no_pit_recommendations": no_pit_recommendations if len(recommendations) > 0 else 0,
            "average_confidence": float(avg_confidence) if len(recommendations) > 0 else 0.0,
            "average_expected_time": float(avg_expected_time) if len(recommendations) > 0 else 0.0,
            "match_rate": float(match_rate) if total_comparisons > 0 else 0.0,
            "matches": matches,
            "total_comparisons": total_comparisons,
        },
        "data_summary": {
            "total_laps": len(race1_laps),
            "total_vehicles": race1_laps["vehicle_id"].nunique(),
            "total_sectors": len(race1_sectors),
            "total_weather_records": len(race1_weather),
        },
    }
    
    report_path = reports_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    
    print(f"\n✓ Validation report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("Race 1 Validation Complete!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Recommendations: {len(recommendations)}")
    if total_comparisons > 0:
        print(f"  Match rate: {match_rate:.1%}")
    print(f"  Average confidence: {avg_confidence:.2%}")
    print(f"\nReports saved to: {reports_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

