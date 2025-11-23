#!/usr/bin/env python3
"""Run baseline tests using subprocess to avoid import timeout."""
import os
import subprocess
import sys
from pathlib import Path

def main():
    """Run baseline comparison via subprocess."""
    # Set environment variables for baseline simulation counts
    if "BASELINE_BASE_SCENARIOS" not in os.environ:
        os.environ["BASELINE_BASE_SCENARIOS"] = "500"
    if "BASELINE_REFINE_SCENARIOS" not in os.environ:
        os.environ["BASELINE_REFINE_SCENARIOS"] = "1000"
    
    # Run Python directly on the validation script
    # But we need to modify it to only run baselines...
    # Actually, simpler: just run the existing script with SKIP flags
    
    print("Running baseline comparisons...")
    print("Note: This uses subprocess to avoid import timeout issues")
    
    # Actually, the simplest: just import and run directly but catch timeout
    import json
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from typing import Optional, Sequence, Any, Union
    
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Import what we can
    from src.grcup.models import load_model, load_hazard_model, load_overtake_model
    from src.grcup.loaders import load_lap_times, load_lap_starts, load_lap_ends, load_results, build_lap_table
    from src.grcup.features import detect_stints
    
    # Now try to import compute_baseline_comparisons with timeout handling
    print("Loading baseline comparison function...")
    try:
        # Use a signal-based timeout (Unix only)
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Import timed out")
        
        # Set 30 second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            from notebooks.validate_walkforward import compute_baseline_comparisons
            signal.alarm(0)  # Cancel timeout
            print("✓ Function loaded successfully")
        except TimeoutError:
            signal.alarm(0)
            raise
    except (ImportError, TimeoutError) as e:
        print(f"✗ Could not load function: {e}")
        print("\nTrying alternative approach: running via subprocess...")
        
        # Fallback: run as subprocess
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
from notebooks.validate_walkforward import compute_baseline_comparisons
# ... rest of code
            """],
            timeout=300,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Subprocess failed: {result.stderr}")
            return
        
        print(result.stdout)
        return
    
    # Continue with normal execution...
    base_dir = Path(__file__).parent
    reports_dir = base_dir / "reports"
    models_dir = base_dir / "models"
    race_dir = base_dir / "Race 2"
    
    # Load walkforward results
    walkforward_path = reports_dir / "walkforward_detailed.json"
    if not walkforward_path.exists():
        walkforward_path = reports_dir / "test" / "walkforward_detailed.json"
    
    if not walkforward_path.exists():
        print(f"✗ Error: Could not find walkforward results")
        return
    
    with open(walkforward_path) as f:
        walkforward_results = json.load(f)
    
    recommendations = walkforward_results.get("recommendations", [])
    print(f"✓ Loaded {len(recommendations)} recommendations")
    
    # Load models
    models_dict = {
        "wear": load_model(models_dir / "wear_quantile_xgb.pkl"),
        "hazard": load_hazard_model(models_dir / "cox_hazard.pkl"),
        "overtake": load_overtake_model(models_dir / "overtake.pkl"),
    }
    
    # Load race data
    race2_laps_raw = load_lap_times(race_dir / "vir_lap_time_R2.csv")
    race2_starts = load_lap_starts(race_dir / "vir_lap_start_R2.csv")
    race2_ends = load_lap_ends(race_dir / "vir_lap_end_R2.csv")
    race2_laps = build_lap_table(race2_laps_raw, race2_starts, race2_ends)
    
    try:
        race2_results = load_results(race_dir / "03_Results GR Cup Race 2 Official_Anonymized.CSV")
    except:
        race2_results = load_results(race_dir / "03_Provisional Results_Race 2_Anonymized.CSV")
    
    # Run baseline comparisons
    print("\nRunning baseline comparisons...")
    baseline_comps = compute_baseline_comparisons(
        race2_laps,
        race2_results,
        recommendations_log=recommendations,
        models=models_dict,
    )
    
    # Save results
    output_path = reports_dir / "baseline_comparisons_only.json"
    with open(output_path, 'w') as f:
        json.dump(baseline_comps, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Display summary
    engine_advantage = baseline_comps.get("engine_advantage", {})
    for name, stats in engine_advantage.items():
        print(f"{name}: {stats.get('time_saved_s', 0):.2f}s")


if __name__ == "__main__":
    main()

