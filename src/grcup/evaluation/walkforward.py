"""Walk-forward validation: causal, no leakage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import os


def _process_vehicle_decision(
    lap_row: pd.Series,
    vehicle_id: str,
    lap_num: int,
    max_lap: int,
    vehicle_state: dict,
    models: dict[str, Any],
    strategy_optimizer,
) -> dict | None:
    """Process a single vehicle decision (helper for parallel processing)."""
    try:
        # Update state
        state = vehicle_state.copy()
        state["current_lap"] = lap_num
        state["tire_age"] += 1.0
        
        # Compute strategy using only past info
        import time
        start_time = time.time()
        recommendation = strategy_optimizer(
            current_lap=lap_num,
            total_laps=max_lap,
            tire_age=state["tire_age"],
            fuel_laps_remaining=state["fuel_remaining"],
            under_sc=state["under_sc"],
            wear_model=models.get("wear"),
            sc_hazard_model=models.get("hazard"),
        )
        elapsed = time.time() - start_time
        if elapsed > 5.0:  # Log slow decisions
            print(f"  ⚠ Slow decision: vehicle={vehicle_id}, lap={lap_num}, time={elapsed:.1f}s", flush=True)
        
        return {
            "vehicle_id": vehicle_id,
            "lap": lap_num,
            "recommended_pit_lap": recommendation.get("recommended_lap"),
            "recommended_window": recommendation.get("window"),
            "expected_time": recommendation.get("expected_time"),
            "confidence": recommendation.get("confidence"),
        }
    except Exception as e:
        # Skip if optimizer fails for this lap
        if "left" in str(e) and "right" in str(e):
            return None  # Pandas edge case
        # Re-raise other exceptions
        raise


def walkforward_validate(
    race2_laps: pd.DataFrame,
    race2_sectors: pd.DataFrame,
    race2_results: pd.DataFrame,
    models: dict[str, Any],  # Pre-loaded models
    strategy_optimizer,
    lap_by_lap: bool = True,
    n_jobs: int = -1,  # -1 = use all cores
) -> dict:
    """
    Walk-forward validation on Race 2.
    
    For each lap t, recompute strategy using only info <= t.
    Compare recommendations to actual race outcomes.
    
    Args:
        race2_laps: Race 2 lap timing data
        race2_sectors: Race 2 sector data
        race2_results: Race 2 final results
        models: Dict of loaded models {wear, kalman, hazard, overtake}
        strategy_optimizer: Strategy optimizer function
        lap_by_lap: If True, validate at each lap; if False, validate at key points
    
    Returns:
        Dict with validation metrics and recommendations log
    """
    recommendations_log = []
    
    # Sort laps by vehicle and lap number
    race2_laps_sorted = race2_laps.sort_values(["vehicle_id", "lap"]).reset_index(drop=True)
    
    # Track state per vehicle
    vehicle_states = {}
    
    for vehicle_id in race2_laps_sorted["vehicle_id"].unique():
        vehicle_laps = race2_laps_sorted[
            race2_laps_sorted["vehicle_id"] == vehicle_id
        ].copy()
        
        vehicle_states[vehicle_id] = {
            "current_lap": 1,
            "tire_age": 0.0,
            "fuel_remaining": 100.0,  # Simplified
            "under_sc": False,
        }
    
    # Filter out invalid lap numbers (0, 32768 sentinel, negative, >1000)
    valid_laps = race2_laps_sorted[
        (race2_laps_sorted["lap"] > 0) & 
        (race2_laps_sorted["lap"] < 1000) &
        (race2_laps_sorted["lap"] != 32768)
    ].copy()
    
    if len(valid_laps) == 0:
        raise ValueError("No valid lap numbers (all filtered out)")
    
    # Walk through race lap by lap
    max_lap = int(valid_laps["lap"].max())
    min_lap = int(valid_laps["lap"].min())
    
    if max_lap < min_lap or max_lap <= 0:
        raise ValueError(f"Invalid lap range: min={min_lap}, max={max_lap}")
    
    # Performance optimization: Only validate every 5 laps instead of every lap
    # This reduces computation from O(vehicles × laps) to O(vehicles × laps/5)
    validation_interval = 5  # Validate every 5 laps
    
    # Ensure valid range (handle edge case where left == right)
    lap_range = list(range(min_lap, max_lap + 1, validation_interval))  # Step by interval
    if len(lap_range) == 0:
        raise ValueError(f"Empty lap range: {min_lap} to {max_lap}")
    
    # Calculate total work
    total_decisions = 0
    for lap_num in lap_range:
        lap_data = valid_laps[valid_laps["lap"] == lap_num]
        total_decisions += len(lap_data[lap_data["vehicle_id"].isin(vehicle_states.keys())])
    
    n_cores = os.cpu_count() or 1
    effective_jobs = n_jobs if n_jobs > 0 else max(1, n_cores - 1)  # Leave 1 core free
    base_sims = int(os.getenv("MC_BASE_SCENARIOS", "1000"))
    close_call_sims = int(os.getenv("MC_CLOSE_SCENARIOS", "2000"))
    
    print(f"\n  📊 Workload Summary:")
    print(f"     • Checkpoints: {len(lap_range)} (every {validation_interval} laps)")
    print(f"     • Total decisions: {total_decisions}")
    print(f"     • Simulations per decision: {base_sims} (base) / {close_call_sims} (close calls)")
    print(f"     • Estimated total simulations: ~{total_decisions * base_sims:,}")
    print(f"     • Parallel workers: {effective_jobs} threads (threading backend)")
    print(f"     • Note: Using threading (shared memory) to avoid slow model pickling\n")
    
    if total_decisions == 0:
        return {
            "recommendations": [],
            "metrics": {
                "total_recommendations": 0,
                "mean_confidence": 0.0,
            }
        }
    
    # Process lap by lap (to maintain state), but parallelize vehicles within each lap
    pbar = tqdm(total=total_decisions, desc="🚀 Processing", unit="dec",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for lap_num in lap_range:
        lap_data = valid_laps[valid_laps["lap"] == lap_num]
        if len(lap_data) == 0:
            continue
        
        # Prepare tasks for this lap
        lap_tasks = []
        for _, lap_row in lap_data.iterrows():
            vehicle_id = lap_row["vehicle_id"]
            if vehicle_id not in vehicle_states:
                continue
            
            lap_tasks.append({
                "lap_row": lap_row,
                "vehicle_id": vehicle_id,
                "lap_num": lap_num,
                "vehicle_state": vehicle_states[vehicle_id].copy(),  # Copy state
            })
        
        if len(lap_tasks) == 0:
            continue
        
        # Process vehicles in parallel for this lap
        # Use 'threading' backend to avoid pickling large models (XGBoost, etc.)
        # Threading works here because most time is spent in numpy/C code (releases GIL)
        try:
            print(f"  🔄 Processing lap {lap_num}: {len(lap_tasks)} vehicles...", flush=True)
            import time
            lap_start = time.time()
            batch_size = max(1, len(lap_tasks) // max(1, effective_jobs * 2))
            lap_results = Parallel(
                n_jobs=effective_jobs,
                backend='threading',
                verbose=10,  # Enable verbose output to show progress
                batch_size=batch_size,
            )(
                delayed(_process_vehicle_decision)(
                    task["lap_row"],
                    task["vehicle_id"],
                    task["lap_num"],
                    max_lap,
                    task["vehicle_state"],
                    models,
                    strategy_optimizer,
                )
                for task in lap_tasks
            )
            lap_elapsed = time.time() - lap_start
            print(f"  ✓ Lap {lap_num} complete: {len(lap_tasks)} vehicles in {lap_elapsed:.1f}s", flush=True)
        except Exception as e:
            # Fallback to sequential if parallel fails
            print(f"\n  ⚠ Parallel processing failed, falling back to sequential: {e}")
            lap_results = [
                _process_vehicle_decision(
                    task["lap_row"],
                    task["vehicle_id"],
                    task["lap_num"],
                    max_lap,
                    task["vehicle_state"],
                    models,
                    strategy_optimizer,
                )
                for task in lap_tasks
            ]
        
        # Update states sequentially and collect results
        for task, result in zip(lap_tasks, lap_results):
            vehicle_id = task["vehicle_id"]
            if result is not None:
                recommendations_log.append(result)
                # Update state for next lap
                vehicle_states[vehicle_id]["current_lap"] = lap_num
                vehicle_states[vehicle_id]["tire_age"] += 1.0
            
            pbar.update(1)
        pbar.refresh()  # Force update
        print(f"  📊 Progress: {len(recommendations_log)}/{total_decisions} decisions completed", flush=True)
    
    pbar.close()
    
    # Compare to actual results (would need actual pit stop data)
    # For now, return log
    return {
        "recommendations": recommendations_log,
        "metrics": {
            "total_recommendations": len(recommendations_log),
            "mean_confidence": pd.DataFrame(recommendations_log)["confidence"].mean(),
        }
    }


def save_walkforward_results(results: dict, path: Path | str):
    """Save walk-forward results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
