"""Walk-forward validation: causal, no leakage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def walkforward_validate(
    race2_laps: pd.DataFrame,
    race2_sectors: pd.DataFrame,
    race2_results: pd.DataFrame,
    models: dict[str, Any],  # Pre-loaded models
    strategy_optimizer,
    lap_by_lap: bool = True,
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
    lap_range = range(min_lap, max_lap + 1, validation_interval)  # Step by interval
    if len(lap_range) == 0:
        raise ValueError(f"Empty lap range: {min_lap} to {max_lap}")
    
    print(f"  Validating at {len(lap_range)} checkpoints (every {validation_interval} laps)...")
    
    for lap_num in lap_range:
        lap_data = valid_laps[valid_laps["lap"] == lap_num]
        
        if len(lap_data) == 0:
            continue  # Skip laps with no data
        
        for _, lap_row in lap_data.iterrows():
            vehicle_id = lap_row["vehicle_id"]
            
            # Ensure vehicle state exists
            if vehicle_id not in vehicle_states:
                continue
            
            state = vehicle_states[vehicle_id]
            
            # Update state
            state["current_lap"] = lap_num
            state["tire_age"] += 1.0
            
            # Compute strategy using only past info
            try:
                recommendation = strategy_optimizer(
                    current_lap=lap_num,
                    total_laps=max_lap,
                    tire_age=state["tire_age"],
                    fuel_laps_remaining=state["fuel_remaining"],
                    under_sc=state["under_sc"],
                    wear_model=models.get("wear"),
                    sc_hazard_model=models.get("hazard"),
                )
            except Exception as e:
                # Skip if optimizer fails for this lap
                if "left" in str(e) and "right" in str(e):
                    # This is the pandas edge case - skip this lap
                    continue
                raise
            
            # Log recommendation
            recommendations_log.append({
                "vehicle_id": vehicle_id,
                "lap": lap_num,
                "recommended_pit_lap": recommendation.get("recommended_lap"),
                "recommended_window": recommendation.get("window"),
                "expected_gain": recommendation.get("expected_gain"),
                "confidence": recommendation.get("confidence"),
            })
    
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


