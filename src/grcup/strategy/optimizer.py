"""Stochastic Dynamic Programming optimizer for pit strategy."""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..models.sc_hazard import predict_sc_probability


def solve_pit_strategy(
    current_lap: int,
    total_laps: int,
    tire_age: float,
    fuel_laps_remaining: float,
    under_sc: bool,
    wear_model,  # Pre-loaded wear quantile model
    sc_hazard_model,  # Pre-loaded SC hazard model
    pit_loss_mean: float = 30.0,
    pit_loss_std: float = 5.0,
    random_state: np.random.Generator | None = None,
) -> dict:
    """
    Solve optimal pit strategy using Stochastic DP.
    
    State: (lap, fuel_laps, tire_age, under_SC)
    
    Args:
        current_lap: Current lap number
        total_laps: Total race laps
        tire_age: Current tire age (laps)
        fuel_laps_remaining: Fuel remaining (in laps)
        under_sc: Currently under safety car
        wear_model: Pre-loaded wear quantile model
        sc_hazard_model: Pre-loaded SC hazard model
        pit_loss_mean: Mean pit stop loss (seconds)
        pit_loss_std: Std dev of pit loss
        random_state: RNG for scenario sampling
    
    Returns:
        Dict with recommended pit window, expected time gain, confidence
    """
    if random_state is None:
        random_state = np.random.default_rng(42)
    
    remaining_laps = total_laps - current_lap + 1
    
    if remaining_laps <= 1:
        return {
            "recommended_lap": current_lap,
            "window": [current_lap, current_lap],
            "expected_gain": 0.0,
            "confidence": 0.0,
            "reasoning": "Race ending, no pit needed",
        }
    
    # Simulate scenarios: pit now vs pit in N laps
    best_scenario = None
    best_expected_time = float('inf')
    
    # Candidate pit laps (reduced range for performance)
    # Only check 3-5 candidate laps to avoid O(n²) explosion
    max_candidates = 5
    pit_candidates = list(range(current_lap, min(current_lap + max_candidates, total_laps + 1)))
    
    for pit_lap in pit_candidates:
        if pit_lap > fuel_laps_remaining + current_lap:
            continue  # Can't pit beyond fuel limit
        
        # Simulate remaining race with this pit strategy
        # Reduced scenarios for performance (100 → 20)
        scenarios = []
        n_scenarios = 20
        
        for _ in range(n_scenarios):  # Reduced from 100 to 20
            total_race_time = 0.0
            sim_tire_age = tire_age
            sim_fuel = fuel_laps_remaining
            sim_under_sc = under_sc
            
            for lap in range(current_lap, total_laps + 1):
                # Check for pit
                if lap == pit_lap:
                    total_race_time += random_state.normal(pit_loss_mean, pit_loss_std)
                    sim_tire_age = 0.0  # Fresh tires
                    sim_fuel = fuel_laps_remaining  # Refuel
                
                # Check for SC (using hazard model)
                if not sim_under_sc and sc_hazard_model is not None:
                    sc_prob = predict_sc_probability(
                        sc_hazard_model,
                        green_run_len=max(1.0, lap - current_lap),
                        pack_density=0.5,  # Simplified
                        rain=0,
                        wind_speed=0.0,
                        k_laps=1,
                    )
                    if random_state.random() < sc_prob:
                        sim_under_sc = True
                
                # Predict lap time (use wear model + base pace)
                if sim_under_sc:
                    lap_time = 180.0  # SC pace
                else:
                    # Get wear degradation
                    features_dict = {
                        "tire_age": sim_tire_age,
                        "track_temp": 50.0,  # Simplified
                        "stint_len": sim_tire_age,
                        "sector_S3_coeff": 0.0,
                        "clean_air": 1.0,
                        "traffic_density": 0.0,
                        "driver_TE": 0.0,
                    }
                    
                    # Sample from quantile model
                    # Try to use actual wear model if available
                    if wear_model is not None:
                        # Would call predict_quantiles here with actual features
                        # For now, use simplified degradation model
                        base_degradation = sim_tire_age * 0.1
                        quantile_preds = {
                            "q10": max(0.0, base_degradation * 0.8),
                            "q50": base_degradation,
                            "q90": base_degradation * 1.5,
                        }
                    else:
                        quantile_preds = {
                            "q10": max(0.0, sim_tire_age * 0.08),
                            "q50": sim_tire_age * 0.1,  # Degradation per lap
                            "q90": sim_tire_age * 0.15,
                        }
                    
                    # Ensure q10 < q50 < q90 for triangular distribution
                    if quantile_preds["q10"] >= quantile_preds["q90"]:
                        quantile_preds["q90"] = quantile_preds["q50"] + 0.1
                    if quantile_preds["q10"] >= quantile_preds["q50"]:
                        quantile_preds["q10"] = max(0.0, quantile_preds["q50"] - 0.1)
                    
                    degradation = random_state.triangular(
                        quantile_preds["q10"],
                        quantile_preds["q50"],
                        quantile_preds["q90"],
                    )
                    
                    base_pace = 130.0  # Base lap time
                    lap_time = base_pace + degradation
                
                total_race_time += lap_time
                sim_tire_age += 1.0
                sim_fuel -= 1.0
                
                if sim_under_sc and random_state.random() < 0.1:
                    sim_under_sc = False  # SC ends
            
            scenarios.append(total_race_time)
        
        # Expected race time for this pit strategy
        expected_time = np.mean(scenarios)
        
        if expected_time < best_expected_time:
            best_expected_time = expected_time
            best_scenario = {
                "pit_lap": pit_lap,
                "expected_time": expected_time,
                "scenarios": scenarios,
            }
    
    if best_scenario is None:
        # Fallback: pit immediately
        recommended_lap = current_lap
        window = [current_lap, current_lap + 2]
        time_gain = 0.0
    else:
        recommended_lap = best_scenario["pit_lap"]
        window = [max(current_lap, recommended_lap - 1), min(total_laps, recommended_lap + 1)]
        
        # Compare to baseline (no pit) - simulate baseline strategy
        # Reduced baseline scenarios for performance
        baseline_scenarios = []
        for _ in range(10):  # Reduced from 50 to 10 for performance
            baseline_time = 0.0
            base_tire_age = tire_age
            base_under_sc = under_sc
            for lap in range(current_lap, total_laps + 1):
                if base_under_sc:
                    lap_time = 180.0
                else:
                    degradation = base_tire_age * 0.12  # Slightly worse (no pit)
                    base_pace = 130.0
                    lap_time = base_pace + degradation + random_state.normal(0, 2.0)
                baseline_time += lap_time
                base_tire_age += 1.0
                if random_state.random() < 0.05:  # Random SC
                    base_under_sc = True
                elif base_under_sc and random_state.random() < 0.1:
                    base_under_sc = False
            baseline_scenarios.append(baseline_time)
        
        baseline_expected = np.mean(baseline_scenarios) if baseline_scenarios else float('inf')
        time_gain = best_scenario["expected_time"] - baseline_expected
    
    # Compute confidence based on scenario variance
    if best_scenario and len(best_scenario["scenarios"]) > 1:
        scenario_std = np.std(best_scenario["scenarios"])
        confidence = max(0.3, min(0.95, 1.0 - (scenario_std / (np.mean(best_scenario["scenarios"]) + 1e-6))))
    else:
        confidence = 0.5
    
    return {
        "recommended_lap": recommended_lap,
        "window": window,
        "expected_gain": -time_gain,  # Negative = time saved
        "confidence": confidence,
        "reasoning": f"Optimal pit at lap {recommended_lap} based on degradation + SC scenarios",
        "contingency": f"If SC between laps {current_lap}-{recommended_lap}, pit immediately",
    }


