"""Stochastic Dynamic Programming optimizer for pit strategy."""
from __future__ import annotations

from typing import Literal

import numpy as np
import os
import pandas as pd

from ..models.sc_hazard import predict_sc_probability
from ..models.wear_quantile_xgb import predict_quantiles, build_feature_vector
from .monte_carlo import ConvergenceMonitor

# Import deterministic optimizer as fallback
try:
    from .deterministic_optimizer import solve_pit_strategy_deterministic
except ImportError:
    solve_pit_strategy_deterministic = None


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
    Solve optimal pit strategy.
    
    Uses deterministic optimizer if USE_DETERMINISTIC=1 env var is set,
    otherwise uses Monte Carlo approach.
    """
    # Check if deterministic mode is enabled
    use_deterministic = os.getenv("USE_DETERMINISTIC", "0") == "1"
    
    if use_deterministic and solve_pit_strategy_deterministic is not None:
        return solve_pit_strategy_deterministic(
            current_lap=current_lap,
            total_laps=total_laps,
            tire_age=tire_age,
            fuel_laps_remaining=fuel_laps_remaining,
            under_sc=under_sc,
            wear_model=wear_model,
            sc_hazard_model=sc_hazard_model,
            pit_loss_mean=pit_loss_mean,
            pit_loss_std=pit_loss_std,
            random_state=random_state,
        )
    
    # Otherwise use Monte Carlo approach (GPU-accelerated if available)
    use_gpu_mc = os.getenv("USE_GPU_MC", "0") == "1"
    scenario_phase = os.getenv("SCENARIO_PHASE")
    track_temp_env = float(os.getenv("SCENARIO_TRACK_TEMP", "50.0"))
    gpu_available = False
    gpu_simulator = None
    if use_gpu_mc:
        try:
            from .gpu_monte_carlo import simulate_strategy_gpu, _check_torch_available
            gpu_available = _check_torch_available()
            if gpu_available:
                gpu_simulator = simulate_strategy_gpu
        except Exception:
            gpu_available = False
        if not gpu_available:
            use_gpu_mc = False
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
        Dict with recommended pit window, expected_time, confidence
    """
    if random_state is None:
        random_state = np.random.default_rng(42)

    base_scenarios = int(os.getenv("MC_BASE_SCENARIOS", "2000"))  # Increased from 1000
    close_call_scenarios = int(os.getenv("MC_CLOSE_SCENARIOS", "5000"))  # Increased from 2000
    # Adaptive convergence window based on scenario count
    convergence_window = min(50, max(10, base_scenarios // 10))  # Scale with scenario count
    convergence_tolerance = 0.1  # Relaxed tolerance for faster convergence
    
    remaining_laps = total_laps - current_lap + 1
    
    # Track length for speed calculations
    track_length_km = float(os.getenv("TRACK_LENGTH_KM", "5.26"))  # Default: VIR full course
    
    if remaining_laps <= 1:
        return {
            "recommended_lap": current_lap,
            "window": [current_lap, current_lap],
            "expected_time": 0.0,
            "confidence": 0.0,
            "avg_lap_time": 0.0,
            "projected_avg_speed_kph": 0.0,
            "pace_delta": 0.0,
            "reasoning": "Race ending, no pit needed",
        }
    
    # Simulate scenarios: pit now vs pit in N laps
    best_scenario = None
    best_expected_time = float('inf')
    candidate_results: list[tuple[int, float, list[float]]] = []  # (pit_lap, expected_time, scenarios)
    
    # Candidate pit laps (reduced range for performance)
    # Only check 3-5 candidate laps to avoid O(n²) explosion
    max_candidates = 5
    pit_candidates = list(range(current_lap, min(current_lap + max_candidates, total_laps + 1)))
    
    for pit_lap in pit_candidates:
        if pit_lap > fuel_laps_remaining + current_lap:
            continue  # Can't pit beyond fuel limit
        
        scenarios = []
        expected_time = float("inf")
        
        if use_gpu_mc:
            try:
                gpu_result = gpu_simulator(
                    current_lap=current_lap,
                    total_laps=total_laps,
                    pit_lap=pit_lap,
                    tire_age=tire_age,
                    wear_model=wear_model,
                    sc_hazard_model=sc_hazard_model,
                    pit_loss_mean=pit_loss_mean,
                    pit_loss_std=pit_loss_std,
                    num_scenarios=base_scenarios,
                    track_temp=track_temp_env,
                    scenario_phase=scenario_phase,
                )
                scenarios = gpu_result["samples"]
                expected_time = gpu_result["mean_time"]
            except Exception:
                # GPU path failed, fall back to CPU Monte Carlo
                use_gpu_mc = False
                scenarios = []
        
        wear_cache = {}
        max_tire_age = min(50, total_laps - current_lap + 1)  # Cap at 50 laps max
        
        if not use_gpu_mc:
            # CPU Monte Carlo (existing logic)
            effective_window = min(convergence_window, base_scenarios // 2)
            base_convergence = ConvergenceMonitor(
                window=effective_window,
                tolerance=convergence_tolerance,
                min_samples=min(20, base_scenarios // 2),
                max_samples=base_scenarios,
            )
            
            for _ in range(base_scenarios):
                total_race_time = 0.0
                sim_tire_age = tire_age
                sim_fuel = fuel_laps_remaining
                sim_under_sc = under_sc
                
                lap_step = 2 if (total_laps - current_lap) > 20 else 1
                lap_range = list(range(current_lap, total_laps + 1, lap_step))
                if lap_range[-1] != total_laps:
                    lap_range.append(total_laps)
                
                for lap in lap_range:
                    if lap == pit_lap:
                        total_race_time += random_state.normal(pit_loss_mean, pit_loss_std)
                        sim_tire_age = 0.0
                        sim_fuel = fuel_laps_remaining
                    
                    if not sim_under_sc and sc_hazard_model is not None:
                        sc_prob = predict_sc_probability(
                            sc_hazard_model,
                            green_run_len=max(1.0, lap - current_lap),
                            pack_density=0.5,
                            rain=0,
                            wind_speed=0.0,
                            k_laps=1,
                        )
                        if random_state.random() < sc_prob:
                            sim_under_sc = True
                    
                    if sim_under_sc:
                        lap_time = 180.0
                    else:
                        fallback_quantiles = {
                            "q10": max(0.0, sim_tire_age * 0.08),
                            "q50": sim_tire_age * 0.1,
                            "q90": sim_tire_age * 0.15,
                        }
                        tire_age_key = int(sim_tire_age)
                        if tire_age_key in wear_cache:
                            quantile_preds = wear_cache[tire_age_key]
                        elif wear_model is not None and tire_age_key <= max_tire_age:
                            try:
                                overrides = {
                                    "tire_age": sim_tire_age,
                                    "track_temp": track_temp_env,
                                    "temp_anomaly": 0.0,
                                    "stint_len": sim_tire_age,
                                    "sector_S3_coeff": 0.0,
                                    "clean_air": 1.0,
                                    "traffic_density": 0.0,
                                    "tire_temp_interaction": sim_tire_age * track_temp_env,
                                    "tire_clean_interaction": sim_tire_age,
                                    "traffic_temp_interaction": 0.0,
                                }
                                feature_row = build_feature_vector(wear_model, overrides)
                                features_df = pd.DataFrame([feature_row])
                                quantile_df = predict_quantiles(wear_model, features_df)
                                quantile_preds = {
                                    "q10": float(quantile_df.iloc[0]["q10"]),
                                    "q50": float(quantile_df.iloc[0]["q50"]),
                                    "q90": float(quantile_df.iloc[0]["q90"]),
                                }
                                wear_cache[tire_age_key] = quantile_preds
                            except Exception:
                                base_degradation = sim_tire_age * 0.1
                                quantile_preds = {
                                    "q10": max(0.0, base_degradation * 0.8),
                                    "q50": base_degradation,
                                    "q90": base_degradation * 1.5,
                                }
                                wear_cache[tire_age_key] = quantile_preds
                        else:
                            base_degradation = sim_tire_age * 0.1
                            quantile_preds = {
                                "q10": max(0.0, base_degradation * 0.8),
                                "q50": base_degradation,
                                "q90": base_degradation * 1.5,
                            }
                        
                        if quantile_preds["q10"] >= quantile_preds["q90"]:
                            quantile_preds["q90"] = quantile_preds["q50"] + 0.1
                        if quantile_preds["q10"] >= quantile_preds["q50"]:
                            quantile_preds["q10"] = max(0.0, quantile_preds["q50"] - 0.1)
                        
                        degradation = random_state.triangular(
                            quantile_preds["q10"],
                            quantile_preds["q50"],
                            quantile_preds["q90"],
                        )
                        base_pace = 130.0
                        lap_time = base_pace + degradation
                    
                    total_race_time += lap_time * lap_step
                    sim_tire_age += lap_step
                    sim_fuel -= lap_step
                    
                    if sim_under_sc and random_state.random() < 0.1:
                        sim_under_sc = False
            
                scenarios.append(total_race_time)
                if base_convergence.update(total_race_time):
                    break
            
            expected_time = np.mean(scenarios)
        
        candidate_results.append((pit_lap, expected_time, scenarios))
        if expected_time < best_expected_time:
            best_expected_time = expected_time
            best_scenario = {
                "pit_lap": pit_lap,
                "expected_time": expected_time,
                "scenarios": scenarios,
            }
    
    # Adaptive Monte Carlo for close calls
    if len(candidate_results) >= 2:
        candidate_results.sort(key=lambda x: x[1])
        top, second = candidate_results[0], candidate_results[1]
        gap = abs(second[1] - top[1])
        if gap < 0.4:  # close call threshold in seconds
            refined: list[tuple[int, float]] = []
            for pit_lap, _, _ in [top, second]:
                refined_scenarios = []
                refine_convergence = ConvergenceMonitor(
                    window=convergence_window,
                    tolerance=convergence_tolerance,
                    min_samples=min(1000, close_call_scenarios),  # Increased from 500
                    max_samples=close_call_scenarios,
                )
                for _ in range(close_call_scenarios):
                    total_race_time = 0.0
                    sim_tire_age = tire_age
                    sim_fuel = fuel_laps_remaining
                    sim_under_sc = under_sc
                    for lap in range(current_lap, total_laps + 1):
                        if lap == pit_lap:
                            total_race_time += random_state.normal(pit_loss_mean, pit_loss_std)
                            sim_tire_age = 0.0
                            sim_fuel = fuel_laps_remaining
                        # SC hazard with optional scenario phase boost
                        if not sim_under_sc and sc_hazard_model is not None:
                            sc_prob = predict_sc_probability(
                                sc_hazard_model,
                                green_run_len=max(1.0, lap - current_lap),
                                pack_density=0.5,
                                rain=0,
                                wind_speed=0.0,
                                k_laps=1,
                            )
                            phase = os.getenv("SCENARIO_SC_PHASE", "").lower()
                            if phase == "early" and lap <= max(current_lap, current_lap + 7):
                                sc_prob = min(1.0, sc_prob + 0.3)
                            elif phase == "late" and lap >= int(0.8 * total_laps):
                                sc_prob = min(1.0, sc_prob + 0.3)
                            if random_state.random() < sc_prob:
                                sim_under_sc = True
                        if sim_under_sc:
                            lap_time = 180.0
                        else:
                            # Get wear degradation using actual model
                            base_degradation = sim_tire_age * 0.1
                            fallback_q10 = max(0.0, base_degradation * 0.8)
                            fallback_q50 = base_degradation
                            fallback_q90 = base_degradation * 1.5
                            if wear_model is not None:
                                try:
                                    overrides = {
                                        "tire_age": sim_tire_age,
                                        "track_temp": track_temp_env,
                                        "temp_anomaly": 0.0,
                                        "stint_len": sim_tire_age,
                                        "sector_S3_coeff": 0.0,
                                        "clean_air": 1.0,
                                        "traffic_density": 0.0,
                                        "tire_temp_interaction": sim_tire_age * track_temp_env,
                                        "tire_clean_interaction": sim_tire_age,
                                        "traffic_temp_interaction": 0.0,
                                    }
                                    features_row = build_feature_vector(wear_model, overrides)
                                    features_df = pd.DataFrame([features_row])
                                    quantile_df = predict_quantiles(wear_model, features_df)
                                    q10 = float(quantile_df.iloc[0]["q10"])
                                    q50 = float(quantile_df.iloc[0]["q50"])
                                    q90 = float(quantile_df.iloc[0]["q90"])
                                except Exception:
                                    q10, q50, q90 = fallback_q10, fallback_q50, fallback_q90
                            else:
                                q10, q50, q90 = fallback_q10, fallback_q50, fallback_q90
                            
                            if q10 >= q90:
                                q90 = q50 + 0.1
                            if q10 >= q50:
                                q10 = max(0.0, q50 - 0.1)
                            degradation = random_state.triangular(q10, q50, q90)
                            base_pace = 130.0
                            lap_time = base_pace + degradation
                        total_race_time += lap_time
                        sim_tire_age += 1.0
                        sim_fuel -= 1.0
                        if sim_under_sc and random_state.random() < 0.1:
                            sim_under_sc = False
                    refined_scenarios.append(total_race_time)
                    if refine_convergence.update(total_race_time):
                        break
                refined_mean = float(np.mean(refined_scenarios)) if refined_scenarios else float('inf')
                refined.append((pit_lap, refined_mean))
            # Select refined best
            refined.sort(key=lambda x: x[1])
            if refined:
                pit_lap_refined, mean_refined = refined[0]
                best_expected_time = mean_refined
                best_scenario = {
                    "pit_lap": pit_lap_refined,
                    "expected_time": mean_refined,
                    "scenarios": [],
                    "close_call": True,
                }

    if best_scenario is None:
        # Fallback: pit immediately
        recommended_lap = current_lap
        window = [current_lap, current_lap + 2]
        expected_time = float('inf')
    else:
        recommended_lap = best_scenario["pit_lap"]
        window = [max(current_lap, recommended_lap - 1), min(total_laps, recommended_lap + 1)]
        expected_time = best_scenario["expected_time"]
    
    # Compute confidence based on scenario variance
    if best_scenario and len(best_scenario["scenarios"]) > 1:
        scenario_std = np.std(best_scenario["scenarios"])
        confidence = max(0.3, min(0.95, 1.0 - (scenario_std / (np.mean(best_scenario["scenarios"]) + 1e-6))))
    else:
        confidence = 0.5
    
    # Compute speed metrics
    # Track length: VIR (Virginia International Raceway) full course is ~5.26 km
    # Using approximate value for GR Cup (could be configurable)
    track_length_km = float(os.getenv("TRACK_LENGTH_KM", "5.26"))  # Default: VIR full course
    
    # Calculate average lap time and projected speed
    avg_lap_time = 0.0
    projected_avg_speed_kph = 0.0
    pace_delta = 0.0
    
    if expected_time != float('inf') and remaining_laps > 0:
        # Average lap time = total time / number of laps
        avg_lap_time = expected_time / remaining_laps
        
        # Convert lap time (seconds) to speed (km/h)
        # Speed = distance / time = track_length_km / (lap_time_hours)
        # lap_time_hours = lap_time_seconds / 3600
        # So: speed_kph = track_length_km / (lap_time_seconds / 3600) = track_length_km * 3600 / lap_time_seconds
        if avg_lap_time > 0:
            projected_avg_speed_kph = (track_length_km * 3600.0) / avg_lap_time
        
        # Calculate pace_delta vs baseline pace (130s lap time = ~145.7 km/h)
        # Baseline pace for comparison (typical GR Cup pace)
        baseline_lap_time = 130.0  # seconds
        baseline_speed_kph = (track_length_km * 3600.0) / baseline_lap_time if baseline_lap_time > 0 else 0.0
        pace_delta = projected_avg_speed_kph - baseline_speed_kph
    
    return {
        "recommended_lap": recommended_lap,
        "window": window,
        "expected_time": expected_time,
        "confidence": (lambda base_conf: max(0.0, min(1.0, base_conf * (1.0 - float(os.getenv("SCENARIO_IMPUTED_CONF_PENALTY", "0.0") or 0.0))))) (confidence),
        "avg_lap_time": avg_lap_time,
        "projected_avg_speed_kph": projected_avg_speed_kph,
        "pace_delta": pace_delta,
        "reasoning": f"Optimal pit at lap {recommended_lap} based on degradation + SC scenarios",
        "contingency": f"If SC between laps {current_lap}-{recommended_lap}, pit immediately",
        "close_call": bool(best_scenario.get("close_call", False)) if isinstance(best_scenario, dict) else False,
    }

