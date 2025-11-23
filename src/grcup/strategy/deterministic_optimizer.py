"""Deterministic rule-based pit strategy optimizer."""
from __future__ import annotations

import numpy as np
import os
from typing import Literal

from ..models.wear_quantile_xgb import predict_quantiles, build_feature_vector
import pandas as pd


def calculate_optimal_stint_length(
    pit_loss: float,
    degradation_rate: float,
    remaining_laps: int,
) -> float:
    """
    Calculate optimal stint length using deterministic formula.
    
    Formula: optimal_stint = sqrt(2 * pit_loss / degradation_rate)
    
    This minimizes total race time by balancing:
    - Pit stop penalty (amortized over stint)
    - Tire degradation cost
    
    Args:
        pit_loss: Pit stop time penalty (seconds)
        degradation_rate: Tire degradation rate (seconds per lap²)
        remaining_laps: Remaining race distance
        
    Returns:
        Optimal stint length (laps)
    """
    if degradation_rate <= 0:
        # No degradation - pit only for fuel
        return float(remaining_laps)
    
    # Optimal stint formula
    optimal_stint = np.sqrt(2.0 * pit_loss / degradation_rate)
    
    # Cap at remaining race distance
    optimal_stint = min(optimal_stint, float(remaining_laps))
    
    # Minimum 5 laps (can't pit too frequently)
    optimal_stint = max(5.0, optimal_stint)
    
    return optimal_stint


def estimate_degradation_rate(
    wear_model,
    tire_age: float,
    track_temp: float = 50.0,
) -> float:
    """
    Estimate degradation rate from wear model.
    
    Uses q50 prediction at different tire ages to estimate rate.
    
    Args:
        wear_model: Pre-loaded wear quantile model
        tire_age: Current tire age
        track_temp: Track temperature
        
    Returns:
        Degradation rate (seconds per lap²)
    """
    if wear_model is None:
        # Fallback: assume 0.1s/lap linear degradation
        return 0.1
    
    try:
        # Sample degradation at tire_age and tire_age+5
        ages = [tire_age, tire_age + 5.0]
        degradation_rates = []
        
        for age in ages:
            overrides = {
                "tire_age": age,
                "track_temp": track_temp,
                "temp_anomaly": 0.0,
                "stint_len": age,
                "sector_S3_coeff": 0.0,
                "clean_air": 1.0,
                "traffic_density": 0.0,
                "tire_temp_interaction": age * track_temp,
                "tire_clean_interaction": age,
                "traffic_temp_interaction": 0.0,
            }
            
            features_df = pd.DataFrame([build_feature_vector(wear_model, overrides)])
            quantile_df = predict_quantiles(wear_model, features_df)
            degradation = float(quantile_df.iloc[0]["q50"])
            degradation_rates.append(degradation)
        
        # Estimate rate: (degradation_2 - degradation_1) / (age_2 - age_1)
        if len(degradation_rates) == 2 and degradation_rates[1] > degradation_rates[0]:
            rate = (degradation_rates[1] - degradation_rates[0]) / 5.0
            # Convert to per-lap² (assuming quadratic degradation)
            # If degradation is linear, rate is already per lap
            # If quadratic, need to estimate: degradation = rate * age²
            # For now, assume linear and convert to quadratic estimate
            rate_per_lap_sq = rate / max(1.0, tire_age)  # Rough estimate
            return max(0.05, min(1.0, rate_per_lap_sq))
        else:
            return 0.1  # Fallback
    except Exception:
        return 0.1  # Fallback


def solve_pit_strategy_deterministic(
    current_lap: int,
    total_laps: int,
    tire_age: float,
    fuel_laps_remaining: float,
    under_sc: bool,
    wear_model,  # Pre-loaded wear quantile model
    sc_hazard_model=None,  # Not used in deterministic
    pit_loss_mean: float = 30.0,
    pit_loss_std: float = 5.0,
    random_state=None,  # Not used in deterministic
) -> dict:
    """
    Solve optimal pit strategy using deterministic rules.
    
    This is a rule-based approach that calculates optimal stint length
    mathematically, avoiding Monte Carlo bias toward early pits.
    
    Args:
        current_lap: Current lap number
        total_laps: Total race laps
        tire_age: Current tire age (laps)
        fuel_laps_remaining: Fuel remaining (in laps)
        under_sc: Currently under safety car
        wear_model: Pre-loaded wear quantile model
        sc_hazard_model: Not used (kept for API compatibility)
        pit_loss_mean: Mean pit stop loss (seconds)
        pit_loss_std: Not used (kept for API compatibility)
        random_state: Not used (kept for API compatibility)
    
    Returns:
        Dict with recommended pit window, expected_time, confidence
    """
    remaining_laps = total_laps - current_lap + 1
    
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
    
    # Get track temperature from environment or default
    track_temp = float(os.getenv("SCENARIO_TRACK_TEMP", "50.0"))
    
    # Estimate degradation rate from wear model
    degradation_rate = estimate_degradation_rate(wear_model, tire_age, track_temp)
    
    # Calculate optimal stint length
    optimal_stint = calculate_optimal_stint_length(
        pit_loss=pit_loss_mean,
        degradation_rate=degradation_rate,
        remaining_laps=remaining_laps,
    )
    
    # Recommended pit lap: current_lap + optimal_stint
    recommended_lap = int(current_lap + optimal_stint)
    
    # Cap at fuel limit
    max_pit_lap = int(current_lap + fuel_laps_remaining)
    recommended_lap = min(recommended_lap, max_pit_lap)
    
    # Cap at total race distance
    recommended_lap = min(recommended_lap, total_laps)
    
    # Ensure at least current_lap + 1
    recommended_lap = max(recommended_lap, current_lap + 1)
    
    # Calculate expected time (simplified deterministic estimate)
    # Stint 1: current_lap to recommended_lap
    stint1_laps = recommended_lap - current_lap
    stint1_time = 0.0
    for lap_offset in range(stint1_laps):
        age = tire_age + lap_offset
        if wear_model:
            try:
                overrides = {
                    "tire_age": age,
                    "track_temp": track_temp,
                    "temp_anomaly": 0.0,
                    "stint_len": age,
                    "sector_S3_coeff": 0.0,
                    "clean_air": 1.0,
                    "traffic_density": 0.0,
                    "tire_temp_interaction": age * track_temp,
                    "tire_clean_interaction": age,
                    "traffic_temp_interaction": 0.0,
                }
                features_df = pd.DataFrame([build_feature_vector(wear_model, overrides)])
                quantile_df = predict_quantiles(wear_model, features_df)
                degradation = float(quantile_df.iloc[0]["q50"])
            except Exception:
                degradation = age * 0.1
        else:
            degradation = age * 0.1
        
        base_pace = 130.0
        stint1_time += base_pace + degradation
    
    # Pit stop
    stint1_time += pit_loss_mean
    
    # Stint 2: recommended_lap to total_laps
    stint2_laps = total_laps - recommended_lap + 1
    stint2_time = 0.0
    for lap_offset in range(stint2_laps):
        age = lap_offset  # Fresh tires
        if wear_model:
            try:
                overrides = {
                    "tire_age": age,
                    "track_temp": track_temp,
                    "temp_anomaly": 0.0,
                    "stint_len": age,
                    "sector_S3_coeff": 0.0,
                    "clean_air": 1.0,
                    "traffic_density": 0.0,
                    "tire_temp_interaction": age * track_temp,
                    "tire_clean_interaction": age,
                    "traffic_temp_interaction": 0.0,
                }
                features_df = pd.DataFrame([build_feature_vector(wear_model, overrides)])
                quantile_df = predict_quantiles(wear_model, features_df)
                degradation = float(quantile_df.iloc[0]["q50"])
            except Exception:
                degradation = age * 0.1
        else:
            degradation = age * 0.1
        
        base_pace = 130.0
        stint2_time += base_pace + degradation
    
    expected_time = stint1_time + stint2_time
    
    # Calculate speed metrics
    track_length_km = float(os.getenv("TRACK_LENGTH_KM", "5.26"))
    avg_lap_time = expected_time / remaining_laps if remaining_laps > 0 else 0.0
    projected_avg_speed_kph = (track_length_km * 3600.0) / avg_lap_time if avg_lap_time > 0 else 0.0
    baseline_lap_time = 130.0
    baseline_speed_kph = (track_length_km * 3600.0) / baseline_lap_time if baseline_lap_time > 0 else 0.0
    pace_delta = projected_avg_speed_kph - baseline_speed_kph
    
    # Confidence based on how close to optimal
    # Higher confidence if optimal_stint is reasonable (8-15 laps)
    if 8.0 <= optimal_stint <= 15.0:
        confidence = 0.9
    elif 5.0 <= optimal_stint < 8.0 or 15.0 < optimal_stint <= 20.0:
        confidence = 0.7
    else:
        confidence = 0.5
    
    return {
        "recommended_lap": recommended_lap,
        "window": [max(current_lap, recommended_lap - 1), min(total_laps, recommended_lap + 1)],
        "expected_time": expected_time,
        "confidence": confidence,
        "avg_lap_time": avg_lap_time,
        "projected_avg_speed_kph": projected_avg_speed_kph,
        "pace_delta": pace_delta,
        "reasoning": f"Optimal pit at lap {recommended_lap} (optimal stint: {optimal_stint:.1f} laps, degradation rate: {degradation_rate:.3f}s/lap²)",
        "contingency": f"If SC between laps {current_lap}-{recommended_lap}, pit immediately",
        "close_call": False,
        "optimal_stint": optimal_stint,
        "degradation_rate": degradation_rate,
    }


