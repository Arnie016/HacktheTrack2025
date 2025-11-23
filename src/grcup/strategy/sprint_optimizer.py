"""Sprint race specific pit strategy optimizer.

For GR Cup NA and other sprint series:
- 20-25 lap races (~45 minutes)
- No refueling
- Single tire compound
- Position > pace (passing is very difficult)
- Expected 0-1 pit stops per race
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def solve_sprint_pit_strategy(
    current_lap: int,
    total_laps: int,
    tire_age: float,
    current_position: int,
    gap_ahead: float,
    gap_behind: float,
    under_sc: bool,
    wear_model,
    sc_hazard_model,
    pit_loss_mean: float = 30.0,
    pit_loss_std: float = 5.0,
    track_position_value: float = 3.0,  # Seconds per position (hard to pass)
    random_state: Optional[np.random.Generator] = None,
) -> dict:
    """
    Solve pit strategy for SPRINT RACES.
    
    Key differences from endurance strategy:
    - No fuel constraint (sprint = no refueling)
    - Position heavily weighted (passing is hard)
    - Tires expected to last full race (22 laps)
    - Only pit if: damage, SC window, or massive undercut opportunity
    
    Args:
        current_lap: Current lap number
        total_laps: Total race laps (typically 20-25)
        tire_age: Current tire age in laps
        current_position: Current track position
        gap_ahead: Gap to car ahead (seconds)
        gap_behind: Gap to car behind (seconds)
        under_sc: Currently under safety car
        wear_model: Tire wear prediction model
        sc_hazard_model: Safety car probability model
        pit_loss_mean: Mean pit stop time loss (seconds)
        pit_loss_std: Std dev of pit loss
        track_position_value: Time value per position (default 3s)
        random_state: RNG for simulations
    
    Returns:
        Dict with:
        - recommended_action: "stay_out" | "pit_now" | "pit_lap_X"
        - confidence: 0-1 confidence score
        - reason: Human-readable explanation
        - expected_position: Predicted final position
        - tire_risk: 0-1 risk of tire failure
    """
    if random_state is None:
        random_state = np.random.default_rng(42)
    
    laps_remaining = total_laps - current_lap
    
    # RULE 1: Sprint races are typically no-stop unless forced
    # Tires should last 22-25 laps easily for GR Cup
    if tire_age <= 15 and laps_remaining <= 10:
        # Fresh enough tires, close to end → STAY OUT
        return {
            "recommended_action": "stay_out",
            "recommended_pit_lap": None,
            "confidence": 0.95,
            "reason": "tires_sufficient_for_sprint",
            "expected_position": current_position,
            "tire_risk": 0.1,
        }
    
    # RULE 2: Safety car = free pit stop (but only if tires are old)
    if under_sc and tire_age >= 10:
        return {
            "recommended_action": "pit_now",
            "recommended_pit_lap": current_lap,
            "confidence": 0.90,
            "reason": "safety_car_free_stop",
            "expected_position": current_position,  # SC neutralizes field
            "tire_risk": 0.0,
        }
    
    # RULE 3: Undercut opportunity (only if big gap advantage)
    # For sprint races, undercut works if:
    # - Your tires are fresh vs competitors on old tires
    # - You have a safe gap behind (>5s) to rejoin without losing position
    # - Car ahead is on OLD tires (>12 laps) and you can gain on pit delta
    
    undercut_gain_per_lap = 0.3  # Fresh tires vs old (conservative for sprint)
    laps_to_undercut = min(5, laps_remaining)  # Look ahead 5 laps max
    
    if gap_ahead <= 5.0 and gap_behind >= 5.0 and tire_age >= 8:
        # Potential undercut window
        # Estimated total gain = (pit loss recovery) + (pace advantage)
        estimated_gain = (undercut_gain_per_lap * laps_to_undercut) - pit_loss_mean
        
        if estimated_gain > gap_ahead:
            return {
                "recommended_action": "pit_now",
                "recommended_pit_lap": current_lap,
                "confidence": 0.70,
                "reason": f"undercut_opportunity_gain_{estimated_gain:.1f}s",
                "expected_position": max(1, current_position - 1),  # Gain 1 position
                "tire_risk": 0.0,
            }
    
    # RULE 4: Tire degradation cliff (emergency pit)
    # If tires are VERY old (>20 laps) and still racing, pit immediately
    if tire_age >= 20:
        return {
            "recommended_action": "pit_now",
            "recommended_pit_lap": current_lap,
            "confidence": 0.85,
            "reason": "tire_age_critical",
            "expected_position": min(20, current_position + 2),  # Lose ~2 positions
            "tire_risk": 0.8,
        }
    
    # RULE 5: Late race SC prediction
    # If SC is likely in next 3-5 laps, hold position and wait
    try:
        # Predict SC probability for next few laps
        sc_prob_next_5 = 0.0
        for future_lap in range(current_lap + 1, min(current_lap + 6, total_laps + 1)):
            # Create minimal feature dict for prediction
            features = {
                "lap": future_lap,
                "tire_age": tire_age + (future_lap - current_lap),
            }
            # Note: predict_sc_probability handles missing features gracefully
            from ..models.sc_hazard import predict_sc_probability
            sc_prob = predict_sc_probability(sc_hazard_model, features)
            sc_prob_next_5 = max(sc_prob_next_5, sc_prob)
        
        if sc_prob_next_5 > 0.3 and tire_age >= 8:
            # High SC probability and tires are aging → wait for free stop
            return {
                "recommended_action": "stay_out",
                "recommended_pit_lap": None,
                "confidence": 0.75,
                "reason": f"await_safety_car_prob_{sc_prob_next_5:.2f}",
                "expected_position": current_position,
                "tire_risk": 0.3,
            }
    except Exception:
        # If SC prediction fails, continue with default logic
        pass
    
    # DEFAULT: HOLD POSITION (sprint races favor track position)
    return {
        "recommended_action": "stay_out",
        "recommended_pit_lap": None,
        "confidence": 0.80,
        "reason": "track_position_priority_sprint",
        "expected_position": current_position,
        "tire_risk": min(0.5, tire_age / 30.0),  # Risk increases with age
    }


def estimate_sprint_tire_life(
    wear_model,
    current_tire_age: float,
    laps_remaining: int,
    track_temp: float = 50.0,
    quantile: float = 0.9,
) -> dict:
    """
    Estimate if tires will last to end of sprint race.
    
    Args:
        wear_model: Trained tire wear model
        current_tire_age: Current tire age (laps)
        laps_remaining: Laps remaining in race
        track_temp: Track temperature (°C)
        quantile: Degradation quantile to check (0.9 = conservative)
    
    Returns:
        Dict with:
        - will_last: bool (tires OK to finish)
        - expected_deg: float (predicted degradation %)
        - risk_level: str ("low" | "medium" | "high")
    """
    try:
        from ..models.wear_quantile_xgb import predict_quantiles
        
        # Predict degradation at end of race
        future_age = current_tire_age + laps_remaining
        features = pd.DataFrame([{
            "tire_age": future_age,
            "track_temp": track_temp,
            "lap": 999,  # Placeholder
        }])
        
        # Get conservative estimate (90th percentile = worse case)
        pred = predict_quantiles(wear_model, features, quantiles=[0.5, 0.9])
        degradation_pct = pred["q90"].iloc[0]
        
        # GR Cup tires typically good for 25-30 laps
        # Sprint races are 20-25 laps → should last with no issues
        if degradation_pct < 30.0:
            risk_level = "low"
            will_last = True
        elif degradation_pct < 50.0:
            risk_level = "medium"
            will_last = True
        else:
            risk_level = "high"
            will_last = False
        
        return {
            "will_last": will_last,
            "expected_deg": float(degradation_pct),
            "risk_level": risk_level,
        }
    except Exception:
        # Fallback: assume tires OK for sprint
        return {
            "will_last": True,
            "expected_deg": 20.0,
            "risk_level": "low",
        }




