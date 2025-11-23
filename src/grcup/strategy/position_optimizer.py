"""Position-aware pit strategy optimization for race position gain."""
from __future__ import annotations

import numpy as np
from typing import Optional


def compute_position_value(
    current_position: int,
    total_cars: int = 20,
    position_value_per_place: float = 3.0,
) -> float:
    """
    Compute time value of track position.
    
    In sprint racing, track position is valuable because passing is difficult.
    Losing 1 position ~ losing 3 seconds (hard to overtake on track).
    
    Args:
        current_position: Current track position (1 = P1)
        total_cars: Total cars in race
        position_value_per_place: Seconds per position (default 3s)
    
    Returns:
        Time value of current position in seconds
    """
    # Leaders have more to lose, backmarkers have less
    # P1 = 60s value, P20 = 3s value (decreasing)
    value_multiplier = (total_cars - current_position + 1) / total_cars
    return position_value_per_place * value_multiplier * current_position


def estimate_position_change_from_pit(
    current_position: int,
    gap_ahead: float,
    gap_behind: float,
    pit_loss_time: float,
    field_gaps: Optional[list[float]] = None,
) -> int:
    """
    Estimate position change after pit stop.
    
    Args:
        current_position: Current position (1 = P1)
        gap_ahead: Gap to car ahead (seconds)
        gap_behind: Gap to car behind (seconds)
        pit_loss_time: Expected pit stop time loss (seconds)
        field_gaps: List of gaps between all positions (optional)
    
    Returns:
        Position change (negative = drop positions)
    """
    if field_gaps is None:
        # Assume average 2s gaps between cars
        field_gaps = [2.0] * 20
    
    # Pit stop loses time, drops positions
    positions_lost = 0
    time_lost_accumulator = 0.0
    
    # Start from car behind, count how many pass us
    for i in range(len(field_gaps)):
        time_lost_accumulator += field_gaps[i]
        
        if time_lost_accumulator <= pit_loss_time:
            positions_lost += 1
        else:
            break
    
    return -positions_lost  # Negative because we drop positions


def optimize_for_position_gain(
    current_lap: int,
    total_laps: int,
    current_position: int,
    gap_ahead: float,
    gap_behind: float,
    pit_loss_mean: float,
    tire_age: float,
    remaining_laps: int,
    degradation_rate: float = 0.1,  # s/lap tire deg
    overtake_difficulty: float = 3.0,  # s needed to overtake
) -> dict:
    """
    Optimize strategy for position gain, not just time.
    
    Key insight: In sprint racing, position > time.
    - If close behind leader (gap < 2s): Aggressive undercut strategy
    - If comfortable gap (gap > 5s): Conservative tire preservation
    - If mid-pack battle: Opportunistic SC-reactive strategy
    
    Args:
        current_lap: Current lap
        total_laps: Total race laps
        current_position: Current position (1 = P1)
        gap_ahead: Gap to car ahead (seconds)
        gap_behind: Gap to car behind (seconds)
        pit_loss_mean: Expected pit loss (seconds)
        tire_age: Current tire age (laps)
        remaining_laps: Laps remaining
        degradation_rate: Tire degradation per lap (s/lap)
        overtake_difficulty: Time needed to attempt overtake (s)
    
    Returns:
        Dict with strategy recommendation and reasoning
    """
    # Strategy modes based on position context
    
    # 1. UNDERCUT OPPORTUNITY: Close behind, can undercut
    if gap_ahead < overtake_difficulty and gap_ahead > 0.5:
        # Close enough to undercut, not close enough to pass on track
        undercut_benefit = degradation_rate * remaining_laps  # Fresh tires gain
        pit_cost = pit_loss_mean - gap_ahead  # Net cost after undercut
        
        if undercut_benefit > pit_cost:
            return {
                "strategy": "aggressive_undercut",
                "recommended_lap": current_lap + 1,
                "reasoning": f"Undercut opportunity: {gap_ahead:.1f}s gap, fresh tire advantage ~{undercut_benefit:.1f}s",
                "expected_position_change": +1,  # Gain 1 position
                "confidence": 0.7,
            }
    
    # 2. DEFENSIVE POSITION: Close car behind, must maintain gap
    if gap_behind < overtake_difficulty and gap_behind > 0.5:
        # Car behind close enough to undercut us
        # Pit earlier to cover their undercut
        return {
            "strategy": "defensive_cover",
            "recommended_lap": current_lap,
            "reasoning": f"Cover undercut threat: {gap_behind:.1f}s gap behind",
            "expected_position_change": 0,  # Maintain position
            "confidence": 0.6,
        }
    
    # 3. CLEAR AIR: Large gaps, optimize for time not position
    if gap_ahead > 5.0 and gap_behind > 5.0:
        # Comfortable gaps, focus on tire management
        optimal_stint_length = int(20.0 / (1.0 + degradation_rate))  # Heuristic
        recommended_lap = current_lap + max(5, optimal_stint_length - tire_age)
        
        return {
            "strategy": "optimal_stint",
            "recommended_lap": min(recommended_lap, total_laps - 2),
            "reasoning": f"Clear air ({gap_ahead:.1f}s ahead, {gap_behind:.1f}s behind), optimize tire life",
            "expected_position_change": 0,
            "confidence": 0.8,
        }
    
    # 4. PACK RACING: Tight mid-pack, wait for SC opportunity
    if current_position > 5 and gap_ahead < 2.0:
        # Mid-pack battle, don't pit yet (wait for SC or settle)
        return {
            "strategy": "hold_position",
            "recommended_lap": current_lap + 5,  # Delay pit
            "reasoning": f"Pack racing (P{current_position}), wait for SC or gap opening",
            "expected_position_change": 0,
            "confidence": 0.5,
        }
    
    # 5. DEFAULT: Standard optimization
    return {
        "strategy": "standard",
        "recommended_lap": current_lap + 10,  # Default mid-stint
        "reasoning": "Standard strategy (no special position factors)",
        "expected_position_change": 0,
        "confidence": 0.6,
    }


def compute_position_aware_objective(
    expected_time: float,
    expected_position_change: int,
    current_position: int,
    position_weight: float = 0.7,  # Weight position over time
) -> float:
    """
    Combine time and position into single objective.
    
    Objective = time_component * (1 - position_weight) + position_component * position_weight
    
    Args:
        expected_time: Expected race time (seconds)
        expected_position_change: Expected position change (negative = drop)
        current_position: Current position
        position_weight: How much to weight position vs time (0-1)
    
    Returns:
        Combined objective (lower is better)
    """
    # Normalize time to ~0-100 range
    time_component = expected_time / 100.0
    
    # Position component: losing positions is bad
    # Each position lost ~ 3 points penalty
    position_component = -expected_position_change * 3.0
    
    # Combined objective
    objective = (
        time_component * (1.0 - position_weight) +
        position_component * position_weight
    )
    
    return objective


def select_best_strategy_with_position(
    strategies: list[dict],
    current_position: int,
    position_weight: float = 0.7,
) -> dict:
    """
    Select best strategy considering both time and position.
    
    Args:
        strategies: List of strategy dicts with expected_time, expected_position_change
        current_position: Current track position
        position_weight: Weight for position vs time (0 = time only, 1 = position only)
    
    Returns:
        Best strategy
    """
    best_strategy = None
    best_objective = float('inf')
    
    for strategy in strategies:
        objective = compute_position_aware_objective(
            expected_time=strategy.get("expected_time", 0.0),
            expected_position_change=strategy.get("expected_position_change", 0),
            current_position=current_position,
            position_weight=position_weight,
        )
        
        if objective < best_objective:
            best_objective = objective
            best_strategy = strategy
    
    if best_strategy:
        best_strategy["position_aware_objective"] = best_objective
    
    return best_strategy

