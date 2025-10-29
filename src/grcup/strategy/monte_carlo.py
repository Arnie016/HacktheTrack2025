"""Monte Carlo simulation for position forecasting (with numba fallback)."""
from __future__ import annotations

import numpy as np

try:
    from numba import jit
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    # Fallback decorator (no-op)
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def _simulate_race_lap_impl(
    current_lap: int,
    total_laps: int,
    pace_mean: float,
    pace_std: float,
    degradation_rate: float,
    pit_lap: int,
    pit_loss: float,
    sc_prob: float,
    rng: np.random.Generator,
) -> float:
    """
    Single race simulation implementation.
    
    Returns:
        Total race time (seconds)
    """
    total_time = 0.0
    tire_age = 0.0
    
    for lap in range(current_lap, total_laps + 1):
        # Pit stop
        if lap == pit_lap:
            total_time += pit_loss
            tire_age = 0.0
        
        # Safety car
        under_sc = rng.random() < sc_prob
        
        if under_sc:
            lap_time = pace_mean * 1.5  # SC pace slower
        else:
            # Degradation
            degradation = tire_age * degradation_rate
            
            # Random variation
            pace = pace_mean + degradation + rng.normal(0, pace_std)
            
            lap_time = max(pace, pace_mean * 0.8)  # Lower bound
        
        total_time += lap_time
        tire_age += 1.0
    
    return total_time


# Use numba if available, otherwise plain function
if USE_NUMBA:
    simulate_race_lap = jit(nopython=True, cache=True)(_simulate_race_lap_impl)
else:
    simulate_race_lap = _simulate_race_lap_impl


def simulate_position_distribution(
    n_sims: int,
    current_positions: dict[str, float],  # {car_id: current_time}
    pace_params: dict[str, dict],  # {car_id: {mean, std, degradation, ...}}
    pit_strategies: dict[str, int],  # {car_id: pit_lap}
    pit_loss_dist: tuple[float, float] = (30.0, 5.0),  # (mean, std)
    sc_prob: float = 0.05,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Simulate race finish positions using Monte Carlo.
    
    Args:
        n_sims: Number of simulations
        current_positions: Current total time per car
        pace_params: Pace parameters per car
        pit_strategies: Pit lap per car
        pit_loss_dist: Pit loss distribution (mean, std)
        sc_prob: Safety car probability per lap
        seed: Random seed
    
    Returns:
        Dict mapping car_id to dict of position probabilities {1: 0.6, 2: 0.3, ...}
    """
    rng = np.random.default_rng(seed)
    car_ids = list(current_positions.keys())
    n_cars = len(car_ids)
    
    # Pre-allocate result arrays
    final_times = {car_id: np.empty(n_sims, dtype=np.float32) for car_id in car_ids}
    
    # Simulate each scenario
    for sim_idx in range(n_sims):
        sim_times = {}
        
        for car_id in car_ids:
            params = pace_params[car_id]
            pit_lap = pit_strategies.get(car_id, -1)
            
            # Sample pit loss
            pit_loss = max(0.0, rng.normal(pit_loss_dist[0], pit_loss_dist[1]))
            
            # Run simulation (with or without numba)
            race_time = _simulate_race_lap_impl(
                current_lap=1,  # Simplified
                total_laps=20,  # Simplified
                pace_mean=params["mean"],
                pace_std=params["std"],
                degradation_rate=params["degradation"],
                pit_lap=pit_lap,
                pit_loss=pit_loss,
                sc_prob=sc_prob,
                rng=rng,
            )
            
            sim_times[car_id] = current_positions[car_id] + race_time
        
        # Rank positions
        sorted_cars = sorted(sim_times.items(), key=lambda x: x[1])
        
        for position, (car_id, _) in enumerate(sorted_cars, start=1):
            final_times[car_id][sim_idx] = position
    
    # Convert to position probabilities
    position_probs = {}
    for car_id in car_ids:
        positions = final_times[car_id]
        probs = {}
        for pos in range(1, n_cars + 1):
            probs[pos] = float(np.mean(positions == pos))
        position_probs[car_id] = probs
    
    return position_probs
