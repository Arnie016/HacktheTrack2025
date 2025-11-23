"""Monte Carlo simulation for position forecasting (with numba fallback)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

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


@dataclass
class ConvergenceMonitor:
    """Lightweight convergence tracker for Monte Carlo estimates."""

    window: int = 200  # Increased from 100 for better stability
    tolerance: float = 0.05  # Tighter tolerance: 0.05s instead of 0.1s
    min_samples: int = 1000  # Increased from 500 for better initial estimate
    max_samples: int | None = None
    samples: list[float] = field(default_factory=list)

    def update(self, value: float) -> bool:
        """Record a new sample and return True if convergence criteria are met."""
        self.samples.append(float(value))
        return self.should_stop()

    def should_stop(self) -> bool:
        if self.max_samples is not None and len(self.samples) >= self.max_samples:
            return True
        if len(self.samples) < self.min_samples or len(self.samples) < 2 * self.window:
            return False
        recent = np.mean(self.samples[-self.window :])
        prev = np.mean(self.samples[-2 * self.window : -self.window])
        return abs(recent - prev) < self.tolerance

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.samples)) if self.samples else float("nan")
    
    @property
    def n_samples(self) -> int:
        return len(self.samples)
    
    def get_stats(self) -> dict:
        """Get comprehensive statistics."""
        if not self.samples:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "n": 0,
                "ci95": (float("nan"), float("nan")),
            }
        arr = np.array(self.samples, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": len(arr),
            "ci95": confidence_interval(self.samples, alpha=0.05),
            "converged": self.should_stop() if len(self.samples) >= self.min_samples else False,
        }


def confidence_interval(samples: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    """Return simple percentile-based CI."""
    if not samples:
        return (float("nan"), float("nan"))
    arr = np.asarray(samples, dtype=float)
    low = float(np.percentile(arr, (alpha / 2) * 100))
    high = float(np.percentile(arr, (1 - alpha / 2) * 100))
    return (low, high)


def simulate_with_antithetic_variates(
    current_lap: int,
    total_laps: int,
    pace_mean: float,
    pace_std: float,
    degradation_rate: float,
    pit_lap: int,
    pit_loss: float,
    sc_prob: float,
    n_scenarios: int,
    seed: int = 42,
) -> tuple[float, float, int]:
    """
    Monte Carlo with antithetic variates for variance reduction.
    
    Technique: For each random number z ~ N(0,1), also simulate with -z.
    This creates negative correlation between pairs, reducing variance ~50%.
    
    Args:
        (same as _simulate_race_lap_impl)
        n_scenarios: Number of scenario pairs (actual sims = 2 * n_scenarios)
        seed: Random seed
    
    Returns:
        (mean_time, std_time, n_samples)
    """
    rng = np.random.default_rng(seed)
    results = []
    
    for _ in range(n_scenarios):
        # Generate random seed for this pair
        pair_seed = rng.integers(0, 2**31)
        
        # Simulation 1: Normal random numbers
        rng1 = np.random.default_rng(pair_seed)
        time1 = _simulate_race_lap_impl(
            current_lap=current_lap,
            total_laps=total_laps,
            pace_mean=pace_mean,
            pace_std=pace_std,
            degradation_rate=degradation_rate,
            pit_lap=pit_lap,
            pit_loss=pit_loss,
            sc_prob=sc_prob,
            rng=rng1,
        )
        results.append(time1)
        
        # Simulation 2: Antithetic (inverted) random numbers
        # Use same seed but invert normal draws
        rng2 = np.random.default_rng(pair_seed)
        time2 = _simulate_race_lap_antithetic(
            current_lap=current_lap,
            total_laps=total_laps,
            pace_mean=pace_mean,
            pace_std=pace_std,
            degradation_rate=degradation_rate,
            pit_lap=pit_lap,
            pit_loss=pit_loss,
            sc_prob=sc_prob,
            rng=rng2,
        )
        results.append(time2)
    
    results_arr = np.array(results, dtype=float)
    mean_time = float(np.mean(results_arr))
    std_time = float(np.std(results_arr))
    n_samples = len(results_arr)
    
    return mean_time, std_time, n_samples


def _simulate_race_lap_antithetic(
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
    Antithetic simulation: invert normal random draws.
    
    If normal sim uses z ~ N(0,1), this uses -z ~ N(0,1).
    Creates negative correlation, reducing variance.
    """
    total_time = 0.0
    tire_age = 0.0
    
    for lap in range(current_lap, total_laps + 1):
        # Pit stop
        if lap == pit_lap:
            total_time += pit_loss
            tire_age = 0.0
        
        # Safety car (keep same boolean logic)
        under_sc = rng.random() < sc_prob
        
        if under_sc:
            lap_time = pace_mean * 1.5
        else:
            # Degradation
            degradation = tire_age * degradation_rate
            
            # ANTITHETIC: Invert normal draw
            z = rng.normal(0, pace_std)
            antithetic_z = -z  # Invert
            
            pace = pace_mean + degradation + antithetic_z
            lap_time = max(pace, pace_mean * 0.8)
        
        total_time += lap_time
        tire_age += 1.0
    
    return total_time


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
