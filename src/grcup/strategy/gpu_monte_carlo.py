"""GPU-accelerated Monte Carlo simulator for pit strategy."""
from __future__ import annotations

from typing import Optional, Sequence, Any, Union

import numpy as np
import pandas as pd

from ..models.wear_quantile_xgb import predict_quantiles, build_feature_vector
from ..models.sc_hazard import predict_sc_probability

try:
    import torch
except ImportError:
    torch = None


def _check_torch_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def _build_feature_row(age: float, track_temp: float, wear_model) -> dict:
    """Build a feature row compatible with the wear model."""
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
    return build_feature_vector(wear_model, overrides)


def _build_degradation_lookup(wear_model, max_age: int, track_temp: float, device) -> tuple:
    """Pre-compute degradation mean/std lookup tensors for ages 0..max_age."""
    ages = list(range(max_age + 1))
    rows = [_build_feature_row(age, track_temp, wear_model) for age in ages]
    features_df = pd.DataFrame(rows)
    quantiles_df = predict_quantiles(wear_model, features_df)
    
    mean = torch.tensor(quantiles_df["q50"].values, device=device, dtype=torch.float32)
    spread = torch.tensor(
        (quantiles_df["q90"] - quantiles_df["q10"]).values,
        device=device,
        dtype=torch.float32,
    )
    std = torch.clamp(spread / 2.0, min=0.05)
    return mean, std


def _compute_sc_probs(sc_hazard_model, lap_numbers: list[int], phase: Optional[str]) -> list[float]:
    """Compute safety-car probabilities for each lap."""
    probs = []
    for idx, lap in enumerate(lap_numbers):
        green_run = max(1.0, lap_numbers[idx] - lap_numbers[0] + 1) if lap_numbers else 1.0
        base_prob = 0.03
        if sc_hazard_model is not None:
            base_prob = predict_sc_probability(
                sc_hazard_model,
                green_run_len=green_run,
                pack_density=0.5,
                rain=0,
                wind_speed=0.0,
                k_laps=1,
            )
        if phase == "early_sc" and idx < 7:
            base_prob = min(1.0, base_prob + 0.3)
        elif phase == "late_sc" and idx >= int(0.8 * len(lap_numbers)):
            base_prob = min(1.0, base_prob + 0.3)
        elif phase == "heavy_traffic":
            base_prob = min(1.0, base_prob + 0.1)
        probs.append(float(base_prob))
    return probs


def simulate_strategy_gpu(
    current_lap: int,
    total_laps: int,
    pit_lap: int,
    tire_age: float,
    wear_model,
    sc_hazard_model,
    pit_loss_mean: float,
    pit_loss_std: float,
    num_scenarios: int,
    track_temp: float,
    base_pace: float = 130.0,
    scenario_phase: Optional[str] = None,
) -> dict:
    """Simulate a candidate strategy entirely on the GPU."""
    if not _check_torch_available():
        raise RuntimeError("Torch with CUDA is required for GPU Monte Carlo")
    
    device = torch.device("cuda")
    scenarios = num_scenarios
    pit_lap = max(pit_lap, current_lap)
    
    stint1_laps = max(0, pit_lap - current_lap)
    stint2_laps = max(0, total_laps - pit_lap + 1)
    max_age = int(max(tire_age + stint1_laps, stint2_laps) + 5)
    
    degrade_mean, degrade_std = _build_degradation_lookup(wear_model, max_age, track_temp, device)
    
    def sample_stint(age_start: float, laps: int, sc_probs: list[float]) -> torch.Tensor:
        if laps <= 0:
            return torch.zeros(scenarios, device=device)
        age_indices = torch.arange(age_start, age_start + laps, device=device)
        age_indices = torch.clamp(age_indices.round().long(), max=degrade_mean.numel() - 1)
        means = degrade_mean[age_indices]
        stds = torch.clamp(degrade_std[age_indices], min=0.02)
        noise = torch.randn(scenarios, laps, device=device)
        degradation = noise * stds.unsqueeze(0) + means.unsqueeze(0)
        lap_times = base_pace + degradation
        lap_times = lap_times + 0.3 * torch.randn_like(lap_times)
        
        if sc_probs:
            sc_tensor = torch.tensor(sc_probs, device=device, dtype=torch.float32)
            sc_random = torch.rand(scenarios, laps, device=device)
            lap_times = torch.where(
                sc_random < sc_tensor.unsqueeze(0),
                torch.full_like(lap_times, base_pace * 1.5),
                lap_times,
            )
        return lap_times.sum(dim=1)
    
    lap_nums_stint1 = list(range(current_lap, pit_lap))
    lap_nums_stint2 = list(range(pit_lap, total_laps + 1))
    sc_probs1 = _compute_sc_probs(sc_hazard_model, lap_nums_stint1, scenario_phase)
    sc_probs2 = _compute_sc_probs(sc_hazard_model, lap_nums_stint2, scenario_phase)
    
    stint1_time = sample_stint(tire_age, stint1_laps, sc_probs1)
    stint2_time = sample_stint(0.0, stint2_laps, sc_probs2)
    
    pit_losses = torch.randn(scenarios, device=device) * pit_loss_std + pit_loss_mean
    pit_losses = torch.clamp(pit_losses, min=pit_loss_mean * 0.5)
    
    total_times = stint1_time + pit_losses + stint2_time
    mean_time = float(total_times.mean().item())
    std_time = float(total_times.std(unbiased=False).item())
    samples = total_times.detach().cpu().numpy()
    
    return {
        "mean_time": mean_time,
        "std_time": std_time,
        "samples": samples,
    }


def simulate_strategy_vectorized(
    current_lap: int,
    total_laps: int,
    initial_tire_age: float,
    initial_fuel_laps: float,
    under_sc: bool,
    pit_schedule: list[int],
    wear_model: dict,
    sc_hazard_model: Optional[Any],
    pit_loss_mean: float = 30.0,
    pit_loss_std: float = 5.0,
    n_scenarios: int = 1000,
    random_state: Optional[np.random.Generator] = None,
    track_temp: float = 50.0,
    scenario_seeds: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """
    Vectorized Monte Carlo simulation (CPU optimized).
    Orders of magnitude faster than row-by-row simulation.
    """
    if random_state is None:
        random_state = np.random.default_rng(42)
    
    # 1. Pre-compute wear lookup table for all possible tire ages
    # Max laps we might see is total_laps + buffer
    max_possible_age = int(total_laps + 30)
    ages = np.arange(max_possible_age + 1)
    
    # Batch predict quantiles (xgboost is fast at batch inference)
    rows = [_build_feature_row(float(a), track_temp, wear_model) for a in ages]
    features_df = pd.DataFrame(rows)
    
    try:
        quantiles_df = predict_quantiles(wear_model, features_df)
        q10_lookup = quantiles_df["q10"].values
        q50_lookup = quantiles_df["q50"].values
        q90_lookup = quantiles_df["q90"].values
    except Exception:
        # Fallback if model fails
        base_deg = ages * 0.1
        q10_lookup = np.maximum(0.0, base_deg * 0.8)
        q50_lookup = base_deg
        q90_lookup = base_deg * 1.5

    # 2. Initialize State Vectors (Shape: [n_scenarios])
    sim_time = np.zeros(n_scenarios, dtype=np.float64)
    sim_tire_age = np.full(n_scenarios, initial_tire_age, dtype=np.float64)
    sim_under_sc = np.full(n_scenarios, under_sc, dtype=bool)
    
    # Pre-compute SC probability profile (simplified)
    # We use a scalar prob per lap for vectorization speed
    sc_probs = np.full(total_laps + 2, 0.02) # Default 2%
    if sc_hazard_model:
        for lap in range(current_lap, total_laps + 1):
            try:
                prob = predict_sc_probability(
                    sc_hazard_model, 
                    green_run_len=max(1.0, lap-current_lap),
                    pack_density=0.5, rain=0, wind_speed=0, k_laps=1
                )
                sc_probs[lap] = max(0.02, prob)
            except:
                pass

    # 3. Vectorized Lap Loop
    # Loop through laps, but process all scenarios in parallel for that lap
    for lap in range(current_lap, total_laps + 1):
        # A. Pit Stop Logic
        if lap in pit_schedule:
            pit_loss = random_state.normal(pit_loss_mean, pit_loss_std, size=n_scenarios)
            sim_time += np.maximum(pit_loss, 15.0)
            sim_tire_age[:] = 0.0
        
        # B. Safety Car Transitions (Vectorized)
        # Chance to enter SC
        prob_enter = sc_probs[lap]
        enter_sc_mask = (random_state.random(n_scenarios) < prob_enter) & (~sim_under_sc)
        sim_under_sc[enter_sc_mask] = True
        
        # Chance to leave SC (fixed ~20% per lap)
        leave_sc_mask = (random_state.random(n_scenarios) < 0.2) & (sim_under_sc)
        sim_under_sc[leave_sc_mask] = False
        
        # C. Calculate Lap Time
        # Look up wear params based on current tire age
        # Clip indices to valid range
        age_indices = np.clip(np.round(sim_tire_age).astype(int), 0, max_possible_age)
        
        q10 = q10_lookup[age_indices]
        q50 = q50_lookup[age_indices]
        q90 = q90_lookup[age_indices]
        
        # Fix crossovers
        bad_mask = q10 > q90
        if np.any(bad_mask):
            q90[bad_mask] = q50[bad_mask] + 0.1
        
        # Sample degradation
        degradation = random_state.triangular(q10, q50, q90)
        
        # Base pace: 180s (SC) vs 130s (Green) - simplified
        base_pace = np.where(sim_under_sc, 180.0, 130.0)
        
        sim_time += (base_pace + degradation)
        sim_tire_age += 1.0
        
    return {
        "mean": float(np.mean(sim_time)),
        "std": float(np.std(sim_time)),
        "samples": sim_time.tolist(),
        "n": n_scenarios
    }
