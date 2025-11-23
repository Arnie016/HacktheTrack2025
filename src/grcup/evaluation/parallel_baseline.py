"""Parallel processing for baseline strategy comparisons."""
from __future__ import annotations

import os
from multiprocessing import Pool, cpu_count
from typing import Callable, Any
import numpy as np


def run_baseline_parallel(
    baseline_strategies: list[str],
    simulation_func: Callable,
    simulation_args: dict[str, Any],
    n_workers: int | None = None,
) -> dict[str, dict]:
    """
    Run baseline comparisons in parallel.
    
    Args:
        baseline_strategies: List of baseline names ["fixed_stint", "fuel_min", ...]
        simulation_func: Function that runs baseline simulation
        simulation_args: Arguments to pass to simulation_func
        n_workers: Number of parallel workers (default: CPU count - 1)
    
    Returns:
        Dict mapping baseline name to results
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave 1 CPU free
    
    # Check if parallel processing is disabled
    if os.getenv("DISABLE_PARALLEL", "0") == "1":
        n_workers = 1
    
    if n_workers == 1:
        # Sequential fallback
        results = {}
        for baseline in baseline_strategies:
            results[baseline] = simulation_func(baseline, **simulation_args)
        return results
    
    # Parallel processing
    with Pool(processes=n_workers) as pool:
        # Create tasks
        tasks = [
            (baseline, simulation_func, simulation_args)
            for baseline in baseline_strategies
        ]
        
        # Run in parallel
        results_list = pool.starmap(_run_baseline_worker, tasks)
    
    # Convert to dict
    results = {
        baseline: result
        for baseline, result in zip(baseline_strategies, results_list)
    }
    
    return results


def _run_baseline_worker(
    baseline_name: str,
    simulation_func: Callable,
    simulation_args: dict,
) -> dict:
    """Worker function for parallel baseline execution."""
    try:
        result = simulation_func(baseline_name, **simulation_args)
        return result
    except Exception as e:
        return {
            "error": str(e),
            "mean_time": float('inf'),
            "scenarios": [],
        }


def parallel_monte_carlo(
    simulation_func: Callable,
    n_scenarios: int,
    simulation_args: dict,
    n_workers: int | None = None,
    batch_size: int = 100,
) -> list[float]:
    """
    Run Monte Carlo scenarios in parallel batches.
    
    Args:
        simulation_func: Function that runs single scenario
        n_scenarios: Total scenarios to run
        simulation_args: Arguments to pass to simulation_func
        n_workers: Number of parallel workers
        batch_size: Scenarios per batch
    
    Returns:
        List of scenario results
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    if os.getenv("DISABLE_PARALLEL", "0") == "1":
        n_workers = 1
    
    if n_workers == 1:
        # Sequential
        return [simulation_func(**simulation_args) for _ in range(n_scenarios)]
    
    # Split into batches
    n_batches = (n_scenarios + batch_size - 1) // batch_size
    batch_sizes = [
        min(batch_size, n_scenarios - i * batch_size)
        for i in range(n_batches)
    ]
    
    with Pool(processes=n_workers) as pool:
        # Create batch tasks
        tasks = [
            (simulation_func, batch_size, simulation_args, i)
            for i, batch_size in enumerate(batch_sizes)
        ]
        
        # Run batches in parallel
        batch_results = pool.starmap(_run_batch_worker, tasks)
    
    # Flatten results
    all_results = []
    for batch_result in batch_results:
        all_results.extend(batch_result)
    
    return all_results


def _run_batch_worker(
    simulation_func: Callable,
    batch_size: int,
    simulation_args: dict,
    batch_idx: int,
) -> list[float]:
    """Worker function for parallel batch execution."""
    # Use different seed per batch
    seed = simulation_args.get("seed", 42) + batch_idx * 10000
    args_with_seed = {**simulation_args, "seed": seed}
    
    results = []
    for i in range(batch_size):
        # Increment seed per scenario within batch
        args_with_seed["seed"] = seed + i
        try:
            result = simulation_func(**args_with_seed)
            results.append(result)
        except Exception:
            # Skip failed scenarios
            continue
    
    return results


def estimate_parallel_speedup(
    n_scenarios: int,
    scenario_time_ms: float = 10.0,
    n_workers: int | None = None,
) -> dict:
    """
    Estimate speedup from parallel processing.
    
    Args:
        n_scenarios: Number of scenarios
        scenario_time_ms: Time per scenario (ms)
        n_workers: Number of workers
    
    Returns:
        Dict with sequential_time, parallel_time, speedup
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    sequential_time = n_scenarios * scenario_time_ms / 1000.0  # seconds
    
    # Amdahl's law: parallel_time = sequential_time / workers + overhead
    overhead_fraction = 0.05  # 5% overhead for communication
    parallel_fraction = 1.0 - overhead_fraction
    
    parallel_time = (
        sequential_time * overhead_fraction +
        sequential_time * parallel_fraction / n_workers
    )
    
    speedup = sequential_time / parallel_time
    
    return {
        "sequential_time_s": sequential_time,
        "parallel_time_s": parallel_time,
        "speedup": speedup,
        "n_workers": n_workers,
        "efficiency": speedup / n_workers,
    }

