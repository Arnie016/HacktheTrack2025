"""Latin Hypercube sampling for scenario generation."""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def generate_lhs_samples(
    n_samples: int,
    n_dims: int,
    bounds: list[tuple[float, float]] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate Latin Hypercube samples.
    
    Args:
        n_samples: Number of samples
        n_dims: Number of dimensions
        bounds: List of (min, max) tuples per dimension (default: [0, 1])
        seed: Random seed
    
    Returns:
        Array of shape (n_samples, n_dims)
    """
    if bounds is None:
        bounds = [(0.0, 1.0)] * n_dims
    
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n=n_samples)
    
    # Transform to bounds
    result = np.zeros_like(samples)
    for i, (min_val, max_val) in enumerate(bounds):
        result[:, i] = samples[:, i] * (max_val - min_val) + min_val
    
    return result


def sample_from_quantiles(
    quantile_preds: dict[str, float],  # {"q10": val, "q50": val, "q90": val}
    random_state: np.random.Generator,
) -> float:
    """
    Sample a single value from quantile predictions.
    
    Uses inverse transform sampling on triangular distribution.
    
    Args:
        quantile_preds: Dict with q10, q50, q90 values
        random_state: RNG for sampling
    
    Returns:
        Sampled pace delta
    """
    q10 = quantile_preds.get("q10", quantile_preds.get("q10", 0.0))
    q50 = quantile_preds.get("q50", quantile_preds.get("q50", 0.0))
    q90 = quantile_preds.get("q90", quantile_preds.get("q90", 0.0))
    
    # Triangular distribution approximation
    # Sample uniform
    u = random_state.uniform()
    
    # Inverse CDF of triangular
    if u < 0.5:
        # Left side (q10 to q50)
        return q10 + np.sqrt(u * 2) * (q50 - q10)
    else:
        # Right side (q50 to q90)
        return q50 + np.sqrt((u - 0.5) * 2) * (q90 - q50)


def sample_pit_loss(
    mean_loss: float,
    std_loss: float,
    random_state: np.random.Generator,
) -> float:
    """Sample pit stop loss from normal distribution."""
    return max(0.0, random_state.normal(mean_loss, std_loss))


def sample_traffic_loss(
    traffic_density: float,
    random_state: np.random.Generator,
    base_loss: float = 0.5,  # Base time loss in traffic
) -> float:
    """Sample traffic-induced time loss."""
    loss = base_loss * traffic_density * random_state.uniform(0.5, 1.5)
    return max(0.0, loss)

