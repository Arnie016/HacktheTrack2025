"""Caching utilities for models and predictions."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


class ModelCache:
    """Simple cache for loaded models."""
    
    def __init__(self):
        self._cache = {}
    
    def get(self, key: str):
        """Get cached model."""
        return self._cache.get(key)
    
    def set(self, key: str, value):
        """Cache model."""
        self._cache[key] = value
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()


# Global cache instance
_model_cache = ModelCache()


def get_cached_model(model_path: Path | str, loader_fn):
    """
    Get model from cache or load from disk.
    
    Args:
        model_path: Path to model file
        loader_fn: Function to load model if not cached
    
    Returns:
        Loaded model
    """
    path_str = str(model_path)
    
    cached = _model_cache.get(path_str)
    if cached is not None:
        return cached
    
    model = loader_fn(model_path)
    _model_cache.set(path_str, model)
    
    return model


# Precomputed wear grid cache
_wear_grid_cache: dict[str, pd.DataFrame] = {}


def get_wear_grid(
    tire_age_bins: list[int],
    temp_bins: list[float],
    sector_coeff_bins: list[float],
    model_data,  # Pre-loaded wear model dict
) -> pd.DataFrame:
    """
    Get or compute wear prediction grid.
    
    Caches grid for O(1) lookup during inference.
    """
    cache_key = f"{len(tire_age_bins)}_{len(temp_bins)}_{len(sector_coeff_bins)}"
    
    if cache_key in _wear_grid_cache:
        return _wear_grid_cache[cache_key]
    
    # Compute grid
    grid_data = []
    
    for tire_age in tire_age_bins:
        for temp in temp_bins:
            for coeff in sector_coeff_bins:
                features_dict = {
                    "tire_age": tire_age,
                    "track_temp": temp,
                    "stint_len": tire_age,
                    "sector_S3_coeff": coeff,
                    "clean_air": 1.0,
                    "traffic_density": 0.0,
                    "driver_TE": 0.0,
                }
                
                from src.grcup.models import predict_quantiles
                
                features_df = pd.DataFrame([features_dict])
                preds = predict_quantiles(model_data, features_df)
                
                grid_data.append({
                    "tire_age": tire_age,
                    "track_temp": temp,
                    "sector_S3_coeff": coeff,
                    **preds.iloc[0].to_dict(),
                })
    
    grid_df = pd.DataFrame(grid_data)
    _wear_grid_cache[cache_key] = grid_df
    
    return grid_df

