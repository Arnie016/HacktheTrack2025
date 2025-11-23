"""Convenience exports for GR Cup model components."""

from importlib import import_module
from typing import Any, Dict, Tuple

from .sc_hazard import (
    prepare_hazard_data,
    train_cox_hazard,
    predict_sc_probability,
    save_hazard_model,
    load_hazard_model,
)
from .wear_quantile_xgb import (
    train_wear_quantile_model,
    predict_quantiles,
    save_model,
    load_model,
    build_feature_vector,
)
from .kalman_pace import (
    KalmanPaceFilter,
    save_kalman_config,
    load_kalman_config,
)

__all__ = [
    "prepare_hazard_data",
    "train_cox_hazard",
    "predict_sc_probability",
    "save_hazard_model",
    "load_hazard_model",
    "train_wear_quantile_model",
    "predict_quantiles",
    "save_model",
    "load_model",
    "build_feature_vector",
    "KalmanPaceFilter",
    "save_kalman_config",
    "load_kalman_config",
    "train_overtake_model",
    "save_overtake_model",
    "load_overtake_model",
    "prepare_overtake_data",
]

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "train_overtake_model": (".overtake_model", "train_overtake_model"),
    "save_overtake_model": (".overtake_model", "save_overtake_model"),
    "load_overtake_model": (".overtake_model", "load_overtake_model"),
    "prepare_overtake_data": (".overtake_model", "prepare_overtake_data"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load heavy modules (e.g., overtake) only when requested."""
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

