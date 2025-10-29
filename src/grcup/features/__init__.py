"""Feature engineering: stints, sectors, traffic."""
from .feature_extractor import (
    build_pace_prediction_features,
    build_wear_training_dataset,
    join_weather_to_laps,
    target_encode_categorical,
)
from .sector_fingerprint import (
    compute_sector_temp_sensitivity,
    compute_traffic_density,
    detect_clean_air_flag,
)
from .stint_detector import (
    compute_tire_age_per_lap,
    detect_stints,
    estimate_pit_loss_empirical,
)

__all__ = [
    "detect_stints",
    "compute_tire_age_per_lap",
    "estimate_pit_loss_empirical",
    "compute_sector_temp_sensitivity",
    "detect_clean_air_flag",
    "compute_traffic_density",
    "join_weather_to_laps",
    "build_wear_training_dataset",
    "build_pace_prediction_features",
    "target_encode_categorical",
]

