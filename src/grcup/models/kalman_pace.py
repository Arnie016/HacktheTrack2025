"""3-regime Kalman filter for online pace estimation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np


class KalmanPaceFilter:
    """Regime-switch Kalman filter for pace estimation.
    
    Regimes: green (normal), pit-out (fresh tires), SC (safety car)
    """
    
    def __init__(
        self,
        initial_pace: float = 130.0,  # seconds
        process_noise: float = 0.1,  # Base process noise
        measurement_noise: float = 0.5,  # Base measurement noise
    ):
        self.initial_pace = initial_pace
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State: [pace_mean, pace_variance]
        self.pace_mean = initial_pace
        self.pace_variance = 1.0  # Initial uncertainty
        
        # Regime tracking
        self.current_regime: Literal["green", "pit-out", "SC"] = "green"
    
    def update(
        self,
        lap_time: float,
        flags: dict | None = None,
        traffic_density: float = 0.0,
    ) -> tuple[float, float]:
        """
        Update filter with new lap time observation.
        
        Args:
            lap_time: Observed lap time (seconds)
            flags: Dict with flag information (e.g., {"FLAG_AT_FL": "FCY"})
            traffic_density: Traffic density proxy (0-1)
        
        Returns:
            (pace_mean, pace_variance) after update
        """
        flags = flags or {}
        
        # Detect regime
        if flags.get("FLAG_AT_FL") == "FCY" or flags.get("safety_car"):
            regime = "SC"
        elif flags.get("pit_out") or flags.get("PIT_TIME"):
            regime = "pit-out"
        else:
            regime = "green"
        
        # Adjust process noise based on regime
        if regime != self.current_regime:
            # Regime change - higher process noise
            process_noise = self.process_noise * 3.0
        else:
            process_noise = self.process_noise
        
        # Adjust measurement noise based on traffic
        measurement_noise = self.measurement_noise * (1.0 + traffic_density)
        
        # Predict step
        # pace_mean stays same, variance increases
        pred_variance = self.pace_variance + process_noise
        
        # Update step (Kalman gain)
        kalman_gain = pred_variance / (pred_variance + measurement_noise)
        
        # Update estimates
        self.pace_mean = self.pace_mean + kalman_gain * (lap_time - self.pace_mean)
        self.pace_variance = (1 - kalman_gain) * pred_variance
        
        self.current_regime = regime
        
        return self.pace_mean, self.pace_variance
    
    def get_state(self) -> dict:
        """Get current filter state."""
        return {
            "pace_mean": float(self.pace_mean),
            "pace_variance": float(self.pace_variance),
            "regime": self.current_regime,
        }
    
    def reset(self, initial_pace: float | None = None):
        """Reset filter to initial state."""
        if initial_pace is not None:
            self.initial_pace = initial_pace
        
        self.pace_mean = self.initial_pace
        self.pace_variance = 1.0
        self.current_regime = "green"


def save_kalman_config(filter: KalmanPaceFilter, path: Path | str):
    """Save Kalman filter configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        "model": "kalman-3reg",
        "initial_pace": filter.initial_pace,
        "process_noise": filter.process_noise,
        "measurement_noise": filter.measurement_noise,
        "version": "1.0",
    }
    
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_kalman_config(path: Path | str) -> KalmanPaceFilter:
    """Load Kalman filter from config."""
    path = Path(path)
    
    with open(path) as f:
        config = json.load(f)
    
    return KalmanPaceFilter(
        initial_pace=config.get("initial_pace", 130.0),
        process_noise=config.get("process_noise", 0.1),
        measurement_noise=config.get("measurement_noise", 0.5),
    )


