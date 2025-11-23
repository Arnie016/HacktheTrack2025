"""Typed dataclasses for race-derived structures."""

from dataclasses import dataclass
from typing import List


@dataclass
class Stint:
    car_id: str
    start_lap: int
    end_lap: int
    best_lap_time_ms: float
    laps: List[int]



