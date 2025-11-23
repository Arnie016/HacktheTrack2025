"""Damage detection model for identifying damage-forced pit stops."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class DamageDetector:
    """
    Detect damage based on lap time anomalies, sector time drops, and telemetry.
    
    Critical for Race 2 accuracy: 40% of actual pits were damage-forced.
    """
    
    def __init__(
        self,
        lap_time_spike_threshold: float = 3.0,  # 3σ above median
        sector_drop_threshold: float = 0.5,  # 0.5s drop in sector time
        speed_drop_threshold: float = 10.0,  # 10 kph drop in top speed
        consecutive_slow_laps: int = 2,  # 2+ slow laps = damage
    ):
        self.lap_time_spike_threshold = lap_time_spike_threshold
        self.sector_drop_threshold = sector_drop_threshold
        self.speed_drop_threshold = speed_drop_threshold
        self.consecutive_slow_laps = consecutive_slow_laps
        
        # Learned from data
        self.baseline_pace = {}  # vehicle_id -> median lap time
        self.baseline_sectors = {}  # vehicle_id -> median s1/s2/s3
        self.baseline_speed = {}  # vehicle_id -> median top speed
    
    def fit(self, laps_df: pd.DataFrame, sectors_df: pd.DataFrame) -> None:
        """
        Learn baseline performance from clean laps (first 3-5 laps typically).
        
        Args:
            laps_df: Lap times with vehicle_id, lap, lap_time_s
            sectors_df: Sector data with s1/s2/s3, top_speed
        """
        # Compute baseline pace per vehicle (first 5 laps, excluding outliers)
        for vehicle_id in laps_df["vehicle_id"].unique():
            vehicle_laps = laps_df[laps_df["vehicle_id"] == vehicle_id].copy()
            
            # Use first 3-8 laps as baseline (clean racing typically)
            baseline_laps = vehicle_laps[
                (vehicle_laps["lap"] >= 3) & 
                (vehicle_laps["lap"] <= 8)
            ]["lap_time_s"].dropna()
            
            if len(baseline_laps) > 0:
                # Median is robust to outliers
                self.baseline_pace[vehicle_id] = baseline_laps.median()
            
            # Compute baseline sectors
            vehicle_sectors = sectors_df[sectors_df["vehicle_id"] == vehicle_id].copy()
            baseline_sectors = vehicle_sectors[
                (vehicle_sectors["lap"] >= 3) & 
                (vehicle_sectors["lap"] <= 8)
            ]
            
            if len(baseline_sectors) > 0:
                self.baseline_sectors[vehicle_id] = {
                    "s1": baseline_sectors.get("s1", pd.Series()).median(),
                    "s2": baseline_sectors.get("s2", pd.Series()).median(),
                    "s3": baseline_sectors.get("s3", pd.Series()).median(),
                }
                self.baseline_speed[vehicle_id] = baseline_sectors.get("top_speed", pd.Series()).median()
    
    def predict_damage_probability(
        self,
        vehicle_id: str,
        current_lap: int,
        lap_times: list[float],  # Recent lap times
        sector_times: Optional[dict] = None,  # s1/s2/s3 for current lap
        top_speed: Optional[float] = None,
        gap_to_leader: Optional[float] = None,
    ) -> float:
        """
        Predict probability of damage (0-1).
        
        Args:
            vehicle_id: Vehicle identifier
            current_lap: Current lap number
            lap_times: Recent lap times (last 3-5 laps)
            sector_times: dict with s1/s2/s3 keys (optional)
            top_speed: Top speed on current lap (optional)
            gap_to_leader: Gap to leader in seconds (optional)
        
        Returns:
            Damage probability (0 = no damage, 1 = certain damage)
        """
        damage_score = 0.0
        
        if vehicle_id not in self.baseline_pace or len(lap_times) == 0:
            return 0.0  # No baseline, can't detect
        
        baseline = self.baseline_pace[vehicle_id]
        current_lap_time = lap_times[-1]
        
        # 1. Lap time spike (sudden slowdown)
        lap_time_delta = current_lap_time - baseline
        lap_time_sigma = lap_time_delta / baseline  # Normalized
        
        if lap_time_sigma > self.lap_time_spike_threshold:
            # 3+ sigma = very likely damage
            damage_score += 0.4
        elif lap_time_sigma > 2.0:
            # 2-3 sigma = possible damage
            damage_score += 0.2
        
        # 2. Consecutive slow laps (persistent damage)
        if len(lap_times) >= self.consecutive_slow_laps:
            recent_laps = lap_times[-self.consecutive_slow_laps:]
            slow_count = sum(1 for lt in recent_laps if (lt - baseline) > 2.0)
            
            if slow_count >= self.consecutive_slow_laps:
                damage_score += 0.3  # Sustained poor pace = damage
        
        # 3. Sector time anomaly (specific corner damage)
        if sector_times and vehicle_id in self.baseline_sectors:
            baseline_sectors = self.baseline_sectors[vehicle_id]
            
            for sector in ["s1", "s2", "s3"]:
                if sector in sector_times and sector in baseline_sectors:
                    sector_baseline = baseline_sectors[sector]
                    sector_current = sector_times[sector]
                    
                    if pd.notna(sector_baseline) and pd.notna(sector_current):
                        sector_delta = sector_current - sector_baseline
                        
                        # Large drop in sector time = damage in that section
                        if sector_delta > self.sector_drop_threshold:
                            damage_score += 0.15  # Each bad sector adds 0.15
        
        # 4. Top speed drop (aero damage, suspension)
        if top_speed is not None and vehicle_id in self.baseline_speed:
            baseline_speed = self.baseline_speed[vehicle_id]
            
            if pd.notna(baseline_speed) and pd.notna(top_speed):
                speed_drop = baseline_speed - top_speed
                
                if speed_drop > self.speed_drop_threshold:
                    damage_score += 0.25  # Significant speed loss = likely damage
        
        # 5. Gap to leader increasing rapidly (falling back)
        if gap_to_leader is not None and gap_to_leader > 20.0:
            # Large gap + slow pace = probable damage
            if lap_time_sigma > 1.5:
                damage_score += 0.15
        
        # Cap at 1.0
        return min(1.0, damage_score)
    
    def should_pit_for_damage(
        self,
        vehicle_id: str,
        current_lap: int,
        lap_times: list[float],
        sector_times: Optional[dict] = None,
        top_speed: Optional[float] = None,
        gap_to_leader: Optional[float] = None,
        damage_threshold: float = 0.6,
    ) -> tuple[bool, float, str]:
        """
        Recommend immediate pit stop if damage detected.
        
        Args:
            damage_threshold: Probability threshold (default 0.6 = 60%)
        
        Returns:
            (should_pit, damage_prob, reason)
        """
        damage_prob = self.predict_damage_probability(
            vehicle_id=vehicle_id,
            current_lap=current_lap,
            lap_times=lap_times,
            sector_times=sector_times,
            top_speed=top_speed,
            gap_to_leader=gap_to_leader,
        )
        
        if damage_prob >= damage_threshold:
            reason = f"Damage detected (prob={damage_prob:.1%})"
            
            # Build detailed reason
            if len(lap_times) > 0 and vehicle_id in self.baseline_pace:
                baseline = self.baseline_pace[vehicle_id]
                delta = lap_times[-1] - baseline
                reason += f" - lap time +{delta:.1f}s vs baseline"
            
            if top_speed and vehicle_id in self.baseline_speed:
                speed_drop = self.baseline_speed[vehicle_id] - top_speed
                if speed_drop > self.speed_drop_threshold:
                    reason += f", speed -{speed_drop:.0f}kph"
            
            return True, damage_prob, reason
        
        return False, damage_prob, "No damage detected"


def create_damage_detector_from_race_data(
    race_laps: pd.DataFrame,
    race_sectors: pd.DataFrame,
) -> DamageDetector:
    """
    Factory function to create and fit damage detector from race data.
    
    Args:
        race_laps: Lap times DataFrame
        race_sectors: Sector times DataFrame
    
    Returns:
        Fitted DamageDetector
    """
    detector = DamageDetector(
        lap_time_spike_threshold=3.0,
        sector_drop_threshold=0.5,
        speed_drop_threshold=10.0,
        consecutive_slow_laps=2,
    )
    
    detector.fit(race_laps, race_sectors)
    
    return detector

