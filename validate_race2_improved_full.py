#!/usr/bin/env python3
"""
PRODUCTION VALIDATION: Race 2 with ALL improvements on ALL 21 vehicles.
"""
import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Enable improvements
os.environ["USE_VARIANCE_REDUCTION"] = "1"
os.environ["MC_BASE_SCENARIOS"] = "1000"
os.environ["MC_CLOSE_SCENARIOS"] = "2000"
os.environ["DISABLE_PARALLEL"] = "0"

print("=" * 70)
print("PRODUCTION VALIDATION - RACE 2 ALL IMPROVEMENTS (21 VEHICLES)")
print("=" * 70)
print()

from src.grcup.loaders import (
    load_lap_times, load_lap_starts, load_lap_ends,
    load_sectors, load_weather, load_results, build_lap_table,
)
from src.grcup.models.damage_detector import create_damage_detector_from_race_data
from src.grcup.strategy.optimizer_improved import solve_pit_strategy_improved
from src.grcup.models.wear_quantile_xgb import load_model
from src.grcup.models.sc_hazard import load_hazard_model

# Load models
print("Loading models...", end=" ", flush=True)
wear_model = load_model("models/wear_quantile_xgb.pkl")
try:
    sc_hazard_model = load_hazard_model("models/sc_hazard_cox.pkl")
except:
    sc_hazard_model = None
print("✓")

# Load Race 2 data
print("Loading Race 2 data...", end=" ", flush=True)
race_dir = Path("Race 2")
race2_lap_times = load_lap_times(race_dir / "vir_lap_time_R2.csv")
race2_lap_starts = load_lap_starts(race_dir / "vir_lap_start_R2.csv")
race2_lap_ends = load_lap_ends(race_dir / "vir_lap_end_R2.csv")
race2_sectors = load_sectors(race_dir / "23_AnalysisEnduranceWithSections_Race 2_Anonymized.CSV")
race2_weather = load_weather(race_dir / "26_Weather_Race 2_Anonymized.CSV")
race2_laps = build_lap_table(race2_lap_times, race2_lap_starts, race2_lap_ends)

try:
    race2_results = load_results(race_dir / "03_Results GR Cup Race 2 Official_Anonymized.CSV")
except:
    race2_results = load_results(race_dir / "03_Provisional Results_Race 2_Anonymized.CSV")
print("✓")

# Build vehicle_id to NUMBER mapping
print("Mapping vehicle_id to NUMBER...", end=" ", flush=True)
vehicle_number_map = {}
if "NUMBER" in race2_sectors.columns:
    sector_nums = pd.to_numeric(race2_sectors["NUMBER"], errors="coerce").dropna().unique()
    for vid in race2_laps["vehicle_id"].unique():
        parts = str(vid).split("-")
        potential_nums = []
        for p in parts:
            try:
                potential_nums.append(int(p))
            except:
                continue
        if potential_nums:
            closest_num = min(sector_nums, key=lambda x: min([abs(x - pn) for pn in potential_nums], default=999))
            vehicle_number_map[vid] = int(closest_num)

number_to_vehicle = {num: vid for vid, num in vehicle_number_map.items()}

# Add vehicle_id to sectors
race2_sectors_mapped = race2_sectors.copy()
# Strip column names
race2_sectors_mapped.columns = [c.strip() for c in race2_sectors_mapped.columns]
race2_sectors_mapped["vehicle_id"] = race2_sectors_mapped["NUMBER"].map(number_to_vehicle)
race2_sectors_mapped = race2_sectors_mapped[race2_sectors_mapped["vehicle_id"].notna()]
race2_sectors_mapped["lap"] = pd.to_numeric(race2_sectors_mapped["LAP_NUMBER"], errors="coerce")
print(f"✓ ({len(vehicle_number_map)} vehicles mapped)")

# Train damage detector
print("Training damage detector...", end=" ", flush=True)
try:
    race1_dir = Path("Race 1")
    race1_lap_times = load_lap_times(race1_dir / "vir_lap_time_R1.csv")
    race1_lap_starts = load_lap_starts(race1_dir / "vir_lap_start_R1.csv")
    race1_lap_ends = load_lap_ends(race1_dir / "vir_lap_end_R1.csv")
    race1_laps = build_lap_table(race1_lap_times, race1_lap_starts, race1_lap_ends)
    
    race1_sectors = load_sectors(race1_dir / "09_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV")
    race1_sectors_mapped = race1_sectors.copy()
    # Strip column names
    race1_sectors_mapped.columns = [c.strip() for c in race1_sectors_mapped.columns]
    # Map Race 1 vehicles too
    race1_vehicle_map = {}
    if "NUMBER" in race1_sectors_mapped.columns:
        r1_sector_nums = pd.to_numeric(race1_sectors_mapped["NUMBER"], errors="coerce").dropna().unique()
        for vid in race1_laps["vehicle_id"].unique():
            parts = str(vid).split("-")
            potential_nums = []
            for p in parts:
                try:
                    potential_nums.append(int(p))
                except:
                    continue
            if potential_nums:
                closest_num = min(r1_sector_nums, key=lambda x: min([abs(x - pn) for pn in potential_nums], default=999))
                race1_vehicle_map[vid] = int(closest_num)
    r1_number_to_vehicle = {num: vid for vid, num in race1_vehicle_map.items()}
    race1_sectors_mapped["vehicle_id"] = race1_sectors_mapped["NUMBER"].map(r1_number_to_vehicle)
    race1_sectors_mapped = race1_sectors_mapped[race1_sectors_mapped["vehicle_id"].notna()]
    race1_sectors_mapped["lap"] = pd.to_numeric(race1_sectors_mapped["LAP_NUMBER"], errors="coerce")
    
    all_laps = pd.concat([race1_laps, race2_laps], ignore_index=True)
    all_sectors = pd.concat([race1_sectors_mapped, race2_sectors_mapped], ignore_index=True)
    
    damage_detector = create_damage_detector_from_race_data(all_laps, all_sectors)
    print(f"✓ ({len(damage_detector.baseline_pace)} vehicles profiled)")
except Exception as e:
    print(f"⚠ Warning: {e}")
    damage_detector = None

# Helper to extract position context
def get_position_context(vehicle_id, lap, sectors_df, laps_df):
    """Extract real position, gaps from data."""
    # Get elapsed time for this vehicle
    vehicle_lap = sectors_df[
        (sectors_df["vehicle_id"] == vehicle_id) &
        (sectors_df["lap"] == lap)
    ]
    
    if len(vehicle_lap) == 0:
        return None, None, None, None
    
    elapsed_s = pd.to_numeric(vehicle_lap["ELAPSED"].apply(
        lambda x: sum([float(p) * (60 ** i) for i, p in enumerate(reversed(str(x).split(":")))]) if pd.notna(x) else np.nan
    ).iloc[0], errors="coerce")
    
    if pd.isna(elapsed_s):
        return None, None, None, None
    
    # Get all vehicles at this lap
    lap_standings = sectors_df[sectors_df["lap"] == lap].copy()
    lap_standings["elapsed_s"] = lap_standings["ELAPSED"].apply(
        lambda x: sum([float(p) * (60 ** i) for i, p in enumerate(reversed(str(x).split(":")))]) if pd.notna(x) else np.nan
    )
    lap_standings = lap_standings[lap_standings["elapsed_s"].notna()].sort_values("elapsed_s")
    
    # Find position
    position = (lap_standings["elapsed_s"] <= elapsed_s).sum()
    
    # Get gaps
    if position > 1:
        ahead_elapsed = lap_standings.iloc[position - 2]["elapsed_s"]
        gap_ahead = elapsed_s - ahead_elapsed
    else:
        gap_ahead = 0.0
    
    if position < len(lap_standings):
        behind_elapsed = lap_standings.iloc[position]["elapsed_s"]
        gap_behind = behind_elapsed - elapsed_s
    else:
        gap_behind = 999.0
    
    gap_to_leader = elapsed_s - lap_standings.iloc[0]["elapsed_s"]
    
    return position, gap_ahead, gap_behind, gap_to_leader

# Helper to get sector times
def get_sector_times(vehicle_id, lap, sectors_df):
    """Extract sector times."""
    vehicle_lap = sectors_df[
        (sectors_df["vehicle_id"] == vehicle_id) &
        (sectors_df["lap"] == lap)
    ]
    
    if len(vehicle_lap) == 0:
        return None, None
    
    s1 = pd.to_numeric(vehicle_lap.get("S1_SECONDS", pd.Series([np.nan])).iloc[0], errors="coerce")
    s2 = pd.to_numeric(vehicle_lap.get("S2_SECONDS", pd.Series([np.nan])).iloc[0], errors="coerce")
    s3 = pd.to_numeric(vehicle_lap.get("S3_SECONDS", pd.Series([np.nan])).iloc[0], errors="coerce")
    top_speed = pd.to_numeric(vehicle_lap.get("TOP_SPEED", vehicle_lap.get("KPH", pd.Series([np.nan]))).iloc[0], errors="coerce")
    
    sector_times = {"s1": s1, "s2": s2, "s3": s3} if not all(pd.isna([s1, s2, s3])) else None
    
    return sector_times, top_speed

# Run walk-forward validation
print()
print("Running walk-forward validation (laps 5, 10, 15)...")
print("-" * 70)

recommendations = []
damage_pits = 0
position_strategies = 0
all_vehicles = race2_laps["vehicle_id"].unique()

for checkpoint_lap in [5, 10, 15]:  # Multiple checkpoints
    print(f"\n### Lap {checkpoint_lap} checkpoint ###")
    
    for vehicle_id in all_vehicles:
        vehicle_laps = race2_laps[race2_laps["vehicle_id"] == vehicle_id].sort_values("lap")
        
        if len(vehicle_laps) < checkpoint_lap:
            continue
        
        lap_window = vehicle_laps[vehicle_laps["lap"] <= checkpoint_lap]
        
        if len(lap_window) < 3:
            continue
        
        # Recent lap times
        recent_lap_times = lap_window["lap_time_s"].tail(5).tolist()
        
        # Position context (REAL DATA)
        position, gap_ahead, gap_behind, gap_to_leader = get_position_context(
            vehicle_id, checkpoint_lap, race2_sectors_mapped, race2_laps
        )
        
        # Sector times (REAL DATA)
        sector_times, top_speed = get_sector_times(
            vehicle_id, checkpoint_lap, race2_sectors_mapped
        )
        
        # Run improved optimizer
        try:
            result = solve_pit_strategy_improved(
                current_lap=checkpoint_lap,
                total_laps=22,
                tire_age=float(checkpoint_lap - 1),
                fuel_laps_remaining=max(1.0, 22 - checkpoint_lap),
                under_sc=False,
                wear_model=wear_model,
                sc_hazard_model=sc_hazard_model,
                damage_detector=damage_detector,
                vehicle_id=vehicle_id,
                recent_lap_times=recent_lap_times,
                current_position=position,
                gap_ahead=gap_ahead,
                gap_behind=gap_behind,
                sector_times=sector_times,
                top_speed=top_speed,
                gap_to_leader=gap_to_leader,
                use_antithetic_variates=True,
                position_weight=0.7,
            )
            
            recommendations.append({
                "vehicle_id": vehicle_id,
                "lap": int(checkpoint_lap),
                "position": int(position) if position else None,
                "gap_ahead": float(gap_ahead) if pd.notna(gap_ahead) else None,
                "gap_behind": float(gap_behind) if pd.notna(gap_behind) else None,
                **{k: (int(v) if isinstance(v, (np.integer, np.int64)) else 
                      float(v) if isinstance(v, (np.floating, np.float64)) else v) 
                   for k, v in result.items()},
            })
            
            if result.get("strategy_type") == "damage_pit":
                damage_pits += 1
            elif result.get("strategy_type") != "standard":
                position_strategies += 1
        
        except Exception as e:
            print(f"  {vehicle_id}: Error - {e}")
            continue

print()
print("-" * 70)
print()

# Summary
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Total recommendations: {len(recommendations)}")
print(f"  Damage-forced pits: {damage_pits} ({damage_pits/len(recommendations)*100:.1f}%)")
print(f"  Position-aware: {position_strategies} ({position_strategies/len(recommendations)*100:.1f}%)")
print(f"  Standard: {len(recommendations) - damage_pits - position_strategies}")
print()

confidences = [r["confidence"] for r in recommendations]
if confidences:
    print(f"Confidence statistics:")
    print(f"  Mean: {np.mean(confidences):.2f}")
    print(f"  Median: {np.median(confidences):.2f}")
    print(f"  High conf (>0.8): {sum(1 for c in confidences if c > 0.8)}")
print()

# Save
output_dir = Path("reports/production")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "race2_full_validation.json"
with open(output_file, "w") as f:
    json.dump({
        "recommendations": recommendations,
        "summary": {
            "total_recommendations": len(recommendations),
            "total_vehicles": len(all_vehicles),
            "checkpoints": [5, 10, 15],
            "damage_pits": damage_pits,
            "position_strategies": position_strategies,
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "damage_detector_active": damage_detector is not None,
        }
    }, f, indent=2)

print(f"✅ Results saved to: {output_file}")
print()
print("=" * 70)
print("PRODUCTION VALIDATION COMPLETE")
print("=" * 70)

