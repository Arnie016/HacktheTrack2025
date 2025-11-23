"""Create a dashboard comparing AI strategy recommendations vs actual race data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grcup.features.stint_detector import detect_stints
from src.grcup.loaders import build_lap_table, load_results, load_lap_times, load_lap_starts, load_lap_ends, load_sectors, load_weather


def load_walkforward_results(results_path: Path) -> dict:
    """Load walkforward validation results."""
    if not results_path.exists():
        return {}
    with open(results_path, "r") as f:
        return json.load(f)


def extract_actual_pits(race_laps: pd.DataFrame, vehicle_id: str) -> list[dict]:
    """Extract actual pit stops for a vehicle."""
    try:
        stints = detect_stints(race_laps, vehicle_id)
        pits = []
        for i, stint in enumerate(stints):
            if i > 0:  # First stint has no prior pit
                pits.append({
                    "lap": stint.start_lap,
                    "stint_length": stint.end_lap - stint.start_lap + 1,
                    "best_lap_time_ms": stint.best_lap_time_ms,
                })
        return sorted(pits, key=lambda x: x["lap"])
    except Exception as e:
        print(f"  Warning: Could not detect pits for {vehicle_id}: {e}")
        return []


def build_comparison_data(
    walkforward_results: dict,
    race_laps: pd.DataFrame,
    race_results: pd.DataFrame,
) -> dict:
    """Build comparison data for dashboard."""
    recommendations = walkforward_results.get("recommendations", [])
    
    # Group recommendations by vehicle
    vehicle_recs = {}
    for rec in recommendations:
        vehicle_id = rec.get("vehicle_id")
        if vehicle_id not in vehicle_recs:
            vehicle_recs[vehicle_id] = []
        vehicle_recs[vehicle_id].append(rec)
    
    # Build comparison for each vehicle
    comparisons = []
    
    for vehicle_id, recs in vehicle_recs.items():
        # Get actual pits
        actual_pits = extract_actual_pits(race_laps, vehicle_id)
        
        # Get vehicle info from results
        vehicle_info = {
            "vehicle_id": vehicle_id,
            "final_position": None,
            "laps_completed": None,
            "status": None,
        }
        
        # Try to match vehicle to results
        for col in race_results.columns:
            if "NUMBER" in col.upper():
                # Extract number from vehicle_id (e.g., "GR86-022-13" -> 13)
                parts = str(vehicle_id).split("-")
                potential_nums = []
                for p in parts:
                    try:
                        potential_nums.append(int(p))
                    except:
                        pass
                
                if potential_nums:
                    vehicle_num = potential_nums[-1]
                    vehicle_row = race_results[
                        race_results[col].astype(str).str.contains(str(vehicle_num), na=False)
                    ]
                    if len(vehicle_row) > 0:
                        pos_col = None
                        laps_col = None
                        status_col = None
                        for c in race_results.columns:
                            if "POSITION" in c.upper() or "POS" in c.upper():
                                pos_col = c
                            if "LAPS" in c.upper():
                                laps_col = c
                            if "STATUS" in c.upper():
                                status_col = c
                        
                        if pos_col:
                            vehicle_info["final_position"] = int(vehicle_row[pos_col].iloc[0]) if pd.notna(vehicle_row[pos_col].iloc[0]) else None
                        if laps_col:
                            vehicle_info["laps_completed"] = int(vehicle_row[laps_col].iloc[0]) if pd.notna(vehicle_row[laps_col].iloc[0]) else None
                        if status_col:
                            vehicle_info["status"] = str(vehicle_row[status_col].iloc[0]) if pd.notna(vehicle_row[status_col].iloc[0]) else None
                        break
        
        # Build timeline of recommendations
        ai_strategy = []
        for rec in sorted(recs, key=lambda x: x.get("lap", 0)):
            ai_strategy.append({
                "lap": rec.get("lap"),
                "recommended_pit_lap": rec.get("recommended_pit_lap"),
                "window": rec.get("recommended_window", []),
                "confidence": rec.get("confidence", 0.0),
                "reasoning": rec.get("reasoning", ""),
                "expected_time": rec.get("expected_time", 0.0),
            })
        
        comparisons.append({
            "vehicle": vehicle_info,
            "ai_strategy": ai_strategy,
            "actual_pits": actual_pits,
            "total_recommendations": len(recs),
        })
    
    return {
        "comparisons": comparisons,
        "summary": {
            "total_vehicles": len(vehicle_recs),
            "total_recommendations": len(recommendations),
            "avg_confidence": np.mean([r.get("confidence", 0.0) for r in recommendations]) if recommendations else 0.0,
        }
    }


def main():
    """Generate dashboard data."""
    # Paths
    base_dir = Path(__file__).parent.parent
    race_dir = base_dir / "Race 2"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Load data
    print("📊 Loading race data...")
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
    
    print("📈 Loading walkforward results...")
    # Try main file first, then check scenario folders
    walkforward_results = load_walkforward_results(reports_dir / "walkforward_detailed.json")
    
    if not walkforward_results or len(walkforward_results.get("recommendations", [])) == 0:
        # Try test folder
        test_results = load_walkforward_results(reports_dir / "test" / "walkforward_detailed.json")
        if test_results and len(test_results.get("recommendations", [])) > 0:
            walkforward_results = test_results
            print(f"  Using test scenario results ({len(walkforward_results.get('recommendations', []))} recommendations)")
        else:
            print("❌ No walkforward results found. Run validation first.")
            return
    
    print("🔍 Building comparison data...")
    comparison_data = build_comparison_data(
        walkforward_results,
        race2_laps,
        race2_results,
    )
    
    # Save JSON
    output_path = reports_dir / "strategy_comparison.json"
    with open(output_path, "w") as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"✅ Dashboard data saved to: {output_path}")
    print(f"   Vehicles analyzed: {comparison_data['summary']['total_vehicles']}")
    print(f"   Total recommendations: {comparison_data['summary']['total_recommendations']}")


if __name__ == "__main__":
    main()

