"""Show detailed scenario information with vehicle and lap context."""
import modal
import json

app = modal.App("grcup-strategy")
reports_volume = modal.Volume.from_name("grcup-reports", create_if_missing=False)

@app.function(volumes={"/reports": reports_volume})
def show_scenario_details_final():
    """Show what each scenario represents."""
    # Read counterfactuals
    with open("/reports/base/counterfactuals.json", "r") as f:
        cf_data = json.load(f)
    
    cf_examples = cf_data.get('examples', [])
    
    print("=" * 100)
    print("SCENARIO DETAILS - What Each Scenario Represents")
    print("=" * 100)
    print("\nEach scenario compares:")
    print("  • Recommended Strategy: What the engine suggested")
    print("  • Actual Strategy: What actually happened in Race 2")
    print("  • Delta Time: actual_time - recommended_time")
    print("    (Negative = recommended was better, Positive = recommended was worse)")
    print("\n" + "=" * 100)
    print(f"{'#':<4} {'Vehicle ID':<20} {'Lap':<6} {'Rec Pit':<8} {'Actual Pos':<12} {'Delta (s)':<12} {'Status':<15}")
    print("-" * 100)
    
    for i, ex in enumerate(cf_examples, 1):
        vehicle_id = ex.get('vehicle_id', 'N/A')
        lap = ex.get('lap', 'N/A')
        rec_pit = ex.get('recommended_pit', 'N/A')
        actual_pos = ex.get('actual_position', 'N/A')
        delta = ex.get('delta_time_s', 0)
        
        if delta < 0:
            status = "✓ IMPROVED"
        elif delta > 0:
            status = "✗ WORSE"
        else:
            status = "= NO CHANGE"
        
        # Format values
        lap_str = str(int(lap)) if lap != 'N/A' and lap is not None else 'N/A'
        rec_pit_str = str(int(rec_pit)) if rec_pit != 'N/A' and rec_pit is not None else 'N/A'
        actual_pos_str = f"{actual_pos:.1f}" if actual_pos != 'N/A' and actual_pos is not None else 'N/A'
        delta_str = f"{delta:+.2f}"
        
        print(f"{i:<4} {vehicle_id:<20} {lap_str:<6} {rec_pit_str:<8} {actual_pos_str:<12} {delta_str:<12} {status:<15}")
    
    print("-" * 100)
    print("\nSUMMARY:")
    print(f"  Total scenarios: {len(cf_examples)}")
    improved = sum(1 for ex in cf_examples if ex.get('delta_time_s', 0) < 0)
    worse = sum(1 for ex in cf_examples if ex.get('delta_time_s', 0) > 0)
    print(f"  Improved (negative delta): {improved}")
    print(f"  Worse (positive delta): {worse}")
    print(f"  No change: {len(cf_examples) - improved - worse}")
    print("=" * 100)
    
    return cf_examples

if __name__ == "__main__":
    show_scenario_details_final.remote()


