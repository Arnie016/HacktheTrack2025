"""Show detailed scenario information."""
import modal
import json

app = modal.App("grcup-strategy")
reports_volume = modal.Volume.from_name("grcup-reports", create_if_missing=False)

@app.function(volumes={"/reports": reports_volume})
def show_scenario_details():
    """Show what each scenario represents."""
    # Read counterfactuals
    with open("/reports/base/counterfactuals.json", "r") as f:
        cf_data = json.load(f)
    
    cf_examples = cf_data.get('examples', [])
    
    print("=" * 90)
    print("SCENARIO DETAILS")
    print("=" * 90)
    
    for i, ex in enumerate(cf_examples, 1):
        print(f"\nScenario {i}:")
        print(f"  Vehicle ID: {ex.get('vehicle_id', 'N/A')}")
        print(f"  Lap: {ex.get('lap', 'N/A')}")
        print(f"  Recommended pit lap: {ex.get('recommended_pit_lap', 'N/A')}")
        print(f"  Alternative pit lap: {ex.get('alternative_pit_lap', 'N/A')}")
        print(f"  Time delta: {ex.get('delta_time_s', 0):+.2f}s")
        print(f"  Position delta: {ex.get('position_delta', 'N/A')}")
        
        # Show any other relevant fields
        for key, value in ex.items():
            if key not in ['vehicle_id', 'lap', 'recommended_pit_lap', 'alternative_pit_lap', 
                          'delta_time_s', 'position_delta', 'scenario']:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 90)
    
    # Also check walkforward recommendations to understand context
    print("\nChecking walkforward recommendations for context...")
    with open("/reports/base/walkforward_detailed.json", "r") as f:
        walkforward = json.load(f)
    
    recommendations = walkforward.get('recommendations', [])
    if recommendations:
        print(f"\nTotal recommendations: {len(recommendations)}")
        print("\nSample recommendations (first 5):")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. Vehicle: {rec.get('vehicle_id', 'N/A')}, Lap: {rec.get('lap', 'N/A')}, "
                  f"Pit lap: {rec.get('recommended_pit_lap', 'N/A')}, "
                  f"Confidence: {rec.get('confidence', 'N/A'):.2f}")
    
    return cf_examples

if __name__ == "__main__":
    show_scenario_details.remote()


