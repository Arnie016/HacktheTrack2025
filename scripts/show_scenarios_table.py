"""Show scenarios table with improvements."""
import modal
import json

app = modal.App("grcup-strategy")
reports_volume = modal.Volume.from_name("grcup-reports", create_if_missing=False)

@app.function(volumes={"/reports": reports_volume})
def show_scenarios_table():
    """Show scenarios in table format."""
    # Read counterfactuals
    with open("/reports/base/counterfactuals.json", "r") as f:
        cf_data = json.load(f)
    
    cf_examples = cf_data.get('examples', [])
    
    print("=" * 80)
    print("SCENARIO IMPROVEMENT TABLE")
    print("=" * 80)
    print(f"{'Scenario':<30} {'Time Delta (s)':<20} {'Status':<15} {'Improvement':<15}")
    print("-" * 80)
    
    improvements = []
    worse_scenarios = []
    
    for i, ex in enumerate(cf_examples, 1):
        scenario = ex.get('scenario', f'Scenario {i}')
        delta = ex.get('delta_time_s', 0)
        
        if delta < 0:
            status = "✓ IMPROVED"
            improvement = f"{abs(delta):.2f}s saved"
            improvements.append(ex)
        elif delta > 0:
            status = "✗ WORSE"
            improvement = f"{delta:.2f}s lost"
            worse_scenarios.append(ex)
        else:
            status = "= NO CHANGE"
            improvement = "0.00s"
        
        print(f"{scenario:<30} {delta:>+10.2f}        {status:<15} {improvement:<15}")
    
    print("-" * 80)
    print(f"\nSUMMARY:")
    print(f"  Total scenarios: {len(cf_examples)}")
    print(f"  Improved: {len(improvements)} ({len(improvements)/len(cf_examples)*100:.1f}%)")
    print(f"  Worse: {len(worse_scenarios)} ({len(worse_scenarios)/len(cf_examples)*100:.1f}%)")
    
    if improvements:
        avg_improvement = sum(ex.get('delta_time_s', 0) for ex in improvements) / len(improvements)
        best_improvement = min(ex.get('delta_time_s', 0) for ex in improvements)
        print(f"\n  Average improvement: {abs(avg_improvement):.2f}s saved")
        print(f"  Best improvement: {abs(best_improvement):.2f}s saved")
    
    if worse_scenarios:
        avg_worse = sum(ex.get('delta_time_s', 0) for ex in worse_scenarios) / len(worse_scenarios)
        worst_case = max(ex.get('delta_time_s', 0) for ex in worse_scenarios)
        print(f"\n  Average loss: {avg_worse:.2f}s lost")
        print(f"  Worst case: {worst_case:.2f}s lost")
    
    print("=" * 80)
    
    return {
        'improvements': len(improvements),
        'worse': len(worse_scenarios),
        'total': len(cf_examples)
    }

if __name__ == "__main__":
    show_scenarios_table.remote()


