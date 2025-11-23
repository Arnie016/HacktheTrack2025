#!/usr/bin/env python3
"""Quick validation script to check if fixes are working."""
import json
from pathlib import Path

def validate_report(report_path: Path):
    """Validate a single validation report."""
    print(f"\n{'='*60}")
    print(f"Validating: {report_path}")
    print(f"{'='*60}")
    
    with open(report_path) as f:
        report = json.load(f)
    
    # Check baseline comparisons
    baseline_comps = report.get("baseline_comparisons", {})
    engine_adv = baseline_comps.get("engine_advantage", {})
    
    print("\n📊 Baseline Comparisons:")
    for baseline_name, data in engine_adv.items():
        time_saved = data.get("time_saved_s", 0)
        ci95 = data.get("ci95", [0, 0])
        
        # Check mirror leader specifically
        if "mirror" in baseline_name.lower():
            if time_saved > 30:
                print(f"  ❌ {baseline_name}: {time_saved:.1f}s (TOO HIGH - should be 5-15s)")
            elif time_saved < 0:
                print(f"  ⚠️  {baseline_name}: {time_saved:.1f}s (negative)")
            else:
                print(f"  ✅ {baseline_name}: {time_saved:.1f}s (reasonable)")
        else:
            print(f"  ✓  {baseline_name}: {time_saved:.1f}s [CI: {ci95[0]:.1f}-{ci95[1]:.1f}]")
    
    # Check counterfactuals
    counterfactuals = report.get("counterfactuals", {})
    examples = counterfactuals.get("examples", [])
    
    print("\n📈 Counterfactuals:")
    if len(examples) == 0:
        print("  ⚠️  No counterfactuals found")
    else:
        delta_times = [ex.get("delta_time_s", 0) for ex in examples]
        unique_values = len(set(delta_times))
        
        if unique_values == 1:
            print(f"  ❌ All {len(examples)} counterfactuals have same value: {delta_times[0]:.1f}s (FAKE DATA!)")
        elif unique_values < len(examples) / 2:
            print(f"  ⚠️  Only {unique_values} unique values out of {len(examples)} (might be fake)")
        else:
            print(f"  ✅ {unique_values} unique values out of {len(examples)} (varied, good!)")
            print(f"     Range: {min(delta_times):.1f}s to {max(delta_times):.1f}s")
    
    # Check wear model
    wear_metrics = report.get("wear_model_metrics", {})
    print("\n🔧 Wear Model:")
    mae = wear_metrics.get("MAE")
    r2 = wear_metrics.get("R2")
    coverage = wear_metrics.get("quantile_coverage_90")
    
    if mae:
        print(f"  MAE: {mae:.2f}s")
    if r2 is not None:
        print(f"  R²: {r2:.3f}")
    if coverage:
        status = "✅" if 0.85 <= coverage <= 0.95 else "⚠️"
        print(f"  Coverage @90%: {coverage:.1%} {status}")
    
    return True

def main():
    """Validate all reports."""
    reports_dir = Path("reports")
    
    # Check base scenario
    base_report = reports_dir / "base" / "validation_report.json"
    if base_report.exists():
        validate_report(base_report)
    else:
        print(f"⚠️  Base report not found: {base_report}")
        print("   Run: python3 scripts/validate_walkforward.py --event R2 --scenario base")
    
    # Check test report if exists
    test_report = reports_dir / "test" / "validation_report.json"
    if test_report.exists():
        validate_report(test_report)

if __name__ == "__main__":
    main()



