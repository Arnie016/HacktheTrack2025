#!/usr/bin/env python3
"""Generate final submission-ready results summary."""
import json
import sys
from pathlib import Path
from typing import Dict, Any

def load_report(report_path: Path) -> Dict[str, Any]:
    """Load validation report JSON."""
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {report_path}: {e}")
        return {}

def aggregate_scenario_results(reports_dir: Path) -> Dict[str, Any]:
    """Aggregate results across all scenarios."""
    scenarios = ['base', 'hot_track', 'heavy_traffic', 'undercut', 'no_weather', 'early_sc', 'late_sc']
    results = {}
    
    for scenario in scenarios:
        report_path = reports_dir / scenario / "validation_report.json"
        if not report_path.exists():
            # Try nested path
            report_path = reports_dir / scenario / scenario / "validation_report.json"
        
        if report_path.exists():
            results[scenario] = load_report(report_path)
        else:
            print(f"⚠️  {scenario}: Report not found")
    
    return results

def generate_summary_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Generate markdown table of results."""
    lines = ["| Scenario | Coverage @90% | MAE | Time Saved | Status |"]
    lines.append("|----------|---------------|-----|------------|--------|")
    
    for scenario, data in results.items():
        wear = data.get('wear_model_metrics', {})
        summary = data.get('summary', {})
        
        coverage = wear.get('quantile_coverage_90', 0.0)
        mae = wear.get('MAE', 0.0)
        time_saved = summary.get('time_saved_mean_s', 0.0)
        
        status = "✅" if coverage >= 0.90 else "⚠️"
        
        lines.append(f"| {scenario:15s} | {coverage:6.2%} | {mae:5.1f}s | {time_saved:6.1f}s | {status} |")
    
    return "\n".join(lines)

def main():
    """Generate submission summary."""
    base_dir = Path(__file__).parent
    reports_dir = base_dir / "reports"
    
    print("=" * 70)
    print("Generating Submission Results Summary")
    print("=" * 70)
    
    # Load base report
    base_report_path = reports_dir / "validation_report.json"
    if not base_report_path.exists():
        base_report_path = reports_dir / "test" / "validation_report.json"
    
    base_report = load_report(base_report_path)
    
    # Aggregate scenario results
    print("\n[1/3] Aggregating scenario results...")
    scenario_results = aggregate_scenario_results(reports_dir)
    print(f"✓ Found {len(scenario_results)} scenario reports")
    
    # Extract key metrics
    print("\n[2/3] Extracting key metrics...")
    wear_metrics = base_report.get('wear_model_metrics', {})
    summary = base_report.get('summary', {})
    walkforward = base_report.get('walkforward', {})
    
    # Generate summary
    print("\n[3/3] Generating summary...")
    
    output = f"""# 🏁 Final Submission Summary

## Quick Stats

- **Quantile Coverage**: {wear_metrics.get('quantile_coverage_90', 0.0):.2%}
- **MAE**: {wear_metrics.get('MAE', 0.0):.2f}s
- **R²**: {wear_metrics.get('R2', 0.0):.3f}
- **Total Recommendations**: {walkforward.get('total_recommendations', 0)}
- **Mean Confidence**: {walkforward.get('mean_confidence', 0.0):.2%}
- **Time Saved (Mean)**: {summary.get('time_saved_mean_s', 0.0):.1f}s
- **Time Saved (CI95)**: {summary.get('time_saved_ci95', [0, 0])[0]:.1f}s - {summary.get('time_saved_ci95', [0, 0])[1]:.1f}s

## Scenario Results

{generate_summary_table(scenario_results)}

## Files Generated

- `SUBMISSION_RESULTS.md` - Full submission document
- `IMPROVEMENT_RECOMMENDATIONS.md` - How to improve the approach
- `reports/` - All validation reports

## Next Steps

1. Review `SUBMISSION_RESULTS.md` for complete results
2. Check `IMPROVEMENT_RECOMMENDATIONS.md` for enhancement ideas
3. Submit reports from `reports/` directory
"""
    
    # Save summary
    summary_path = base_dir / "SUBMISSION_SUMMARY.md"
    with open(summary_path, 'w') as f:
        f.write(output)
    
    print(f"\n✓ Summary saved to: {summary_path}")
    print("\n" + "=" * 70)
    print("Submission Summary Generated!")
    print("=" * 70)
    print("\nKey Files:")
    print(f"  • {summary_path}")
    print(f"  • {base_dir / 'SUBMISSION_RESULTS.md'}")
    print(f"  • {base_dir / 'IMPROVEMENT_RECOMMENDATIONS.md'}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()


