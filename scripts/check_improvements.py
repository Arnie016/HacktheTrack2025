"""Check actual improvements from validation."""
import modal
import json

app = modal.App("grcup-strategy")
reports_volume = modal.Volume.from_name("grcup-reports", create_if_missing=False)

@app.function(volumes={"/reports": reports_volume})
def check_improvements():
    """Check improvement metrics."""
    # Read validation report
    with open("/reports/base/validation_report.json", "r") as f:
        report = json.load(f)
    
    # Read walkforward detailed results
    with open("/reports/base/walkforward_detailed.json", "r") as f:
        walkforward = json.load(f)
    
    # Read counterfactuals
    with open("/reports/base/counterfactuals.json", "r") as f:
        cf_data = json.load(f)
    
    print("=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)
    
    # Analyze recommendations
    recommendations = walkforward.get('recommendations', [])
    if recommendations:
        print(f"\n📊 Strategy Recommendations: {len(recommendations)} total")
        
        # Count early vs late pit recommendations
        early_pits = sum(1 for r in recommendations if r.get('recommended_pit_lap', 0) < 20)
        late_pits = sum(1 for r in recommendations if r.get('recommended_pit_lap', 0) >= 20)
        print(f"  Early pit (< lap 20): {early_pits}")
        print(f"  Late pit (≥ lap 20): {late_pits}")
        
        # Average confidence
        confidences = [r.get('confidence', 0) for r in recommendations if r.get('confidence')]
        if confidences:
            print(f"  Avg confidence: {sum(confidences)/len(confidences):.2f}")
            print(f"  High confidence (>0.8): {sum(1 for c in confidences if c > 0.8)}")
    
    # Counterfactuals - show actual improvements (negative delta = time saved)
    cf_examples = cf_data.get('examples', [])
    if cf_examples:
        print(f"\n🔄 Counterfactual Analysis: {len(cf_examples)} scenarios")
        
        improvements = [ex for ex in cf_examples if ex.get('delta_time_s', 0) < 0]
        worse = [ex for ex in cf_examples if ex.get('delta_time_s', 0) > 0]
        
        print(f"  Scenarios with improvement (negative delta): {len(improvements)}")
        if improvements:
            avg_improvement = sum(ex.get('delta_time_s', 0) for ex in improvements) / len(improvements)
            print(f"  Average time saved: {abs(avg_improvement):.2f}s")
            print(f"  Best improvement: {abs(min(ex.get('delta_time_s', 0) for ex in improvements)):.2f}s")
        
        print(f"  Scenarios that were worse: {len(worse)}")
        if worse:
            avg_worse = sum(ex.get('delta_time_s', 0) for ex in worse) / len(worse)
            print(f"  Average time lost: {avg_worse:.2f}s")
    
    # Baseline comparisons
    baseline = report.get('baseline_comparisons', {})
    if baseline:
        print(f"\n📈 Baseline Comparisons:")
        engine_adv = baseline.get('engine_advantage', {})
        if engine_adv:
            for baseline_name, metrics in engine_adv.items():
                time_saved = metrics.get('time_saved_s', 0)
                if time_saved != 0:
                    print(f"  vs {baseline_name}: {time_saved:+.2f}s")
    
    # Summary
    summary = report.get('summary', {})
    print(f"\n✅ Summary:")
    print(f"  Expected positions gain: {summary.get('expected_positions_gain', 'N/A')}")
    print(f"  Quantile coverage: {summary.get('quantile_coverage_90', 0):.1%}")
    
    print("=" * 70)
    
    return {
        'recommendations': len(recommendations),
        'counterfactuals': len(cf_examples),
        'improvements': len(improvements) if cf_examples else 0,
    }

if __name__ == "__main__":
    check_improvements.remote()


