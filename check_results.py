"""Quick script to check validation results."""
import modal
import json

app = modal.App("grcup-strategy")
reports_volume = modal.Volume.from_name("grcup-reports", create_if_missing=False)

@app.function(volumes={"/reports": reports_volume})
def read_results():
    """Read and print validation results."""
    report_path = "/reports/base/validation_report.json"
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
        
        print("=" * 60)
        print("VALIDATION RESULTS SUMMARY")
        print("=" * 60)
        print(f"\nWalk-Forward Validation:")
        print(f"  Total recommendations: {report.get('walkforward', {}).get('total_recommendations', 'N/A')}")
        conf = report.get('walkforward', {}).get('mean_confidence', 'N/A')
        if conf != 'N/A':
            print(f"  Mean confidence: {conf:.2f}")
        else:
            print(f"  Mean confidence: {conf}")

        summary = report.get('summary', {})
        print(f"\nSummary Metrics:")
        print(f"  Time saved (mean): {summary.get('time_saved_mean_s', 'N/A')}s")
        ci95 = summary.get('time_saved_ci95', [])
        if ci95 and len(ci95) == 2:
            print(f"  Time saved (CI95): {ci95[0]:.1f}s - {ci95[1]:.1f}s")
        cov = summary.get('quantile_coverage_90', 'N/A')
        if cov != 'N/A':
            print(f"  Quantile coverage @90%: {cov:.2%}")
        else:
            print(f"  Quantile coverage @90%: {cov}")
        pos_gain = summary.get('expected_positions_gain', 'N/A')
        print(f"  Expected positions gain: {pos_gain}")

        # Baseline comparisons
        baseline = report.get('baseline_comparisons', {})
        if baseline:
            print(f"\nBaseline Comparisons:")
            engine_adv = baseline.get('engine_advantage', {})
            if engine_adv:
                vs_fixed = engine_adv.get('vs_fixed_stint', {})
                if vs_fixed:
                    saved = vs_fixed.get('time_saved_s', 'N/A')
                    print(f"  vs Fixed Stint: {saved}s saved")
                vs_leader = engine_adv.get('vs_follow_leader', {})
                if vs_leader:
                    saved = vs_leader.get('time_saved_s', 'N/A')
                    print(f"  vs Follow Leader: {saved}s saved")

        # Counterfactuals details
        cf = report.get('counterfactuals', {})
        print(f"\nCounterfactuals: {cf.get('n_examples', 'N/A')} examples")
        cf_examples = cf.get('examples', [])
        if cf_examples:
            print(f"  Sample improvements:")
            for i, ex in enumerate(cf_examples[:3], 1):
                delta = ex.get('delta_time_s', 0)
                print(f"    {i}. {delta:+.1f}s ({ex.get('scenario', 'N/A')})")

        print("=" * 60)
        return report
    except Exception as e:
        print(f"Error reading report: {e}")
        return None

if __name__ == "__main__":
    read_results.remote()

