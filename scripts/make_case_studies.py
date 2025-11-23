"""Print top Δtime/Δposition examples across scenarios for video script."""
from __future__ import annotations

import json
from pathlib import Path


def main(reports_dir: str = "reports"):
    reports_path = Path(reports_dir)
    cf_path = reports_path / "counterfactuals.json"
    wf_path = reports_path / "walkforward_detailed.json"

    if not cf_path.exists() or not wf_path.exists():
        print("Reports not found. Run validation first.")
        return

    with open(cf_path) as f:
        cf = json.load(f)
    with open(wf_path) as f:
        wf = json.load(f)

    examples = cf.get("examples") or cf.get("examples", []) or cf.get("examples", [])
    if not examples:
        examples = cf.get("examples", [])

    # Flatten if needed
    if isinstance(examples, dict) and "examples" in examples:
        examples = examples["examples"]

    # Rank by absolute time gain descending
    ranked = sorted(
        examples,
        key=lambda x: abs(x.get("delta_time_s", 0.0)),
        reverse=True,
    )

    top = ranked[:2]
    print("Top case studies:")
    for i, ex in enumerate(top, 1):
        vid = ex.get("vehicle_id")
        lap = ex.get("lap")
        dt = ex.get("delta_time_s")
        dp = ex.get("delta_position", 0)
        rec = ex.get("recommendation")
        print(f"{i}. Car {vid} lap {lap}: Δtime {dt:+.2f}s, Δpos {dp:+d}; rec={rec}")


if __name__ == "__main__":
    main()








