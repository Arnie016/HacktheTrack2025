"""Walk-forward validation on Race 2."""
import argparse
import sys
from pathlib import Path

# Add notebooks to path
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))


def main():
    ap = argparse.ArgumentParser(description="Validate models with walk-forward on Race 2")
    ap.add_argument("--event", required=True, help="Race event (R2)")
    ap.add_argument("--outdir", default="reports", help="Output directory")
    ap.add_argument("--scenario", default="base", help="Named scenario (base, early_sc, late_sc, hot_track, heavy_traffic, undercut, no_weather)")
    args = ap.parse_args()

    # Use validation script from notebooks
    from validate_walkforward import main as validate_main
    
    # Override reports_dir if needed
    import validate_walkforward
    validate_walkforward.reports_dir = Path(args.outdir)
    # Plumb scenario selection into notebook module (consumed within validate_walkforward)
    try:
        setattr(validate_walkforward, "scenario", args.scenario)
    except Exception:
        pass
    
    validate_main()


if __name__ == "__main__":
    main()
