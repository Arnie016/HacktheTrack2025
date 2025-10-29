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
    args = ap.parse_args()

    # Use validation script from notebooks
    from validate_walkforward import main as validate_main
    
    # Override reports_dir if needed
    import validate_walkforward
    validate_walkforward.reports_dir = Path(args.outdir)
    
    validate_main()


if __name__ == "__main__":
    main()
