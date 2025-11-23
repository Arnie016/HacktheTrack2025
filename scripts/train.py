"""Train models on Race 1 data."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add notebooks to path for training script
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return "unknown"


def get_package_versions() -> dict:
    """Get versions of key packages."""
    packages = ["xgboost", "pandas", "numpy", "lifelines", "numba", "sklearn"]
    versions = {}
    
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=True
        )
        
        for line in result.stdout.split("\n"):
            for pkg in packages:
                if line.startswith(pkg + "=="):
                    versions[pkg] = line.split("==")[1]
    except:
        pass
    
    return versions


def main():
    ap = argparse.ArgumentParser(description="Train models on race data")
    ap.add_argument("--event", required=True, help="Race event (R1, R2)")
    ap.add_argument("--outdir", default="models", help="Output directory")
    args = ap.parse_args()

    # Use training script from notebooks
    from train_models import main as train_main
    
    # Override models_dir if needed
    import train_models
    train_models.models_dir = Path(args.outdir)
    
    train_main()


if __name__ == "__main__":
    main()

