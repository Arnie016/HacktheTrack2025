import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grcup.features.telemetry import build_telemetry_features

def preprocess_telemetry(race_dir: Path, raw_filename: str, output_filename: str):
    raw_path = race_dir / raw_filename
    output_path = race_dir / output_filename
    
    if not raw_path.exists():
        print(f"Skipping {raw_filename} (not found)")
        return

    print(f"Processing {raw_path} ({raw_path.stat().st_size / 1024 / 1024:.1f} MB)...")
    try:
        features = build_telemetry_features(raw_path)
        features.to_csv(output_path, index=False)
        print(f"✓ Saved features to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"✗ Failed: {e}")

def main():
    base_dir = Path(__file__).parent.parent
    
    # Race 1
    preprocess_telemetry(
        base_dir / "Race 1",
        "R1_vir_telemetry_data.csv",
        "R1_telemetry_features.csv"
    )
    
    # Race 2
    preprocess_telemetry(
        base_dir / "Race 2",
        "R2_vir_telemetry_data.csv",
        "R2_telemetry_features.csv"
    )

if __name__ == "__main__":
    main()





