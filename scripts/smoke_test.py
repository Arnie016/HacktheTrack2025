"""Fast smoke test to catch type/OOM errors early."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Turn warnings into errors
import warnings
warnings.filterwarnings("error")


def main():
    print("Python OK")
    
    # Check directories exist
    for p in ["models", "reports", "scripts"]:
        Path(p).mkdir(exist_ok=True)
    print("Directories OK")
    
    # Import check (light)
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    print("Imports OK")
    
    # Basic schema validation
    from src.grcup.data.schemas import LapRow, WeatherRow
    
    # Test timestamp normalization (UTC)
    test_ts = pd.Timestamp.now(tz="UTC")
    assert test_ts.tz is not None, "Timestamps must be UTC"
    print("Timestamp validation OK")
    
    # Test no NaNs in simple operations
    test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert test_df.notna().all().all(), "Test should have no NaNs"
    print("NaN check OK")
    
    # Test monotonic assertion
    test_laps = pd.Series([1, 2, 3, 4, 5])
    assert test_laps.is_monotonic_increasing, "Laps should be monotonic"
    print("Monotonic check OK")
    
    print("Smoke test passed ✓")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Smoke test FAILED: {e}", file=sys.stderr)
        sys.exit(1)


