# Race 1 Improvements Validation

## Overview

Created validation script to test improvements made to Race 1 strategy/stage.

## Files Created

1. **`validate_race1_improvements.py`** - Main validation script for Race 1
   - Tests improved strategy on Race 1 data
   - Compares AI recommendations to actual Race 1 outcomes
   - Generates validation report

## Current Issue

**Filesystem timeout** - Python can't read files from disk (timeout error 60). This affects:
- Model loading (`src.grcup.models`)
- Evaluation module loading (`src.grcup.evaluation`)
- Any file I/O operations

## What the Script Does (When Filesystem Works)

1. **Loads Models**
   - Wear quantile XGBoost model
   - SC hazard (Cox) model
   - Overtake model

2. **Loads Race 1 Data**
   - Lap times, starts, ends
   - Sectors
   - Weather
   - Results
   - Telemetry features (if available)

3. **Runs Walk-Forward Validation**
   - Generates recommendations for each vehicle at checkpoints
   - Uses improved strategy solver
   - Compares to actual Race 1 outcomes

4. **Computes Statistics**
   - Total recommendations
   - Pit vs no-pit recommendations
   - Average confidence
   - Match rate (AI vs actual)

5. **Saves Results**
   - `reports/race1_validation/validation_report.json`
   - `reports/race1_validation/walkforward_detailed.json`

## How to Run (After Fixing Filesystem)

```bash
# Basic run
python3 validate_race1_improvements.py

# With environment variables for faster testing
BASELINE_BASE_SCENARIOS=500 BASELINE_REFINE_SCENARIOS=1000 python3 validate_race1_improvements.py
```

## Expected Output

```
Race 1 Improvements Validation
======================================================================

[1/5] Loading trained models...
  Loading wear model... ✓
  Loading SC hazard model... ✓
  Loading overtake model... ✓

[2/5] Loading Race 1 data...
  Loading lap timing files... ✓ (X laps, Y vehicles)
  Loading sectors... ✓ (Z records)
  Loading weather... ✓ (W records)
  Loading results... ✓ (V entries)

[3/5] Running walk-forward validation on Race 1...
  Processing recommendations... ✓ (N recommendations, avg confidence: X.XX)

[4/5] Computing statistics...
  Total recommendations: N
  Pit recommendations: X
  No-pit recommendations: Y
  Average confidence: Z.ZZ%
  Average expected time: T.TTs

[5/5] Comparing to actual Race 1 outcomes...
  Recommendations matching actual: M/T (XX.X%)

✓ Validation report saved to: reports/race1_validation/validation_report.json

Race 1 Validation Complete!
======================================================================
```

## What to Check After Running

1. **Match Rate** - How often AI recommendations match actual Race 1 strategy
2. **Confidence** - Average confidence of recommendations
3. **Pit Recommendations** - Number of pit vs no-pit recommendations
4. **Expected Time** - Average expected race time with AI strategy

## Next Steps

1. **Fix filesystem timeout issue** (check disk health, network mounts, etc.)
2. **Run validation script** once filesystem is working
3. **Compare results** to Race 2 validation to see improvements
4. **Update submission documents** with Race 1 validation results

## Notes

- Script uses timeout protection for imports (30 seconds)
- Falls back gracefully if walkforward_validate can't be imported
- Saves intermediate results to avoid data loss
- Single-threaded to prevent freezing issues


