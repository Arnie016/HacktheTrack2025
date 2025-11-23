# GR Cup Strategy Engine - Project Summary & Validation Guide

## What is "Mirror Leader"?

**Mirror Leader** is a baseline strategy that pits whenever the race leader pits. It's a reactive strategy used to compare against the optimizer:

- **How it works**: 
  1. Identifies the race leader (POSITION=1 in results)
  2. Detects when the leader pitted using `detect_stints()` (looks for large lap time spikes >30s)
  3. Simulates pitting at those same lap numbers
  4. Compares total race time vs the optimizer's strategy

- **Why it's a good baseline**: 
  - Reactive (follows the leader)
  - Reasonable strategy (leader usually knows what they're doing)
  - Should be close to optimal, so advantage should be **small** (5-15s, not 40s)

- **Previous bug**: Used fake multiplier (0.52x) instead of real simulation → inflated to 40s
- **Fixed**: Now actually simulates mirror leader strategy using Monte Carlo

---

## Project Overview

### Goal
Real-time pit strategy optimizer for Toyota GR Cup racing that:
- Trains on Race 1 data
- Walk-forward validates on Race 2 (causal, no leakage)
- Provides optimal pit stop timing recommendations

### Core Components

#### 1. **Models** (trained on Race 1)
- **Wear Quantile XGBoost**: Predicts tire degradation → lap time quantiles (q10, q50, q90)
- **Kalman Pace Filter**: Online pace estimation (3 regimes: green, pit-out, SC)
- **Cox SC Hazard**: Safety car probability based on race conditions
- **Overtake Logistic**: Position gain probability

#### 2. **Strategy Optimizer** (`src/grcup/strategy/optimizer.py`)
- **Stochastic Dynamic Programming**: Evaluates pit strategies using Monte Carlo
- **Input**: Current lap, tire age, fuel, SC status, models
- **Output**: Recommended pit lap, expected race time, confidence
- **Adaptive MC**: Re-simulates close calls (top-2 within 0.4s) with 100 scenarios

#### 3. **Validation** (`notebooks/validate_walkforward.py`)
- **Walk-forward**: Validates at each lap using only past information
- **Baseline comparisons**: Compares vs 3 baselines (fixed_stint, fuel_min, mirror_leader)
- **Counterfactuals**: Simulates "what if" scenarios
- **Calibration**: CQR-adjusted quantile coverage

---

## What We Fixed Today

### Problem: Fake Data
- **Baseline comparisons** used arbitrary multipliers (0.52x, 0.65x) instead of real simulations
- **Counterfactuals** used `expected_gain` directly (all showed 169s)
- **Mirror leader** showed 40s advantage (unrealistic)

### Solution: Real Simulations
1. ✅ Removed fake baseline comparison from optimizer
2. ✅ Implemented `simulate_baseline_strategy()` - actually simulates all 3 baselines
3. ✅ Implemented `simulate_race_time()` - Monte Carlo simulation with pit schedule
4. ✅ Implemented `get_leader_pit_schedule()` - detects leader's actual pit stops
5. ✅ Fixed `compute_baseline_comparisons()` - uses real simulations, not multipliers
6. ✅ Fixed `compute_counterfactuals()` - simulates actual vs recommended strategies
7. ✅ Fixed Python 3.9 compatibility (replaced `|` with `Optional[]`)

---

## How to Validate Everything

### Step 1: Install Dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run Validation
```bash
python3 scripts/validate_walkforward.py --event R2 --outdir reports/test --scenario base
```

### Step 3: Check Results

#### A. **Baseline Comparisons** (`reports/test/validation_report.json`)
Look for `baseline_comparisons.engine_advantage`:

```json
{
  "vs_fixed_stint": {
    "time_saved_s": 78.7,  // Should be largest (fixed is worst)
    "ci95": [12.0, 169.2]
  },
  "vs_fuel_min": {
    "time_saved_s": 51.2,  // Medium (fuel-optimized)
    "ci95": [7.8, 110.0]
  },
  "vs_mirror_leader": {
    "time_saved_s": 5-15,  // ✅ Should be SMALL (5-15s, not 40s!)
    "ci95": [2.0, 20.0]
  }
}
```

**Validation checks**:
- ✅ Mirror leader advantage < 20s (realistic)
- ✅ Fixed stint > Fuel min > Mirror leader (hierarchy makes sense)
- ✅ CI95 ranges are reasonable (not 0-200s)

#### B. **Counterfactuals** (`reports/test/counterfactuals.json`)
Look for `examples[].delta_time_s`:

```json
{
  "examples": [
    {"delta_time_s": 12.5},  // ✅ Varied values
    {"delta_time_s": -3.2},  // ✅ Some negative (actual was better)
    {"delta_time_s": 8.7},   // ✅ Not all 169s!
    ...
  ]
}
```

**Validation checks**:
- ✅ Values are varied (not all identical)
- ✅ Range is reasonable (-20s to +50s, not all 169s)
- ✅ Some negative values (actual strategy sometimes better)

#### C. **Wear Model Metrics**
```json
{
  "wear_model_metrics": {
    "MAE": 15.6,              // ✅ Reasonable
    "R2": 0.303,              // ✅ Positive
    "quantile_coverage_90": 0.86-0.93  // ✅ Close to 90%
  }
}
```

### Step 4: Run All Scenarios
```bash
for scenario in base early_sc late_sc hot_track heavy_traffic undercut no_weather; do
  python3 scripts/validate_walkforward.py --event R2 --scenario $scenario --outdir reports/$scenario
done
```

**Check**: All scenarios should complete without errors

### Step 5: Smoke Test
```bash
PYTHONWARNINGS=error python3 scripts/smoke_test.py
```

---

## Key Files Changed

1. **`src/grcup/strategy/optimizer.py`**
   - Already correct: returns `expected_time` (not `expected_gain`)
   - No fake baseline comparison

2. **`src/grcup/evaluation/walkforward.py`**
   - Updated to use `expected_time` instead of `expected_gain`

3. **`notebooks/validate_walkforward.py`**
   - Added `simulate_race_time()` - Monte Carlo simulation
   - Added `get_leader_pit_schedule()` - detect leader pits
   - Added `simulate_baseline_strategy()` - simulate baselines
   - Fixed `compute_baseline_comparisons()` - real simulations
   - Fixed `compute_counterfactuals()` - real simulations

4. **`src/grcup/data/schemas.py`**
   - Fixed Python 3.9 compatibility (`Optional[]` instead of `|`)

5. **`requirements.txt`**
   - Fixed numpy version (1.26.4 for Python 3.9 compatibility)

---

## Expected Results After Fix

### Before (Fake Data)
- Mirror leader: **40.9s** advantage (unrealistic)
- Counterfactuals: All **169.2s** (identical, fake)
- Baselines: Arbitrary multipliers

### After (Real Simulations)
- Mirror leader: **5-15s** advantage (realistic)
- Counterfactuals: **Varied** values (-20s to +50s)
- Baselines: Real Monte Carlo simulations

---

## Red Flags to Watch For

❌ **Mirror leader > 30s** → Simulation might be broken
❌ **All counterfactuals identical** → Still using fake data
❌ **CI95 ranges huge** (0-200s) → Simulation variance too high
❌ **Negative time_saved** → Optimizer worse than baseline (should be rare)
❌ **All baselines same value** → Not actually simulating different strategies

✅ **Mirror leader 5-15s** → Good!
✅ **Counterfactuals varied** → Good!
✅ **Fixed > Fuel > Mirror** → Makes sense!
✅ **CI95 reasonable** → Good!

---

## Quick Validation Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Validation runs without errors
- [ ] Mirror leader advantage < 20s
- [ ] Counterfactuals have varied values
- [ ] Baseline hierarchy makes sense (Fixed > Fuel > Mirror)
- [ ] CI95 ranges are reasonable
- [ ] All scenarios complete successfully
- [ ] Smoke test passes

---

## Questions?

If you see:
- **Mirror leader still 40s**: Check if `compute_baseline_comparisons()` is calling `simulate_baseline_strategy()`
- **All counterfactuals 169s**: Check if `compute_counterfactuals()` is using `simulate_race_time()`
- **Python errors**: Check Python version (needs 3.9+) and type hints

