# Simulation & Comparison Improvements

## Summary of Changes

### 1. ✅ Increased Simulation Counts (COMPLETED)

**Baseline Comparisons:**
- Base scenarios: **100 → 500** (5x increase)
- Refined scenarios (close calls): **200 → 1000** (5x increase)

**Optimizer:**
- Standard scenarios: **100 → 500** (5x increase)
- Close call refinement: **100 → 1000** (10x increase)

**Counterfactuals:**
- Simulation count: **100 → 500** (5x increase)

**Benefits:**
- **Better statistical power**: More samples = tighter confidence intervals
- **More reliable comparisons**: Reduced variance in baseline vs optimizer comparisons
- **Consistency**: All components now use similar simulation counts

### 2. ✅ Enhanced Debugging (COMPLETED)

Added detailed debug output to diagnose the 0.0 baseline comparison issue:
- Per-checkpoint comparison logging (first 3 recommendations)
- Warning messages for invalid simulation results
- Total recommendations vs valid optimizer times tracking
- Scenario count tracking per baseline

### 3. ✅ Feature Coverage Expansion (COMPLETED)

- **Weather joins upgraded**: added `air_temp_c`, `humidity_pct`, `wind_speed_kph`, `pressure_hpa`, `rain_intensity`, and `wind_direction_deg` with imputation flags.
- **Race context merged**: lap-level `gap_to_leader_s`, `gap_to_car_ahead_s`, `lap_position`, `position_fraction`, `flag_state_code`, `pit_time_s`, `top_speed`, `s1/s2/s3` splits, and heuristic `damage_indicator`.
- **Results and telemetry wiring**: final classification (`final_position`, `status_classified`, `laps_completed`) and physics metrics (accel/jerk/throttle) now flow automatically through both training and inference.
- **Model defaults stored**: wear quantile models persist median defaults so optimizers can consume richer feature sets without real-time telemetry gaps.
- **Smoke test**: `build_wear_training_dataset` on Race 1 now emits 51 columns (vs 18), confirming CSV data is fully leveraged.

### 3. 🔄 Additional Recommendations

#### A. Variance Reduction Techniques (Not Yet Implemented)

**Common Random Numbers (CRN):**
- ✅ Already partially implemented with shared `scenario_seeds`
- Could be improved by ensuring identical random streams for all strategies

**Antithetic Variates:**
- For each scenario, also run with inverted random numbers
- Reduces variance by ~50% with minimal computational cost

**Control Variates:**
- Use a simpler analytical model as control
- Adjust Monte Carlo estimates based on control model error

#### B. Position-Based Metrics (Not Yet Implemented)

Currently only comparing **time saved**, but could also track:
- **Position gain probability**: P(finish position improves)
- **Expected position**: E[final_position]
- **Position variance**: Var[final_position]

#### C. Scenario Quality Improvements

**More Realistic SC Modeling:**
- Current: Fixed 0.1 probability SC ends per lap
- Better: Time-based SC duration (2-5 laps typical)
- Better: Track position-dependent SC probability

**Traffic Modeling:**
- Currently assumes clean air
- Could model traffic density from actual race data
- Affects both pace and SC probability

**Weather Integration:**
- Currently hardcoded: `rain=0, wind_speed=0.0`
- Could use actual weather data from Race 2
- Weather affects tire degradation and SC probability

## Expected Impact

### Before Improvements:
- Baseline comparisons: **0.0** (bug/issue)
- Confidence intervals: Wide (±50-100s)
- Statistical power: Low (100 scenarios)

### After Improvements:
- Baseline comparisons: **Should show realistic values** (5-80s range expected)
- Confidence intervals: **Tighter** (±10-30s with 500 scenarios)
- Statistical power: **5x better** (500 vs 100 scenarios)

## Performance Considerations

**Computational Cost:**
- Baseline comparisons: ~5x slower (500 vs 100 scenarios)
- Optimizer: ~5x slower per recommendation
- **Total runtime**: May increase from ~5min to ~25min for full validation

**Mitigation Options:**
1. **Parallelization**: Run scenarios in parallel (multiprocessing)
2. **Sampling**: Use stratified sampling instead of pure random
3. **Early stopping**: Stop refinement if confidence interval is already tight
4. **Caching**: Cache simulation results for identical states

## Next Steps

1. **Run validation** to see if 0.0 issue is resolved:
   ```bash
   python scripts/validate_walkforward.py --event R2 --outdir reports/test --scenario base
   ```

2. **Check debug output** for:
   - Valid optimizer times count
   - Baseline simulation success rates
   - First few comparison deltas

3. **If still seeing 0.0**, investigate:
   - Are `recommendations_log` entries valid?
   - Are baseline simulations actually running?
   - Are `expected_time` values being set correctly?

4. **Consider implementing** (if needed):
   - Antithetic variates for variance reduction
   - Position-based metrics
   - Weather/traffic integration

## Quick Reference: Simulation Counts

| Component | Old | New | Increase |
|-----------|-----|-----|----------|
| Baseline base | 100 | 500 | 5x |
| Baseline refined | 200 | 1000 | 5x |
| Optimizer base | 100 | 500 | 5x |
| Optimizer refined | 100 | 1000 | 10x |
| Counterfactuals | 100 | 500 | 5x |


