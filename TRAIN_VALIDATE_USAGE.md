# Train & Validate One-Shot Script

## Quick Start

**Basic usage (train + validate base scenario):**
```bash
python scripts/train_and_validate.py
```

**Train + validate all scenarios:**
```bash
python scripts/train_and_validate.py --all-scenarios
```

**Using Makefile:**
```bash
make train-validate        # Train + validate base scenario
make train-validate-all    # Train + validate all scenarios
```

## Examples

### 1. Basic Run
```bash
python scripts/train_and_validate.py
```
- Trains models on Race 1
- Validates on Race 2 (base scenario)
- Saves models to `models/`
- Saves reports to `reports/base/`

### 2. All Scenarios
```bash
python scripts/train_and_validate.py --all-scenarios
```
- Trains once
- Validates all 7 scenarios:
  - `base`, `early_sc`, `late_sc`, `hot_track`, `heavy_traffic`, `undercut`, `no_weather`
- Saves reports to `reports/{scenario}/` for each

### 3. Custom CQR Parameters
```bash
python scripts/train_and_validate.py \
    --cqr-alpha 0.10 \
    --cqr-scale 2.2 \
    --cqr-band-scale 1.5
```
- Uses custom CQR calibration parameters
- `--cqr-alpha`: Target miscoverage (0.10 = 90% coverage)
- `--cqr-scale`: Scale factor for training
- `--cqr-band-scale`: Band scale for validation

### 4. Custom Directories
```bash
python scripts/train_and_validate.py \
    --models-dir custom_models/ \
    --reports-dir custom_reports/
```

### 5. Skip Training (Use Existing Models)
```bash
python scripts/train_and_validate.py --skip-training
```
- Skips training step
- Uses existing models from `models/`
- Only runs validation

### 6. Skip Validation (Only Train)
```bash
python scripts/train_and_validate.py --skip-validation
```
- Only trains models
- Skips validation step

### 7. Single Scenario
```bash
python scripts/train_and_validate.py --scenario hot_track
```
- Trains models
- Validates only `hot_track` scenario

## Output

The script provides:
1. **Training output**: Model training progress and metrics
2. **Validation output**: Walk-forward validation results per scenario
3. **Summary**: Time saved, coverage metrics, elapsed time

Example output:
```
======================================================================
 GR Cup Training & Validation Pipeline
======================================================================
Training event: R1
Validation event: R2
Models directory: models
Reports directory: reports

[1/3] Training models...
============================================================
GR Cup Model Training Pipeline
============================================================
...
✓ Training completed successfully

[2/3] Validating scenarios: base
  Running scenario: base
...
✓ base: Time saved=7.9s, Coverage=93.25%

✓ Validation completed

======================================================================
 Validation Summary
======================================================================
Successful scenarios: 1/1

Results:
  base           : Time saved=   7.9s, Coverage=93.3%

[3/3] Pipeline Summary

Total time: 245.3 seconds (4.1 minutes)
Models saved to: models
Reports saved to: reports

======================================================================
 Pipeline Complete
======================================================================
```

## Environment Variables

You can also set simulation counts via env vars:
```bash
MC_BASE_SCENARIOS=2000 \
MC_CLOSE_SCENARIOS=3000 \
python scripts/train_and_validate.py --all-scenarios
```

## Exit Codes

- `0`: Success
- `1`: Error (training failed or validation failed for all scenarios)

## Files Created

**Models:**
- `models/wear_quantile_xgb.pkl`
- `models/cox_hazard.pkl`
- `models/overtake.pkl`
- `models/kalman_config.json`
- `models/metadata.json`

**Reports (per scenario):**
- `reports/{scenario}/validation_report.json`
- `reports/{scenario}/walkforward_detailed.json`
- `reports/{scenario}/counterfactuals.json`
- `reports/{scenario}/ablation_report.json`








