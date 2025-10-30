# GR Cup Real-Time Strategy Engine

Real-time strategy engine for Toyota GR Cup racing: wear quantiles + online pace + SC hazard → stochastic DP policy.

## Planning & Automation

**Goal:** CPU-only pipeline that trains on Race 1, walk-forward validates on Race 2, and auto-commits artifacts nightly.

### Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install

# tiny run
PYTHONWARNINGS=error python scripts/smoke_test.py
```

### Train & Validate (full)

```bash
make train    # or: python scripts/train.py --event R1 --outdir models/
make validate # or: python scripts/validate_walkforward.py --event R2 --outdir reports/
make all      # train + validate
```

### CI/Automation

* **CI** runs on every push/PR (lint + smoke test).
* **Nightly job** (02:00 UTC) trains/validates on CPU and **auto-commits** updated `models/` and `reports/` to `bot/artifacts` branch (opens/updates PR).
* **Pre-deadline release** (00:00 + 00:40 UTC Nov 25, 2025) makes a tarball before the submission cutoff.

### Submission Artifacts

* `models/` – pickled models + metadata.json (versions, git SHA, RNG seeds)
* `reports/` – `validation_report.json`, `ablation_report.json`, `counterfactuals.json`
* One-click bundle on the **Releases** page

## Models

* **Wear Quantile XGB** (monotonic on tire_age) → quantile predictions
* **Kalman Pace Filter** (3-regime) → online pace estimation
* **Cox SC Hazard** → safety car probability
* **Overtake Logistic** → position gain odds

## Strategy

Stochastic Dynamic Programming optimizer using model outputs with uncertainty propagation.

## Validation

Walk-forward on Race 2 (causal, no leakage) with counterfactuals, calibration, and ablations.

### Calibration Summary

90% band calibrated to ~**93% empirical coverage** on Race 2 using **CQR α=0.10** with runtime scaling (`CQR_SCALE=2.2`, `CQR_BAND_SCALE=1.5`).

Validated on Race 2 with 401 high-quality laps:
- **MAE**: 15.6s
- **RMSE**:  (see report)
- **R²**: 0.303
- **Quantile Coverage @90%**: 93.25% ✅

Reproduce:
```bash
python notebooks/train_models.py --cqr-alpha 0.10 --cqr-scale 2.2
CQR_SCALE=2.2 CQR_BAND_SCALE=1.5 python scripts/validate_walkforward.py --event R2 --outdir reports/
```

The model uses Conformalized Quantile Regression (CQR) for calibrated uncertainty, enabling risk-aware strategy decisions.


