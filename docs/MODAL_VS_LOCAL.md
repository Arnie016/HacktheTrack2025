# Modal vs Local Scripts Comparison

## Two Scripts Available

### 1. Local Script: `scripts/train_and_validate.py`
**Runs on your local machine**

- Uses local CPU/RAM
- Trains models locally
- Validates locally
- Saves to local directories (`models/`, `reports/`)

**Usage:**
```bash
python scripts/train_and_validate.py
python scripts/train_and_validate.py --all-scenarios
```

**Best for:**
- Quick testing
- Development/debugging
- When you have good local hardware
- When you want immediate access to files

---

### 2. Modal Script: `scripts/train_and_validate_modal.py`
**Runs on Modal cloud infrastructure**

- Uses Modal's cloud CPU/RAM (4 CPU, 8GB RAM)
- Trains models on Modal
- Validates on Modal
- Saves to Modal volumes (`grcup-models`, `grcup-reports`)
- Can download results locally with `--download-results`

**Usage:**
```bash
python scripts/train_and_validate_modal.py
python scripts/train_and_validate_modal.py --all-scenarios --download-results
```

**Best for:**
- Production runs
- When local machine is slow/limited
- Parallel scenario runs
- When you want cloud scalability
- Automated/nightly runs

---

## Comparison

| Feature | Local Script | Modal Script |
|---------|-------------|--------------|
| **Compute** | Your machine | Modal cloud |
| **CPU** | Your CPU cores | 4 CPU cores |
| **RAM** | Your RAM | 8GB RAM |
| **Storage** | Local disk | Modal volumes |
| **Cost** | Free | ~$0.36/hour |
| **Speed** | Depends on hardware | Consistent |
| **Parallel** | Limited | Better |
| **Access** | Immediate | Via download |
| **Best for** | Development | Production |

---

## When to Use Which

### Use **Local Script** when:
- ✅ Testing/debugging code changes
- ✅ Quick iterations
- ✅ Your machine is fast enough
- ✅ You want immediate file access
- ✅ No internet/Modal access needed

### Use **Modal Script** when:
- ✅ Production runs
- ✅ Your machine is slow/limited
- ✅ Running all scenarios in parallel
- ✅ Automated/nightly jobs
- ✅ Need consistent performance
- ✅ Want cloud scalability

---

## Quick Examples

**Local (development):**
```bash
# Quick test
python scripts/train_and_validate.py --scenario base

# All scenarios locally
python scripts/train_and_validate.py --all-scenarios
```

**Modal (production):**
```bash
# Train + validate on Modal
python scripts/train_and_validate_modal.py

# All scenarios + download results
python scripts/train_and_validate_modal.py --all-scenarios --download-results

# Skip training, only validate
python scripts/train_and_validate_modal.py --skip-training
```

---

## Modal Volumes

**Models volume:** `grcup-models`
- Contains: `wear_quantile_xgb.pkl`, `cox_hazard.pkl`, `overtake.pkl`, `kalman_config.json`, `metadata.json`

**Reports volume:** `grcup-reports`
- Contains: `{scenario}/validation_report.json`, `{scenario}/walkforward_detailed.json`, etc.

**Download manually:**
```bash
# Download models
python3 -m modal volume download grcup-models /models ./models

# Download reports
python3 -m modal volume download grcup-reports /reports/base ./reports/base
```

---

## Makefile Targets

**Local:**
```bash
make train-validate        # Local: train + validate base
make train-validate-all    # Local: train + validate all scenarios
```

**Modal (add to Makefile if needed):**
```bash
make train-validate-modal        # Modal: train + validate base
make train-validate-modal-all   # Modal: train + validate all scenarios
```








