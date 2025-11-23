# Using Modal with GR Cup Strategy Engine

## Setup

1. **Install Modal CLI:**
   ```bash
   python3 -m pip install modal
   ```

2. **Authenticate:**
   ```bash
   python3 -m modal token new
   ```
   This opens a browser to authenticate with Modal.

   **Note:** Use `python3 -m modal` instead of just `modal` to avoid PATH issues.

## Usage

### Train Models

Train all models on Race 1 data:

```bash
python3 -m modal run grcup_modal.py::train_models --event R1 --cqr-alpha 0.10 --cqr-scale 0.90
```

### Validate Single Scenario

Run walk-forward validation on Race 2 for a specific scenario:

```bash
python3 -m modal run grcup_modal.py::validate_walkforward --event R2 --scenario base --cqr-scale 2.2 --cqr-band-scale 1.5
```

Available scenarios:
- `base` - Default scenario
- `early_sc` - Early safety car
- `late_sc` - Late safety car
- `hot_track` - High temperature
- `heavy_traffic` - Heavy traffic conditions
- `undercut` - Undercut scenario
- `no_weather` - No weather data

### Validate All Scenarios (Parallel)

Run all scenarios in parallel on Modal:

```bash
python3 -m modal run grcup_modal.py::validate_all_scenarios --event R2 --cqr-scale 2.2 --cqr-band-scale 1.5
```

This runs all 7 scenarios concurrently, much faster than sequential runs.

### Deploy Dashboard

Deploy the Flask dashboard as a web endpoint:

```bash
python3 -m modal deploy grcup_modal.py::serve_dashboard
```

Then access it via the URL provided by Modal.

## Modal CLI Commands

### Check Status

```bash
# List all running functions
python3 -m modal app list

# View logs
python3 -m modal app logs grcup-strategy

# View function runs
python3 -m modal function logs grcup-strategy::train_models
```

### Access Volumes

Models and reports are stored in Modal volumes:

```bash
# Download models locally
python3 -m modal volume download grcup-models /models ./models

# Download reports locally
python3 -m modal volume download grcup-reports /reports ./reports
```

### Interactive Shell

Run a shell in Modal to debug:

```bash
python3 -m modal run grcup_modal.py::train_models --interactive
```

## Costs

Modal charges per second of compute time:
- CPU: ~$0.0001/second per CPU
- Memory: ~$0.000001/second per GB
- Example: 4 CPU, 8GB RAM for 1 hour = ~$0.50

Volumes are free for first 10GB, then ~$0.10/GB/month.

## Tips

1. **Parallel Validation**: Use `validate_all_scenarios` to run all scenarios at once (much faster).

2. **Volume Persistence**: Models and reports persist in volumes, so you can download them later or share between runs.

3. **Monitoring**: Use Modal dashboard (modal.com) to monitor runs, costs, and logs.

4. **Local Development**: Test locally first, then run on Modal for full dataset or parallel processing.

5. **Environment Variables**: Modal functions inherit your local env vars, or set them in the function definition.

