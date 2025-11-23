# GR Cup Strategy Dashboard

Quick web UI to visualize model predictions, calibration metrics, and strategy recommendations.

## Setup

```bash
# Install Flask if needed
pip install flask==3.0.0

# Or install all requirements
pip install -r requirements.txt
```

## Run

```bash
# Option 1: Direct
python app.py

# Option 2: Via script
python scripts/run_dashboard.py
```

Then open http://localhost:5000 in your browser.

## Features

- **Model Metrics**: R², MAE, RMSE, sample count
- **Calibration**: Coverage percentage with visual bars
- **Strategy Performance**: Time saved vs baselines with confidence intervals
- **Coverage Charts**: By tire age and temperature buckets
- **Counterfactuals**: Top strategy recommendations with expected gains

## What It Shows

- **Wear Model**: How well the tire wear model predicts lap times
- **Coverage**: Whether 90% prediction bands actually cover ~90% of real laps
- **Time Saved**: How much faster your strategy is vs fixed pit stops
- **Bucket Analysis**: Coverage broken down by tire age and track temperature

