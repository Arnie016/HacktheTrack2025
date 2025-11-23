# Modal GPU Options for GR Cup Strategy Engine

## Current Setup: CPU-Only

Your current `grcup_modal.py` functions run on **CPU only** (no GPU specified). This is fine for:
- XGBoost training (CPU is often sufficient for small-medium datasets)
- Data processing and validation
- Monte Carlo simulations

## Available GPUs in Modal

Modal offers these NVIDIA GPUs:

### High-End (Training/Inference)
- **H100** - Latest flagship, ~$4-8/hour
  - 80GB HBM3 memory
  - Best for large model training
- **A100-80GB** - ~$2-4/hour
  - 80GB memory, Ampere architecture
  - Good for training large models
- **A100-40GB** - ~$1.5-3/hour
  - 40GB memory
  - Balanced performance/cost

### Mid-Range (Inference/Batch)
- **L40S** - ~$1-2/hour
  - 48GB memory, Ada Lovelace
  - Good for inference workloads
- **A10G** - ~$0.75-1.5/hour
  - 24GB memory
  - Cost-effective for inference

### Budget (Light Workloads)
- **T4** - ~$0.40-0.80/hour
  - 16GB memory
  - Good for small models, testing
- **L4** - ~$0.50-1/hour
  - 24GB memory
  - Better than T4 for similar price

## When to Use GPU

### For Your Project:

**XGBoost Training:**
- XGBoost supports GPU via `tree_method='gpu_hist'`
- GPU helps with **large datasets** (100K+ samples)
- Your Race 1 data is probably small enough that CPU is fine
- **Recommendation**: Keep CPU unless training takes >30min

**Monte Carlo Simulations:**
- Your simulations are CPU-bound (numpy/pandas)
- GPU won't help unless you rewrite with CUDA
- **Recommendation**: Keep CPU

**Parallel Validation:**
- Running 7 scenarios in parallel benefits from **more CPU cores**, not GPU
- **Recommendation**: Increase CPU cores (8-16) instead of GPU

## How to Add GPU

### Option 1: Add GPU to Training Function

```python
@app.function(
    image=image,
    volumes={"/models": models_volume, "/reports": reports_volume},
    gpu="T4",  # Add this line
    timeout=3600,
    cpu=4.0,
    memory=8192,
)
def train_models(...):
    # XGBoost will auto-detect GPU if available
    # Make sure to use tree_method='gpu_hist' in XGBoost config
```

### Option 2: Use A100 for Faster Training

```python
@app.function(
    gpu="A100-40GB",  # More expensive but faster
    ...
)
```

### Option 3: Use T4 for Budget Testing

```python
@app.function(
    gpu="T4",  # Cheapest option
    ...
)
```

## Cost Comparison

**Current (CPU-only):**
- 4 CPU cores, 8GB RAM: ~$0.0001/second = **~$0.36/hour**

**With GPU:**
- T4: ~$0.40/hour + CPU = **~$0.76/hour** (2x cost)
- A100-40GB: ~$1.50/hour + CPU = **~$1.86/hour** (5x cost)
- H100: ~$4/hour + CPU = **~$4.36/hour** (12x cost)

## Recommendation for Your Project

**Keep CPU-only** because:
1. Your dataset is small (Race 1/2 data)
2. XGBoost CPU training is fast enough
3. Monte Carlo is CPU-bound
4. Save money for parallel scenario runs

**Only add GPU if:**
- Training takes >30 minutes on CPU
- You're doing hyperparameter sweeps (100+ runs)
- You want to experiment with larger models

## Example: GPU-Enabled Training

If you want to try GPU acceleration:

```python
@app.function(
    image=image,
    volumes={"/models": models_volume, "/reports": reports_volume},
    gpu="T4",  # Add GPU
    timeout=3600,
    cpu=4.0,
    memory=8192,
)
def train_models(event: str = "R1", cqr_alpha: float = 0.10, cqr_scale: float = 0.90):
    """Train all models on Race data with GPU acceleration."""
    # ... existing code ...
    
    # In train_wear_quantile_model, ensure XGBoost uses GPU:
    # xgb.XGBRegressor(tree_method='gpu_hist', ...)
```

## Checking GPU Availability

In your function, you can check if GPU is available:

```python
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("No GPU available")
```

## Summary

- **Current**: CPU-only (no GPU)
- **Cost**: ~$0.36/hour
- **Recommendation**: Keep CPU unless training is slow
- **If adding GPU**: Use `gpu="T4"` for budget or `gpu="A100-40GB"` for speed

