"""Ablation studies: remove features/models and measure impact."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def run_ablations(
    base_features: pd.DataFrame,
    base_metrics: dict,
    ablations: list[dict],  # [{"name": "no_weather", "remove": ["track_temp"]}, ...]
    wear_model,
    predict_fn,
) -> pd.DataFrame:
    """
    Run ablation studies.
    
    Args:
        base_features: Full feature set
        base_metrics: Metrics with full feature set (MAE, RMSE, etc.)
        ablations: List of ablation configs
        wear_model: Trained model
        predict_fn: Prediction function
    
    Returns:
        DataFrame with ablation results
    """
    results = []
    
    # Baseline (full features)
    results.append({
        "ablation": "baseline",
        "features_removed": [],
        **base_metrics,
    })
    
    # Run each ablation
    for ablation_config in ablations:
        name = ablation_config["name"]
        remove_cols = ablation_config.get("remove", [])
        
        # Remove features
        ablated_features = base_features.drop(columns=remove_cols, errors="ignore")
        
        # Re-train or re-predict (simplified - would need full retrain)
        # For now, just report feature removal
        results.append({
            "ablation": name,
            "features_removed": remove_cols,
            "n_features_removed": len(remove_cols),
            "note": "Would require model retraining",
        })
    
    return pd.DataFrame(results)


def compare_to_baseline(
    ablation_results: pd.DataFrame,
    metric_col: str = "MAE",
) -> pd.DataFrame:
    """
    Compute delta vs baseline for each ablation.
    
    Args:
        ablation_results: DataFrame from run_ablations
        metric_col: Metric to compare
    
    Returns:
        DataFrame with deltas
    """
    baseline_val = ablation_results[
        ablation_results["ablation"] == "baseline"
    ][metric_col].iloc[0]
    
    ablation_results[f"{metric_col}_delta"] = (
        ablation_results[metric_col] - baseline_val
    )
    
    ablation_results[f"{metric_col}_pct_change"] = (
        ablation_results[f"{metric_col}_delta"] / baseline_val * 100
    )
    
    return ablation_results


def save_ablation_report(results: pd.DataFrame, path: Path | str):
    """Save ablation results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    results_dict = results.to_dict(orient="records")
    
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)


