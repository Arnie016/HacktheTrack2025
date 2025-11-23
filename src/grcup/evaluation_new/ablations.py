"""Ablation studies and baseline comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def run_ablations(
    base_model,
    features_df,
    actuals,
    ablation_configs: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """
    Run ablation studies by removing features.
    
    Args:
        base_model: Base model to use
        features_df: Feature DataFrame
        actuals: Actual target values
        ablation_configs: List of configs, each with 'name' and 'features_to_remove'
    
    Returns:
        List of ablation results
    """
    results = []
    
    for config in ablation_configs:
        name = config.get("name", "unknown")
        features_to_remove = config.get("features_to_remove", [])
        
        # Create ablated features
        ablated_features = features_df.drop(columns=features_to_remove, errors="ignore")
        
        # Run model (simplified - would need actual model prediction)
        result = {
            "name": name,
            "features_removed": features_to_remove,
            "n_features": len(ablated_features.columns),
            "note": "Ablation study - would require model retraining",
        }
        results.append(result)
    
    return results


def save_ablation_report(
    ablations: list[Dict[str, Any]],
    path: Path | str,
):
    """Save ablation report to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "ablations": ablations,
    }
    
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def compare_to_baseline(
    model_predictions,
    baseline_predictions,
    actuals,
) -> Dict[str, Any]:
    """
    Compare model predictions to baseline.
    
    Returns:
        Dictionary with comparison metrics
    """
    # Simplified comparison
    return {
        "model_mae": None,  # Would compute MAE
        "baseline_mae": None,
        "improvement": None,
        "note": "Baseline comparison - requires actual predictions",
    }

