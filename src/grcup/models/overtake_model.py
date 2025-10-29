"""Logistic regression for overtake probability."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def prepare_overtake_data(
    sectors_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare training data for overtake model.
    
    Features:
    - delta_pace (pace difference vs car ahead)
    - tire_age_diff (tire age difference)
    - gap_seconds (current gap)
    - sector_s3_coeff (straight/exit speed proxy)
    - restart_flag (SC restart = 1, else 0)
    
    Target:
    - overtake (1 if position gained, 0 otherwise)
    """
    # This is simplified - real implementation would track positions lap-by-lap
    # For now, use sector times to proxy for pace differences
    
    overtake_data = []
    
    # Group by lap
    for lap_num in sectors_df["LAP_NUMBER"].unique():
        lap_sectors = sectors_df[sectors_df["LAP_NUMBER"] == lap_num]
        
        if len(lap_sectors) < 2:
            continue
        
        # Sort by S3 time (straight/exit speed)
        lap_sectors_sorted = lap_sectors.sort_values("S3_SECONDS")
        
        # Compare each car to the one ahead
        for i in range(len(lap_sectors_sorted) - 1):
            car_ahead = lap_sectors_sorted.iloc[i]
            car_behind = lap_sectors_sorted.iloc[i + 1]
            
            delta_pace = car_behind["S3_SECONDS"] - car_ahead["S3_SECONDS"]
            gap_seconds = abs(delta_pace)
            
            # Only consider gaps < 2.5s (cap extrapolation)
            if gap_seconds > 2.5:
                continue
            
            # Tire age diff (simplified - would need stint data)
            tire_age_diff = 0.0  # TODO: compute from stint data
            
            # Sector S3 coefficient (speed proxy)
            sector_s3_coeff = car_behind["S3_SECONDS"]
            
            # Restart flag (check if previous lap had SC)
            prev_lap = sectors_df[sectors_df["LAP_NUMBER"] == lap_num - 1]
            restart_flag = 1 if (prev_lap["FLAG_AT_FL"] == "FCY").any() else 0
            
            # Target: did car behind pass car ahead? (simplified heuristic)
            # In reality, need position tracking
            overtake_target = 1 if delta_pace < -0.1 else 0  # Faster = passed
            
            overtake_data.append({
                "delta_pace": delta_pace,
                "tire_age_diff": tire_age_diff,
                "gap_seconds": gap_seconds,
                "sector_s3_coeff": sector_s3_coeff,
                "restart_flag": restart_flag,
                "overtake": overtake_target,
            })
    
    return pd.DataFrame(overtake_data)


def train_overtake_model(
    overtake_df: pd.DataFrame,
    balance_ratio: float = 2.0,
) -> LogisticRegression:
    """
    Train logistic regression for overtake probability.
    
    Args:
        overtake_df: DataFrame with features and overtake target
        balance_ratio: Ratio of negative samples to positive samples (default 2.0)
    
    Returns:
        Fitted LogisticRegression model
    """
    features = ["delta_pace", "tire_age_diff", "gap_seconds", "sector_s3_coeff", "restart_flag"]
    
    X = overtake_df[features].values
    y = overtake_df["overtake"].values
    
    # Remove NaNs
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        raise ValueError("No valid training data")
    
    # Check for class balance
    unique_classes = np.unique(y)
    
    if len(unique_classes) < 2:
        # Only one class - create a fallback model that predicts baseline probability
        # This prevents pipeline failure while indicating the issue
        from sklearn.dummy import DummyClassifier
        
        print(f"Warning: Only one class in overtake data ({unique_classes[0]}). Using fallback model.")
        model = DummyClassifier(strategy="constant", constant=unique_classes[0])
        model.fit(X, y)
        return model
    
    # Balance the classes if needed
    positive_count = int(np.sum(y == 1))
    negative_count = int(np.sum(y == 0))
    
    if positive_count == 0 or negative_count == 0:
        # Edge case: after filtering, one class disappeared
        from sklearn.dummy import DummyClassifier
        print(f"Warning: Class imbalance after filtering. Using fallback model.")
        model = DummyClassifier(strategy="constant", constant=unique_classes[0])
        model.fit(X, y)
        return model
    
    # Balance by oversampling minority class or undersampling majority
    if positive_count < negative_count:
        # Oversample positives
        target_neg_count = int(positive_count * balance_ratio)
        neg_indices = np.where(y == 0)[0]
        if len(neg_indices) > target_neg_count:
            neg_selected = np.random.choice(neg_indices, size=target_neg_count, replace=False)
        else:
            neg_selected = neg_indices
        
        pos_indices = np.where(y == 1)[0]
        selected_indices = np.concatenate([neg_selected, pos_indices])
        X = X[selected_indices]
        y = y[selected_indices]
    elif negative_count < positive_count:
        # Oversample negatives
        target_pos_count = int(negative_count * balance_ratio)
        pos_indices = np.where(y == 1)[0]
        if len(pos_indices) > target_pos_count:
            pos_selected = np.random.choice(pos_indices, size=target_pos_count, replace=False)
        else:
            pos_selected = pos_indices
        
        neg_indices = np.where(y == 0)[0]
        selected_indices = np.concatenate([pos_selected, neg_indices])
        X = X[selected_indices]
        y = y[selected_indices]
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    
    return model


def predict_overtake_probability(
    model: LogisticRegression,
    delta_pace: float,
    tire_age_diff: float,
    gap_seconds: float,
    sector_s3_coeff: float,
    restart_flag: int = 0,
) -> float:
    """
    Predict probability of overtake.
    
    Args:
        model: Fitted logistic model (or DummyClassifier fallback)
        delta_pace: Pace difference (negative = faster)
        tire_age_diff: Tire age difference
        gap_seconds: Current gap
        sector_s3_coeff: Sector 3 time
        restart_flag: SC restart flag
    
    Returns:
        Probability of overtake (0-1)
    """
    # Cap extrapolation: gap must be < 2.5s
    if gap_seconds > 2.5:
        return 0.0
    
    X = np.array([[delta_pace, tire_age_diff, gap_seconds, sector_s3_coeff, restart_flag]])
    
    # Handle both LogisticRegression and DummyClassifier
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0, 1]  # Probability of class 1 (overtake)
    else:
        # Fallback: return baseline probability based on training data
        prob = 0.1 if model.classes_[0] == 1 else 0.9
    
    return float(prob)


def save_overtake_model(model: LogisticRegression, path: Path | str):
    """Save overtake model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_overtake_model(path: Path | str) -> LogisticRegression:
    """Load overtake model from disk."""
    path = Path(path)
    
    with open(path, "rb") as f:
        return pickle.load(f)


