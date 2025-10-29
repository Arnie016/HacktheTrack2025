"""IO utilities for loading/saving models and data."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


def save_json(data: dict, path: Path | str, indent: int = 2):
    """Save dict to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Path | str) -> dict:
    """Load dict from JSON file."""
    path = Path(path)
    
    with open(path) as f:
        return json.load(f)


def save_pickle(obj: object, path: Path | str):
    """Save object to pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path | str):
    """Load object from pickle file."""
    path = Path(path)
    
    with open(path, "rb") as f:
        return pickle.load(f)


def save_parquet(df: pd.DataFrame, path: Path | str, **kwargs):
    """Save DataFrame to Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(path, **kwargs)


def load_parquet(path: Path | str, **kwargs) -> pd.DataFrame:
    """Load DataFrame from Parquet."""
    path = Path(path)
    
    return pd.read_parquet(path, **kwargs)


