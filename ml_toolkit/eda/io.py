import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
import os

def load_dataset(
    path: str,
    outcome_mapping: Optional[Dict] = None,
    outcome_col_index: int = -1,
    feature_prefix: str = "feature_"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load dataset and return DataFrame with schema info.
    """
    # Try path as-is first
    if os.path.exists(path):
        df = pd.read_csv(path, header=None)
    else:
        # If not found, try relative to current working directory
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            df = pd.read_csv(cwd_path, header=None)
        else:
            # Try going up one level (for notebooks directory)
            parent_path = Path.cwd().parent / path
            if parent_path.exists():
                df = pd.read_csv(parent_path, header=None)
            else:
                raise FileNotFoundError(f"Dataset not found at: {path}, {cwd_path}, or {parent_path}")
    
    n_cols = df.shape[1]
    n_features = n_cols - 1
    
    feature_names = [f"{feature_prefix}{i}" for i in range(n_features)]
    outcome_col_name = "outcome"
    
    df.columns = feature_names + [outcome_col_name]
    
    if outcome_mapping:
        df[outcome_col_name] = df[outcome_col_name].map(outcome_mapping)
    
    schema = {
        "n_samples": len(df),
        "n_features": n_features,
        "feature_names": feature_names,
        "target_name": outcome_col_name,
        "target_values": df[outcome_col_name].unique().tolist()
    }
    
    return df, schema
