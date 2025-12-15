"""
Data loading utilities - reusable across projects
"""

import pandas as pd
from pathlib import Path

def load_dataset(path, outcome_col_index=-1, outcome_mapping=None, feature_prefix="feature"):
    """
    Generic dataset loader - works on any CSV
    
    Args:
        path: Path to CSV file
        outcome_col_index: Index of target column (-1 for last)
        outcome_mapping: Dict to map target labels to integers
        feature_prefix: Prefix for feature column names
        
    Returns:
        df: DataFrame with standardized column names
        schema: Dict with dataset metadata
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    df = pd.read_csv(path, header=None)
    
    n_cols = df.shape[1]
    if outcome_col_index < 0:
        outcome_col_index = n_cols + outcome_col_index
    
    new_cols = [f"{feature_prefix}_{i+1}" for i in range(n_cols)]
    new_cols[outcome_col_index] = "outcome"
    df.columns = new_cols
    
    if outcome_mapping is not None:
        unmapped = set(df['outcome'].unique()) - set(outcome_mapping.keys())
        if unmapped:
            raise ValueError(f"Unmapped outcome values: {unmapped}")
        df['outcome'] = df['outcome'].map(outcome_mapping).astype(int)
    
    feature_cols = [c for c in df.columns if c != 'outcome']
    schema = {
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'target_name': 'outcome',
        'target_values': sorted(df['outcome'].unique()),
    }
    
    return df, schema
