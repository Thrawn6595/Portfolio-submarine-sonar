import pandas as pd
from sklearn.model_selection import train_test_split
from ..config import DEFAULT_SEED, OUTCOME_MAPPING

def load_sonar_raw(path="data/raw/sonar.csv", outcome_col_index=-1):
    df = pd.read_csv(path, header=None)
    n_cols = df.shape[1]
    if outcome_col_index < 0:
        outcome_col_index = n_cols + outcome_col_index
    new_cols = [f"feature_{i+1}" for i in range(n_cols)]
    new_cols[outcome_col_index] = "outcome"
    df.columns = new_cols
    df['outcome'] = df['outcome'].map(OUTCOME_MAPPING).astype(int)
    return df

def split_train_test(df, target_col="outcome", test_size=0.2, seed=DEFAULT_SEED, stratify=True):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
