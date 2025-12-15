"""
Bivariate analysis - works on any dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_with_target(df, target='outcome'):
    """Calculate correlation of each feature with target"""
    features = [c for c in df.columns if c != target]
    correlations = df[features].corrwith(df[target]).abs()
    return correlations.sort_values(ascending=False)

def plot_correlation_heatmap(df, figsize=(14, 12)):
    """Plot correlation heatmap"""
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    return fig

def count_high_correlations(corr_matrix, threshold=0.85):
    """Count highly correlated feature pairs"""
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr = []
    for column in upper.columns:
        high = upper[column][upper[column].abs() > threshold]
        for idx in high.index:
            high_corr.append({
                'feature_1': column,
                'feature_2': idx,
                'correlation': upper.loc[idx, column]
            })
    
    return pd.DataFrame(high_corr)
