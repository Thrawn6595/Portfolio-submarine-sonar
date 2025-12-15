"""
Univariate analysis - works on any dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def check_normality(df, features, alpha=0.05):
    """Test normality using Shapiro-Wilk test"""
    results = []
    for col in features:
        stat, p_value = stats.shapiro(df[col])
        results.append({
            'feature': col,
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > alpha
        })
    return pd.DataFrame(results).sort_values('p_value')

def plot_distributions(df, features, n_cols=4, figsize=(16, 12)):
    """Plot histograms for features"""
    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(features):
        axes[idx].hist(df[col], bins=20, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col}', fontsize=10)
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
    
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

def get_summary_stats(df, features):
    """Get descriptive statistics for features"""
    return df[features].describe().T
