"""
Reusable visualization utilities with Economist styling.
Factory pattern for consistent, publication-quality plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Optional, Tuple
import pandas as pd

# Economist color palette
ECONOMIST_COLORS = {
    'primary_red': '#E3120B',
    'teal': '#00847E', 
    'blue': '#0F5499',
    'dark_gray': '#363636',
    'light_gray': '#D0D0CE',
    'background': '#FFFFFF'
}

class EconomistStyler:
    """Template for Economist-style plots."""
    
    @staticmethod
    def configure():
        """Set matplotlib params to Economist style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Palatino', 'Georgia', 'Times New Roman'],
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'grid.alpha': 0.3,
            'grid.color': ECONOMIST_COLORS['light_gray'],
            'axes.edgecolor': ECONOMIST_COLORS['dark_gray'],
            'axes.linewidth': 0.8,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })

class RadarChartFactory:
    """Factory for creating radar/spider charts."""
    
    @staticmethod
    def create_class_signature(
        data: pd.DataFrame,
        features: List[str],
        target_col: str,
        class_0_label: str = "Class 0",
        class_1_label: str = "Class 1",
        title: str = "Class Signatures",
        figsize: Tuple[int, int] = (12, 12)
    ):
        """
        Create radar plot showing mean feature values by class.
        
        Args:
            data: DataFrame with features and target
            features: List of feature names
            target_col: Target column name
            class_0_label: Label for class 0
            class_1_label: Label for class 1
            title: Plot title
            figsize: Figure size
        """
        EconomistStyler.configure()
        
        # Calculate means by class
        class_0_means = data[data[target_col] == 0][features].mean().values
        class_1_means = data[data[target_col] == 1][features].mean().values
        
        # Normalize to 0-1 range for better visualization
        all_vals = np.concatenate([class_0_means, class_1_means])
        val_min, val_max = all_vals.min(), all_vals.max()
        
        class_0_norm = (class_0_means - val_min) / (val_max - val_min)
        class_1_norm = (class_1_means - val_min) / (val_max - val_min)
        
        # Number of features
        num_features = len(features)
        
        # Angles for radar
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
        
        # Close the plot
        class_0_norm = np.concatenate([class_0_norm, [class_0_norm[0]]])
        class_1_norm = np.concatenate([class_1_norm, [class_1_norm[0]]])
        angles += angles[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, class_0_norm, 'o-', linewidth=2, 
                color=ECONOMIST_COLORS['blue'], label=class_0_label, alpha=0.8)
        ax.fill(angles, class_0_norm, alpha=0.25, color=ECONOMIST_COLORS['blue'])
        
        ax.plot(angles, class_1_norm, 'o-', linewidth=2,
                color=ECONOMIST_COLORS['primary_red'], label=class_1_label, alpha=0.8)
        ax.fill(angles, class_1_norm, alpha=0.25, color=ECONOMIST_COLORS['primary_red'])
        
        # Feature labels
        feature_labels = [f.replace('feature_', '') for f in features]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels, size=8)
        
        # Grid
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Title and legend
        ax.set_title(title, size=16, weight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fancybox=False)
        
        plt.tight_layout()
        return fig, ax

class KDEPlotFactory:
    """Factory for KDE distribution plots."""
    
    @staticmethod
    def create_class_comparison(
        data: pd.DataFrame,
        features: List[str],
        target_col: str,
        class_0_label: str = "Class 0",
        class_1_label: str = "Class 1",
        n_cols: int = 3,
        figsize: Optional[Tuple[int, int]] = None
    ):
        """
        Create KDE plots comparing feature distributions by class.
        
        Args:
            data: DataFrame with features and target
            features: List of features to plot
            target_col: Target column name
            class_0_label: Label for class 0
            class_1_label: Label for class 1
            n_cols: Number of columns in subplot grid
            figsize: Figure size (auto-calculated if None)
        """
        EconomistStyler.configure()
        
        n_features = len(features)
        n_rows = int(np.ceil(n_features / n_cols))
        
        if figsize is None:
            figsize = (n_cols * 5, n_rows * 3)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            # Get data by class
            class_0_data = data[data[target_col] == 0][feature].dropna()
            class_1_data = data[data[target_col] == 1][feature].dropna()
            
            # Plot KDE
            if len(class_0_data) > 1:
                class_0_data.plot.kde(ax=ax, color=ECONOMIST_COLORS['blue'], 
                                     linewidth=2, label=class_0_label, alpha=0.7)
                ax.fill_between(class_0_data.plot.kde().get_lines()[0].get_xdata(),
                               class_0_data.plot.kde().get_lines()[0].get_ydata(),
                               alpha=0.2, color=ECONOMIST_COLORS['blue'])
            
            if len(class_1_data) > 1:
                class_1_data.plot.kde(ax=ax, color=ECONOMIST_COLORS['primary_red'],
                                     linewidth=2, label=class_1_label, alpha=0.7)
            
            ax.set_title(feature.replace('feature_', 'Feature '), fontsize=11, weight='bold')
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig, axes

class OutlierPlotFactory:
    """Factory for outlier visualization."""
    
    @staticmethod
    def create_boxplot_comparison(
        data: pd.DataFrame,
        features: List[str],
        target_col: str,
        n_cols: int = 3,
        figsize: Optional[Tuple[int, int]] = None
    ):
        """Create box plots showing outliers by class."""
        EconomistStyler.configure()
        
        n_features = len(features)
        n_rows = int(np.ceil(n_features / n_cols))
        
        if figsize is None:
            figsize = (n_cols * 4, n_rows * 3)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            data_to_plot = [
                data[data[target_col] == 0][feature].dropna(),
                data[data[target_col] == 1][feature].dropna()
            ]
            
            bp = ax.boxplot(data_to_plot, labels=['Rock', 'Mine'],
                           patch_artist=True, widths=0.6)
            
            # Color boxes
            bp['boxes'][0].set_facecolor(ECONOMIST_COLORS['blue'])
            bp['boxes'][1].set_facecolor(ECONOMIST_COLORS['primary_red'])
            
            for box in bp['boxes']:
                box.set_alpha(0.6)
            
            ax.set_title(feature.replace('feature_', 'Feature '), fontsize=11, weight='bold')
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig, axes

def detect_outliers_iqr(data: pd.Series) -> pd.Series:
    """Detect outliers using IQR method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

def detect_outliers_zscore(data: pd.Series, threshold: float = 3) -> pd.Series:
    """Detect outliers using Z-score method."""
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold
