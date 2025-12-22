"""Class separability analyzer using KDE."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .base import BaseAnalyzer
from ..results import AnalysisResult


class SeparabilityResult(AnalysisResult):
    """KDE separability analysis result."""
    
    def __init__(self, findings, implications, decision, sep_df, df, target_col, feature_cols, config):
        super().__init__(findings, implications, decision)
        self.sep_df = sep_df
        self.df = df
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.config = config
    
    def plot_all(self):
        """Generate comprehensive KDE grid for all features."""
        n_features = len(self.feature_cols)
        n_cols = 6
        n_rows = int(np.ceil(n_features / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 28))
        axes = axes.flatten()
        
        colors = {
            self.config['class_labels'][0]: self.config['colors']['class_0'],
            self.config['class_labels'][1]: self.config['colors']['class_1']
        }
        
        for idx, feature in enumerate(self.feature_cols):
            ax = axes[idx]
            
            # Get data by class
            class_0_data = self.df[self.df[self.target_col] == 0][feature].dropna()
            class_1_data = self.df[self.df[self.target_col] == 1][feature].dropna()
            
            # Plot KDE
            if len(class_0_data) > 1:
                class_0_data.plot.kde(ax=ax, color=colors[self.config['class_labels'][0]], 
                                     linewidth=2, label=self.config['class_labels'][0], alpha=0.75)
            if len(class_1_data) > 1:
                class_1_data.plot.kde(ax=ax, color=colors[self.config['class_labels'][1]],
                                     linewidth=2, label=self.config['class_labels'][1], alpha=0.75)
            
            # Styling
            feature_num = int(feature.replace('feature_', ''))
            ax.set_title(f'F{feature_num}', fontsize=9, weight='bold',
                        family='sans-serif', pad=4, loc='left')
            ax.set_xlabel('Signal strength', fontsize=7, family='sans-serif')
            ax.set_ylabel('Density', fontsize=7, family='sans-serif')
            
            if idx == 0:
                ax.legend(frameon=False, fontsize=8, loc='upper right')
            else:
                ax.legend().set_visible(False)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_position(('outward', 3))
            ax.spines['bottom'].set_position(('outward', 3))
            ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.3)
            ax.tick_params(labelsize=7)
            
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_family('sans-serif')
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        # Overall title
        plt.tight_layout(rect=[0, 0, 1, 0.978])
        fig.text(0.05, 0.990, 'Distribution patterns vary across the sonar frequency spectrum',
                fontsize=13, weight='bold', family='sans-serif', ha='left')
        fig.text(0.05, 0.982, f'Kernel density estimates for all {n_features} frequency bands showing {self.config["class_labels"][1]} vs {self.config["class_labels"][0]} signatures',
                fontsize=10, style='italic', family='sans-serif', ha='left', color='#666666')
        
        plt.show()
    
    def separation_summary(self):
        """Print separation statistics."""
        print(f"\n{'='*70}")
        print("DISTRIBUTIONAL SEPARATION ANALYSIS")
        print(f"{'='*70}")
        print(f"  Features with strong separation (>0.5): {self.findings['strong_sep_count']}")
        print(f"  Features with moderate separation (0.3-0.5): {self.findings['moderate_sep_count']}")
        print(f"  Features with weak separation (<0.3): {self.findings['weak_sep_count']}")
        print(f"\nTop 10 most separated features:")
        print(self.sep_df.head(10).to_string(index=False))
        print(f"{'='*70}\n")


class DistributionAnalyzer(BaseAnalyzer):
    """Analyzes class separability via KDE patterns."""
    
    def __init__(self, df, target_col, feature_cols, config):
        super().__init__(df, target_col)
        self.feature_cols = feature_cols
        self.config = config
    
    def prepare_data(self):
        """Calculate separation metric for each feature."""
        separation_scores = []
        
        for feature in self.feature_cols:
            class_0_data = self.df[self.df[self.target_col] == 0][feature]
            class_1_data = self.df[self.df[self.target_col] == 1][feature]
            
            # Separation metric: difference in means / pooled std
            mean_diff = abs(class_1_data.mean() - class_0_data.mean())
            pooled_std = np.sqrt((class_0_data.std()**2 + class_1_data.std()**2) / 2)
            separation = mean_diff / pooled_std if pooled_std > 0 else 0
            
            separation_scores.append({
                'feature': feature,
                'separation': separation
            })
        
        return pd.DataFrame(separation_scores).sort_values('separation', ascending=False)
    
    def compute_metrics(self, sep_df):
        """Compute separation statistics."""
        strong_count = int((sep_df['separation'] > 0.5).sum())
        moderate_count = int(((sep_df['separation'] > 0.3) & (sep_df['separation'] <= 0.5)).sum())
        weak_count = int((sep_df['separation'] <= 0.3).sum())
        
        avg_separation = sep_df['separation'].mean()
        
        return {
            'strong_sep_count': strong_count,
            'moderate_sep_count': moderate_count,
            'weak_sep_count': weak_count,
            'avg_separation': avg_separation,
            'top_features': sep_df.head(10)['feature'].tolist()
        }
    
    def interpret_findings(self, findings):
        """Interpret separation patterns."""
        implications = [
            f"{findings['strong_sep_count']} features show clear distributional separation",
            f"{findings['moderate_sep_count']} features exhibit moderate overlap",
            f"{findings['weak_sep_count']} features heavily overlapping - need feature combinations",
            f"Average separation: {findings['avg_separation']:.3f} (moderate overall)",
            "No single feature perfectly separates classes",
            "Partial separation suggests non-linear models required"
        ]
        return implications
    
    def formulate_decision(self, findings, implications):
        if findings['strong_sep_count'] > 10:
            return ("Strong separability detected. Prioritize SVM (RBF kernel) and Random Forest "
                   "to capture non-linear boundaries. Linear models likely insufficient given "
                   "overlapping distributions across multiple features.")
        else:
            return ("Moderate separability. Classification feasible but challenging. Ensemble "
                   "methods and non-linear SVMs essential. Consider feature engineering if "
                   "initial model performance insufficient.")
    
    def create_result(self, findings, implications, decision):
        sep_df = self.prepare_data()
        return SeparabilityResult(
            findings, implications, decision, 
            sep_df, self.df, self.target_col, self.feature_cols, self.config
        )


