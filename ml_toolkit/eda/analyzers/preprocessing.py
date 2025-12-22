"""Preprocessing requirements analyzer."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from .base import BaseAnalyzer
from ..results import AnalysisResult


class OutlierResult:
    """Outlier analysis result."""
    
    def __init__(self, findings, df, feature_cols, config):
        self.findings = findings
        self.df = df
        self.feature_cols = feature_cols
        self.config = config
    
    def plot(self, top_n=6):
        """Plot box plots for top features."""
        top_features = self.feature_cols[:top_n]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_features):
            ax = axes[idx]
            
            data_to_plot = [
                self.df[self.df['outcome'] == 0][feature].dropna(),
                self.df[self.df['outcome'] == 1][feature].dropna()
            ]
            
            bp = ax.boxplot(data_to_plot, labels=['Rock', 'Mine'], patch_artist=True, widths=0.6)
            
            bp['boxes'][0].set_facecolor(self.config['colors']['class_0'])
            bp['boxes'][1].set_facecolor(self.config['colors']['class_1'])
            
            for box in bp['boxes']:
                box.set_alpha(0.6)
            
            ax.set_title(feature.replace('feature_', 'Feature '), fontsize=11, weight='bold', family='sans-serif')
            ax.set_ylabel('Value', fontsize=9, family='sans-serif')
            ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.suptitle('Outlier detection via box plots (top 6 features)', fontsize=13, weight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """Print outlier findings."""
        print(f"\n{'='*70}")
        print("OUTLIER ASSESSMENT")
        print(f"{'='*70}")
        print(f"  Total outliers (IQR method): {self.findings['total_outliers_iqr']}")
        print(f"  Total outliers (Z-score >3): {self.findings['total_outliers_zscore']}")
        print(f"  Features with outliers: {self.findings['features_with_outliers']}")
        print(f"\n  Decision: {self.findings['decision']}")
        print(f"{'='*70}\n")


class ScalingResult:
    """Scaling requirements result."""
    
    def __init__(self, findings, df, feature_cols):
        self.findings = findings
        self.df = df
        self.feature_cols = feature_cols
    
    def analyze(self):
        """Print scaling analysis."""
        print(f"\n{'='*70}")
        print("SCALING REQUIREMENTS")
        print(f"{'='*70}")
        print(f"  Overall min: {self.findings['overall_min']:.4f}")
        print(f"  Overall max: {self.findings['overall_max']:.4f}")
        print(f"  Average range: {self.findings['avg_range']:.4f}")
        print(f"  Range std dev: {self.findings['range_std']:.4f}")
        print(f"\n  Recommendation: {self.findings['recommendation']}")
        print(f"{'='*70}\n")


class SummaryStatsResult:
    """Summary statistics result."""
    
    def __init__(self, summary_df, top_features):
        self.summary_df = summary_df
        self.top_features = top_features
    
    def display(self):
        """Display summary statistics table."""
        print(f"\n{'='*70}")
        print(f"SUMMARY STATISTICS (Top {len(self.top_features)} Features)")
        print(f"{'='*70}")
        print(self.summary_df.to_string())
        print(f"{'='*70}\n")


class PreprocessingResult(AnalysisResult):
    """Combined preprocessing analysis result."""
    
    def __init__(self, findings, implications, decision, sub_results):
        super().__init__(findings, implications, decision)
        self.sub_results = sub_results
    
    def __getitem__(self, key):
        """Dict-like access to sub-results."""
        return self.sub_results[key]


class PreprocessingAnalyzer(BaseAnalyzer):
    """Analyzes outliers, scaling needs, and summary statistics."""
    
    def __init__(self, df, feature_cols, config):
        super().__init__(df)
        self.feature_cols = feature_cols
        self.config = config
    
    def prepare_data(self):
        """Prepare numeric features."""
        return self.df[self.feature_cols]
    
    def compute_metrics(self, features):
        """Run all preprocessing analyses."""
        outliers = self._analyze_outliers(features)
        scaling = self._analyze_scaling(features)
        summary = self._compute_summary_stats(features)
        
        return {
            'outliers': outliers,
            'scaling': scaling,
            'summary_stats': summary
        }
    
    def _analyze_outliers(self, features):
        """Detect outliers using IQR and Z-score."""
        outlier_counts = {'IQR': 0, 'Z-score': 0}
        features_with_outliers = 0
        
        for col in features.columns[:10]:  # Check top 10 features
            # IQR method
            Q1 = features[col].quantile(0.25)
            Q3 = features[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = ((features[col] < (Q1 - 1.5 * IQR)) | 
                           (features[col] > (Q3 + 1.5 * IQR))).sum()
            
            # Z-score method
            z_scores = np.abs(stats.zscore(features[col]))
            z_outliers = (z_scores > 3).sum()
            
            outlier_counts['IQR'] += iqr_outliers
            outlier_counts['Z-score'] += z_outliers
            
            if iqr_outliers > 0:
                features_with_outliers += 1
        
        findings = {
            'total_outliers_iqr': int(outlier_counts['IQR']),
            'total_outliers_zscore': int(outlier_counts['Z-score']),
            'features_with_outliers': features_with_outliers,
            'decision': 'Keep outliers - genuine sonar readings. Use robust scaling if needed.'
        }
        
        return OutlierResult(findings, self.df, self.feature_cols, self.config)
    
    def _analyze_scaling(self, features):
        """Assess scaling requirements."""
        ranges = features.max() - features.min()
        
        findings = {
            'overall_min': float(features.min().min()),
            'overall_max': float(features.max().max()),
            'avg_range': float(ranges.mean()),
            'range_std': float(ranges.std()),
            'recommendation': ('StandardScaler required for distance-based models (SVM, KNN). '
                             'Not required for tree-based models (Random Forest).')
        }
        
        return ScalingResult(findings, self.df, self.feature_cols)
    
    def _compute_summary_stats(self, features):
        """Compute summary statistics for top features."""
        # Get top 12 by correlation if target available
        if hasattr(self, 'target_col') and self.target_col:
            correlations = self.df[self.feature_cols + [self.target_col]].corr()[self.target_col].abs().sort_values(ascending=False)
            top_features = correlations.iloc[1:13].index.tolist()
        else:
            top_features = self.feature_cols[:12]
        
        summary = features[top_features].describe().T
        summary['range'] = summary['max'] - summary['min']
        summary['cv'] = summary['std'] / summary['mean']
        
        return SummaryStatsResult(summary[['mean', 'std', 'min', 'max', 'range', 'cv']].round(4), top_features)
    
    def interpret_findings(self, findings):
        """Synthesize preprocessing implications."""
        implications = [
            f"Outliers present but appear genuine ({findings['outliers'].findings['total_outliers_iqr']} via IQR)",
            "Scaling required for SVM, KNN; not for tree-based models",
            f"Feature ranges: {findings['scaling'].findings['overall_min']:.2f} to {findings['scaling'].findings['overall_max']:.2f}",
            "No imputation needed (data complete)",
            "Summary statistics show reasonable variance across features"
        ]
        return implications
    
    def formulate_decision(self, findings, implications):
        return ("Apply StandardScaler for SVM and KNN pipelines. Keep outliers as genuine data. "
               "No imputation or feature removal required. Tree-based models can use raw features.")
    
    def create_result(self, findings, implications, decision):
        return PreprocessingResult(
            findings={k: v.findings if hasattr(v, 'findings') else {} for k, v in findings.items()},
            implications=implications,
            decision=decision,
            sub_results=findings
        )
