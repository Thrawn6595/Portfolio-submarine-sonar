"""Class distribution analyzer."""

import matplotlib.pyplot as plt
import numpy as np
from .base import BaseAnalyzer
from ..results import AnalysisResult


class ClassDistributionResult(AnalysisResult):
    """Class distribution analysis results."""
    
    def __init__(self, findings, implications, decision, plot_data, config):
        super().__init__(findings, implications, decision)
        self.plot_data = plot_data
        self.config = config
    
    def plot(self):
        """Generate class distribution bar chart."""
        counts = self.plot_data['counts']
        proportions = self.plot_data['proportions']
        labels = self.plot_data['labels']
        colors = self.plot_data['colors']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bars = ax.bar(labels, counts.values, color=colors, alpha=0.85, width=0.6)
        
        # Add count and percentage labels
        for i, (bar, count, prop) in enumerate(zip(bars, counts.values, proportions.values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({prop:.1%})',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    family='sans-serif')
        
        # Title
        ax.text(0.0, 1.08, 'Training set exhibits balanced class distribution',
                transform=ax.transAxes, fontsize=13, weight='bold',
                family='sans-serif', ha='left')
        ax.text(0.0, 1.03, f'{len(self.plot_data["df"])} samples from stratified split',
                transform=ax.transAxes, fontsize=10, style='italic',
                family='sans-serif', ha='left', color='#666666')
        
        # Styling
        ax.spines['left'].set_position(('outward', 5))
        ax.spines['bottom'].set_position(('outward', 5))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_ylim(0, max(counts.values) * 1.18)
        ax.set_ylabel('Count', fontsize=11, family='sans-serif')
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_family('sans-serif')
        
        plt.tight_layout()
        plt.show()


class ClassDistributionAnalyzer(BaseAnalyzer):
    """Analyzes class balance and asymmetric risk."""
    
    def __init__(self, df, target_col, config):
        super().__init__(df, target_col)
        self.config = config
    
    def prepare_data(self):
        return self.df[self.target_col].value_counts().sort_index()
    
    def compute_metrics(self, counts):
        proportions = counts / len(self.df)
        balance_ratio = min(proportions) / max(proportions)
        
        return {
            'class_0_count': int(counts.iloc[0]),
            'class_1_count': int(counts.iloc[1]),
            'class_0_pct': proportions.iloc[0],
            'class_1_pct': proportions.iloc[1],
            'balance_ratio': balance_ratio,
            'is_balanced': abs(proportions.iloc[0] - 0.5) < 0.1,
            'baseline_accuracy': max(proportions)
        }
    
    def interpret_findings(self, findings):
        implications = []
        
        if findings['is_balanced']:
            implications.append("Classes are balanced - no resampling required")
            implications.append("Sufficient examples of both classes for learning")
        else:
            implications.append(f"Class imbalance detected ({findings['balance_ratio']:.2f} ratio)")
            implications.append("Consider class weights or resampling techniques")
        
        implications.append(f"Baseline accuracy: {findings['baseline_accuracy']:.1%} (majority class)")
        implications.append("Model must significantly exceed baseline to be useful")
        
        # Asymmetric risk context
        implications.append("Asymmetric costs: False negatives >> False positives")
        implications.append("Optimize for recall via F2 score + threshold tuning")
        
        return implications
    
    def formulate_decision(self, findings, implications):
        if findings['is_balanced']:
            return ("Proceed with natural class distribution. Train using F2 score "
                   "(recall-weighted metric), then optimize decision threshold post-training "
                   "to achieve target recall ≥98% while maintaining acceptable precision.")
        else:
            return (f"Address imbalance via class weights or threshold tuning. "
                   f"F2 optimization still recommended given asymmetric costs.")
    
    def create_result(self, findings, implications, decision):
        # Prepare plot data
        counts = self.prepare_data()
        proportions = counts / len(self.df)
        
        class_labels = [self.config['class_labels'][i] for i in counts.index]
        colors = [self.config['colors'][f'class_{i}'] for i in counts.index]
        
        plot_data = {
            'counts': counts,
            'proportions': proportions,
            'labels': class_labels,
            'colors': colors,
            'df': self.df
        }
        
        return ClassDistributionResult(findings, implications, decision, plot_data, self.config)
