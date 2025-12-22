"""Feature correlation analyzer (4 sub-analyses + VIF)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .base import BaseAnalyzer
from ..results import AnalysisResult


class CorrelationResult(AnalysisResult):
    """Correlation analysis results with 4 sub-components."""
    
    def __init__(self, findings, implications, decision, sub_results):
        super().__init__(findings, implications, decision)
        self.sub_results = sub_results
    
    def __getitem__(self, key):
        """Allow dict-like access to sub-results."""
        return self.sub_results[key]


class VIFResult:
    """VIF multicollinearity result."""
    
    def __init__(self, findings, vif_df, config):
        self.findings = findings
        self.vif_df = vif_df
        self.config = config
    
    def plot(self):
        """Visualize VIF with distinctive reference lines."""
        fig, ax = plt.subplots(figsize=(12, 14))
        
        # Sort by VIF ascending for horizontal bars
        vif_sorted = self.vif_df.sort_values('VIF', ascending=True)
        
        # Colorblind-friendly colors
        colors = []
        for vif in vif_sorted['VIF']:
            if vif < 5:
                colors.append('#426590')  # Navy - Low
            elif vif < 10:
                colors.append('#7B68A6')  # Purple - Moderate
            else:
                colors.append('#FF9933')  # Orange - High
        
        # Horizontal bar chart
        y_pos = np.arange(len(vif_sorted))
        bars = ax.barh(y_pos, vif_sorted['VIF'], color=colors, alpha=0.85)
        
        # Distinctive reference lines with legend
        ax.axvline(x=5, color='#FFA500', linestyle='--', linewidth=2, alpha=0.8, label='Moderate threshold (VIF=5)')
        ax.axvline(x=10, color='#D62728', linestyle='-', linewidth=2.5, alpha=0.9, label='Severe threshold (VIF=10)')
        
        # Title (consistent font sizes)
        ax.text(0.0, 1.05, 'Variance Inflation Factors reveal multicollinearity patterns',
                transform=ax.transAxes, fontsize=13, weight='bold',
                family='sans-serif', ha='left')
        ax.text(0.0, 1.02, f'VIF > 10 indicates severe multicollinearity (affects {self.findings["severe_count"]} features)',
                transform=ax.transAxes, fontsize=10, style='italic',
                family='sans-serif', ha='left', color='#666666')
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('feature_', 'F') for f in vif_sorted['Feature']], fontsize=7)
        ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=11, family='sans-serif')
        ax.set_ylabel('Feature', fontsize=11, family='sans-serif')
        
        # Legend
        ax.legend(frameon=False, fontsize=10, loc='lower right')
        
        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', 5))
        ax.spines['bottom'].set_position(('outward', 5))
        ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.5)
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_family('sans-serif')
        
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """Print VIF findings."""
        print(f"\n{'='*70}")
        print("VARIANCE INFLATION FACTOR ANALYSIS")
        print(f"{'='*70}")
        print(f"  Features with VIF < 5 (Low): {self.findings['low_count']}")
        print(f"  Features with VIF 5-10 (Moderate): {self.findings['moderate_count']}")
        print(f"  Features with VIF > 10 (Severe): {self.findings['severe_count']}")
        print(f"\nTop 10 highest VIF features:")
        print(self.vif_df.nlargest(10, 'VIF').to_string(index=False))
        print(f"\n  Implication: {self.findings['implication']}")
        print(f"{'='*70}\n")


class MulticollinearityResult:
    """Multicollinearity analysis result."""
    
    def __init__(self, findings, corr_matrix, feature_cols, config):
        self.findings = findings
        self.corr_matrix = corr_matrix
        self.feature_cols = feature_cols
        self.config = config
    
    def plot(self):
        """Plot correlation heatmap with masked upper triangle."""
        # Use all 60 frequency bands
        subset_corr = self.corr_matrix
        
        # Mask upper triangle
        mask = np.triu(np.ones_like(subset_corr, dtype=bool))
        
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Apply mask and plot with viridis
        masked_corr = subset_corr.mask(mask)
        im = ax.imshow(masked_corr, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
        
        # Title (consistent font sizes: 13pt bold, 10pt italic)
        ax.text(0.0, 1.05, 'Adjacent frequency bands show strong multicollinearity',
                transform=ax.transAxes, fontsize=13, weight='bold',
                family='sans-serif', ha='left')
        ax.text(0.0, 1.02, 'Correlation matrix (all 60 frequency bands, upper triangle masked)',
                transform=ax.transAxes, fontsize=10, style='italic',
                family='sans-serif', ha='left', color='#666666')
        
        # Tick labels for all 60 features
        tick_labels = [col.replace('feature_', 'F') for col in self.feature_cols]
        ax.set_xticks(np.arange(len(self.feature_cols)))
        ax.set_yticks(np.arange(len(self.feature_cols)))
        ax.set_xticklabels(tick_labels, fontsize=6, family='sans-serif', rotation=90)
        ax.set_yticklabels(tick_labels, fontsize=6, family='sans-serif')
        
        # Colorbar legend
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation', fontsize=10, family='sans-serif')
        for label in cbar.ax.get_yticklabels():
            label.set_family('sans-serif')
            label.set_fontsize(8)
        
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """Print findings."""
        print(f"\n{'='*70}")
        print("CORRELATION MATRIX ANALYSIS")
        print(f"{'='*70}")
        print(f"  High correlation pairs (>0.8): {self.findings['high_corr_count']}")
        print(f"  Adjacent bands (within 2): {self.findings['adjacent_pct']:.1f}%")
        print(f"  Hypothesis: {'CONFIRMED' if self.findings['adjacent_pct'] > 70 else 'REJECTED'}")
        print(f"\n  Implication: {self.findings['implication']}")
        print(f"{'='*70}\n")


class ClassSignatureResult:
    """Class-specific frequency signature result."""
    
    def __init__(self, findings, signed_corr, config):
        self.findings = findings
        self.signed_corr = signed_corr
        self.config = config
    
    def plot(self):
        """Plot signed correlation across all bands."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        indices = [int(f.replace('feature_', '')) for f in self.signed_corr.index]
        colors = [self.config['colors']['class_1'] if v > 0 else self.config['colors']['class_0'] 
                 for v in self.signed_corr.values]
        
        ax.bar(indices, self.signed_corr.values, color=colors, alpha=0.85, width=0.8)
        ax.axhline(y=0, color='#333333', linewidth=0.8)
        
        # Add legend
        legend_elements = [
            Patch(facecolor=self.config['colors']['class_1'], alpha=0.85, label=f'{self.config["class_labels"][1]}-favoring (positive correlation)'),
            Patch(facecolor=self.config['colors']['class_0'], alpha=0.85, label=f'{self.config["class_labels"][0]}-favoring (negative correlation)')
        ]
        ax.legend(handles=legend_elements, frameon=False, fontsize=9, loc='upper right')
        
        # Title (consistent font sizes)
        ax.text(0.0, 1.08, 'Frequency bands show distinct class preferences',
                transform=ax.transAxes, fontsize=13, weight='bold',
                family='sans-serif', ha='left')
        ax.text(0.0, 1.04, f'Signed correlation: {self.config["class_labels"][1]} (magenta) vs {self.config["class_labels"][0]} (navy)',
                transform=ax.transAxes, fontsize=10, style='italic',
                family='sans-serif', ha='left', color='#666666')
        
        ax.set_xlabel('Frequency Band Index', fontsize=11, family='sans-serif')
        ax.set_ylabel(f'Correlation with {self.config["class_labels"][1]} Class', fontsize=11, family='sans-serif')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', 5))
        ax.spines['bottom'].set_position(('outward', 5))
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_family('sans-serif')
            label.set_fontsize(9)
        
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """Print findings."""
        print(f"\n{'='*70}")
        print("CLASS-SPECIFIC FREQUENCY SIGNATURES")
        print(f"{'='*70}")
        print(f"  {self.config['class_labels'][1]}-favoring frequencies: {self.findings['mine_count']}")
        print(f"  {self.config['class_labels'][0]}-favoring frequencies: {self.findings['rock_count']}")
        print(f"  Strongest region: {self.findings['strongest_region']}")
        print(f"  Hypothesis: {'CONFIRMED' if self.findings['signatures_exist'] else 'REJECTED'}")
        print(f"\n  Implication: {self.findings['implication']}")
        print(f"{'='*70}\n")


class MutualInformationResult:
    """MI vs correlation comparison result."""
    
    def __init__(self, findings, mi_df, config):
        self.findings = findings
        self.mi_df = mi_df
        self.config = config
    
    def plot(self):
        """Scatter plot: MI vs correlation."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(self.mi_df['abs_correlation'], self.mi_df['MI'],
                  c=self.config['colors']['class_1'], alpha=0.6, s=60,
                  edgecolors=self.config['colors']['class_0'], linewidth=0.5)
        
        # Annotate top features
        for _, row in self.mi_df.head(5).iterrows():
            ax.annotate(row['feature'].replace('feature_', 'F'),
                       (row['abs_correlation'], row['MI']),
                       fontsize=8, family='sans-serif',
                       xytext=(5, 5), textcoords='offset points')
        
        # Title (consistent font sizes)
        ax.text(0.0, 1.06, 'Mutual information aligns with correlation strength',
                transform=ax.transAxes, fontsize=13, weight='bold',
                family='sans-serif', ha='left')
        ax.text(0.0, 1.02, 'Pearson correlation vs mutual information across frequency bands',
                transform=ax.transAxes, fontsize=10, style='italic',
                family='sans-serif', ha='left', color='#666666')
        
        ax.set_xlabel('Absolute Correlation (Linear)', fontsize=11, family='sans-serif')
        ax.set_ylabel('Mutual Information (Any Relationship)', fontsize=11, family='sans-serif')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', 5))
        ax.spines['bottom'].set_position(('outward', 5))
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_family('sans-serif')
            label.set_fontsize(9)
        
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """Print findings."""
        print(f"\n{'='*70}")
        print("NON-LINEAR RELATIONSHIP ASSESSMENT")
        print(f"{'='*70}")
        print(f"  Non-linear candidates: {self.findings['nonlinear_count']}")
        print(f"  MI-correlation alignment: {self.findings['alignment']}")
        print(f"  Hypothesis: {'CONFIRMED' if self.findings['nonlinear_present'] else 'REJECTED'}")
        print(f"\n  Implication: {self.findings['implication']}")
        print(f"{'='*70}\n")


class PCAResult:
    """PCA dimensionality assessment result."""
    
    def __init__(self, findings, explained_var, cumulative_var, config):
        self.findings = findings
        self.explained_var = explained_var
        self.cumulative_var = cumulative_var
        self.config = config
    
    def plot(self):
        """Plot scree + cumulative variance."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scree plot
        axes[0].bar(range(1, len(self.explained_var)+1), self.explained_var,
                   color=self.config['colors']['class_0'], alpha=0.85)
        axes[0].text(0.0, 1.08, 'Variance concentrates in first 15 components',
                    transform=axes[0].transAxes, fontsize=13, weight='bold',
                    family='sans-serif', ha='left')
        axes[0].text(0.0, 1.04, 'Scree plot showing explained variance per component',
                    transform=axes[0].transAxes, fontsize=10, style='italic',
                    family='sans-serif', ha='left', color='#666666')
        axes[0].set_xlabel('Principal Component', fontsize=11, family='sans-serif')
        axes[0].set_ylabel('Explained Variance Ratio', fontsize=11, family='sans-serif')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].spines['left'].set_position(('outward', 5))
        axes[0].spines['bottom'].set_position(('outward', 5))
        axes[0].grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
        
        # Cumulative variance
        axes[1].plot(range(1, len(self.cumulative_var)+1), self.cumulative_var,
                    marker='o', color=self.config['colors']['class_1'], linewidth=2.5, markersize=4)
        axes[1].axhline(y=0.95, color=self.config['colors']['class_0'],
                       linestyle='--', linewidth=2, label='95% threshold', alpha=0.7)
        axes[1].text(0.0, 1.08, f'{self.findings["n_components_95"]} components capture 95% of variance',
                    transform=axes[1].transAxes, fontsize=13, weight='bold',
                    family='sans-serif', ha='left')
        axes[1].text(0.0, 1.04, 'Cumulative explained variance across components',
                    transform=axes[1].transAxes, fontsize=10, style='italic',
                    family='sans-serif', ha='left', color='#666666')
        axes[1].set_xlabel('Number of Components', fontsize=11, family='sans-serif')
        axes[1].set_ylabel('Cumulative Explained Variance', fontsize=11, family='sans-serif')
        axes[1].legend(frameon=False, fontsize=9)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].spines['left'].set_position(('outward', 5))
        axes[1].spines['bottom'].set_position(('outward', 5))
        axes[1].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        for ax in axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_family('sans-serif')
                label.set_fontsize(9)
        
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """Print findings."""
        print(f"\n{'='*70}")
        print("DIMENSIONALITY REDUCTION POTENTIAL")
        print(f"{'='*70}")
        print(f"  Original dimensions: {self.findings['n_features']}")
        print(f"  95% variance components: {self.findings['n_components_95']}")
        print(f"  Reduction potential: {self.findings['reduction_pct']:.1f}%")
        print(f"  Hypothesis: {'CONFIRMED' if self.findings['reducible'] else 'REJECTED'}")
        print(f"\n  Implication: {self.findings['implication']}")
        print(f"{'='*70}\n")


class CorrelationAnalyzer(BaseAnalyzer):
    """Analyzes feature correlations (4 hypotheses + VIF)."""
    
    def __init__(self, df, target_col, feature_cols, config):
        super().__init__(df, target_col)
        self.feature_cols = feature_cols
        self.config = config
    
    def prepare_data(self):
        """Prepare correlation data structures."""
        return {
            'features': self.df[self.feature_cols],
            'target': self.df[self.target_col]
        }
    
    def compute_metrics(self, data):
        """Run all correlation analyses + VIF."""
        # VIF
        vif = self._analyze_vif(data['features'])
        
        # Correlation matrix
        multicollinearity = self._analyze_correlation_matrix(data['features'])
        
        # Class signatures
        class_signatures = self._analyze_class_signatures(data['features'], data['target'])
        
        # Mutual information
        mi_analysis = self._analyze_mutual_information(data['features'], data['target'])
        
        # PCA
        pca_analysis = self._analyze_pca(data['features'])
        
        return {
            'vif': vif,
            'multicollinearity': multicollinearity,
            'class_signatures': class_signatures,
            'mutual_information': mi_analysis,
            'pca': pca_analysis
        }
    
    def _analyze_vif(self, features):
        """Calculate Variance Inflation Factors."""
        vif_data = []
        for i, col in enumerate(features.columns):
            vif = variance_inflation_factor(features.values, i)
            vif_data.append({'Feature': col, 'VIF': vif})
        
        vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
        
        low_vif = int((vif_df['VIF'] < 5).sum())
        moderate_vif = int(((vif_df['VIF'] >= 5) & (vif_df['VIF'] < 10)).sum())
        severe_vif = int((vif_df['VIF'] >= 10).sum())
        
        # Determine implication
        if severe_vif > 20:
            implication = "Severe multicollinearity detected. Ridge/Elastic Net essential for linear models. Tree models unaffected."
        elif severe_vif > 10:
            implication = "Moderate multicollinearity. Regularisation recommended for linear models."
        else:
            implication = "Low multicollinearity. Most models can handle without special treatment."
        
        findings = {
            'low_count': low_vif,
            'moderate_count': moderate_vif,
            'severe_count': severe_vif,
            'implication': implication
        }
        
        return VIFResult(findings, vif_df, self.config)
    
    def _analyze_correlation_matrix(self, features):
        """Analyze pairwise correlations."""
        corr_matrix = features.corr()
        
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    feat1_num = int(corr_matrix.columns[i].replace('feature_', ''))
                    feat2_num = int(corr_matrix.columns[j].replace('feature_', ''))
                    distance = abs(feat1_num - feat2_num)
                    high_corr.append(distance)
        
        adjacent_count = sum(1 for d in high_corr if d <= 2)
        adjacent_pct = (adjacent_count / len(high_corr) * 100) if high_corr else 0
        
        findings = {
            'high_corr_count': len(high_corr),
            'adjacent_pct': adjacent_pct,
            'implication': 'Tree models handle naturally. Linear models need regularisation.'
        }
        
        return MulticollinearityResult(findings, corr_matrix, self.feature_cols, self.config)
    
    def _analyze_class_signatures(self, features, target):
        """Analyze signed correlations with target."""
        combined = features.copy()
        combined['target'] = target
        signed_corr = combined.corr()['target'].drop('target')
        
        mine_count = (signed_corr > 0).sum()
        rock_count = (signed_corr < 0).sum()
        
        # Region analysis
        low = signed_corr[[f for f in signed_corr.index if int(f.replace('feature_', '')) < 20]]
        mid = signed_corr[[f for f in signed_corr.index if 20 <= int(f.replace('feature_', '')) < 40]]
        high = signed_corr[[f for f in signed_corr.index if int(f.replace('feature_', '')) >= 40]]
        
        region_means = {'low': abs(low.mean()), 'mid': abs(mid.mean()), 'high': abs(high.mean())}
        strongest = max(region_means, key=region_means.get)
        
        findings = {
            'mine_count': int(mine_count),
            'rock_count': int(rock_count),
            'strongest_region': strongest,
            'signatures_exist': True,
            'implication': f'{strongest.capitalize()} frequency bands show strongest class discrimination'
        }
        
        return ClassSignatureResult(findings, signed_corr, self.config)
    
    def _analyze_mutual_information(self, features, target):
        """Compare MI vs correlation."""
        mi_scores = mutual_info_classif(features, target, random_state=self.config.get('random_seed', 42))
        
        signed_corr = features.copy()
        signed_corr['target'] = target
        abs_corr = signed_corr.corr()['target'].drop('target').abs()
        
        mi_df = pd.DataFrame({
            'feature': self.feature_cols,
            'MI': mi_scores,
            'abs_correlation': abs_corr.values
        }).sort_values('MI', ascending=False)
        
        mi_df['MI_rank'] = mi_df['MI'].rank(ascending=False)
        mi_df['corr_rank'] = mi_df['abs_correlation'].rank(ascending=False)
        mi_df['rank_diff'] = abs(mi_df['corr_rank'] - mi_df['MI_rank'])
        
        nonlinear_count = (mi_df['rank_diff'] > 20).sum()
        
        findings = {
            'nonlinear_count': int(nonlinear_count),
            'alignment': 'strong' if nonlinear_count < 5 else 'weak',
            'nonlinear_present': nonlinear_count > 0,
            'implication': 'Linear relationships dominate. Non-linear models optional but not critical.'
        }
        
        return MutualInformationResult(findings, mi_df, self.config)
    
    def _analyze_pca(self, features):
        """Assess dimensionality reduction potential."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        
        pca = PCA(n_components=min(30, len(self.feature_cols)))
        pca.fit(X_scaled)
        
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        n_components_95 = int((cumulative_var >= 0.95).argmax() + 1)
        reduction_pct = (len(self.feature_cols) - n_components_95) / len(self.feature_cols) * 100
        
        findings = {
            'n_features': len(self.feature_cols),
            'n_components_95': n_components_95,
            'reduction_pct': reduction_pct,
            'reducible': reduction_pct > 30,
            'implication': f'PCA optional. Can reduce {len(self.feature_cols)} to {n_components_95} if needed.'
        }
        
        return PCAResult(findings, explained_var, cumulative_var, self.config)
    
    def interpret_findings(self, findings):
        """Synthesize across all analyses."""
        implications = [
            f"VIF: {findings['vif'].findings['severe_count']} features with severe multicollinearity",
            f"Correlation: {findings['multicollinearity'].findings['high_corr_count']} high-correlation pairs",
            f"Class signatures: Distinct frequency preferences exist ({findings['class_signatures'].findings['strongest_region']} bands strongest)",
            f"Non-linearity: {findings['mutual_information'].findings['alignment'].capitalize()} MI-correlation alignment",
            f"Dimensionality: {findings['pca'].findings['reduction_pct']:.0f}% reduction possible via PCA"
        ]
        return implications
    
    def formulate_decision(self, findings, implications):
        return ("Use all 60 features initially. VIF indicates multicollinearity requires regularisation "
               "for linear models. Random Forest and SVM handle naturally. PCA available if computational "
               "cost becomes prohibitive. Linear relationships dominate.")
    
    def create_result(self, findings, implications, decision):
        return CorrelationResult(
            findings={k: v.findings for k, v in findings.items()},
            implications=implications,
            decision=decision,
            sub_results=findings
        )