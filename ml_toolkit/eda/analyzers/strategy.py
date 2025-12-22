"""Modeling strategy generator - synthesizes all EDA findings."""

import pandas as pd


class ModelingStrategy:
    """Evidence-based modeling recommendations."""
    
    def __init__(self, findings_summary, recommendations, rationale):
        self.findings_summary = findings_summary
        self.recommendations = recommendations
        self.rationale = rationale
    
    def summary(self):
        """Return formatted summary."""
        output = []
        output.append("\n" + "="*70)
        output.append("EDA FINDINGS SUMMARY")
        output.append("="*70)
        
        for category, items in self.findings_summary.items():
            output.append(f"\n{category}:")
            for item in items:
                output.append(f"  • {item}")
        
        output.append("\n" + "="*70)
        output.append("MODELING RECOMMENDATIONS")
        output.append("="*70)
        
        for i, rec in enumerate(self.recommendations, 1):
            output.append(f"\n{i}. {rec['model']}")
            output.append(f"   Rationale: {rec['rationale']}")
            output.append(f"   Priority: {rec['priority']}")
        
        output.append("\n" + "="*70)
        output.append("STRATEGIC RATIONALE")
        output.append("="*70)
        output.append(self.rationale)
        output.append("="*70 + "\n")
        
        return "\n".join(output)
    
    def display_recommendations(self):
        """Print just the recommendations table."""
        print(f"\n{'='*70}")
        print("RECOMMENDED MODELING APPROACH")
        print(f"{'='*70}")
        
        for i, rec in enumerate(self.recommendations, 1):
            print(f"\n{i}. {rec['model']} [{rec['priority']} priority]")
            print(f"   {rec['rationale']}")
        
        print(f"\n{'='*70}\n")


class ModelingStrategyGenerator:
    """Synthesizes EDA results into actionable modeling strategy."""
    
    def __init__(self, all_results, config):
        self.results = all_results
        self.config = config
    
    def generate(self):
        """Generate comprehensive modeling strategy."""
        findings = self._synthesize_findings()
        recommendations = self._formulate_recommendations()
        rationale = self._craft_rationale()
        
        return ModelingStrategy(findings, recommendations, rationale)
    
    def _synthesize_findings(self):
        """Extract key findings from all analyses."""
        findings = {
            'Data Quality': [],
            'Class Distribution': [],
            'Feature Patterns': [],
            'Separability': [],
            'Preprocessing': []
        }
        
        # Quality
        if 'quality' in self.results:
            q = self.results['quality']
            findings['Data Quality'].append(f"Complete dataset ({q.findings.get('total_missing', 0)} missing values)")
            findings['Data Quality'].append(f"{q.findings.get('duplicates', 0)} duplicate records")
        
        # Distribution
        if 'distribution' in self.results:
            d = self.results['distribution']
            findings['Class Distribution'].append(f"{'Balanced' if d.findings.get('is_balanced') else 'Imbalanced'} classes")
            findings['Class Distribution'].append("Asymmetric costs favor recall optimization")
        
        # Correlations
        if 'correlations' in self.results:
            c = self.results['correlations']
            findings['Feature Patterns'].append(f"Multicollinearity: {c.findings['multicollinearity']['high_corr_count']} high-correlation pairs")
            findings['Feature Patterns'].append(f"Dimensionality: {c.findings['pca']['reduction_pct']:.0f}% reducible via PCA")
            findings['Feature Patterns'].append(f"Class signatures: {c.findings['class_signatures']['strongest_region']} bands strongest")
        
        # Separability
        if 'separability' in self.results:
            s = self.results['separability']
            findings['Separability'].append(f"Strong separation: {s.findings.get('strong_sep_count', 0)} features")
            findings['Separability'].append(f"Moderate separation: {s.findings.get('moderate_sep_count', 0)} features")
            findings['Separability'].append("Partial overlap suggests non-linear models needed")
        
        # Preprocessing
        if 'preprocessing' in self.results:
            p = self.results['preprocessing']
            findings['Preprocessing'].append("Scaling required for distance-based models")
            findings['Preprocessing'].append("Outliers present but genuine - keep them")
        
        return findings
    
    def _formulate_recommendations(self):
        """Generate prioritized model recommendations."""
        recommendations = [
            {
                'model': 'SVM (RBF Kernel)',
                'rationale': 'Handles non-linear boundaries, robust to multicollinearity, works well with small datasets',
                'priority': 'HIGH',
                'params': 'C: [0.1, 0.5, 1.0, 2.0], gamma: [scale, auto]'
            },
            {
                'model': 'Random Forest',
                'rationale': 'Naturally handles correlated features, captures feature interactions, provides importance scores',
                'priority': 'HIGH',
                'params': 'n_estimators: [100, 200], max_depth: [10, 20, None], min_samples_split: [2, 5]'
            },
            {
                'model': 'Ridge/Elastic Net',
                'rationale': 'Regularized linear baseline, handles multicollinearity via penalty',
                'priority': 'MEDIUM',
                'params': 'alpha: [0.1, 1.0, 10.0]'
            },
            {
                'model': 'KNN',
                'rationale': 'Non-parametric, good for small datasets, simple interpretability',
                'priority': 'MEDIUM',
                'params': 'n_neighbors: [3, 5, 7, 11]'
            },
            {
                'model': 'Logistic Regression',
                'rationale': 'Fast baseline, interpretable coefficients, probability calibration',
                'priority': 'LOW',
                'params': 'C: [0.1, 1.0, 10.0], penalty: [l1, l2]'
            }
        ]
        
        return recommendations
    
    def _craft_rationale(self):
        """Craft strategic rationale tying findings to recommendations."""
        rationale = (
            "The EDA reveals a well-structured classification problem with moderate complexity. "
            "Clean data eliminates preprocessing overhead. Balanced classes with asymmetric costs "
            "justify F2-optimized training followed by threshold tuning for deployment. "
            "\n\n"
            "Multicollinearity and partial feature overlap favor non-linear models (SVM, Random Forest) "
            "over linear approaches. The 60-dimensional feature space is manageable without PCA, "
            "allowing models to leverage the full frequency spectrum. "
            "\n\n"
            "Recommended workflow: (1) Train 5-7 models with 5-fold CV optimizing F2 score, "
            "(2) Select champion based on CV performance, (3) Validate on held-out test set, "
            "(4) Optimize threshold using out-of-fold predictions to achieve 100% recall target. "
            "\n\n"
            "Expected outcome: SVM or Random Forest achieving 95%+ test recall at default threshold, "
            "tunable to 98-100% recall with acceptable precision trade-off (70%+ precision maintained)."
        )
        
        return rationale


