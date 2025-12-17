"""
EDA Workflow - Reproducible analysis pipeline for binary classification.

Design Patterns:
- Template: Base analyzer structure
- Factory: Consistent visualization generation
- Strategy: Pluggable analysis methods

Usage:
    eda = EDAWorkflow(df_train, config)
    quality_results = eda.assess_quality()
    quality_results.summarize()
    quality_results.plot()
"""

from .analyzers.quality import DataQualityAnalyzer
from .analyzers.distribution import ClassDistributionAnalyzer
from .analyzers.correlation import CorrelationAnalyzer
from .analyzers.separability import DistributionAnalyzer
from .analyzers.preprocessing import PreprocessingAnalyzer
from .analyzers.strategy import ModelingStrategyGenerator


class EDAWorkflow:
    """
    Orchestrates complete EDA workflow.
    
    Args:
        df: Training DataFrame (already split)
        config: Dict with target_col, class_labels, colors, etc.
    """
    
    def __init__(self, df, config):
        self.df = df
        self.target_col = config['target_col']
        self.feature_cols = [c for c in df.columns if c != self.target_col]
        self.config = config
        self.results = {}  # Store all results for final strategy
        
    def assess_quality(self):
        """Section 1: Data Quality Assessment."""
        analyzer = DataQualityAnalyzer(self.df, self.target_col)
        result = analyzer.analyze()
        self.results['quality'] = result
        return result
    
    def analyze_class_distribution(self):
        """Section 2: Class Distribution & Asymmetric Risk."""
        analyzer = ClassDistributionAnalyzer(self.df, self.target_col, self.config)
        result = analyzer.analyze()
        self.results['distribution'] = result
        return result
    
    def analyze_correlations(self):
        """Section 3: Feature Correlation Patterns (4 sub-analyses)."""
        analyzer = CorrelationAnalyzer(self.df, self.target_col, self.feature_cols, self.config)
        result = analyzer.analyze()
        self.results['correlations'] = result
        return result
    
    def analyze_separability(self):
        """Section 4: Class Separability (KDE patterns)."""
        analyzer = DistributionAnalyzer(self.df, self.target_col, self.feature_cols, self.config)
        result = analyzer.analyze()
        self.results['separability'] = result
        return result
    
    def assess_preprocessing(self):
        """Section 5: Preprocessing Requirements."""
        analyzer = PreprocessingAnalyzer(self.df, self.feature_cols, self.config)
        result = analyzer.analyze()
        self.results['preprocessing'] = result
        return result
    
    def generate_modeling_strategy(self):
        """Section 6: Evidence-Based Modeling Recommendations."""
        generator = ModelingStrategyGenerator(self.results, self.config)
        return generator.generate()


class AnalysisResult:
    """Base class for analysis results with consistent interface."""
    
    def __init__(self, findings, implications, decision):
        self.findings = findings
        self.implications = implications
        self.decision = decision
    
    def summarize(self):
        """Print structured summary."""
        print("\n" + "="*70)
        print("FINDINGS")
        print("="*70)
        for key, value in self.findings.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*70)
        print("IMPLICATIONS")
        print("="*70)
        for implication in self.implications:
            print(f"  • {implication}")
        
        print("\n" + "="*70)
        print("DECISION")
        print("="*70)
        print(f"  {self.decision}")
        print("="*70 + "\n")
    
    def plot(self):
        """Generate visualization (implemented by subclasses)."""
        raise NotImplementedError("Subclass must implement plot()")
