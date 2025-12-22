"""EDA Workflow - Reproducible analysis pipeline."""

from .analyzers.quality import DataQualityAnalyzer
from .analyzers.distribution import ClassDistributionAnalyzer
from .analyzers.correlation import CorrelationAnalyzer
from .analyzers.separability import DistributionAnalyzer
from .analyzers.preprocessing import PreprocessingAnalyzer
from .analyzers.strategy import ModelingStrategyGenerator


class EDAWorkflow:
    """Orchestrates complete EDA workflow."""
    
    def __init__(self, df, config):
        self.df = df
        self.target_col = config['target_col']
        self.feature_cols = [c for c in df.columns if c != self.target_col]
        self.config = config
        self.results = {}
        
    def assess_quality(self):
        analyzer = DataQualityAnalyzer(self.df, self.target_col)
        result = analyzer.analyze()
        self.results['quality'] = result
        return result
    
    def analyze_class_distribution(self):
        analyzer = ClassDistributionAnalyzer(self.df, self.target_col, self.config)
        result = analyzer.analyze()
        self.results['distribution'] = result
        return result
    
    def analyze_correlations(self):
        analyzer = CorrelationAnalyzer(self.df, self.target_col, self.feature_cols, self.config)
        result = analyzer.analyze()
        self.results['correlations'] = result
        return result
    
    def analyze_separability(self):
        analyzer = DistributionAnalyzer(self.df, self.target_col, self.feature_cols, self.config)
        result = analyzer.analyze()
        self.results['separability'] = result
        return result
    
    def assess_preprocessing(self):
        analyzer = PreprocessingAnalyzer(self.df, self.feature_cols, self.config)
        result = analyzer.analyze()
        self.results['preprocessing'] = result
        return result
    
    def generate_modeling_strategy(self):
        generator = ModelingStrategyGenerator(self.results, self.config)
        return generator.generate()
