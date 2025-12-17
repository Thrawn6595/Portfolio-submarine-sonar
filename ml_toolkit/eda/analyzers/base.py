"""Base analyzer using Template Pattern."""

from abc import ABC, abstractmethod


class BaseAnalyzer(ABC):
    """Template for all analyzers."""
    
    def __init__(self, df, target_col=None):
        self.df = df
        self.target_col = target_col
        self.feature_cols = [c for c in df.columns if c != target_col] if target_col else df.columns.tolist()
    
    def analyze(self):
        """Template method - defines analysis workflow."""
        data = self.prepare_data()
        findings = self.compute_metrics(data)
        implications = self.interpret_findings(findings)
        decision = self.formulate_decision(findings, implications)
        
        return self.create_result(findings, implications, decision)
    
    @abstractmethod
    def prepare_data(self):
        """Prepare data for analysis."""
        pass
    
    @abstractmethod
    def compute_metrics(self, data):
        """Compute relevant metrics."""
        pass
    
    @abstractmethod
    def interpret_findings(self, findings):
        """Translate metrics to implications."""
        pass
    
    @abstractmethod
    def formulate_decision(self, findings, implications):
        """Make actionable recommendation."""
        pass
    
    @abstractmethod
    def create_result(self, findings, implications, decision):
        """Package results."""
        pass
