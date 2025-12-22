"""Data quality analyzer."""

import pandas as pd
from .base import BaseAnalyzer
from ..results import AnalysisResult


class DataQualityResult(AnalysisResult):
    """Quality analysis results."""
    
    def plot(self):
        print("Quality assessment complete. No visualization generated.")


class DataQualityAnalyzer(BaseAnalyzer):
    """Analyzes data completeness and quality."""
    
    def prepare_data(self):
        return self.df
    
    def compute_metrics(self, data):
        missing = data.isnull().sum()
        total_missing = int(missing.sum())
        missing_cols = int((missing > 0).sum())
        duplicates = int(data.duplicated().sum())
        zero_frac = (data.select_dtypes(include='number') == 0).mean()
        sparse_cols = int((zero_frac >= 0.95).sum())
        variance = data.select_dtypes(include='number').var()
        low_var_cols = int((variance <= 1e-8).sum())
        
        return {
            'total_missing': total_missing,
            'missing_cols': missing_cols,
            'missing_pct': (total_missing / (len(data) * len(data.columns))) * 100,
            'duplicates': duplicates,
            'duplicate_pct': (duplicates / len(data)) * 100,
            'sparse_cols': sparse_cols,
            'low_variance_cols': low_var_cols,
            'total_samples': len(data),
            'total_features': len(data.columns)
        }
    
    def interpret_findings(self, findings):
        implications = []
        if findings['total_missing'] == 0:
            implications.append("Complete dataset - no imputation required")
        else:
            implications.append(f"{findings['missing_cols']} features require imputation strategy")
        
        if findings['duplicates'] > 0:
            implications.append(f"{findings['duplicates']} duplicate rows may bias model")
        else:
            implications.append("No duplicate records detected")
        
        if findings['sparse_cols'] > 0:
            implications.append(f"{findings['sparse_cols']} sparse features may have limited signal")
        
        if findings['low_variance_cols'] > 0:
            implications.append(f"{findings['low_variance_cols']} near-constant features should be removed")
        
        return implications
    
    def formulate_decision(self, findings, implications):
        if findings['total_missing'] == 0 and findings['duplicates'] == 0 and findings['low_variance_cols'] == 0:
            return "Proceed with complete cases. Data quality is high - no preprocessing required."
        else:
            actions = []
            if findings['total_missing'] > 0:
                actions.append("implement imputation")
            if findings['duplicates'] > 0:
                actions.append("remove duplicates")
            if findings['low_variance_cols'] > 0:
                actions.append("drop low-variance features")
            return f"Data quality issues detected. Required actions: {', '.join(actions)}"
    
    def create_result(self, findings, implications, decision):
        return DataQualityResult(findings, implications, decision)
