"""Base result classes for EDA workflow."""


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
