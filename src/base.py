# Base classes and interfaces
from typing import List
from dataclasses import dataclass


@dataclass
class AspectSentiment:
    """Data class for aspect-sentiment pairs"""
    aspect: str
    sentiment: str
    confidence: float
    text_span: tuple = None
    vader_breakdown: dict = None  # Optional field, used by LexiconABSA to compare results with the default results from Vader

class ABSAAnalyzer:
    """Base interface for all ABSA implementations"""

    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Analyze text and extract aspect-sentiment pairs.

        Args:
            text: Input text to analyze.
        Returns:
            List of AspectSentiment objects.
        """
        raise NotImplementedError("Subclasses must implement this method.")