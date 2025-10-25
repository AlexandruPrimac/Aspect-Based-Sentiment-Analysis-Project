"""
Implementation 2: Transformer-Based ABSA using a pre-trained model.

This version uses a Hugging Face transformer fine-tuned for aspect-based
sentiment analysis (ABSA). The default model is:
    -> "yangheng/deberta-v3-base-absa-v1.1"

This model expects input in the form:
    "<sentence> [SEP] <aspect>"

and outputs a sentiment label for that aspect (e.g., Positive / Negative / Neutral).
"""

from typing import List

import torch
from transformers import pipeline

from src.base import ABSAAnalyzer, AspectSentiment


class TransformerABSA(ABSAAnalyzer):
    def __init__(self, model_name: str = "yangheng/deberta-v3-base-absa-v1.1"):
        # Select GPU if available, else fallback to CPU
        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading model: {model_name} (device: {'GPU' if device == 0 else 'CPU'})")

        # Initialize Hugging Face for text classification.
        # The model is trained to take (text + aspect) and predict sentiment.
        self.classifier = pipeline("text-classification", model=model_name, device=device)

    # -------------------------------------------------------------
    # Simple aspect extraction helper
    # -------------------------------------------------------------
    def _extract_aspects(self, text: str) -> List[str]:
        """
        Rough heuristic for aspect extraction:
        - Extracts candidate noun-like words using regex
        - Filters out short words and stop words
        """
        import re
        stop_words = {"the", "a", "an", "and", "is", "are", "was", "were", "but", "or"}
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]

    # -------------------------------------------------------------
    # Main ABSA analysis method
    # -------------------------------------------------------------
    def analyze(self, text: str, aspects: List[str] = None) -> List[AspectSentiment]:
        """
        Analyze a given text and return aspect–sentiment pairs.

        If no explicit aspects are provided, they are automatically extracted
        using a simple heuristic method (_extract_aspects).
        """
        if not text.strip():
            return []

        # Either use provided aspects or automatically extract them
        aspects = aspects or self._extract_aspects(text)
        results = []

        for aspect in aspects:
            try:
                # The pre-trained model expects "sentence [SEP] aspect"
                pred = self.classifier(f"{text} [SEP] {aspect}")[0]

                # Convert model output into AspectSentiment object
                results.append(
                    AspectSentiment(
                        aspect=aspect,
                        sentiment=pred["label"].lower(),
                        confidence=pred["score"]
                    )
                )

            except Exception as e:
                # Keep analysis robust against unexpected errors
                print(f"Error on aspect '{aspect}': {e}")

        # Return results sorted by confidence (most confident first)
        return sorted(results, key=lambda x: x.confidence, reverse=True)
