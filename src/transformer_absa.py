"""
Implementation 2: Transformer-Based ABSA using pre-trained model
Uses yangheng/deberta-v3-base-absa-v1.1 from Hugging Face
"""

from typing import List
from transformers import pipeline
import torch

from src.base import ABSAAnalyzer, AspectSentiment


class TransformerABSA(ABSAAnalyzer):
    def __init__(self, model_name: str = "yangheng/deberta-v3-base-absa-v1.1"):
        # Automatically pick GPU or CPU
        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading model: {model_name} (device: {'GPU' if device == 0 else 'CPU'})")

        # Hugging Face text classification pipeline
        self.classifier = pipeline("text-classification", model=model_name, device=device)

    def _extract_aspects(self, text: str) -> List[str]:
        """
        Very simple aspect extraction: pick likely nouns or noun phrases. simple at the moment, expand later if its good
        """
        import re
        stop_words = {"the", "a", "an", "and", "is", "are", "was", "were", "but", "or"}
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]

    def analyze(self, text: str, aspects: List[str] = None) -> List[AspectSentiment]:
        if not text.strip():
            return []

        aspects = aspects or self._extract_aspects(text)
        results = []

        for aspect in aspects:
            try:
                # Model expects "text [SEP] aspect"
                pred = self.classifier(f"{text} [SEP] {aspect}")[0]
                results.append(
                    AspectSentiment(
                        aspect=aspect,
                        sentiment=pred["label"].lower(),
                        confidence=pred["score"]
                    )
                )
            except Exception as e:
                print(f"Error on aspect '{aspect}': {e}")

        # Sort by confidence
        return sorted(results, key=lambda x: x.confidence, reverse=True)
