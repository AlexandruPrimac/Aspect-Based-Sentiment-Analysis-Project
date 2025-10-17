"""
Implementation 2: Transformer-Based ABSA using pre-trained model
Uses yangheng/deberta-v3-base-absa-v1.1 from Hugging Face
"""

from typing import List
from transformers import pipeline
import torch

from src.base import ABSAAnalyzer, AspectSentiment


class TransformerABSA(ABSAAnalyzer):
    """
    Transformer-based ABSA implementation using DeBERTa model with pipeline API.

    This model expects input in the format: "text [SEP] aspect"
    It classifies the sentiment toward a specific aspect in the text.
    """

    def __init__(self, model_name: str = "yangheng/deberta-v3-base-absa-v1.1", device: int = None):
        """
        Initialize the transformer-based ABSA model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on (0 for cuda:0, -1 for CPU). Auto-detected if None.
        """
        print(f"Loading transformer model: {model_name}")

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 0  # Use first CUDA device
            elif torch.backends.mps.is_available():
                device = 0  # MPS support
            else:
                device = -1  # CPU

        device_name = f"cuda:{device}" if device >= 0 else "cpu"
        print(f"Using device: {device_name}")

        # Load classifier using pipeline API
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device=device
        )

        print(f"Model loaded successfully.")

    def _extract_candidate_aspects(self, text: str) -> List[tuple]:
        """
        Extract candidate aspects from text.
        Returns list of (aspect_text, start_pos, end_pos) tuples.
        """
        import re

        # Simple approach: extract words that could be aspects
        # Remove common stop words and very short words
        stop_words = {'the', 'a', 'an', 'and', 'but', 'or', 'was', 'is', 'are', 'were', 'been', 'be', 'have', 'has', 'had'}

        # Split into words and clean
        words = re.findall(r'\b\w+\b', text.lower())
        aspects = []

        # Track positions for single and multi-word aspects
        text_lower = text.lower()

        for i, word in enumerate(words):
            # Skip stop words and very short words
            if word in stop_words or len(word) <= 2:
                continue

            # Single word aspects
            start = text_lower.find(word)
            if start != -1:
                aspects.append((word, start, start + len(word)))

            # Two-word phrases (noun phrases often have 2 words)
            if i < len(words) - 1:
                next_word = words[i + 1]
                if next_word not in stop_words:
                    phrase = f"{word} {next_word}"
                    start = text_lower.find(phrase)
                    if start != -1:
                        aspects.append((phrase, start, start + len(phrase)))

        # Remove duplicates while preserving order
        seen = set()
        unique_aspects = []
        for aspect in aspects:
            if aspect[0] not in seen:
                seen.add(aspect[0])
                unique_aspects.append(aspect)

        return unique_aspects

    def _classify_aspect_sentiment(self, text: str, aspect: str) -> tuple:
        """
        Classify sentiment toward a specific aspect in the text.

        Args:
            text: The input text
            aspect: The aspect term to analyze

        Returns:
            (sentiment_label, confidence_score)
        """
        # Format input as expected by the model: "text [SEP] aspect"
        input_text = f"{text} [SEP] {aspect}"

        # Get prediction from pipeline
        result = self.classifier(input_text)[0]

        # Extract label and confidence
        sentiment_label = result['label'].lower()
        confidence = result['score']

        return sentiment_label, confidence

    def analyze(self, text: str, aspects: List[str] = None) -> List[AspectSentiment]:
        """
        Analyze text and extract aspect-sentiment pairs.

        Args:
            text: Input text to analyze
            aspects: Optional list of aspect terms. If None, will auto-extract.

        Returns:
            List of AspectSentiment objects
        """
        if not text or not text.strip():
            return []

        results = []

        try:
            # If aspects not provided, extract them
            if aspects is None:
                candidate_aspects = self._extract_candidate_aspects(text)
                aspects_to_analyze = [asp[0] for asp in candidate_aspects]
                aspect_positions = {asp[0]: (asp[1], asp[2]) for asp in candidate_aspects}
            else:
                aspects_to_analyze = aspects
                aspect_positions = {}

            # Classify sentiment for each aspect
            for aspect in aspects_to_analyze:
                try:
                    sentiment, confidence = self._classify_aspect_sentiment(text, aspect)

                    # Filter out very low-confidence predictions (optional)
                    # Adjust threshold based on your needs
                    if confidence > 0.3:  # Confidence threshold
                        text_span = aspect_positions.get(aspect, None)
                        results.append(
                            AspectSentiment(
                                aspect=aspect,
                                sentiment=sentiment,
                                confidence=confidence,
                                text_span=text_span
                            )
                        )
                except Exception as e:
                    print(f"Error classifying aspect '{aspect}': {e}")
                    continue

            # Sort by confidence (highest first)
            results.sort(key=lambda x: x.confidence, reverse=True)

        except Exception as e:
            print(f"Error in analyze: {e}")
            return []

        return results

    def analyze_with_aspects(self, text: str, aspects: List[str]) -> List[AspectSentiment]:
        """
        Analyze sentiment for specific, pre-defined aspects.
        Useful when you know which aspects to look for.

        Args:
            text: Input text to analyze
            aspects: List of aspect terms to analyze

        Returns:
            List of AspectSentiment objects
        """
        return self.analyze(text, aspects=aspects)

    def analyze_batch(self, text: str, aspects: List[str]) -> List[AspectSentiment]:
        """
        Batch process multiple aspects for better performance.

        Args:
            text: Input text to analyze
            aspects: List of aspect terms to analyze

        Returns:
            List of AspectSentiment objects
        """
        if not text or not text.strip() or not aspects:
            return []

        results = []

        try:
            # Prepare batch inputs
            inputs = [f"{text} [SEP] {aspect}" for aspect in aspects]

            # Batch prediction (much faster!)
            predictions = self.classifier(inputs)

            # Process results
            for aspect, pred in zip(aspects, predictions):
                sentiment = pred['label'].lower()
                confidence = pred['score']

                # Filter low confidence (optional)
                if confidence > 0.3:
                    results.append(
                        AspectSentiment(
                            aspect=aspect,
                            sentiment=sentiment,
                            confidence=confidence,
                            text_span=None
                        )
                    )

            # Sort by confidence
            results.sort(key=lambda x: x.confidence, reverse=True)

        except Exception as e:
            print(f"Error in batch analyze: {e}")
            return []

        return results


# Example usage
# if __name__ == "__main__":
#     # Initialize the analyzer
#     analyzer = TransformerABSA()
#
#     # Example 1: Your example sentence
#     sentence = "The food was exceptional, although the service was a bit slow."
#     print(f"\nAnalyzing: '{sentence}'")
#
#     # Method 1: Analyze with specific aspects
#     aspects = ["food", "service"]
#     results = analyzer.analyze_with_aspects(sentence, aspects)
#     for result in results:
#         print(f"  {result}")
#
#     # Example 2: Auto-extract aspects
#     text2 = "The pizza was delicious but the service was terrible."
#     print(f"\nAnalyzing: '{text2}' (auto-extract)")
#     results2 = analyzer.analyze(text2)
#     for result in results2:
#         print(f"  {result}")
#
#     # Example 3: Batch processing (faster for multiple aspects)
#     text3 = "The laptop has amazing battery life, great performance, but the screen quality is disappointing."
#     aspects3 = ["battery life", "performance", "screen quality", "laptop"]
#     print(f"\nAnalyzing: '{text3}' (batch)")
#     results3 = analyzer.analyze_batch(text3, aspects3)
#     for result in results3:
#         print(f"  {result}")
#
#     # Example 4: Complex review
#     text4 = "Great ambiance and fantastic food, but overpriced and slow service."
#     print(f"\nAnalyzing: '{text4}'")
#     results4 = analyzer.analyze(text4)
#     for result in results4:
#         print(f"  {result}")