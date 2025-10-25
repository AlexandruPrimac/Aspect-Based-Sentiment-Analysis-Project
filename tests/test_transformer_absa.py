import unittest
from src.transformer_absa import TransformerABSA
from src.base import AspectSentiment


class TestTransformerABSA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize the TransformerABSA model once for all tests."""
        cls.analyzer = TransformerABSA()

    def test_basic_positive_negative(self):
        """Test that the model can correctly classify clear sentiments."""
        text = "The pizza was delicious but the service was terrible."
        results = self.analyzer.analyze(text, aspects=["pizza", "service"])

        aspects = {r.aspect.lower(): r.sentiment for r in results}
        self.assertIn("pizza", aspects)
        self.assertIn("service", aspects)
        # The model label outputs may vary in capitalization or language
        self.assertEqual(aspects["pizza"], "positive")
        self.assertEqual(aspects["service"], "negative")

    def test_neutral_sentence(self):
        """The model should handle neutral sentiment correctly."""
        text = "The meeting was okay."
        results = self.analyzer.analyze(text, aspects=["meeting"])
        self.assertTrue(any(r.sentiment == "neutral" for r in results))

    def test_empty_input(self):
        """Empty string should return an empty list."""
        results = self.analyzer.analyze("")
        self.assertEqual(results, [])

    def test_auto_aspect_extraction(self):
        """Ensure the _extract_aspects method generates valid aspect candidates."""
        text = "The laptop is fast and powerful."
        aspects = self.analyzer._extract_aspects(text)
        self.assertIn("laptop", aspects)
        self.assertIn("fast", aspects)  # depending on regex extraction

    def test_output_structure(self):
        """Ensure outputs are valid AspectSentiment instances."""
        text = "The phone is amazing."
        results = self.analyzer.analyze(text, aspects=["phone"])

        for r in results:
            self.assertIsInstance(r, AspectSentiment)
            self.assertTrue(hasattr(r, "aspect"))
            self.assertTrue(hasattr(r, "sentiment"))
            self.assertTrue(hasattr(r, "confidence"))

    def test_multiple_aspects(self):
        """Check if model handles multiple aspects in a single run."""
        text = "The battery life is excellent but the camera is poor."
        aspects = ["battery life", "camera"]
        results = self.analyzer.analyze(text, aspects=aspects)
        result_aspects = {r.aspect.lower(): r.sentiment for r in results}

        self.assertIn("battery life", result_aspects)
        self.assertIn("camera", result_aspects)
        self.assertEqual(result_aspects["battery life"], "positive")
        self.assertEqual(result_aspects["camera"], "negative")

    def test_error_resilience(self):
        """Ensure model handles invalid aspects gracefully."""
        text = "The product was fine."
        results = self.analyzer.analyze(text, aspects=["", None, "product"])
        self.assertTrue(any(isinstance(r, AspectSentiment) for r in results))


if __name__ == "__main__":
    unittest.main()
