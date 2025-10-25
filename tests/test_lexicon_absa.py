"""
Unit tests and demonstration for the Lexicon-Based ABSA implementation.
Uses pytest-style structure for automated testing, but can also be run manually.
"""
import sys, os

import unittest

from src.base import AspectSentiment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lexicon_absa import LexiconABSA


# -----------------------------
#  Automated Tests
# -----------------------------

class TestLexiconABSA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize LexiconABSA once for all tests."""
        cls.analyzer = LexiconABSA(debug=False)

    def test_basic_positive_negative(self):
        """Check if the model correctly identifies positive and negative aspects."""
        text = "The pizza was delicious but the service was terrible."
        results = self.analyzer.analyze(text)
        aspects = {r.aspect.lower(): r.sentiment for r in results}

        self.assertIn("pizza", aspects)
        self.assertIn("service", aspects)
        self.assertEqual(aspects["pizza"], "positive")
        self.assertEqual(aspects["service"], "negative")

    def test_negation_handling(self):
        """Ensure that negations flip sentiment polarity correctly."""
        text = "The food was not good."
        results = self.analyzer.analyze(text)
        found = any(r.aspect.lower() == "food" and r.sentiment == "negative" for r in results)
        self.assertTrue(found)

    def test_intensifier_effect(self):
        """Check if intensifiers increase sentiment strength."""
        text1 = "The movie was good."
        text2 = "The movie was very good."
        res1 = self.analyzer.analyze(text1)
        res2 = self.analyzer.analyze(text2)

        s1 = next(r for r in res1 if r.aspect.lower() == "movie")
        s2 = next(r for r in res2 if r.aspect.lower() == "movie")

        self.assertEqual(s1.sentiment, "positive")
        self.assertEqual(s2.sentiment, "positive")
        self.assertGreaterEqual(s2.confidence, s1.confidence)

    def test_softener_effect(self):
        """Check if softeners reduce sentiment confidence."""
        text1 = "The service was bad."
        text2 = "The service was somewhat bad."
        res1 = self.analyzer.analyze(text1)
        res2 = self.analyzer.analyze(text2)

        s1 = next(r for r in res1 if r.aspect.lower() == "service")
        s2 = next(r for r in res2 if r.aspect.lower() == "service")

        self.assertEqual(s1.sentiment, "negative")
        self.assertEqual(s2.sentiment, "negative")
        self.assertLessEqual(s2.confidence, s1.confidence)

    def test_empty_input(self):
        """Empty or whitespace input should return an empty list."""
        results = self.analyzer.analyze("   ")
        self.assertEqual(results, [])

    def test_output_structure(self):
        """Ensure all outputs are valid AspectSentiment objects."""
        text = "The phone is great."
        results = self.analyzer.analyze(text)

        for r in results:
            self.assertIsInstance(r, AspectSentiment)
            self.assertTrue(hasattr(r, "aspect"))
            self.assertTrue(hasattr(r, "sentiment"))
            self.assertTrue(hasattr(r, "confidence"))


if __name__ == "__main__":
    unittest.main()


