"""
Unit tests and demonstration for the Lexicon-Based ABSA implementation.
Uses pytest-style structure for automated testing, but can also be run manually.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lexicon_absa import LexiconABSA


# -----------------------------
#  Automated Tests
# -----------------------------

def test_multiple_aspects():
    """
    Verify that multiple aspects are detected with correct sentiment polarity.
    """
    analyzer = LexiconABSA()
    text = "The pizza was great but the service was bad."
    results = analyzer.analyze(text)

    # Ensure we get at least 2 aspects
    assert isinstance(results, list)
    assert len(results) >= 2

    # Check for both aspects
    aspects = [r.aspect for r in results]
    assert "pizza" in aspects
    assert "service" in aspects

    # Verify polarity consistency
    sentiments = {r.aspect: r.sentiment for r in results}
    assert sentiments["pizza"] == "positive"
    assert sentiments["service"] == "negative"


def test_negation_handling():
    """
    Test that negations flip the expected sentiment polarity.
    """
    analyzer = LexiconABSA()
    text = "The pizza was not good."
    results = analyzer.analyze(text)
    pizza_result = next((r for r in results if r.aspect == "pizza"), None)

    assert pizza_result is not None
    assert pizza_result.sentiment == "negative"
    assert pizza_result.confidence > 0.3


def test_neutral_sentence():
    """
    Ensure neutral sentiment is correctly classified.
    """
    analyzer = LexiconABSA()
    text = "The food was okay."
    results = analyzer.analyze(text)
    food_result = next((r for r in results if r.aspect == "food"), None)

    assert food_result is not None
    assert food_result.sentiment == "neutral"


def test_intensifier_effect():
    """
    Ensure adverbs like 'extremely' increase sentiment strength.
    """
    analyzer = LexiconABSA()
    base_text = "The service was bad."
    intense_text = "The service was extremely bad."

    base_conf = analyzer.analyze(base_text)[0].confidence
    intense_conf = analyzer.analyze(intense_text)[0].confidence

    assert intense_conf >= base_conf, "Intensifier should increase confidence"


def test_emoji_handling():
    """
    Verify that emojis are replaced and do not break the pipeline.
    """
    analyzer = LexiconABSA()
    text = "The pizza was 💘 delicious."
    results = analyzer.analyze(text)
    assert any("pizza" in r.aspect for r in results)
    assert any(r.sentiment == "positive" for r in results)


def test_vader_breakdown_structure():
    """
    Check that vader_breakdown field contains all expected sentiment scores.
    """
    analyzer = LexiconABSA()
    text = "The pizza was great."
    results = analyzer.analyze(text)
    breakdown = results[0].vader_breakdown

    assert isinstance(breakdown, dict)
    for key in ["neg", "neu", "pos", "compound"]:
        assert key in breakdown
