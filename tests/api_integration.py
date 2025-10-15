from src.base import ABSAAnalyzer
from src.lexicon_absa import LexiconABSA


def main():
    analyzer = LexiconABSA()
    assert isinstance(analyzer, ABSAAnalyzer), "LexiconABSA should inherit from ABSAAnalyzer"

    text = "The pizza was delicious but the service was terrible."
    results = analyzer.analyze(text)

    # Print results
    print("\nUnified API Test:")
    for r in results:
        print(f"Aspect: {r.aspect:10s} | Sentiment: {r.sentiment:8s} | Confidence: {r.confidence:.2f}")

    # Check expected structure
    assert all(hasattr(r, "aspect") for r in results)
    assert all(hasattr(r, "sentiment") for r in results)
    assert all(hasattr(r, "confidence") for r in results)
    print("\n Unified API works correctly for LexiconABSA!")

if __name__ == "__main__":
    main()