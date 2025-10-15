from src.transformer_absa import TransformerABSA
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_integration():
    """Test that TransformerABSA works with your base classes"""

    # Initialize analyzer
    print("Initializing TransformerABSA...")
    analyzer = TransformerABSA()

    # Test 1: Your example
    print("\n" + "=" * 60)
    print("Test 1: Your example sentence")
    print("=" * 60)
    sentence = "The food was exceptional, although the service was a bit slow."
    aspects = ["food", "service"]

    print(f"Text: {sentence}")
    print(f"Aspects: {aspects}")

    results = analyzer.analyze_with_aspects(sentence, aspects)

    print("\nResults:")
    for result in results:
        print(f"  Aspect: {result.aspect}")
        print(f"  Sentiment: {result.sentiment}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Text span: {result.text_span}")
        print(f"  Vader breakdown: {result.vader_breakdown}")
        print()

    # Test 2: Compare with auto-extraction
    print("=" * 60)
    print("Test 2: Auto-extract aspects")
    print("=" * 60)
    text2 = "The pizza was delicious but the service was terrible."

    print(f"Text: {text2}")
    results2 = analyzer.analyze(text2)

    print("\nResults:")
    for result in results2:
        print(f"  {result.aspect:15} → {result.sentiment:8} ({result.confidence:.3f})")

    # Test 3: Batch processing
    print("\n" + "=" * 60)
    print("Test 3: Batch processing")
    print("=" * 60)
    text3 = "Great ambiance and fantastic food, but overpriced and slow service."
    aspects3 = ["ambiance", "food", "price", "service"]

    print(f"Text: {text3}")
    print(f"Aspects: {aspects3}")

    results3 = analyzer.analyze_batch(text3, aspects3)

    print("\nResults:")
    for result in results3:
        print(f"  {result.aspect:15} → {result.sentiment:8} ({result.confidence:.3f})")


if __name__ == "__main__":
    test_integration()