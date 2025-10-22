"""
Runs each of the three ABSA implementations on the test_samples.json dataset
and compares results with expected sentiments.
"""

import sys
import os
import json

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Available implementations
from src.llm_absa import OllamaABSA
from src.transformer_absa import TransformerABSA
from src.lexicon_absa import LexiconABSA


def normalize(aspect: str) -> str:
    """Simplify aspect names for fairer comparison."""
    aspect = aspect.lower().strip()
    # remove generic or context words the model might add
    for word in ["the", "a", "an", "app", "system", "team", "product", "item"]:
        aspect = aspect.replace(word, "")
    # clean up punctuation and extra spaces
    aspect = aspect.replace("-", " ").replace("_", " ")
    aspect = " ".join(aspect.split())
    return aspect.strip()


def aspect_match(expected: str, predicted: str) -> bool:
    """Match aspects leniently using normalization and substring logic."""
    e = normalize(expected)
    p = normalize(predicted)
    return e in p or p in e


def main():
    # Locate and load dataset
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_path = os.path.join(project_root, "data", "test_samples.json")

    if not os.path.exists(data_path):
        print(f" Dataset not found: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Choose which implementation to run
    analyzer = LexiconABSA()  # Change to TransformerABSA() or LexiconABSA() as needed

    total = 0
    correct = 0

    print(f"\n=== Running Dataset Evaluation using {analyzer.__class__.__name__} ===")

    for sample in samples:
        text = sample["text"]
        expected = sample["expected"]

        print("\n────────────────────────────────────")
        print(f"Text: {text}")

        # Run analyzer
        results = analyzer.analyze(text)

        # Skip empty outputs gracefully
        if not results:
            print("  ⚠️ No aspects detected.")
            continue

        # Print details and compute matches
        for exp in expected:
            total += 1
            exp_aspect = exp["aspect"]
            exp_sent = exp["sentiment"]

            # Try to find a matching predicted aspect
            matched = False
            for pred in results:
                if aspect_match(exp_aspect, pred.aspect):
                    matched = True
                    pred_sent = pred.sentiment
                    correct_flag = exp_sent == pred_sent
                    correct += correct_flag

                    print(f"  Aspect: {exp_aspect:15s} | Expected: {exp_sent:8s} | Predicted: {pred_sent:8s} | {'✅' if correct_flag else '❌'}")
                    if hasattr(pred, "confidence") and pred.confidence is not None:
                        print(f"     Confidence: {pred.confidence:.2f}")
                    if hasattr(pred, "vader_breakdown") and pred.vader_breakdown:
                        print(f"     → VADER scores: {pred.vader_breakdown}")
                    break

            if not matched:
                print(f"  Aspect: {exp_aspect:15s} | Expected: {exp_sent:8s} | Predicted: missing | ❌")

    # Compute and display final accuracy
    accuracy = correct / total if total else 0.0
    print("\n────────────────────────────────────")
    print(f"Overall accuracy: {accuracy:.2%} ({correct}/{total} correct)")
    print("====================================\n")


if __name__ == "__main__":
    main()
