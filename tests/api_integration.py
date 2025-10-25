"""
Runs each of the three ABSA implementations on the test_samples.json dataset
and compares results with expected sentiments.

This script serves as the *integration and evaluation harness* for the project.
It tests the unified API design by running all ABSAAnalyzer subclasses using the
same interface, verifying that results are comparable and consistent.
"""

import json
import os
import sys
import time

# -------------------------------------------------------------------------
# Adjust the Python path so imports work correctly no matter where script runs.
# This ensures "src" can be imported when running this file directly.
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -------------------------------------------------------------------------
# Import available ABSA implementations from the project source
# -------------------------------------------------------------------------
from src.lexicon_absa import LexiconABSA
from src.transformer_absa import TransformerABSA
from src.llm_absa import OllamaABSA



def normalize(aspect: str) -> str:
    """
    Normalize aspect names for consistent comparison across models.
    Example:
        "The pizza" -> "pizza"
        "App system" -> "system"

    This helps account for small naming differences between model outputs
    (e.g., "service quality" vs. "service").
    """
    aspect = aspect.lower().strip()
    # Remove generic filler words the model might add
    for word in ["the", "a", "an", "app", "system", "team", "product", "item"]:
        aspect = aspect.replace(word, "")
    # Normalize punctuation and spacing
    aspect = aspect.replace("-", " ").replace("_", " ")
    aspect = " ".join(aspect.split())
    return aspect.strip()


def aspect_match(expected: str, predicted: str) -> bool:
    """
    Check whether a predicted aspect matches an expected one.

    Uses lenient substring matching after normalization, since
    model outputs may include partial or rephrased versions.
    Example:
        expected="battery life" and predicted="battery" -> match ✅
    """
    e = normalize(expected)
    p = normalize(predicted)
    return e in p or p in e


def main():
    """
    Main integration test entry point.

    Steps:
    1. Load dataset (test_samples.json)
    2. Choose which ABSA model to run
    3. Execute analyzer.analyze(text) on each sample
    4. Compare predicted sentiments with expected ground truth
    5. Print detailed results and final accuracy summary
    """

    # ---------------------------------------------------------------------
    # Locate and load the dataset (stored under /data/test_samples.json)
    # Choose what dataset you want: test(faster) or evaluation(slower)
    # ---------------------------------------------------------------------
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    # data_path = os.path.join(project_root, "data", "test_samples.json")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_path = os.path.join(project_root, "data", "evaluation_data.json")

    if not os.path.exists(data_path):
        print(f" Dataset not found: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # ---------------------------------------------------------------------
    # Select which ABSA implementation to evaluate.
    # Swap between OllamaABSA(), TransformerABSA(), or LexiconABSA() as needed.
    # ---------------------------------------------------------------------
    analyzer = OllamaABSA()  # Change here to test other implementations

    total = 0
    correct = 0

    print(f"\n=== Running Dataset Evaluation using {analyzer.__class__.__name__} ===")

    # ---------------------------------------------------------------------
    # Start timing for full evaluation
    # ---------------------------------------------------------------------
    start_time = time.time()

    # ---------------------------------------------------------------------
    # Iterate over all dataset samples
    # Each sample contains:
    #   - text: review or sentence
    #   - expected: list of known aspect-sentiment pairs
    # ---------------------------------------------------------------------
    for sample in samples:
        text = sample["text"]
        expected = sample["expected"]

        print("\n────────────────────────────────────")
        print(f"Text: {text}")

        # Run analyzer using the unified API
        results = analyzer.analyze(text)

        # Record time for each individual sample (optional)
        sample_start = time.time()
        results = analyzer.analyze(text)
        sample_end = time.time()

        sample_duration = sample_end - sample_start
        print(f" Processing time: {sample_duration:.2f}s")

        # Skip gracefully if model fails to produce any output
        if not results:
            print("  ⚠️ No aspects detected.")
            continue

        # -------------------------------------------------------------
        # Compare predicted results with expected reference data
        # -------------------------------------------------------------
        for exp in expected:
            total += 1
            exp_aspect = exp["aspect"]
            exp_sent = exp["sentiment"]

            matched = False

            # Search for a predicted aspect that matches expected aspect
            for pred in results:
                if aspect_match(exp_aspect, pred.aspect):
                    matched = True
                    pred_sent = pred.sentiment
                    correct_flag = exp_sent == pred_sent
                    correct += correct_flag

                    print(
                        f"  Aspect: {exp_aspect:15s} | Expected: {exp_sent:8s} | Predicted: {pred_sent:8s} | {'✅' if correct_flag else '❌'}")

                    # Optional debugging info — depends on which ABSA model was used
                    if hasattr(pred, "confidence") and pred.confidence is not None:
                        print(f"     Confidence: {pred.confidence:.2f}")
                    if hasattr(pred, "vader_breakdown") and pred.vader_breakdown:
                        print(f"     → VADER scores: {pred.vader_breakdown}")
                    break

            # If no matching aspect was found
            if not matched:
                print(f"  Aspect: {exp_aspect:15s} | Expected: {exp_sent:8s} | Predicted: missing | ❌")

    # ---------------------------------------------------------------------
    # Compute global accuracy metric
    # ---------------------------------------------------------------------\
    end_time = time.time()
    total_duration = end_time - start_time
    avg_duration = total_duration / len(samples) if samples else 0
    accuracy = correct / total if total else 0.0

    print("\n────────────────────────────────────")
    print(f"Overall accuracy: {accuracy:.2%} ({correct}/{total} correct)")
    print(f"Total runtime: {total_duration:.2f} seconds")
    print(f"Average time per sample: {avg_duration:.2f} seconds")
    print("====================================\n")


# Entry point for standalone script execution
if __name__ == "__main__":
    main()
