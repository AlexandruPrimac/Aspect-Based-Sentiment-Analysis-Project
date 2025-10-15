"""
Runs the ABSA analyzer on the test_samples.json dataset
and compares results with expected sentiments.
Also prints full VADER sentiment breakdown for transparency.
"""

import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lexicon_absa import LexiconABSA


def main():
    # Locate and load dataset
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_path = os.path.join(project_root, "data", "test_samples.json")

    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    analyzer = LexiconABSA()

    total = 0
    correct = 0

    print("\n=== Running Dataset Evaluation ===")

    for sample in samples:
        text = sample["text"]
        expected = sample["expected"]

        print("\n────────────────────────────────────")
        print(f"Text: {text}")

        # Run your analyzer
        results = analyzer.analyze(text)

        # Make aspect → sentiment map
        result_map = {r.aspect.lower(): r for r in results}

        # Print details per aspect
        for exp in expected:
            aspect = exp["aspect"].lower()
            exp_sent = exp["sentiment"]

            # Get result object if exists
            pred_result = result_map.get(aspect)
            if pred_result:
                pred_sent = pred_result.sentiment
                vader = pred_result.vader_breakdown
                print(f"  Aspect: {aspect:15s} | Expected: {exp_sent:8s} | Predicted: {pred_sent:8s} | {'✅' if exp_sent == pred_sent else '❌'}")
                print(f"     → VADER scores: {vader}")
            else:
                pred_sent = "missing"
                print(f"  Aspect: {aspect:15s} | Expected: {exp_sent:8s} | Predicted: missing | ❌")

            total += 1
            correct += int(exp_sent == pred_sent)

    accuracy = correct / total if total else 0
    print("\n────────────────────────────────────")
    print(f"Overall accuracy: {accuracy:.2%} ({correct}/{total} correct)")
    print("====================================\n")


if __name__ == "__main__":
    main()
