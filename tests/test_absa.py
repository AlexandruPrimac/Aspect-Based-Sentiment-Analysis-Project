from src.leixcon_absa import LexiconABSA

analyzer = LexiconABSA()

texts = [
    "The pizza was delicious but the service was terrible.",
    "The pizza was not good.",
    "The service isn't bad.",
    "He playfully good football.",
    "The pizza was not very good but the service was extremely bad."
]

for t in texts:
    print("\nText:", t)
    for r in analyzer.analyze(t):
        print(f"Aspect: {r.aspect:10s} | Sentiment: {r.sentiment:8s} | Confidence: {r.confidence:.2f}")
