from src.leixcon_absa import LexiconABSA

analyzer = LexiconABSA()

texts = [
    "The pizza was 💘 delicious but the service was terrible.",
    "The pizza was not good.",
    "The service isn't bad.",
    "He playfully good football.",
    "The pizza was not very good but the service was extremely bad.",
    "The pizza was not very good :(, but the service was extremely bad.",

]

for t in texts:
    print(f"\nText: {t}")
    results = analyzer.analyze(t)
    for r in results:
        print(f"\nAspect: {r.aspect:10s} | Sentiment: {r.sentiment:8s} | Confidence: {r.confidence:.2f}")
        print(f"   VADER scores: {r.vader_breakdown}\n")
