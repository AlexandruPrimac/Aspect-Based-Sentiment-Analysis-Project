from src.leixcon_absa import LexiconABSA

analyzer = LexiconABSA()

# text = "KdG is a great school, but it is hard sometimes."
text = "The pizza was delicious but the service was terrible."
results = analyzer.analyze(text)

for r in results:
    print(f"Aspect: {r.aspect:10s} | Sentiment: {r.sentiment:8s} | Confidence: {r.confidence:.2f}")
