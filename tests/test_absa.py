from src.leixcon_absa import LexiconABSA

analyzer = LexiconABSA()

text = "KdG is a great school, but it is hard sometimes."
results = analyzer.analyze(text)

print(results)