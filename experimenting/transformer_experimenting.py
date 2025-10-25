"""
In this directory I mainly tried to experiment how spacy, vader and the transformer works, before working on the actual implementation. Trying to inspect the code and output, so that I can understand how it works.
"""

from transformers import pipeline

classifier = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")
sentence = "The food was exceptional, although the service was a bit slow."

sentence = "The food was exceptional, although the service was a bit slow."
aspects = ["food", "service"]

for aspect in aspects:
    combined = f"{sentence} [SEP] {aspect}"
    res = classifier(combined)
    print(f"{aspect} → {res[0]['label']} ({res[0]['score']:.2f})")
