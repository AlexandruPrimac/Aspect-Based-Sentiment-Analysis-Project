# Data class for structured resuls
from dataclasses import dataclass
from typing import List

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class AspectSentiment:
    aspect: str
    sentiment: str
    confidence: float
    text_span: tuple = None


class LexiconABSA:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")  # Load English tokenizer, tagger, parser and NER
        self.vader = SentimentIntensityAnalyzer()  # Load Vader sentiment analyzer

    def analyze(self, text: str) -> List[AspectSentiment]:
        doc = self.nlp(text)
        results = []

        # Step 1: I identify aspects (simple: all nouns)
        aspects = [token for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        print("Noun phrases:", aspects)

        for aspect in aspects:
            # Step 2: Find adjectives or verbs related
            opinion_options = []

            # Direct adjectives next to the aspect
            opinion_options += [t for t in aspect.lefts if t.pos_ == "ADJ"]
            opinion_options += [t for t in aspect.rights if t.pos_ == "ADJ"]

            # If the aspect is subject of a verb (nsubj), I check the verb’s children for adjectives
            if aspect.dep_ == "nsubj" and aspect.head.pos_ in ["VERB", "AUX"]:
                opinion_options += [t for t in aspect.head.children if t.pos_ == "ADJ"]

            # If the aspect's *head* is an adjective (copular construction), I use it directly
            if aspect.head.pos_ == "ADJ":
                opinion_options.append(aspect.head)

            print(f"Aspect '{aspect.text}' → Adjectives found:", opinion_options)

            # Step 3: I get sentiment for each opinion word
            for opinion in opinion_options:
                vs = self.vader.polarity_scores(opinion.text)["compound"]
                sentiment = (
                    "positive" if vs > 0.05 else "negative" if vs < -0.05 else "neutral"
                )
                results.append(
                    AspectSentiment(
                        aspect=aspect.text,
                        sentiment=sentiment,
                        confidence=abs(vs),
                        text_span=(aspect.idx, aspect.idx + len(aspect.text)),
                    )
                )
        return results
