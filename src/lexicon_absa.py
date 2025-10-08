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
    vader_breakdown: dict = None


class LexiconABSA:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")  # Load English tokenizer, tagger, parser and NER
        self.vader = SentimentIntensityAnalyzer()  # Load Vader sentiment analyzer
        self.negations = {"not", "no", "never", "n't"}  # Common negetions (maybe add more later)
        self.emoji_map = {
            "💘": "love",
            "❤️": "love",
            "😡": "angry",
            "😢": "sad",
            "😂": "laughing",
            ":(": "sad",
            ":)": "happy",
            "😔": "sad"
        }

    def analyze(self, text: str) -> List[AspectSentiment]:
        for emo, word in self.emoji_map.items():
            text = text.replace(emo, f" {word} ")

        doc = self.nlp(text)
        results = []

        # Step 1: I identify the aspects
        aspects = [
            token for token in doc
            if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in self.emoji_map.values()
        ]

        print(f"Processed text: {text}")
        print("Noun phrases:", aspects)

        for aspect in aspects:
            # Step 2: I find the adjectives, adverbs or verbs related
            opinion_options = []
            related_verbs = []
            modifiers = []
            negations_found = []

            # Direct adjectives next to the aspect
            opinion_options += [t for t in aspect.lefts if t.pos_ == "ADJ"]
            opinion_options += [t for t in aspect.rights if t.pos_ == "ADJ"]

            # If the aspect is subject of a verb (nsubj), I check the verb’s children for adjectives
            if aspect.dep_ == "nsubj" and aspect.head.pos_ in ["VERB", "AUX"]:
                related_verbs.append(aspect.head)
                opinion_options += [t for t in aspect.head.children if t.pos_ == "ADJ"]

            # If the aspect's *head* is an adjective (copular construction), I use it directly
            if aspect.head.pos_ == "ADJ":
                opinion_options.append(aspect.head)

            # Collect nearby adverbs (modifiers)
            for token in doc:
                if token.pos_ == "ADV" and abs(token.i - aspect.i) <= 3:
                    modifiers.append(token)

            # Prints
            print(f"\n--- Aspect: '{aspect.text}' ---")
            print(f"  Related verbs:      {[t.text for t in related_verbs]}")
            print(f"  Nearby adverbs:     {[t.text for t in modifiers]}")
            print(f"  Related adjectives: {[t.text for t in opinion_options]}")

            # Step 3: I get sentiment for each opinion word
            for opinion in opinion_options:
                vader_scores = self.vader.polarity_scores(opinion.text)
                vs = vader_scores["compound"]  # keep all scores for results

                #  Intensifier & softener detection (adverb modifiers) !!!! idk if correct
                for adv in modifiers:
                    adv_lower = adv.text.lower()
                    if adv_lower in ["very", "extremely", "really", "so", "super", "highly"]:
                        vs *= 1.2
                        print(f"  Intensifier found near '{opinion.text}': {adv.text} (+20%)")
                    elif adv_lower in ["slightly", "somewhat", "barely", "a bit", "kind of"]:
                        vs *= 0.8
                        print(f"  Softener found near '{opinion.text}': {adv.text} (-20%)")

                # Check if negation near opinion
                negs = [child.text for child in opinion.children if child.lower_ in self.negations]
                if opinion.i > 0 and doc[opinion.i - 1].lower_ in self.negations:
                    negs.append(doc[opinion.i - 1].text)
                if negs:
                    vs = -vs
                    print(f"  Negation affecting '{opinion.text}': {negs}")

                sentiment = (
                    "positive" if vs > 0.05 else "negative" if vs < -0.05 else "neutral"
                )

                results.append(
                    AspectSentiment(
                        aspect=aspect.text,
                        sentiment=sentiment,
                        confidence=abs(vs),
                        text_span=(aspect.idx, aspect.idx + len(aspect.text)),
                        vader_breakdown=vader_scores
                    )
                )

        return results
