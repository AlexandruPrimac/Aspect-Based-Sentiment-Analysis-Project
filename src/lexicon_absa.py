# src/lexicon_absa.py
"""
Lexicon-based ABSA implementation using spaCy + VADER.
- Extracts noun phrases as aspects
- Associates each aspect with nearby adjectives or verbs
- Handles negations, intensifiers, softenings, and emojis
- Aggregates results for cleaner structured output
"""

from dataclasses import dataclass
from typing import List
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.base import ABSAAnalyzer, AspectSentiment


class LexiconABSA(ABSAAnalyzer):
    def __init__(self, debug: bool = False):
        self.nlp = spacy.load("en_core_web_sm")
        self.vader = SentimentIntensityAnalyzer()
        self.debug = debug

        # Negations, intensifiers, and softeners
        self.negations = {"not", "no", "never", "n't"}
        self.intensifiers = {"very", "extremely", "really", "so", "super", "highly", "too"}
        self.softeners = {"slightly", "somewhat", "barely", "a bit", "kind of", "rather"}

        # Emoji mapping for pre-normalization
        self.emoji_map = {
            "💘": "love", "❤️": "love", "💕": "love", "😍": "love", "😘": "love",
            "😡": "angry", "🤬": "furious", "😢": "sad", "😭": "crying", "😂": "laughing",
            "🤣": "laughing", "😔": "sad", "🙂": "smile", "😊": "smile", ":)": "smile",
            ":(": "sad", ":/": "disappointed", "😒": "annoyed", "😩": "tired",
            "😃": "happy", "😆": "happy"
        }

    # -------------------------- #
    #  Main ABSA Analysis Method #
    # -------------------------- #
    def analyze(self, text: str) -> List[AspectSentiment]:
        # Replace emojis with mapped words for lexical analyzers
        for emo, word in self.emoji_map.items():
            text = text.replace(emo, f" {word} ")

        doc = self.nlp(text)
        results = []

        # STEP 1 — Aspect extraction via noun chunks
        aspects = []
        for chunk in doc.noun_chunks:
            if not any(w.text.lower() in self.emoji_map.values() for w in chunk):
                aspects.append(chunk)

        if self.debug:
            print(f"\nProcessed text: {text}")
            print("Extracted aspects:", [a.text for a in aspects])

        # STEP 2 — Find related opinion words using dependency parsing
        pairs = []
        for token in doc:
            # adjectival modifier: good food
            if token.dep_ == "amod" and token.head.pos_ in ("NOUN", "PROPN"):
                pairs.append((token.head, token))
            # adjectival complement: food is good
            elif token.dep_ == "acomp" and token.head.pos_ in ("VERB", "AUX"):
                for subj in token.head.children:
                    if subj.dep_ in ("nsubj", "nsubjpass"):
                        pairs.append((subj, token))
            # verb opinions: camera disappoints
            elif token.pos_ == "VERB":
                for subj in token.children:
                    if subj.dep_ in ("nsubj", "nsubjpass") and subj.pos_ in ("NOUN", "PROPN"):
                        pairs.append((subj, token))

        if self.debug:
            print("Aspect-opinion pairs:", [(a.text, o.text) for a, o in pairs])

        # STEP 3 — Compute sentiment for each aspect–opinion pair
        for aspect, opinion in pairs:
            opinion_phrase = " ".join(t.text for t in opinion.subtree)
            vader_scores = self.vader.polarity_scores(opinion_phrase)
            vs = vader_scores["compound"]

            # Intensifiers / softeners near the opinion
            modifiers = [t for t in doc if t.pos_ == "ADV" and abs(t.i - opinion.i) <= 3]
            for adv in modifiers:
                adv_lower = adv.text.lower()
                if adv_lower in self.intensifiers:
                    vs *= 1.2
                    if self.debug:
                        print(f"  Intensifier near '{opinion.text}': {adv.text} (+20%)")
                elif adv_lower in self.softeners:
                    vs *= 0.8
                    if self.debug:
                        print(f"  Softener near '{opinion.text}': {adv.text} (-20%)")

            # Negation detection via dependency relations
            if self._has_negation(opinion):
                vs = -vs
                if self.debug:
                    print(f"  Negation flips sentiment near '{opinion.text}'")

            # Clamp and classify sentiment
            vs = max(min(vs, 1.0), -1.0)
            if vs > 0.3:
                sentiment = "positive"
            elif vs < -0.3:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            confidence = min(1.0, abs(vs) + 0.1 * len(modifiers))

            # Handle Token vs Span differences
            if hasattr(aspect, "start_char"):
                start = aspect.start_char
                end = aspect.end_char
            else:
                start = aspect.idx
                end = aspect.idx + len(aspect.text)

            result = AspectSentiment(
                aspect=aspect.text,
                sentiment=sentiment,
                confidence=confidence,
                text_span=(start, end),
                vader_breakdown=vader_scores
            )
            results.append(result)

        # STEP 4 — Aggregate by aspect (keep highest-confidence sentiment)
        final_results = self._aggregate_results(results)

        if self.debug:
            print("\nFinal aggregated results:")
            for r in final_results:
                print(f"  {r.aspect:<15} → {r.sentiment} ({r.confidence:.2f})")

        return final_results

    # -------------------------- #
    #      Helper Functions      #
    # -------------------------- #
    def _has_negation(self, token):
        """Detect if a token or its head has a negation dependency nearby."""
        if any(child.dep_ == "neg" for child in token.children):
            return True
        if token.head is not None and any(child.dep_ == "neg" for child in token.head.children):
            return True
        if token.i > 0 and token.doc[token.i - 1].lower_ in self.negations:
            return True
        return False

    def _aggregate_results(self, results: List[AspectSentiment]) -> List[AspectSentiment]:
        """Aggregate multiple opinions for same aspect, keep highest confidence."""
        merged = {}
        for r in results:
            key = r.aspect.lower()
            if key not in merged or r.confidence > merged[key].confidence:
                merged[key] = r
        return list(merged.values())
