# ------------------------------------------------------------- #
# Helper Functions
# ------------------------------------------------------------- #
from typing import List

from src.base import AspectSentiment


def has_negation(token, negation_words=None):
    """Detect if a token or its syntactic head is negated."""
    negation_words = negation_words or {"not", "no", "never", "n't"}
    if any(child.dep_ == "neg" for child in token.children):
        return True
    if token.head and any(child.dep_ == "neg" for child in token.head.children):
        return True
    if token.i > 0 and token.doc[token.i - 1].lower_ in negation_words:
        return True
    return False


def aggregate_results(results: List[AspectSentiment]) -> List[AspectSentiment]:
    """
    Merge multiple AspectSentiment entries for the same aspect
    and keep only the highest-confidence one.
    """
    merged = {}
    for r in results:
        key = r.aspect.lower()
        if key not in merged or r.confidence > merged[key].confidence:
            merged[key] = r
    return list(merged.values())
