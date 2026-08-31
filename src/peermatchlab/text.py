"""Small deterministic text-vector utilities with no runtime dependencies."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "using",
        "with",
    }
)


def tokenize(text: str, *, stopwords: frozenset[str] = _DEFAULT_STOPWORDS) -> tuple[str, ...]:
    """Tokenize Unicode text using case folding and conservative punctuation removal."""

    return tuple(
        token
        for token in _TOKEN_PATTERN.findall(text.casefold())
        if len(token) > 1 and token not in stopwords
    )


def cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Compute sparse cosine similarity, returning zero for an empty vector."""

    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


@dataclass(frozen=True, slots=True)
class TfIdfSpace:
    """A fitted TF-IDF vocabulary for a closed matching run."""

    inverse_document_frequency: Mapping[str, float]

    @classmethod
    def fit(cls, texts: Iterable[str]) -> TfIdfSpace:
        """Fit smooth IDF values over the provided evidence corpus."""

        documents = [set(tokenize(text)) for text in texts]
        document_count = len(documents)
        frequencies: Counter[str] = Counter()
        for terms in documents:
            frequencies.update(terms)
        idf = {
            term: math.log((1 + document_count) / (1 + count)) + 1.0
            for term, count in frequencies.items()
        }
        return cls(inverse_document_frequency=idf)

    def transform(self, text: str) -> dict[str, float]:
        """Create a sublinear-TF, IDF-weighted sparse vector."""

        counts = Counter(tokenize(text))
        return {
            term: (1.0 + math.log(count)) * self.inverse_document_frequency[term]
            for term, count in counts.items()
            if term in self.inverse_document_frequency
        }


def set_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    """Return Jaccard similarity after Unicode-aware normalization."""

    left_set = {value.casefold().strip() for value in left if value.strip()}
    right_set = {value.casefold().strip() for value in right if value.strip()}
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)
