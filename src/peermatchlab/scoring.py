"""Explainable content and preference scoring."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from peermatchlab.models import Conflict, Document, Expert, MatchScore
from peermatchlab.text import TfIdfSpace, cosine, set_overlap


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """Non-negative weights normalized by :class:`MatchScorer`."""

    content: float = 0.50
    topics: float = 0.20
    bid: float = 0.15
    recency: float = 0.10
    seniority: float = 0.05

    def __post_init__(self) -> None:
        values = (self.content, self.topics, self.bid, self.recency, self.seniority)
        if any(not _finite_number(value) for value in values):
            raise ValueError("score weights must be finite numbers")
        if any(value < 0 for value in values):
            raise ValueError("score weights must not be negative")
        if sum(values) == 0:
            raise ValueError("at least one score weight must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> ScoreWeights:
        """Create weights from a strict component mapping."""

        if not isinstance(values, Mapping):
            raise ValueError("score weights must be a mapping")
        if any(not isinstance(name, str) for name in values):
            raise ValueError("score weight names must be strings")
        allowed = ("content", "topics", "bid", "recency", "seniority")
        unknown = set(values) - set(allowed)
        if unknown:
            raise ValueError(f"unknown score weights: {sorted(unknown)}")
        return cls(**{name: values.get(name, 0.0) for name in allowed})

    def normalized(self) -> dict[str, float]:
        """Return weights whose sum is exactly one within floating-point precision."""

        values = {
            "content": self.content,
            "topics": self.topics,
            "bid": self.bid,
            "recency": self.recency,
            "seniority": self.seniority,
        }
        total = sum(values.values())
        return {name: value / total for name, value in values.items()}


class MatchScorer:
    """Fit a run-local text space and score every eligible pair."""

    def __init__(
        self,
        documents: Iterable[Document],
        experts: Iterable[Expert],
        *,
        conflicts: Iterable[Conflict] = (),
        weights: ScoreWeights | None = None,
        current_year: int = 2026,
        publication_half_life: float = 6.0,
    ) -> None:
        document_items = tuple(documents)
        expert_items = tuple(experts)
        self.documents = {document.id: document for document in document_items}
        self.experts = {expert.id: expert for expert in expert_items}
        if len(self.documents) == 0:
            raise ValueError("at least one document is required")
        if len(self.experts) == 0:
            raise ValueError("at least one expert is required")
        if len(self.documents) != len(document_items):
            raise ValueError("document identifiers must be unique")
        if len(self.experts) != len(expert_items):
            raise ValueError("expert identifiers must be unique")
        if not _finite_number(publication_half_life) or publication_half_life <= 0:
            raise ValueError("publication_half_life must be a positive finite number")
        if isinstance(current_year, bool) or not isinstance(current_year, int):
            raise ValueError("current_year must be an integer")
        if not 1800 <= current_year <= 2200:
            raise ValueError("current_year must be between 1800 and 2200")
        self.current_year = current_year
        self.publication_half_life = publication_half_life
        if weights is not None and not isinstance(weights, ScoreWeights):
            raise ValueError("weights must be a ScoreWeights instance")
        self.weights = (weights or ScoreWeights()).normalized()
        conflict_items = tuple(conflicts)
        unknown_documents = sorted(
            {item.document_id for item in conflict_items if item.document_id not in self.documents}
        )
        unknown_experts = sorted(
            {item.expert_id for item in conflict_items if item.expert_id not in self.experts}
        )
        if unknown_documents or unknown_experts:
            raise ValueError(
                "conflicts reference unknown identifiers: "
                f"documents={unknown_documents}, experts={unknown_experts}"
            )
        self.conflicts = {
            (conflict.document_id, conflict.expert_id): conflict.reason
            for conflict in conflict_items
        }
        all_text = [document.text for document in self.documents.values()]
        all_text.extend(expert.text for expert in self.experts.values())
        self.text_space = TfIdfSpace.fit(all_text)
        self.document_vectors = {
            key: self.text_space.transform(document.text)
            for key, document in self.documents.items()
        }
        self.expert_vectors = {
            key: self.text_space.transform(expert.text) for key, expert in self.experts.items()
        }

    def _recency(self, expert: Expert) -> float:
        dated = [item.year for item in expert.publications if item.year is not None]
        if not dated:
            return 0.0
        return sum(
            math.pow(0.5, max(0, self.current_year - year) / self.publication_half_life)
            for year in dated
        ) / len(dated)

    def score(self, document_id: str, expert_id: str) -> MatchScore:
        """Score one pair and retain a human-readable evidence summary."""

        document = self.documents[document_id]
        expert = self.experts[expert_id]
        conflict_reason = self.conflicts.get((document_id, expert_id))
        if conflict_reason is not None:
            return MatchScore(
                document_id=document_id,
                expert_id=expert_id,
                total=0.0,
                content=0.0,
                topics=0.0,
                bid=0.0,
                recency=0.0,
                seniority=expert.seniority,
                eligible=False,
                reasons=(f"excluded: {conflict_reason}",),
            )
        if expert.capacity == 0:
            return MatchScore(
                document_id=document_id,
                expert_id=expert_id,
                total=0.0,
                content=0.0,
                topics=0.0,
                bid=0.0,
                recency=0.0,
                seniority=expert.seniority,
                eligible=False,
                reasons=("excluded: zero capacity",),
            )

        content = cosine(self.document_vectors[document_id], self.expert_vectors[expert_id])
        topics = set_overlap(
            (*document.topics, *document.keywords), (*expert.topics, *expert.keywords)
        )
        raw_bid = expert.bids.get(document_id, 0.0)
        bid = (raw_bid + 1.0) / 2.0
        recency = self._recency(expert)
        components = {
            "content": content,
            "topics": topics,
            "bid": bid,
            "recency": recency,
            "seniority": expert.seniority,
        }
        total = sum(components[name] * self.weights[name] for name in components)
        reasons: list[str] = []
        shared_topics = sorted(
            {value.casefold() for value in document.topics}
            & {value.casefold() for value in expert.topics}
        )
        if shared_topics:
            reasons.append("shared topics: " + ", ".join(shared_topics[:4]))
        if raw_bid > 0:
            reasons.append(f"positive preference ({raw_bid:.2f})")
        elif raw_bid < 0:
            reasons.append(f"negative preference ({raw_bid:.2f})")
        if content >= 0.5:
            reasons.append(f"strong text similarity ({content:.2f})")
        if not reasons:
            reasons.append("eligible with limited direct evidence")
        return MatchScore(
            document_id=document_id,
            expert_id=expert_id,
            total=total,
            content=content,
            topics=topics,
            bid=bid,
            recency=recency,
            seniority=expert.seniority,
            reasons=tuple(reasons),
        )

    def matrix(self) -> tuple[MatchScore, ...]:
        """Score all pairs in deterministic document/expert order."""

        return tuple(
            self.score(document_id, expert_id)
            for document_id in sorted(self.documents)
            for expert_id in sorted(self.experts)
        )
