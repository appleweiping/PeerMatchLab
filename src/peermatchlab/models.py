"""Domain models shared by scoring, assignment, and reporting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


class DataValidationError(ValueError):
    """Raised when an input object cannot participate in matching."""


def _clean_terms(values: tuple[str, ...]) -> tuple[str, ...]:
    if any(not isinstance(value, str) for value in values):
        raise DataValidationError("terms must be strings")
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise DataValidationError("mapping value must be an object")
    return MappingProxyType(dict(values))


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


@dataclass(frozen=True, slots=True)
class Publication:
    """A dated text record used as evidence for an expert profile."""

    title: str
    abstract: str = ""
    year: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not isinstance(self.abstract, str):
            raise DataValidationError("publication title and abstract must be strings")
        if not self.title.strip():
            raise DataValidationError("publication title must not be empty")
        if self.year is not None:
            if isinstance(self.year, bool) or not isinstance(self.year, int):
                raise DataValidationError("publication year must be an integer")
            if not 1800 <= self.year <= 2200:
                raise DataValidationError("publication year must be between 1800 and 2200")


@dataclass(frozen=True, slots=True)
class Document:
    """A document that needs one or more qualified experts."""

    id: str
    title: str
    abstract: str = ""
    topics: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    required_experts: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) for value in (self.id, self.title, self.abstract)):
            raise DataValidationError("document id, title, and abstract must be strings")
        if not self.id.strip():
            raise DataValidationError("document id must not be empty")
        if not self.title.strip():
            raise DataValidationError(f"document {self.id!r} must have a title")
        if self.required_experts is not None:
            if isinstance(self.required_experts, bool) or not isinstance(
                self.required_experts, int
            ):
                raise DataValidationError("required_experts must be an integer")
            if self.required_experts < 1:
                raise DataValidationError("required_experts must be positive")
        object.__setattr__(self, "topics", _clean_terms(self.topics))
        object.__setattr__(self, "keywords", _clean_terms(self.keywords))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def text(self) -> str:
        """Return the canonical text used for content scoring."""

        return " ".join((self.title, self.abstract, *self.topics, *self.keywords))


@dataclass(frozen=True, slots=True)
class Expert:
    """An expert candidate with capacity and optional preference signals."""

    id: str
    name: str
    summary: str = ""
    topics: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    publications: tuple[Publication, ...] = ()
    capacity: int = 3
    institution: str | None = None
    regions: tuple[str, ...] = ()
    seniority: float = 0.5
    bids: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) for value in (self.id, self.name, self.summary)):
            raise DataValidationError("expert id, name, and summary must be strings")
        if not self.id.strip():
            raise DataValidationError("expert id must not be empty")
        if not self.name.strip():
            raise DataValidationError(f"expert {self.id!r} must have a name")
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise DataValidationError("expert capacity must be an integer")
        if self.capacity < 0:
            raise DataValidationError("expert capacity must not be negative")
        if not _finite_number(self.seniority):
            raise DataValidationError("expert seniority must be a finite number")
        if not 0.0 <= self.seniority <= 1.0:
            raise DataValidationError("expert seniority must be in [0, 1]")
        if self.institution is not None and not isinstance(self.institution, str):
            raise DataValidationError("expert institution must be a string or null")
        if any(not isinstance(publication, Publication) for publication in self.publications):
            raise DataValidationError("expert publications must contain Publication objects")
        if not isinstance(self.bids, Mapping):
            raise DataValidationError("expert bids must be an object")
        if any(not isinstance(key, str) or not key.strip() for key in self.bids):
            raise DataValidationError("bid document identifiers must be non-empty strings")
        if any(not _finite_number(value) for value in self.bids.values()):
            raise DataValidationError("bid values must be finite numbers")
        if any(not -1.0 <= value <= 1.0 for value in self.bids.values()):
            raise DataValidationError("bid values must be in [-1, 1]")
        object.__setattr__(self, "topics", _clean_terms(self.topics))
        object.__setattr__(self, "keywords", _clean_terms(self.keywords))
        object.__setattr__(self, "regions", _clean_terms(self.regions))
        object.__setattr__(self, "bids", _freeze_mapping(self.bids))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def text(self) -> str:
        """Return a stable text representation of all expertise evidence."""

        publication_text = " ".join(
            f"{publication.title} {publication.abstract}" for publication in self.publications
        )
        return " ".join((self.summary, *self.topics, *self.keywords, publication_text)).strip()


@dataclass(frozen=True, slots=True)
class Conflict:
    """A hard exclusion between a document and an expert."""

    document_id: str
    expert_id: str
    reason: str = "declared conflict"

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) for value in (self.document_id, self.expert_id, self.reason)
        ):
            raise DataValidationError("conflict fields must be strings")
        if not self.document_id.strip() or not self.expert_id.strip():
            raise DataValidationError("conflict identifiers must not be empty")


@dataclass(frozen=True, slots=True)
class MatchScore:
    """A decomposed document-expert score suitable for audit and explanation."""

    document_id: str
    expert_id: str
    total: float
    content: float
    topics: float
    bid: float
    recency: float
    seniority: float
    eligible: bool = True
    reasons: tuple[str, ...] = ()
    affinity: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not isinstance(self.expert_id, str):
            raise DataValidationError("score identifiers must be strings")
        values = (self.total, self.content, self.topics, self.bid, self.recency, self.seniority)
        if any(not _finite_number(value) for value in values):
            raise DataValidationError("score values must be finite numbers")
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise DataValidationError("score values must be in [0, 1]")
        if self.affinity is not None:
            if not _finite_number(self.affinity) or not 0.0 <= self.affinity <= 1.0:
                raise DataValidationError("affinity score must be a finite number in [0, 1]")
            object.__setattr__(self, "affinity", float(self.affinity))
        if not isinstance(self.eligible, bool):
            raise DataValidationError("eligible must be a boolean")
        if any(not isinstance(reason, str) for reason in self.reasons):
            raise DataValidationError("score reasons must be strings")

    def component_map(self) -> dict[str, float]:
        """Return score components in a serialization-friendly form."""

        components = {
            "content": self.content,
            "topics": self.topics,
            "bid": self.bid,
            "recency": self.recency,
            "seniority": self.seniority,
        }
        if self.affinity is not None:
            components["affinity"] = self.affinity
        return components


@dataclass(frozen=True, slots=True)
class Assignment:
    """A selected expert for a document, including the evidence score."""

    document_id: str
    expert_id: str
    score: float
    rank: int
    components: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not isinstance(self.expert_id, str):
            raise DataValidationError("assignment identifiers must be strings")
        if not self.document_id.strip() or not self.expert_id.strip():
            raise DataValidationError("assignment identifiers must not be empty")
        if not _finite_number(self.score):
            raise DataValidationError("assignment score must be a finite number")
        if not 0.0 <= self.score <= 1.0:
            raise DataValidationError("assignment score must be in [0, 1]")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise DataValidationError("assignment rank must be an integer")
        if self.rank < 1:
            raise DataValidationError("assignment rank must be positive")
        if not isinstance(self.components, Mapping):
            raise DataValidationError("assignment components must be an object")
        if any(not isinstance(key, str) for key in self.components):
            raise DataValidationError("assignment component names must be strings")
        if any(not _finite_number(value) for value in self.components.values()):
            raise DataValidationError("assignment components must be finite numbers")
        if any(not 0.0 <= value <= 1.0 for value in self.components.values()):
            raise DataValidationError("assignment components must be in [0, 1]")
        object.__setattr__(self, "components", _freeze_mapping(self.components))


@dataclass(frozen=True, slots=True)
class MatchPlan:
    """Complete assignment output with explicit unmet demand."""

    assignments: tuple[Assignment, ...]
    unmet: Mapping[str, int]
    strategy: str
    total_score: float

    def __post_init__(self) -> None:
        if any(not isinstance(item, Assignment) for item in self.assignments):
            raise DataValidationError("plan assignments must contain Assignment objects")
        if not isinstance(self.unmet, Mapping):
            raise DataValidationError("unmet must be an object")
        if any(not isinstance(key, str) or not key.strip() for key in self.unmet):
            raise DataValidationError("unmet document identifiers must be non-empty strings")
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in self.unmet.values()
        ):
            raise DataValidationError("unmet counts must be integers")
        if any(value < 0 for value in self.unmet.values()):
            raise DataValidationError("unmet counts must not be negative")
        if not isinstance(self.strategy, str) or not self.strategy.strip():
            raise DataValidationError("plan strategy must be a non-empty string")
        if not _finite_number(self.total_score):
            raise DataValidationError("plan total_score must be a finite number")
        calculated_total = sum(item.score for item in self.assignments)
        if not math.isclose(self.total_score, calculated_total, rel_tol=1e-12, abs_tol=1e-12):
            raise DataValidationError("plan total_score must equal the assignment score sum")
        object.__setattr__(self, "unmet", _freeze_mapping(self.unmet))

    def for_document(self, document_id: str) -> tuple[Assignment, ...]:
        """Return assignments for one document in display rank order."""

        return tuple(
            sorted(
                (item for item in self.assignments if item.document_id == document_id),
                key=lambda item: item.rank,
            )
        )
