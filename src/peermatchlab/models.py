"""Domain models shared by scoring, assignment, and reporting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class DataValidationError(ValueError):
    """Raised when an input object cannot participate in matching."""


def _identifier_is_valid(value: object) -> bool:
    """Return whether an identifier is stable in JSON, JSONL, and CSV artifacts."""

    return isinstance(value, str) and bool(value) and value == value.strip() and value.isprintable()


class FeasibilityStatus(StrEnum):
    """What one assignment run proves about its complete demand."""

    SATISFIED = "satisfied"
    INFEASIBLE = "infeasible"
    NOT_CERTIFIED = "not_certified"


class UnmetReason(StrEnum):
    """Stable machine-readable evidence associated with unmet demand."""

    HARD_CONFLICT = "hard_conflict"
    OTHER_INELIGIBLE = "other_ineligible"
    ZERO_CAPACITY = "zero_capacity"
    MINIMUM_SCORE = "minimum_score"
    SPARSE_SCORE_MATRIX = "sparse_score_matrix"
    CANDIDATE_SCARCITY = "candidate_scarcity"
    EXPERT_CAPACITY = "expert_capacity"
    SENIORITY_FLOOR = "seniority_floor"
    INSTITUTION_DIVERSITY = "institution_diversity"
    GLOBAL_CAPACITY_COUPLING = "global_capacity_coupling"
    GREEDY_NOT_CERTIFIED = "greedy_not_certified"


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
    id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not isinstance(self.abstract, str):
            raise DataValidationError("publication title and abstract must be strings")
        if not self.title.strip():
            raise DataValidationError("publication title must not be empty")
        if self.id is not None and not _identifier_is_valid(self.id):
            raise DataValidationError(
                "publication id must be null or a non-empty printable string without "
                "surrounding whitespace"
            )
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
        if not _identifier_is_valid(self.id):
            raise DataValidationError(
                "document id must not be empty or contain surrounding whitespace/control characters"
            )
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
        if not _identifier_is_valid(self.id):
            raise DataValidationError(
                "expert id must not be empty or contain surrounding whitespace/control characters"
            )
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
        if any(not _identifier_is_valid(key) for key in self.bids):
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
        if not _identifier_is_valid(self.document_id) or not _identifier_is_valid(self.expert_id):
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
        if not _identifier_is_valid(self.document_id) or not _identifier_is_valid(self.expert_id):
            raise DataValidationError(
                "score identifiers must be printable strings without surrounding whitespace"
            )
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
        if not _identifier_is_valid(self.document_id) or not _identifier_is_valid(self.expert_id):
            raise DataValidationError(
                "assignment identifiers must be printable strings without surrounding whitespace"
            )
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
class DemandDiagnostic:
    """Constraint evidence for one document's requested expert slots.

    ``reason_codes`` are overlapping signals, not a minimal unsatisfiable core.
    The numeric ``evidence`` makes each signal inspectable by downstream tools.
    """

    document_id: str
    requested: int
    assigned: int
    unmet: int
    reason_codes: tuple[UnmetReason, ...] = ()
    evidence: Mapping[str, int] = field(default_factory=dict)
    saturated_experts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _identifier_is_valid(self.document_id):
            raise DataValidationError("diagnostic document_id must be a non-empty string")
        counts = (self.requested, self.assigned, self.unmet)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise DataValidationError("diagnostic demand counts must be integers")
        if self.requested < 1 or self.assigned < 0 or self.unmet < 0:
            raise DataValidationError(
                "diagnostic requested must be positive and result counts non-negative"
            )
        if self.assigned + self.unmet != self.requested:
            raise DataValidationError("diagnostic assigned plus unmet must equal requested")
        try:
            reasons = tuple(UnmetReason(value) for value in self.reason_codes)
        except (TypeError, ValueError) as error:
            raise DataValidationError("diagnostic contains an unknown reason code") from error
        if len(reasons) != len(set(reasons)):
            raise DataValidationError("diagnostic reason codes must be unique")
        if self.unmet == 0 and reasons:
            raise DataValidationError("satisfied diagnostics must not contain reason codes")
        if not isinstance(self.evidence, Mapping):
            raise DataValidationError("diagnostic evidence must be an object")
        if any(not isinstance(key, str) or not key.strip() for key in self.evidence):
            raise DataValidationError("diagnostic evidence names must be non-empty strings")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.evidence.values()
        ):
            raise DataValidationError("diagnostic evidence values must be non-negative integers")
        if any(not _identifier_is_valid(value) for value in self.saturated_experts):
            raise DataValidationError("saturated expert identifiers must be non-empty strings")
        if len(self.saturated_experts) != len(set(self.saturated_experts)):
            raise DataValidationError("saturated expert identifiers must be unique")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "evidence", _freeze_mapping(self.evidence))
        object.__setattr__(self, "saturated_experts", tuple(sorted(self.saturated_experts)))


@dataclass(frozen=True, slots=True)
class AssignmentDiagnostics:
    """Run-level feasibility result plus per-document constraint evidence."""

    status: FeasibilityStatus
    requested: int
    assigned: int
    unmet: int
    documents: tuple[DemandDiagnostic, ...]

    def __post_init__(self) -> None:
        try:
            status = FeasibilityStatus(self.status)
        except (TypeError, ValueError) as error:
            raise DataValidationError("diagnostic status is not supported") from error
        counts = (self.requested, self.assigned, self.unmet)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise DataValidationError("run diagnostic counts must be integers")
        if any(value < 0 for value in counts):
            raise DataValidationError("run diagnostic counts must not be negative")
        if self.assigned + self.unmet != self.requested:
            raise DataValidationError("run diagnostic assigned plus unmet must equal requested")
        if any(not isinstance(value, DemandDiagnostic) for value in self.documents):
            raise DataValidationError("run diagnostics must contain DemandDiagnostic objects")
        document_ids = [value.document_id for value in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise DataValidationError("run diagnostic document identifiers must be unique")
        if sum(value.requested for value in self.documents) != self.requested:
            raise DataValidationError("run diagnostic requested total is inconsistent")
        if sum(value.assigned for value in self.documents) != self.assigned:
            raise DataValidationError("run diagnostic assigned total is inconsistent")
        if status is FeasibilityStatus.SATISFIED and self.unmet:
            raise DataValidationError("satisfied diagnostic status cannot contain unmet demand")
        if status is not FeasibilityStatus.SATISFIED and not self.unmet:
            raise DataValidationError("an incomplete diagnostic status requires unmet demand")
        object.__setattr__(self, "status", status)
        ordered_documents = tuple(sorted(self.documents, key=lambda item: item.document_id))
        object.__setattr__(self, "documents", ordered_documents)

    @property
    def certified(self) -> bool:
        """Whether the result proves feasibility or global infeasibility."""

        return self.status is not FeasibilityStatus.NOT_CERTIFIED

    def for_document(self, document_id: str) -> DemandDiagnostic | None:
        """Return the diagnostic for one document, when present."""

        return next((item for item in self.documents if item.document_id == document_id), None)


@dataclass(frozen=True, slots=True)
class MatchPlan:
    """Complete assignment output with explicit unmet demand."""

    assignments: tuple[Assignment, ...]
    unmet: Mapping[str, int]
    strategy: str
    total_score: float
    diagnostics: AssignmentDiagnostics | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(item, Assignment) for item in self.assignments):
            raise DataValidationError("plan assignments must contain Assignment objects")
        if not isinstance(self.unmet, Mapping):
            raise DataValidationError("unmet must be an object")
        if any(not _identifier_is_valid(key) for key in self.unmet):
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
        if self.diagnostics is not None:
            if not isinstance(self.diagnostics, AssignmentDiagnostics):
                raise DataValidationError("plan diagnostics must be AssignmentDiagnostics or null")
            assignment_counts: dict[str, int] = {}
            for assignment in self.assignments:
                assignment_counts[assignment.document_id] = (
                    assignment_counts.get(assignment.document_id, 0) + 1
                )
            diagnostic_ids = {item.document_id for item in self.diagnostics.documents}
            if diagnostic_ids != set(assignment_counts) | set(self.unmet):
                raise DataValidationError(
                    "plan diagnostics do not cover the plan document identifiers"
                )
            if any(
                item.assigned != assignment_counts.get(item.document_id, 0)
                or item.unmet != self.unmet.get(item.document_id, 0)
                for item in self.diagnostics.documents
            ):
                raise DataValidationError(
                    "plan diagnostics do not match assignments and unmet counts"
                )
        object.__setattr__(self, "unmet", _freeze_mapping(self.unmet))

    def for_document(self, document_id: str) -> tuple[Assignment, ...]:
        """Return assignments for one document in display rank order."""

        return tuple(
            sorted(
                (item for item in self.assignments if item.document_id == document_id),
                key=lambda item: item.rank,
            )
        )
