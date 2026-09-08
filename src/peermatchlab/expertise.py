"""Deterministic, explainable TF-IDF and BM25 expertise generation.

The lightweight scorer in :mod:`peermatchlab.scoring` remains the convenient
end-to-end default.  This module is the explicit information-retrieval layer:
it preserves individual profile/publication evidence, fits corpus statistics,
and records term-level contributions for every paper-reviewer score.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from peermatchlab.io import load_json_text
from peermatchlab.models import (
    DataValidationError,
    Document,
    Expert,
    Publication,
    _identifier_is_valid,
)

ExpertiseModelName = Literal["tfidf", "bm25"]
ExpertiseAggregation = Literal["aggregate", "max", "average"]

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_DEFAULT_STOPWORDS = (
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
)
_SUBMISSION_FIELDS = frozenset({"title", "abstract", "topics", "keywords"})
_PROFILE_FIELDS = frozenset({"summary", "topics", "keywords"})
_PUBLICATION_FIELDS = frozenset({"title", "abstract"})
_MODEL_SCHEMA_VERSION = 1
_MAX_MODEL_BYTES = 64 * 1024 * 1024
_MAX_SCORE_CONTRIBUTIONS = 1_000_000
_MAX_EVIDENCE_TOKENS = 1_000_000
_MAX_TEXT_PARTS = len(_SUBMISSION_FIELDS)
_LIMIT_MAXIMUMS = {
    name: 1_000_000
    for name in (
        "minimum_token_length",
        "minimum_evidence_tokens",
        "max_submissions",
        "max_reviewers",
        "max_publications_per_reviewer",
        "max_vocabulary_terms",
    )
} | {
    "max_total_publications": 10_000_000,
    "max_input_file_bytes": _MAX_MODEL_BYTES,
    "max_model_file_bytes": _MAX_MODEL_BYTES,
    "max_document_characters": _MAX_MODEL_BYTES,
    "max_document_bytes": _MAX_MODEL_BYTES,
    "max_scanned_matches": 10_000_000,
    "max_token_characters": _MAX_EVIDENCE_TOKENS,
    "max_tokens_per_document": _MAX_EVIDENCE_TOKENS,
    "max_total_tokens": 10_000_000,
    "max_pairs": 10_000_000,
    "max_total_contributions": 10_000_000,
}


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _positive_integer(value: object, name: str, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > min(maximum, sys.maxsize - 1)
    ):
        raise DataValidationError(
            f"{name} must be an integer between 1 and {min(maximum, sys.maxsize - 1)}"
        )
    return int(value)


def _bounded_tuple(values: Iterable[Any], limit: int, label: str) -> tuple[Any, ...]:
    """Consume at most ``limit + 1`` values so hostile iterables stay bounded."""

    try:
        iterator = iter(values)
    except TypeError as error:
        raise DataValidationError(f"{label} must be iterable") from error
    items = tuple(islice(iterator, limit + 1))
    if len(items) > limit:
        raise DataValidationError(f"{label} exceeds its configured limit")
    return items


def _read_bounded_utf8(path: str | Path, *, max_bytes: int, label: str) -> tuple[str, int]:
    source = Path(path)
    with source.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise DataValidationError(f"{label} exceeds the {max_bytes}-byte hard limit")
    try:
        return data.decode("utf-8"), len(data)
    except UnicodeDecodeError as error:
        raise DataValidationError(f"{label} must be UTF-8") from error


def _fields(value: object, *, name: str, allowed: frozenset[str]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value or len(value) > len(allowed):
        raise DataValidationError(f"{name} must be a non-empty array of field names")
    if any(type(item) is not str or not item for item in value):
        raise DataValidationError(f"{name} must contain non-empty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise DataValidationError(f"{name} must not contain duplicate fields")
    unknown = set(result) - allowed
    if unknown:
        raise DataValidationError(f"unknown {name}: {sorted(unknown)}")
    return result


@dataclass(frozen=True, slots=True)
class ExpertiseConfig:
    """Closed, serializable controls for one expertise model.

    Tokenization uses Unicode alphanumeric runs, Unicode case-folding, an
    explicit minimum token length, and the configured stopword set.  Fields,
    publication dates, empty content, duplicates, and every resource ceiling
    are therefore part of the reproducibility contract rather than ambient
    preprocessing choices.
    """

    model: ExpertiseModelName = "tfidf"
    aggregation: ExpertiseAggregation = "aggregate"
    submission_fields: tuple[str, ...] = ("title", "abstract", "topics", "keywords")
    profile_fields: tuple[str, ...] = ("summary", "topics", "keywords")
    publication_fields: tuple[str, ...] = ("title", "abstract")
    include_profile: bool = True
    require_submission_abstract: bool = False
    require_publication_abstract: bool = False
    minimum_publication_year: int | None = None
    maximum_publication_year: int | None = None
    undated_publications: Literal["include", "exclude"] = "include"
    stopwords: tuple[str, ...] = _DEFAULT_STOPWORDS
    minimum_token_length: int = 2
    minimum_evidence_tokens: int = 1
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    minimum_output_score: float = 0.0
    max_submissions: int = 10_000
    max_reviewers: int = 100_000
    max_publications_per_reviewer: int = 10_000
    max_total_publications: int = 500_000
    max_input_file_bytes: int = 64 * 1024 * 1024
    max_model_file_bytes: int = 64 * 1024 * 1024
    max_document_characters: int = 5_000_000
    max_document_bytes: int = 16 * 1024 * 1024
    max_scanned_matches: int = 200_000
    max_token_characters: int = 100_000
    max_tokens_per_document: int = 50_000
    max_total_tokens: int = 2_000_000
    max_vocabulary_terms: int = 500_000
    max_pairs: int = 1_000_000
    max_total_contributions: int = 1_000_000

    def __post_init__(self) -> None:
        if type(self.model) is not str or self.model not in {"tfidf", "bm25"}:
            raise DataValidationError("expertise model must be 'tfidf' or 'bm25'")
        if type(self.aggregation) is not str or self.aggregation not in {
            "aggregate",
            "max",
            "average",
        }:
            raise DataValidationError(
                "expertise aggregation must be 'aggregate', 'max', or 'average'"
            )
        object.__setattr__(
            self,
            "submission_fields",
            _fields(self.submission_fields, name="submission_fields", allowed=_SUBMISSION_FIELDS),
        )
        object.__setattr__(
            self,
            "profile_fields",
            _fields(self.profile_fields, name="profile_fields", allowed=_PROFILE_FIELDS),
        )
        object.__setattr__(
            self,
            "publication_fields",
            _fields(
                self.publication_fields,
                name="publication_fields",
                allowed=_PUBLICATION_FIELDS,
            ),
        )
        if not isinstance(self.include_profile, bool):
            raise DataValidationError("include_profile must be a boolean")
        if not isinstance(self.require_submission_abstract, bool):
            raise DataValidationError("require_submission_abstract must be a boolean")
        if not isinstance(self.require_publication_abstract, bool):
            raise DataValidationError("require_publication_abstract must be a boolean")
        if self.require_publication_abstract and "abstract" not in self.publication_fields:
            raise DataValidationError(
                "publication_fields must include 'abstract' when "
                "require_publication_abstract is true"
            )
        if self.require_submission_abstract and "abstract" not in self.submission_fields:
            raise DataValidationError(
                "submission_fields must include 'abstract' when require_submission_abstract is true"
            )
        for name in ("minimum_publication_year", "maximum_publication_year"):
            year = getattr(self, name)
            if year is not None and (
                isinstance(year, bool) or not isinstance(year, int) or not 1800 <= year <= 2200
            ):
                raise DataValidationError(
                    f"{name} must be null or an integer between 1800 and 2200"
                )
            if year is not None:
                object.__setattr__(self, name, int(year))
        if (
            self.minimum_publication_year is not None
            and self.maximum_publication_year is not None
            and self.minimum_publication_year > self.maximum_publication_year
        ):
            raise DataValidationError(
                "minimum_publication_year must not exceed maximum_publication_year"
            )
        if type(self.undated_publications) is not str or self.undated_publications not in {
            "include",
            "exclude",
        }:
            raise DataValidationError("undated_publications must be 'include' or 'exclude'")
        for name in (
            "minimum_token_length",
            "minimum_evidence_tokens",
            "max_submissions",
            "max_reviewers",
            "max_publications_per_reviewer",
            "max_total_publications",
            "max_input_file_bytes",
            "max_model_file_bytes",
            "max_document_characters",
            "max_document_bytes",
            "max_scanned_matches",
            "max_token_characters",
            "max_tokens_per_document",
            "max_total_tokens",
            "max_vocabulary_terms",
            "max_pairs",
            "max_total_contributions",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(getattr(self, name), name, maximum=_LIMIT_MAXIMUMS[name]),
            )
        if not isinstance(self.stopwords, (list, tuple)):
            raise DataValidationError("stopwords must be an array of non-empty strings")
        stopwords = _bounded_tuple(
            self.stopwords,
            _LIMIT_MAXIMUMS["max_vocabulary_terms"],
            "expertise stopwords",
        )
        normalized_stopwords: set[str] = set()
        for item in stopwords:
            if type(item) is not str or not item or len(item) > self.max_token_characters:
                raise DataValidationError("stopwords must contain bounded non-empty strings")
            try:
                if len(item.encode("utf-8")) > self.max_document_bytes:
                    raise DataValidationError("stopword exceeds max_document_bytes")
            except UnicodeEncodeError as error:
                raise DataValidationError("stopwords must contain valid Unicode") from error
            normalized = item.casefold().strip()
            if not normalized or len(normalized) > self.max_token_characters:
                raise DataValidationError("stopwords must contain bounded non-empty strings")
            normalized_stopwords.add(normalized)
        object.__setattr__(self, "stopwords", tuple(sorted(normalized_stopwords)))
        if not _finite_number(self.bm25_k1) or float(self.bm25_k1) <= 0.0:
            raise DataValidationError("bm25_k1 must be a positive finite number")
        if not _finite_number(self.bm25_b) or not 0.0 <= float(self.bm25_b) <= 1.0:
            raise DataValidationError("bm25_b must be a finite number in [0, 1]")
        if (
            not _finite_number(self.minimum_output_score)
            or not 0.0 <= float(self.minimum_output_score) <= 1.0
        ):
            raise DataValidationError("minimum_output_score must be a finite number in [0, 1]")
        object.__setattr__(self, "bm25_k1", float(self.bm25_k1))
        object.__setattr__(self, "bm25_b", float(self.bm25_b))
        object.__setattr__(self, "minimum_output_score", float(self.minimum_output_score))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExpertiseConfig:
        """Parse an expertise configuration and reject unknown controls."""

        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise DataValidationError("expertise configuration must be an object with string keys")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise DataValidationError(f"unknown expertise configuration keys: {sorted(unknown)}")
        converted = dict(value)
        for name in ("submission_fields", "profile_fields", "publication_fields", "stopwords"):
            if name in converted:
                raw = converted[name]
                if not isinstance(raw, list):
                    raise DataValidationError(f"{name} must be a JSON array")
                converted[name] = tuple(raw)
        return cls(**converted)

    @classmethod
    def from_json(cls, path: str | Path) -> ExpertiseConfig:
        """Load a strict UTF-8 JSON configuration."""

        text, byte_count = _read_bounded_utf8(
            path, max_bytes=_MAX_MODEL_BYTES, label="expertise configuration"
        )
        value = load_json_text(text)
        if not isinstance(value, Mapping):
            raise DataValidationError("expertise configuration must be a JSON object")
        result = cls.from_mapping(value)
        if byte_count > result.max_input_file_bytes:
            raise DataValidationError("expertise configuration exceeds max_input_file_bytes")
        return result

    def as_dict(self) -> dict[str, object]:
        """Return the complete configuration in its stable persisted form."""

        return {
            "model": self.model,
            "aggregation": self.aggregation,
            "submission_fields": list(self.submission_fields),
            "profile_fields": list(self.profile_fields),
            "publication_fields": list(self.publication_fields),
            "include_profile": self.include_profile,
            "require_submission_abstract": self.require_submission_abstract,
            "require_publication_abstract": self.require_publication_abstract,
            "minimum_publication_year": self.minimum_publication_year,
            "maximum_publication_year": self.maximum_publication_year,
            "undated_publications": self.undated_publications,
            "stopwords": list(self.stopwords),
            "minimum_token_length": self.minimum_token_length,
            "minimum_evidence_tokens": self.minimum_evidence_tokens,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "minimum_output_score": self.minimum_output_score,
            "max_submissions": self.max_submissions,
            "max_reviewers": self.max_reviewers,
            "max_publications_per_reviewer": self.max_publications_per_reviewer,
            "max_total_publications": self.max_total_publications,
            "max_input_file_bytes": self.max_input_file_bytes,
            "max_model_file_bytes": self.max_model_file_bytes,
            "max_document_characters": self.max_document_characters,
            "max_document_bytes": self.max_document_bytes,
            "max_scanned_matches": self.max_scanned_matches,
            "max_token_characters": self.max_token_characters,
            "max_tokens_per_document": self.max_tokens_per_document,
            "max_total_tokens": self.max_total_tokens,
            "max_vocabulary_terms": self.max_vocabulary_terms,
            "max_pairs": self.max_pairs,
            "max_total_contributions": self.max_total_contributions,
        }


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    """One normalized sparse input document with its original selected fields."""

    id: str
    owner_id: str
    kind: Literal["submission", "profile", "publication"]
    fields: Mapping[str, str]
    tokens: tuple[str, ...]
    year: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.id) is not str
            or type(self.owner_id) is not str
            or not _identifier_is_valid(self.id)
            or not _identifier_is_valid(self.owner_id)
        ):
            raise DataValidationError(
                "evidence identifiers must be non-empty and contain no surrounding "
                "whitespace or control characters"
            )
        if type(self.kind) is not str or self.kind not in {
            "submission",
            "profile",
            "publication",
        }:
            raise DataValidationError("evidence kind is not supported")
        if not isinstance(self.fields, Mapping):
            raise DataValidationError("evidence fields must map strings to strings")
        field_rows = _bounded_tuple(self.fields.items(), _MAX_TEXT_PARTS, "evidence fields")
        if any(type(key) is not str or type(value) is not str for key, value in field_rows):
            raise DataValidationError("evidence fields must map strings to strings")
        if len({key for key, _value in field_rows}) != len(field_rows):
            raise DataValidationError("evidence fields must use unique names")
        tokens = _bounded_tuple(self.tokens, _MAX_EVIDENCE_TOKENS, "evidence tokens")
        if any(type(token) is not str or not token for token in tokens):
            raise DataValidationError("evidence tokens must be non-empty strings")
        if self.year is not None and (
            isinstance(self.year, bool)
            or not isinstance(self.year, int)
            or not 1800 <= self.year <= 2200
        ):
            raise DataValidationError("evidence year must be null or between 1800 and 2200")
        object.__setattr__(self, "fields", MappingProxyType(dict(field_rows)))
        object.__setattr__(self, "tokens", tokens)
        if self.year is not None:
            object.__setattr__(self, "year", int(self.year))

    def as_dict(self) -> dict[str, object]:
        """Serialize selected fields and exact fitted tokens."""

        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "kind": self.kind,
            "year": self.year,
            "fields": dict(self.fields),
            "tokens": list(self.tokens),
        }


def _reviewer_evidence_id_is_valid(evidence_id: str, owner_id: str) -> bool:
    if evidence_id == f"profile:{owner_id}":
        return True
    prefix = f"publication:{owner_id}:"
    if not evidence_id.startswith(prefix):
        return False
    suffix = evidence_id[len(prefix) :]
    if suffix.startswith("id:"):
        return _identifier_is_valid(suffix[3:])
    if not suffix.startswith("position:"):
        return False
    position = suffix[len("position:") :]
    return position.isascii() and position.isdigit() and not position.startswith("0")


def _validate_evidence_namespace(item: EvidenceDocument) -> None:
    if item.kind == "submission":
        if item.id != item.owner_id or item.year is not None:
            raise DataValidationError(
                "submission evidence must use its owner id and cannot have a year"
            )
        return
    if not _reviewer_evidence_id_is_valid(item.id, item.owner_id):
        raise DataValidationError("reviewer evidence owner or kind namespace is invalid")
    if item.kind == "profile":
        if item.id != f"profile:{item.owner_id}" or item.year is not None:
            raise DataValidationError(
                "profile evidence must use its owner id and cannot have a year"
            )
    elif not item.id.startswith(f"publication:{item.owner_id}:"):
        raise DataValidationError("publication evidence id does not match its owner")


def _validate_evidence_for_config(item: EvidenceDocument, config: ExpertiseConfig) -> None:
    _validate_evidence_namespace(item)
    if item.kind == "submission":
        selected_fields = config.submission_fields
    elif item.kind == "profile":
        selected_fields = config.profile_fields
        if not config.include_profile:
            raise DataValidationError("profile evidence is forbidden when include_profile is false")
    else:
        selected_fields = config.publication_fields
    if set(item.fields) - set(selected_fields):
        raise DataValidationError("evidence contains a field excluded by its configuration")
    if any(not value or value != value.strip() for value in item.fields.values()):
        raise DataValidationError("evidence fields must contain normalized non-empty text")
    expected_tokens = _tokenize_parts(
        (item.fields[name] for name in selected_fields if name in item.fields), config
    )
    if item.tokens != expected_tokens:
        raise DataValidationError("evidence tokens disagree with its selected fields")
    if item.kind != "submission" and len(item.tokens) < config.minimum_evidence_tokens:
        raise DataValidationError("reviewer evidence is below minimum_evidence_tokens")
    if (
        item.kind == "submission"
        and config.require_submission_abstract
        and not item.fields.get("abstract", "").strip()
    ):
        raise DataValidationError("submission evidence is missing a required abstract")
    if (
        item.kind == "publication"
        and config.require_publication_abstract
        and not item.fields.get("abstract", "").strip()
    ):
        raise DataValidationError("publication evidence is missing a required abstract")
    if item.kind == "publication" and (
        (item.year is None and config.undated_publications == "exclude")
        or (
            item.year is not None
            and config.minimum_publication_year is not None
            and item.year < config.minimum_publication_year
        )
        or (
            item.year is not None
            and config.maximum_publication_year is not None
            and item.year > config.maximum_publication_year
        )
    ):
        raise DataValidationError("publication evidence is excluded by date filters")


@dataclass(frozen=True, slots=True)
class TermContribution:
    """One term's additive contribution to an unnormalized pair score."""

    term: str
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.term, str) or not self.term.strip():
            raise DataValidationError("contribution term must be a non-empty string")
        if not _finite_number(self.value) or float(self.value) < 0.0:
            raise DataValidationError("contribution value must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ExpertiseScore:
    """A paper-reviewer affinity with an exact additive explanation."""

    document_id: str
    expert_id: str
    score: float
    raw_score: float
    model: ExpertiseModelName
    aggregation: ExpertiseAggregation
    evidence_count: int
    selected_evidence_id: str | None
    contributions: tuple[TermContribution, ...]

    def __post_init__(self) -> None:
        if not _identifier_is_valid(self.document_id) or not _identifier_is_valid(self.expert_id):
            raise DataValidationError(
                "expertise score identifiers must be non-empty and contain no surrounding "
                "whitespace or control characters"
            )
        if not _finite_number(self.score) or not 0.0 <= float(self.score) <= 1.0:
            raise DataValidationError("expertise score must be finite and in [0, 1]")
        if not _finite_number(self.raw_score) or float(self.raw_score) < 0.0:
            raise DataValidationError("raw expertise score must be finite and non-negative")
        if self.model not in {"tfidf", "bm25"} or self.aggregation not in {
            "aggregate",
            "max",
            "average",
        }:
            raise DataValidationError("expertise score model or aggregation is invalid")
        if self.selected_evidence_id is not None and not _identifier_is_valid(
            self.selected_evidence_id
        ):
            raise DataValidationError(
                "selected_evidence_id must be null or a valid non-empty identifier"
            )
        if (
            isinstance(self.evidence_count, bool)
            or not isinstance(self.evidence_count, int)
            or self.evidence_count < 0
        ):
            raise DataValidationError("evidence_count must be a non-negative integer")
        contributions = _bounded_tuple(
            self.contributions,
            _MAX_SCORE_CONTRIBUTIONS,
            "term contributions",
        )
        if any(not isinstance(item, TermContribution) for item in contributions):
            raise DataValidationError("term contributions must be finite and non-negative")
        if len({item.term for item in contributions}) != len(contributions):
            raise DataValidationError("term contributions must use unique terms")
        if any(float(item.value) == 0.0 for item in contributions):
            raise DataValidationError("zero-valued term contributions must be omitted")
        selected = self.selected_evidence_id
        if self.evidence_count == 0 and (
            selected is not None
            or float(self.score) != 0.0
            or float(self.raw_score) != 0.0
            or contributions
        ):
            raise DataValidationError(
                "scores without evidence must be exactly zero and unexplained"
            )
        if not math.isclose(
            sum(item.value for item in contributions),
            float(self.raw_score),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise DataValidationError("term contributions must add to raw_score")
        expected_score = (
            float(self.raw_score)
            if self.model == "tfidf"
            else float(self.raw_score) / (1.0 + float(self.raw_score))
        )
        if not math.isclose(float(self.score), expected_score, rel_tol=1e-12, abs_tol=1e-12):
            raise DataValidationError("expertise score disagrees with its model normalization")
        if self.aggregation == "average" and selected is not None:
            raise DataValidationError("average expertise scores cannot select evidence")
        if (
            self.aggregation == "aggregate"
            and self.evidence_count > 0
            and selected != f"aggregate:{self.expert_id}"
        ):
            raise DataValidationError(
                "aggregate expertise scores must select their reviewer aggregate"
            )
        if (
            self.aggregation == "max"
            and self.evidence_count > 0
            and (selected is None or not _reviewer_evidence_id_is_valid(selected, self.expert_id))
        ):
            raise DataValidationError(
                "max expertise scores must select evidence owned by their reviewer"
            )
        object.__setattr__(self, "contributions", contributions)

    def as_dict(self) -> dict[str, object]:
        """Return a stable explanation record."""

        return {
            "document_id": self.document_id,
            "expert_id": self.expert_id,
            "score": self.score,
            "raw_score": self.raw_score,
            "model": self.model,
            "aggregation": self.aggregation,
            "evidence_count": self.evidence_count,
            "selected_evidence_id": self.selected_evidence_id,
            "contributions": [
                {"term": item.term, "value": item.value} for item in self.contributions
            ],
        }


@dataclass(frozen=True, slots=True)
class ExpertiseCorpus:
    """Normalized query documents and reviewer evidence for one run."""

    submissions: tuple[EvidenceDocument, ...]
    reviewer_evidence: Mapping[str, tuple[EvidenceDocument, ...]]
    filters: Mapping[str, int]
    config: ExpertiseConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, ExpertiseConfig):
            raise DataValidationError("expertise corpus requires an ExpertiseConfig")
        submissions = _bounded_tuple(
            self.submissions, self.config.max_submissions, "submission evidence"
        )
        if not submissions or any(not isinstance(item, EvidenceDocument) for item in submissions):
            raise DataValidationError("expertise corpus requires submission evidence")
        for item in submissions:
            if item.kind != "submission":
                raise DataValidationError("expertise corpus queries must be submissions")
            _validate_evidence_for_config(item, self.config)
        if len({item.id for item in submissions}) != len(submissions):
            raise DataValidationError("submission evidence identifiers must be unique")

        if not isinstance(self.reviewer_evidence, Mapping):
            raise DataValidationError("reviewer_evidence must be a mapping")
        reviewer_rows = _bounded_tuple(
            self.reviewer_evidence.items(), self.config.max_reviewers, "reviewer evidence"
        )
        if not reviewer_rows:
            raise DataValidationError("expertise corpus requires reviewer evidence mappings")
        normalized_reviewers: dict[str, tuple[EvidenceDocument, ...]] = {}
        total_publications = 0
        total_tokens = sum(len(item.tokens) for item in submissions)
        for reviewer_id, raw_evidence in reviewer_rows:
            if not _identifier_is_valid(reviewer_id) or reviewer_id in normalized_reviewers:
                raise DataValidationError("reviewer evidence keys must be unique valid identifiers")
            evidence = _bounded_tuple(
                raw_evidence,
                self.config.max_publications_per_reviewer + 1,
                f"reviewer {reviewer_id!r} evidence",
            )
            if any(not isinstance(item, EvidenceDocument) for item in evidence):
                raise DataValidationError("reviewer evidence must contain EvidenceDocument objects")
            for item in evidence:
                if item.owner_id != reviewer_id or item.kind == "submission":
                    raise DataValidationError("reviewer evidence owner or kind is invalid")
                _validate_evidence_for_config(item, self.config)
                total_tokens += len(item.tokens)
            if len({item.id for item in evidence}) != len(evidence):
                raise DataValidationError("reviewer evidence identifiers must be unique")
            profile_count = sum(item.kind == "profile" for item in evidence)
            publication_count = sum(item.kind == "publication" for item in evidence)
            if profile_count > 1 or publication_count > self.config.max_publications_per_reviewer:
                raise DataValidationError("reviewer evidence kinds exceed configured limits")
            total_publications += publication_count
            if total_publications > self.config.max_total_publications:
                raise DataValidationError("publication count exceeds max_total_publications")
            normalized_reviewers[reviewer_id] = evidence
        if total_tokens > self.config.max_total_tokens:
            raise DataValidationError("normalized corpus exceeds max_total_tokens")

        if not isinstance(self.filters, Mapping):
            raise DataValidationError("expertise corpus filters must be a mapping")
        filter_rows = _bounded_tuple(self.filters.items(), 64, "expertise corpus filters")
        normalized_filters: dict[str, int] = {}
        for name, count in filter_rows:
            if (
                not isinstance(name, str)
                or not name
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise DataValidationError("expertise corpus filters must be non-negative counts")
            normalized_filters[name] = count
        object.__setattr__(self, "submissions", submissions)
        object.__setattr__(self, "reviewer_evidence", MappingProxyType(normalized_reviewers))
        object.__setattr__(self, "filters", MappingProxyType(normalized_filters))


def _validated_text_parts(parts: Iterable[str], config: ExpertiseConfig) -> tuple[str, ...]:
    normalized = _bounded_tuple(parts, _MAX_TEXT_PARTS, "document text fields")
    if any(type(text) is not str for text in normalized):
        raise DataValidationError("document text fields must be plain strings")
    character_count = 0
    byte_count = 0
    for part_number, text in enumerate(normalized):
        separator = 1 if part_number else 0
        character_count += len(text) + separator
        if character_count > config.max_document_characters:
            raise DataValidationError(
                f"document exceeds max_document_characters {config.max_document_characters}"
            )
        byte_count += separator
        try:
            for start in range(0, len(text), 4096):
                byte_count += len(text[start : start + 4096].encode("utf-8"))
                if byte_count > config.max_document_bytes:
                    raise DataValidationError(
                        f"document exceeds max_document_bytes {config.max_document_bytes}"
                    )
        except UnicodeEncodeError as error:
            raise DataValidationError("document text must be valid Unicode") from error
    return cast(tuple[str, ...], normalized)


def _tokenize_parts(parts: Iterable[str], config: ExpertiseConfig) -> tuple[str, ...]:
    normalized = _validated_text_parts(parts, config)
    tokens: list[str] = []
    scanned_matches = 0
    stopwords = frozenset(config.stopwords)
    for text in normalized:
        for match in _TOKEN_PATTERN.finditer(text):
            scanned_matches += 1
            if scanned_matches > config.max_scanned_matches:
                raise DataValidationError(
                    f"document exceeds max_scanned_matches {config.max_scanned_matches}"
                )
            if match.end() - match.start() > config.max_token_characters:
                raise DataValidationError(
                    f"token exceeds max_token_characters {config.max_token_characters}"
                )
            token = match.group(0).casefold()
            if len(token) > config.max_token_characters:
                raise DataValidationError(
                    f"token exceeds max_token_characters {config.max_token_characters}"
                )
            if len(token) < config.minimum_token_length or token in stopwords:
                continue
            if len(tokens) >= config.max_tokens_per_document:
                raise DataValidationError(
                    f"document exceeds max_tokens_per_document {config.max_tokens_per_document}"
                )
            tokens.append(token)
    return tuple(tokens)


def _tokenize(text: str, config: ExpertiseConfig) -> tuple[str, ...]:
    return _tokenize_parts((text,), config)


def _selected_fields(
    values: Mapping[str, str], selected: Sequence[str], config: ExpertiseConfig
) -> Mapping[str, str]:
    raw = tuple(values[name] for name in selected if values.get(name, ""))
    _validated_text_parts(raw, config)
    result: dict[str, str] = {}
    for name in selected:
        value = values.get(name, "")
        if value:
            stripped = value.strip()
            if stripped:
                result[name] = stripped
    return MappingProxyType(result)


def _evidence(
    *,
    evidence_id: str,
    owner_id: str,
    kind: Literal["submission", "profile", "publication"],
    values: Mapping[str, str],
    selected: Sequence[str],
    config: ExpertiseConfig,
    year: int | None = None,
) -> EvidenceDocument:
    fields = _selected_fields(values, selected, config)
    return EvidenceDocument(
        id=evidence_id,
        owner_id=owner_id,
        kind=kind,
        fields=fields,
        tokens=_tokenize_parts(fields.values(), config),
        year=year,
    )


def _publication_allowed(publication: Publication, config: ExpertiseConfig) -> bool:
    if config.require_publication_abstract and not publication.abstract.strip():
        return False
    if publication.year is None:
        return config.undated_publications == "include"
    if (
        config.minimum_publication_year is not None
        and publication.year < config.minimum_publication_year
    ):
        return False
    return not (
        config.maximum_publication_year is not None
        and publication.year > config.maximum_publication_year
    )


def build_expertise_corpus(
    documents: Iterable[Document],
    experts: Iterable[Expert],
    *,
    config: ExpertiseConfig | None = None,
) -> ExpertiseCorpus:
    """Build deterministic sparse documents from validated local domain objects."""

    active = config or ExpertiseConfig()
    if not isinstance(active, ExpertiseConfig):
        raise DataValidationError("config must be an ExpertiseConfig")
    document_items = _bounded_tuple(documents, active.max_submissions, "max_submissions")
    expert_items = _bounded_tuple(experts, active.max_reviewers, "max_reviewers")
    if not document_items or any(not isinstance(item, Document) for item in document_items):
        raise DataValidationError("expertise generation requires Document objects")
    if not expert_items or any(not isinstance(item, Expert) for item in expert_items):
        raise DataValidationError("expertise generation requires Expert objects")
    publications_by_reviewer: dict[str, tuple[Publication, ...]] = {}
    total_publications = 0
    for expert in expert_items:
        publications = _bounded_tuple(
            expert.publications,
            active.max_publications_per_reviewer,
            f"reviewer {expert.id!r} publications",
        )
        if any(not isinstance(item, Publication) for item in publications):
            raise DataValidationError("expert publications must contain Publication objects")
        total_publications += len(publications)
        if total_publications > active.max_total_publications:
            raise DataValidationError("publication count exceeds max_total_publications")
        publications_by_reviewer[expert.id] = publications
    if len({item.id for item in document_items}) != len(document_items):
        raise DataValidationError("submission identifiers must be unique")
    if len({item.id for item in expert_items}) != len(expert_items):
        raise DataValidationError("reviewer identifiers must be unique")
    if active.require_submission_abstract:
        for item in document_items:
            _validated_text_parts((item.abstract,), active)
            if not item.abstract.strip():
                raise DataValidationError("every submission must contain an abstract")

    submissions = tuple(
        _evidence(
            evidence_id=item.id,
            owner_id=item.id,
            kind="submission",
            values={
                "title": item.title,
                "abstract": item.abstract,
                "topics": " ".join(item.topics),
                "keywords": " ".join(item.keywords),
            },
            selected=active.submission_fields,
            config=active,
        )
        for item in sorted(document_items, key=lambda value: value.id)
    )
    filters: Counter[str] = Counter()
    reviewer_evidence: dict[str, tuple[EvidenceDocument, ...]] = {}
    total_tokens = sum(len(item.tokens) for item in submissions)
    for expert in sorted(expert_items, key=lambda value: value.id):
        evidence_items: list[EvidenceDocument] = []
        if active.include_profile:
            profile = _evidence(
                evidence_id=f"profile:{expert.id}",
                owner_id=expert.id,
                kind="profile",
                values={
                    "summary": expert.summary,
                    "topics": " ".join(expert.topics),
                    "keywords": " ".join(expert.keywords),
                },
                selected=active.profile_fields,
                config=active,
            )
            if len(profile.tokens) >= active.minimum_evidence_tokens:
                evidence_items.append(profile)
            else:
                filters["empty_profiles"] += 1
        seen_publications: set[tuple[str, str, int | None]] = set()
        seen_publication_ids: set[str] = set()
        for position, publication in enumerate(publications_by_reviewer[expert.id], start=1):
            _validated_text_parts((publication.title, publication.abstract), active)
            if publication.id is not None:
                if publication.id in seen_publication_ids:
                    raise DataValidationError(
                        f"reviewer {expert.id!r} contains duplicate publication id: "
                        f"{publication.id}"
                    )
                seen_publication_ids.add(publication.id)
            duplicate_key = (
                publication.title.casefold().strip(),
                publication.abstract.casefold().strip(),
                publication.year,
            )
            if duplicate_key in seen_publications:
                filters["duplicate_publications"] += 1
                continue
            seen_publications.add(duplicate_key)
            if not _publication_allowed(publication, active):
                filters["date_or_content_filtered_publications"] += 1
                continue
            identifier = (
                f"id:{publication.id}" if publication.id is not None else f"position:{position}"
            )
            item = _evidence(
                evidence_id=f"publication:{expert.id}:{identifier}",
                owner_id=expert.id,
                kind="publication",
                values={"title": publication.title, "abstract": publication.abstract},
                selected=active.publication_fields,
                config=active,
                year=publication.year,
            )
            if len(item.tokens) < active.minimum_evidence_tokens:
                filters["empty_publications"] += 1
                continue
            evidence_items.append(item)
        evidence_ids = [item.id for item in evidence_items]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise DataValidationError(
                f"reviewer {expert.id!r} produced duplicate evidence identifiers"
            )
        reviewer_evidence[expert.id] = tuple(evidence_items)
        total_tokens += sum(len(item.tokens) for item in evidence_items)
        if total_tokens > active.max_total_tokens:
            raise DataValidationError("normalized corpus exceeds max_total_tokens")
    return ExpertiseCorpus(
        submissions=submissions,
        reviewer_evidence=reviewer_evidence,
        filters=dict(filters),
        config=active,
    )


def _index_documents(
    reviewer_evidence: Mapping[str, tuple[EvidenceDocument, ...]],
    aggregation: ExpertiseAggregation,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if aggregation == "aggregate":
        return tuple(
            (reviewer_id, tuple(token for item in evidence for token in item.tokens))
            for reviewer_id, evidence in reviewer_evidence.items()
            if evidence
        )
    return tuple(
        (item.id, item.tokens)
        for evidence in reviewer_evidence.values()
        for item in evidence
        if item.tokens
    )


def _fit_idf(
    index: Sequence[tuple[str, tuple[str, ...]]], model: ExpertiseModelName
) -> dict[str, float]:
    document_count = len(index)
    frequencies: Counter[str] = Counter()
    for _document_id, tokens in index:
        frequencies.update(set(tokens))
    if model == "tfidf":
        return {
            term: math.log((1.0 + document_count) / (1.0 + count)) + 1.0
            for term, count in frequencies.items()
        }
    return {
        term: math.log(1.0 + (document_count - count + 0.5) / (count + 0.5))
        for term, count in frequencies.items()
    }


def _tfidf_pair(
    query: tuple[str, ...], evidence: tuple[str, ...], idf: Mapping[str, float]
) -> tuple[float, dict[str, float]]:
    query_counts = Counter(query)
    evidence_counts = Counter(evidence)
    query_weights = {
        term: (1.0 + math.log(count)) * idf[term]
        for term, count in query_counts.items()
        if term in idf
    }
    evidence_weights = {
        term: (1.0 + math.log(count)) * idf[term]
        for term, count in evidence_counts.items()
        if term in idf
    }
    query_norm = math.sqrt(sum(value * value for value in query_weights.values()))
    evidence_norm = math.sqrt(sum(value * value for value in evidence_weights.values()))
    if query_norm == 0.0 or evidence_norm == 0.0:
        return 0.0, {}
    denominator = query_norm * evidence_norm
    contributions = {
        term: value * evidence_weights[term] / denominator
        for term, value in query_weights.items()
        if term in evidence_weights
    }
    return sum(contributions.values()), contributions


def _bm25_pair(
    query: tuple[str, ...],
    evidence: tuple[str, ...],
    idf: Mapping[str, float],
    *,
    average_length: float,
    k1: float,
    b: float,
) -> tuple[float, dict[str, float]]:
    if not query or not evidence or average_length == 0.0:
        return 0.0, {}
    counts = Counter(evidence)
    length_normalizer = k1 * (1.0 - b + b * len(evidence) / average_length)
    contributions = {
        term: idf[term] * (counts[term] * (k1 + 1.0)) / (counts[term] + length_normalizer)
        for term in sorted(set(query))
        if term in idf and term in counts
    }
    return sum(contributions.values()), contributions


@dataclass(frozen=True, slots=True)
class ExpertiseModel:
    """A fitted, versioned sparse expertise index."""

    config: ExpertiseConfig
    reviewer_evidence: Mapping[str, tuple[EvidenceDocument, ...]]
    inverse_document_frequency: Mapping[str, float]
    document_count: int
    average_document_length: float

    def __post_init__(self) -> None:
        if not isinstance(self.config, ExpertiseConfig):
            raise DataValidationError("expertise model requires an ExpertiseConfig")
        if not isinstance(self.reviewer_evidence, Mapping):
            raise DataValidationError("model reviewer evidence must be a mapping")
        reviewer_rows = _bounded_tuple(
            self.reviewer_evidence.items(), self.config.max_reviewers, "model reviewers"
        )
        if not reviewer_rows:
            raise DataValidationError("model requires at least one reviewer")
        reviewers: dict[str, tuple[EvidenceDocument, ...]] = {}
        total_publications = 0
        total_tokens = 0
        for reviewer_id, raw_evidence in reviewer_rows:
            if not _identifier_is_valid(reviewer_id) or reviewer_id in reviewers:
                raise DataValidationError("model reviewer ids must be unique valid identifiers")
            reviewer_id = cast(str, reviewer_id)
            evidence = _bounded_tuple(
                raw_evidence,
                self.config.max_publications_per_reviewer + 1,
                f"model reviewer {reviewer_id!r} evidence",
            )
            if any(not isinstance(item, EvidenceDocument) for item in evidence):
                raise DataValidationError("model reviewer evidence has an invalid value")
            for item in evidence:
                if item.owner_id != reviewer_id or item.kind == "submission":
                    raise DataValidationError("model reviewer evidence owner or kind is invalid")
                _validate_evidence_for_config(item, self.config)
                total_tokens += len(item.tokens)
            if len({item.id for item in evidence}) != len(evidence):
                raise DataValidationError("model reviewer evidence ids must be unique")
            profile_count = sum(item.kind == "profile" for item in evidence)
            publication_count = sum(item.kind == "publication" for item in evidence)
            if profile_count > 1 or publication_count > self.config.max_publications_per_reviewer:
                raise DataValidationError("model reviewer evidence kinds exceed configured limits")
            total_publications += publication_count
            if total_publications > self.config.max_total_publications:
                raise DataValidationError("model publication evidence exceeds configured limits")
            reviewers[reviewer_id] = evidence
        if total_tokens > self.config.max_total_tokens:
            raise DataValidationError("model reviewer evidence exceeds max_total_tokens")
        if not isinstance(self.inverse_document_frequency, Mapping):
            raise DataValidationError("model inverse_document_frequency must be a mapping")
        idf_rows = _bounded_tuple(
            self.inverse_document_frequency.items(),
            self.config.max_vocabulary_terms,
            "model vocabulary",
        )
        idf: dict[str, float] = {}
        for term, value in idf_rows:
            if (
                not isinstance(term, str)
                or not term
                or not _finite_number(value)
                or float(value) <= 0
            ):
                raise DataValidationError("model inverse_document_frequency is invalid")
            idf[term] = float(value)
        if isinstance(self.document_count, bool) or not isinstance(self.document_count, int):
            raise DataValidationError("model document_count must be a non-negative integer")
        if self.document_count < 0 or not _finite_number(self.average_document_length):
            raise DataValidationError("model statistics are invalid")
        average_length = float(self.average_document_length)
        if average_length < 0:
            raise DataValidationError("model statistics are invalid")
        expected_index = _index_documents(reviewers, self.config.aggregation)
        if self.document_count != len(expected_index):
            raise DataValidationError("model document_count disagrees with reviewer evidence")
        expected_average = (
            sum(len(tokens) for _document_id, tokens in expected_index) / len(expected_index)
            if expected_index
            else 0.0
        )
        if not math.isclose(average_length, expected_average, rel_tol=0.0, abs_tol=1e-12):
            raise DataValidationError("model average_document_length disagrees with evidence")
        expected_idf = _fit_idf(expected_index, self.config.model)
        if idf.keys() != expected_idf.keys() or any(
            not math.isclose(idf[term], expected_idf[term], rel_tol=0.0, abs_tol=1e-12)
            for term in idf
        ):
            raise DataValidationError("model IDF statistics disagree with reviewer evidence")
        object.__setattr__(self, "reviewer_evidence", MappingProxyType(reviewers))
        object.__setattr__(self, "inverse_document_frequency", MappingProxyType(idf))
        object.__setattr__(self, "average_document_length", average_length)

    @classmethod
    def fit(cls, corpus: ExpertiseCorpus, *, config: ExpertiseConfig) -> ExpertiseModel:
        """Fit global sparse statistics over aggregate or atomic reviewer evidence."""

        if corpus.config != config:
            raise DataValidationError("corpus and model expertise configurations must match")
        index = _index_documents(corpus.reviewer_evidence, config.aggregation)
        idf = _fit_idf(index, config.model)
        if len(idf) > config.max_vocabulary_terms:
            raise DataValidationError("fitted vocabulary exceeds max_vocabulary_terms")
        average_length = (
            sum(len(tokens) for _document_id, tokens in index) / len(index) if index else 0.0
        )
        result = cls(
            config=config,
            reviewer_evidence=corpus.reviewer_evidence,
            inverse_document_frequency=idf,
            document_count=len(index),
            average_document_length=average_length,
        )
        replayed = cls.from_mapping(result.to_dict())
        if replayed.to_dict() != result.to_dict():
            raise DataValidationError("fitted expertise model cannot be replayed exactly")
        return result

    def _pair(
        self, query: tuple[str, ...], evidence: tuple[str, ...]
    ) -> tuple[float, dict[str, float]]:
        if self.config.model == "tfidf":
            return _tfidf_pair(query, evidence, self.inverse_document_frequency)
        return _bm25_pair(
            query,
            evidence,
            self.inverse_document_frequency,
            average_length=self.average_document_length,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )

    def _reviewer_score(self, submission: EvidenceDocument, reviewer_id: str) -> ExpertiseScore:
        evidence = self.reviewer_evidence[reviewer_id]
        pair_scores: list[tuple[str, float, dict[str, float]]] = []
        if self.config.aggregation == "aggregate" and evidence:
            tokens = tuple(token for item in evidence for token in item.tokens)
            raw, contributions = self._pair(submission.tokens, tokens)
            pair_scores.append((f"aggregate:{reviewer_id}", raw, contributions))
        else:
            for item in evidence:
                raw, contributions = self._pair(submission.tokens, item.tokens)
                pair_scores.append((item.id, raw, contributions))

        selected: str | None = None
        combined: dict[str, float] = {}
        if not pair_scores:
            raw_score = 0.0
        elif self.config.aggregation == "max":
            selected, raw_score, combined = max(pair_scores, key=lambda item: item[1])
        elif self.config.aggregation == "average":
            raw_score = sum(item[1] for item in pair_scores) / len(pair_scores)
            for _evidence_id, _score, contributions in pair_scores:
                for term, value in contributions.items():
                    combined[term] = combined.get(term, 0.0) + value / len(pair_scores)
        else:
            selected, raw_score, combined = pair_scores[0]
        score = raw_score if self.config.model == "tfidf" else raw_score / (1.0 + raw_score)
        term_contributions = tuple(
            TermContribution(term=term, value=value)
            for term, value in sorted(combined.items(), key=lambda item: (-item[1], item[0]))
        )
        return ExpertiseScore(
            document_id=submission.owner_id,
            expert_id=reviewer_id,
            score=max(0.0, min(1.0, score)),
            raw_score=raw_score,
            model=self.config.model,
            aggregation=self.config.aggregation,
            evidence_count=len(evidence),
            selected_evidence_id=selected,
            contributions=term_contributions,
        )

    def score(self, submissions: Iterable[EvidenceDocument]) -> tuple[ExpertiseScore, ...]:
        """Score normalized submissions against every fitted reviewer."""

        items = _bounded_tuple(submissions, self.config.max_submissions, "submission evidence")
        if any(not isinstance(item, EvidenceDocument) for item in items):
            raise DataValidationError("expertise queries must be EvidenceDocument objects")
        if any(item.kind != "submission" for item in items):
            raise DataValidationError("expertise queries must be submission evidence documents")
        for item in items:
            _validate_evidence_for_config(item, self.config)
        for reviewer_id, evidence in self.reviewer_evidence.items():
            if not _identifier_is_valid(reviewer_id):
                raise DataValidationError("model reviewer id is invalid")
            for item in evidence:
                if item.owner_id != reviewer_id or item.kind == "submission":
                    raise DataValidationError("model reviewer evidence owner or kind is invalid")
                _validate_evidence_for_config(item, self.config)
        if len({item.owner_id for item in items}) != len(items):
            raise DataValidationError("submission evidence identifiers must be unique")
        if (
            any(len(item.tokens) > self.config.max_tokens_per_document for item in items)
            or sum(len(item.tokens) for item in items)
            + sum(
                len(item.tokens)
                for evidence in self.reviewer_evidence.values()
                for item in evidence
            )
            > self.config.max_total_tokens
        ):
            raise DataValidationError("submission evidence exceeds configured token limits")
        if len(items) * len(self.reviewer_evidence) > self.config.max_pairs:
            raise DataValidationError("candidate matrix exceeds max_pairs")
        scores: list[ExpertiseScore] = []
        contribution_count = 0
        for submission in sorted(items, key=lambda item: item.owner_id):
            for reviewer_id in sorted(self.reviewer_evidence):
                score = self._reviewer_score(submission, reviewer_id)
                contribution_count += len(score.contributions)
                if contribution_count > self.config.max_total_contributions:
                    raise DataValidationError("score explanations exceed max_total_contributions")
                scores.append(score)
        return tuple(scores)

    def score_documents(self, documents: Iterable[Document]) -> tuple[ExpertiseScore, ...]:
        """Normalize domain documents with the fitted configuration, then score them."""

        items = _bounded_tuple(documents, self.config.max_submissions, "submission count")
        if not items or any(not isinstance(item, Document) for item in items):
            raise DataValidationError("expertise scoring requires Document objects")
        if len({item.id for item in items}) != len(items):
            raise DataValidationError("submission identifiers must be unique")
        if self.config.require_submission_abstract and any(
            not item.abstract.strip() for item in items
        ):
            raise DataValidationError("every submission must contain an abstract")
        submissions = tuple(
            _evidence(
                evidence_id=item.id,
                owner_id=item.id,
                kind="submission",
                values={
                    "title": item.title,
                    "abstract": item.abstract,
                    "topics": " ".join(item.topics),
                    "keywords": " ".join(item.keywords),
                },
                selected=self.config.submission_fields,
                config=self.config,
            )
            for item in sorted(items, key=lambda value: value.id)
        )
        return self.score(submissions)

    def to_dict(self) -> dict[str, object]:
        """Serialize the exact fitted state under a versioned schema."""

        return {
            "schema_version": _MODEL_SCHEMA_VERSION,
            "config": self.config.as_dict(),
            "statistics": {
                "document_count": self.document_count,
                "average_document_length": self.average_document_length,
                "inverse_document_frequency": dict(sorted(self.inverse_document_frequency.items())),
            },
            "reviewers": [
                {
                    "id": reviewer_id,
                    "evidence": [item.as_dict() for item in evidence],
                }
                for reviewer_id, evidence in sorted(self.reviewer_evidence.items())
            ],
        }

    def save(self, path: str | Path) -> None:
        """Atomically persist stable JSON that is guaranteed to be loadable."""

        try:
            serialized = json.dumps(
                self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        except RecursionError as error:
            raise DataValidationError("JSON value exceeds the supported nesting depth") from error
        payload = (serialized + "\n").encode("utf-8")
        if len(payload) > self.config.max_model_file_bytes:
            raise DataValidationError("expertise model exceeds max_model_file_bytes")
        destination = Path(path)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name or 'model'}-", dir=destination.parent
        )
        temporary = Path(temporary_name)
        installed = False
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            installed = True
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if not installed:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> ExpertiseModel:
        """Load and validate a version-one model with a pre-parse byte ceiling."""

        text, byte_count = _read_bounded_utf8(
            path, max_bytes=_MAX_MODEL_BYTES, label="expertise model"
        )
        value = load_json_text(text)
        model = cls.from_mapping(value)
        if byte_count > model.config.max_model_file_bytes:
            raise DataValidationError("expertise model exceeds max_model_file_bytes")
        return model

    @classmethod
    def from_mapping(cls, value: object) -> ExpertiseModel:
        """Reconstruct a fitted model without silently accepting schema drift."""

        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "config",
            "statistics",
            "reviewers",
        }:
            raise DataValidationError("expertise model has unexpected top-level fields")
        schema_version = value.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != _MODEL_SCHEMA_VERSION
        ):
            raise DataValidationError("unsupported expertise model schema_version")
        raw_config = value.get("config")
        if not isinstance(raw_config, Mapping):
            raise DataValidationError("expertise model config must be an object")
        config = ExpertiseConfig.from_mapping(raw_config)
        statistics = value.get("statistics")
        if not isinstance(statistics, Mapping) or set(statistics) != {
            "document_count",
            "average_document_length",
            "inverse_document_frequency",
        }:
            raise DataValidationError("expertise model statistics are invalid")
        document_count = statistics.get("document_count")
        average_length = statistics.get("average_document_length")
        raw_idf = statistics.get("inverse_document_frequency")
        if (
            isinstance(document_count, bool)
            or not isinstance(document_count, int)
            or document_count < 0
        ):
            raise DataValidationError("model document_count must be a non-negative integer")
        if not _finite_number(average_length):
            raise DataValidationError(
                "model average_document_length must be finite and non-negative"
            )
        if not isinstance(average_length, (int, float)) or isinstance(average_length, bool):
            raise DataValidationError(
                "model average_document_length must be finite and non-negative"
            )
        normalized_average_length = float(average_length)
        if normalized_average_length < 0.0:
            raise DataValidationError(
                "model average_document_length must be finite and non-negative"
            )
        if not isinstance(raw_idf, Mapping) or any(
            not isinstance(term, str)
            or not term
            or not _finite_number(number)
            or float(number) <= 0.0
            for term, number in raw_idf.items()
        ):
            raise DataValidationError("model inverse_document_frequency is invalid")
        if len(raw_idf) > config.max_vocabulary_terms:
            raise DataValidationError("model vocabulary exceeds max_vocabulary_terms")
        rows = value.get("reviewers")
        if not isinstance(rows, list) or not rows or len(rows) > config.max_reviewers:
            raise DataValidationError("model reviewers must be an in-bounds array")
        reviewers: dict[str, tuple[EvidenceDocument, ...]] = {}
        total_tokens = 0
        total_publications = 0
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"id", "evidence"}:
                raise DataValidationError("model reviewer record is invalid")
            reviewer_id = row.get("id")
            evidence_rows = row.get("evidence")
            if not _identifier_is_valid(reviewer_id) or reviewer_id in reviewers:
                raise DataValidationError("model reviewer ids must be unique valid identifiers")
            normalized_reviewer_id = cast(str, reviewer_id)
            if (
                not isinstance(evidence_rows, list)
                or len(evidence_rows) > config.max_publications_per_reviewer + 1
            ):
                raise DataValidationError("model reviewer evidence is not an in-bounds array")
            evidence: list[EvidenceDocument] = []
            for evidence_row in evidence_rows:
                evidence.append(_evidence_from_mapping(evidence_row, config))
                if (
                    evidence[-1].owner_id != normalized_reviewer_id
                    or evidence[-1].kind == "submission"
                ):
                    raise DataValidationError("model reviewer evidence owner or kind is invalid")
                total_tokens += len(evidence[-1].tokens)
                if total_tokens > config.max_total_tokens:
                    raise DataValidationError("model evidence exceeds max_total_tokens")
                if evidence[-1].kind == "publication":
                    total_publications += 1
                    if total_publications > config.max_total_publications:
                        raise DataValidationError(
                            "model publication evidence exceeds max_total_publications"
                        )
            if len({item.id for item in evidence}) != len(evidence):
                raise DataValidationError("model reviewer evidence ids must be unique")
            if (
                sum(item.kind == "profile" for item in evidence) > 1
                or sum(item.kind == "publication" for item in evidence)
                > config.max_publications_per_reviewer
            ):
                raise DataValidationError("model reviewer evidence kinds exceed configured limits")
            reviewers[normalized_reviewer_id] = tuple(evidence)
        expected_index = _index_documents(reviewers, config.aggregation)
        if document_count != len(expected_index):
            raise DataValidationError("model document_count disagrees with reviewer evidence")
        expected_average = (
            sum(len(tokens) for _document_id, tokens in expected_index) / len(expected_index)
            if expected_index
            else 0.0
        )
        if not math.isclose(
            normalized_average_length, expected_average, rel_tol=0.0, abs_tol=1e-12
        ):
            raise DataValidationError(
                "model average_document_length disagrees with reviewer evidence"
            )
        expected_idf = _fit_idf(expected_index, config.model)
        idf = {str(term): float(number) for term, number in raw_idf.items()}
        if idf.keys() != expected_idf.keys() or any(
            not math.isclose(idf[term], expected_idf[term], rel_tol=0.0, abs_tol=1e-12)
            for term in idf
        ):
            raise DataValidationError("model IDF statistics disagree with reviewer evidence")
        return cls(
            config=config,
            reviewer_evidence=reviewers,
            inverse_document_frequency=idf,
            document_count=document_count,
            average_document_length=normalized_average_length,
        )


def _evidence_from_mapping(value: object, config: ExpertiseConfig) -> EvidenceDocument:
    if not isinstance(value, Mapping) or set(value) != {
        "id",
        "owner_id",
        "kind",
        "year",
        "fields",
        "tokens",
    }:
        raise DataValidationError("model evidence record is invalid")
    fields = value.get("fields")
    tokens = value.get("tokens")
    if not isinstance(fields, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in fields.items()
    ):
        raise DataValidationError("model evidence fields are invalid")
    if (
        not isinstance(tokens, list)
        or len(tokens) > config.max_tokens_per_document
        or any(not isinstance(token, str) or not token for token in tokens)
    ):
        raise DataValidationError("model evidence tokens are invalid")
    evidence_id = value.get("id")
    owner_id = value.get("owner_id")
    kind = value.get("kind")
    year = value.get("year")
    if not _identifier_is_valid(evidence_id) or not _identifier_is_valid(owner_id):
        raise DataValidationError("model evidence identifiers are invalid")
    evidence_id = cast(str, evidence_id)
    owner_id = cast(str, owner_id)
    if kind not in {"profile", "publication"}:
        raise DataValidationError("model evidence kind is invalid")
    if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
        raise DataValidationError("model evidence year is invalid")
    result = EvidenceDocument(
        id=evidence_id,
        owner_id=owner_id,
        kind=kind,
        fields=dict(fields),
        tokens=tuple(tokens),
        year=year,
    )
    _validate_evidence_for_config(result, config)
    return result


@dataclass(frozen=True, slots=True)
class ExpertiseRun:
    """All normalized evidence, fitted state, and pair scores for one run."""

    corpus: ExpertiseCorpus
    model: ExpertiseModel
    scores: tuple[ExpertiseScore, ...]


def generate_expertise(
    documents: Iterable[Document],
    experts: Iterable[Expert],
    *,
    config: ExpertiseConfig | None = None,
) -> ExpertiseRun:
    """Build, fit, and score a complete local expertise run."""

    active = config or ExpertiseConfig()
    corpus = build_expertise_corpus(documents, experts, config=active)
    model = ExpertiseModel.fit(corpus, config=active)
    return ExpertiseRun(corpus=corpus, model=model, scores=model.score(corpus.submissions))
