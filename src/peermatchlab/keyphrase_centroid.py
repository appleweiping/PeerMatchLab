"""Offline, bounded pairwise training of mean-keyphrase reviewer centroids.

This is an original, dependency-free local protocol. It does not run the frozen
OpenReview Expertise PyTorch model or claim its checkpoint/score equivalence.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from peermatchlab.affinity import Affinity
from peermatchlab.expertise_io import _install_directory_no_replace
from peermatchlab.io import (
    load_conflicts_text,
    load_documents_text,
    load_experts_text,
    load_json_text,
)
from peermatchlab.models import (
    Conflict,
    DataValidationError,
    Document,
    Expert,
    _identifier_is_valid,
)

PROTOCOL = "peermatch-keyphrase-centroid-v1"
_MAX_SOURCE = 16 * 1024 * 1024
_MAX_LABEL_SOURCE = 2 * 1024 * 1024
_MAX_MODEL = 8 * 1024 * 1024
_MAX_TERMS = 4096
_MAX_ENTITY_TERMS = 256
_MAX_PAIRS = 10_000
_MAX_WORK = 20_000_000


def _json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise DataValidationError("centroid artifact cannot be encoded as strict JSON") from error


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: str | Path, maximum: int, label: str) -> bytes:
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(maximum + 1)
    except OSError as error:
        raise DataValidationError(f"cannot read {label}") from error
    if len(raw) > maximum:
        raise DataValidationError(f"{label} exceeds {maximum} bytes")
    return raw


def _decode(raw: bytes, label: str) -> str:
    try:
        return raw.decode("utf-8", "strict")
    except UnicodeError as error:
        raise DataValidationError(f"{label} must be strict UTF-8") from error


def _source_limit(name: str) -> int:
    if name == "keyphrase-manifest.json":
        return 64 * 1024
    if name in {"train", "validation", "conflicts"}:
        return _MAX_LABEL_SOURCE
    return _MAX_SOURCE


def _json_lines(raw: bytes, label: str, maximum: int) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(_decode(raw, label).splitlines(), 1):
        if not line or len(line.encode("utf-8")) > 64 * 1024:
            raise DataValidationError(f"{label} has blank or oversized line {number}")
        if len(rows) >= maximum:
            raise DataValidationError(f"{label} exceeds {maximum} rows")
        try:
            value = load_json_text(line)
        except (ValueError, RecursionError) as error:
            raise DataValidationError(f"invalid {label} line {number}") from error
        if not isinstance(value, dict):
            raise DataValidationError(f"{label} line {number} must be an object")
        rows.append(value)
    if not rows:
        raise DataValidationError(f"{label} is empty")
    return tuple(rows)


def _finite_real(value: object) -> bool:
    if not isinstance(value, (int, float)) or type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class CentroidConfig:
    """Hard ceilings for one deterministic pairwise-SGD fit."""

    dimensions: int = 8
    epochs: int = 8
    learning_rate: float = 0.1
    l2: float = 0.0
    seed: int = 17
    max_work: int = _MAX_WORK

    def __post_init__(self) -> None:
        for name, low, high in (
            ("dimensions", 1, 32),
            ("epochs", 1, 50),
            ("seed", 0, 2**32 - 1),
            ("max_work", 1, _MAX_WORK),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise DataValidationError(f"{name} must be an integer in [{low}, {high}]")
        for name, lower_float, upper_float in (("learning_rate", 0.0, 1.0), ("l2", 0.0, 1.0)):
            float_value = getattr(self, name)
            if not _finite_real(float_value) or not lower_float <= float_value <= upper_float:
                raise DataValidationError(
                    f"{name} must be finite in [{lower_float}, {upper_float}]"
                )
        if self.learning_rate == 0:
            raise DataValidationError("learning_rate must be positive")

    def as_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CentroidTriplet:
    document_id: str
    positive_expert_id: str
    negative_expert_id: str


def _triplets(raw: bytes, label: str, maximum: int) -> tuple[CentroidTriplet, ...]:
    rows = _json_lines(raw, label, maximum)
    result: list[CentroidTriplet] = []
    seen: set[CentroidTriplet] = set()
    labels: dict[tuple[str, str], bool] = {}
    for row in rows:
        if set(row) != {"document_id", "positive_expert_id", "negative_expert_id"}:
            raise DataValidationError(f"{label} triplet schema is invalid")
        item = CentroidTriplet(
            row["document_id"], row["positive_expert_id"], row["negative_expert_id"]
        )
        if (
            any(
                not _identifier_is_valid(value)
                for value in (item.document_id, item.positive_expert_id, item.negative_expert_id)
            )
            or item.positive_expert_id == item.negative_expert_id
        ):
            raise DataValidationError(f"{label} triplet IDs must be valid and distinct")
        if item in seen:
            raise DataValidationError(f"{label} repeats a triplet")
        seen.add(item)
        for reviewer, positive in (
            (item.positive_expert_id, True),
            (item.negative_expert_id, False),
        ):
            pair = (item.document_id, reviewer)
            if pair in labels and labels[pair] is not positive:
                raise DataValidationError(f"{label} has contradictory pair labels")
            labels[pair] = positive
        result.append(item)
    return tuple(
        sorted(
            result,
            key=lambda row: (row.document_id, row.positive_expert_id, row.negative_expert_id),
        )
    )


def _safe_term(value: object) -> bool:
    if (
        type(value) is not str
        or len(value) <= 1
        or not value.isalnum()
        or value != value.casefold()
    ):
        return False
    try:
        return len(value.encode("utf-8", "strict")) <= 128
    except UnicodeError:
        return False


def _safe_score(value: object) -> bool:
    return _finite_real(value) and isinstance(value, (int, float)) and value >= 0


def _keyphrase_terms(
    raw: bytes, expected_count: int
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    records = _json_lines(raw, "keyphrase records", 10_000)
    if len(records) != expected_count:
        raise DataValidationError("keyphrase manifest record count mismatch")
    seen: set[tuple[str, str, str]] = set()
    submissions: dict[str, tuple[str, ...]] = {}
    reviewers: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    profiles: set[str] = set()
    for row in records:
        if set(row) != {"kind", "owner_id", "evidence_id", "token_count", "keyphrases"}:
            raise DataValidationError("keyphrase record schema is invalid")
        kind, owner, evidence = row["kind"], row["owner_id"], row["evidence_id"]
        if kind not in {"submission", "profile", "publication"} or not all(
            _identifier_is_valid(value) for value in (owner, evidence)
        ):
            raise DataValidationError("keyphrase record identity is invalid")
        identity = (kind, owner, evidence)
        if identity in seen or (kind in {"submission", "profile"} and evidence != owner):
            raise DataValidationError("keyphrase record identity is duplicate or inconsistent")
        seen.add(identity)
        if type(row["token_count"]) is not int or not 0 <= row["token_count"] <= 100_000:
            raise DataValidationError("keyphrase token_count is invalid")
        phrases = row["keyphrases"]
        if not isinstance(phrases, list) or len(phrases) > 1000:
            raise DataValidationError("keyphrase list exceeds limit")
        terms: list[str] = []
        for phrase in phrases:
            if not isinstance(phrase, dict) or set(phrase) != {"term", "score"}:
                raise DataValidationError("keyphrase term schema is invalid")
            term, score = phrase["term"], phrase["score"]
            if not _safe_term(term) or not _safe_score(score) or term in terms:
                raise DataValidationError("keyphrase term or score is invalid")
            terms.append(term)
        if kind == "submission":
            submissions[owner] = tuple(terms)
        else:
            if kind == "profile":
                profiles.add(owner)
            reviewers[owner].append(tuple(terms))
    combined: dict[str, tuple[str, ...]] = {}
    for owner, groups in reviewers.items():
        combined_terms = tuple(dict.fromkeys(term for group in groups for term in group))
        if len(combined_terms) > _MAX_ENTITY_TERMS:
            raise DataValidationError("reviewer keyphrases exceed per-entity limit")
        combined[owner] = combined_terms
    if set(reviewers) != profiles:
        raise DataValidationError("reviewer keyphrase profile records are missing")
    return submissions, combined


@dataclass(frozen=True, slots=True)
class CentroidInputs:
    documents: tuple[Document, ...]
    experts: tuple[Expert, ...]
    conflicts: tuple[Conflict, ...]
    submissions: Mapping[str, tuple[str, ...]]
    reviewers: Mapping[str, tuple[str, ...]]
    train: tuple[CentroidTriplet, ...]
    validation: tuple[CentroidTriplet, ...]
    raw: Mapping[str, bytes]
    paths: Mapping[str, Path]

    @property
    def hashes(self) -> dict[str, str]:
        return {name: _sha(data) for name, data in sorted(self.raw.items())}

    @property
    def holdout_ids(self) -> tuple[str, ...]:
        labelled = {item.document_id for item in (*self.train, *self.validation)}
        return tuple(sorted(set(self.submissions) - labelled))


def load_centroid_inputs(
    keyphrases_directory: str | Path,
    documents_path: str | Path,
    experts_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    *,
    conflicts_path: str | Path | None = None,
) -> CentroidInputs:
    """Load exact byte snapshots and disjoint, fully referenced label partitions."""
    paths = {
        "keyphrases.jsonl": Path(keyphrases_directory) / "keyphrases.jsonl",
        "keyphrase-manifest.json": Path(keyphrases_directory) / "manifest.json",
        "documents": Path(documents_path),
        "experts": Path(experts_path),
        "train": Path(train_path),
        "validation": Path(validation_path),
    }
    if conflicts_path is not None:
        paths["conflicts"] = Path(conflicts_path)
    if len({path.resolve(strict=False) for path in paths.values()}) != len(paths):
        raise DataValidationError("centroid input paths must be distinct")
    raw = {name: _read(path, _source_limit(name), name) for name, path in paths.items()}
    try:
        manifest = load_json_text(_decode(raw["keyphrase-manifest.json"], "keyphrase manifest"))
    except (ValueError, RecursionError) as error:
        raise DataValidationError("keyphrase manifest is invalid") from error
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "schema_version",
            "algorithm",
            "config",
            "records",
            "documents_sha256",
            "experts_sha256",
            "keyphrases_sha256",
            "keyphrases_bytes",
        }
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["algorithm"] != "lexical-undirected-pagerank-v1"
        or not isinstance(manifest["config"], dict)
    ):
        raise DataValidationError("unsupported keyphrase manifest")
    if (
        type(manifest["records"]) is not int
        or not 1 <= manifest["records"] <= 10_000
        or type(manifest["keyphrases_bytes"]) is not int
        or manifest["keyphrases_bytes"] != len(raw["keyphrases.jsonl"])
        or manifest["keyphrases_sha256"] != _sha(raw["keyphrases.jsonl"])
        or manifest["documents_sha256"] != _sha(raw["documents"])
        or manifest["experts_sha256"] != _sha(raw["experts"])
    ):
        raise DataValidationError("keyphrase manifest source hashes/counts disagree")
    documents = load_documents_text(_decode(raw["documents"], "documents"), max_records=10_000)
    experts = load_experts_text(_decode(raw["experts"], "experts"), max_records=10_000)
    conflicts = (
        load_conflicts_text(_decode(raw["conflicts"], "conflicts"), max_records=10_000)
        if "conflicts" in raw
        else ()
    )
    if len({(item.document_id, item.expert_id) for item in conflicts}) != len(conflicts):
        raise DataValidationError("duplicate conflict pairs")
    submissions, reviewers = _keyphrase_terms(raw["keyphrases.jsonl"], manifest["records"])
    if set(submissions) != {item.id for item in documents} or set(reviewers) != {
        item.id for item in experts
    }:
        raise DataValidationError("keyphrase submission/reviewer IDs do not match source inputs")
    if any(not terms or len(terms) > _MAX_ENTITY_TERMS for terms in submissions.values()) or any(
        not terms for terms in reviewers.values()
    ):
        raise DataValidationError("submission/reviewer keyphrases are missing or exceed limit")
    train = _triplets(raw["train"], "train", 2000)
    validation = _triplets(raw["validation"], "validation", 500)
    if {item.document_id for item in train} & {item.document_id for item in validation}:
        raise DataValidationError("train and validation submission IDs overlap")
    conflict_pairs = {(item.document_id, item.expert_id) for item in conflicts}
    for item in (*train, *validation):
        if (
            item.document_id not in submissions
            or item.positive_expert_id not in reviewers
            or item.negative_expert_id not in reviewers
        ):
            raise DataValidationError("triplet references missing keyphrase evidence")
        if (item.document_id, item.positive_expert_id) in conflict_pairs:
            raise DataValidationError("positive triplet pair is a declared hard conflict")
    if any(
        item.document_id not in submissions or item.expert_id not in reviewers for item in conflicts
    ):
        raise DataValidationError("conflict references unknown submission/reviewer")
    result = CentroidInputs(
        documents,
        experts,
        conflicts,
        MappingProxyType(submissions),
        MappingProxyType(reviewers),
        train,
        validation,
        MappingProxyType(raw),
        MappingProxyType(paths),
    )
    if not result.holdout_ids:
        raise DataValidationError(
            "centroid scoring requires at least one unlabeled holdout submission"
        )
    return result


def _assert_captured_inputs(inputs: CentroidInputs) -> None:
    """Reject replaced fields that no longer describe the exact source snapshot."""
    required = {
        "keyphrases.jsonl",
        "keyphrase-manifest.json",
        "documents",
        "experts",
        "train",
        "validation",
    }
    names = required | ({"conflicts"} if "conflicts" in inputs.paths else set())
    if (
        set(inputs.paths) != names
        or set(inputs.raw) != names
        or any(not isinstance(path, Path) for path in inputs.paths.values())
        or any(
            type(raw) is not bytes or len(raw) > _source_limit(name)
            for name, raw in inputs.raw.items()
        )
    ):
        raise DataValidationError("centroid input snapshot paths or bytes are malformed")
    for name, path in inputs.paths.items():
        if _read(path, len(inputs.raw[name]), name) != inputs.raw[name]:
            raise DataValidationError(f"centroid source changed after loading: {name}")
    fresh = load_centroid_inputs(
        inputs.paths["keyphrases.jsonl"].parent,
        inputs.paths["documents"],
        inputs.paths["experts"],
        inputs.paths["train"],
        inputs.paths["validation"],
        conflicts_path=inputs.paths.get("conflicts"),
    )
    if fresh != inputs:
        raise DataValidationError("centroid input snapshot differs from exact source bytes")


def _centroid(
    indices: tuple[int, ...],
    weights: list[list[float]] | tuple[tuple[float, ...], ...],
    dimensions: int,
) -> tuple[float, ...]:
    if not indices:
        raise DataValidationError("entity has no trained keyphrase coordinates")
    return tuple(
        math.fsum(weights[index][axis] for index in indices) / len(indices)
        for axis in range(dimensions)
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def _softplus_negative(margin: float) -> float:
    return (
        math.log1p(math.exp(-margin)) if margin >= 0.0 else -margin + math.log1p(math.exp(margin))
    )


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _update(
    weights: list[list[float]],
    query: tuple[int, ...],
    positive: tuple[int, ...],
    negative: tuple[int, ...],
    *,
    dimensions: int,
    learning_rate: float,
    l2: float,
) -> float:
    """One exact synchronous gradient step for softplus(-q dot (p - n))."""
    q = _centroid(query, weights, dimensions)
    p = _centroid(positive, weights, dimensions)
    n = _centroid(negative, weights, dimensions)
    margin = _dot(q, tuple(a - b for a, b in zip(p, n, strict=True)))
    factor = _sigmoid(-margin)
    gradients: dict[int, list[float]] = {}
    for indices, vector, sign in (
        (query, tuple(a - b for a, b in zip(p, n, strict=True)), -1.0),
        (positive, q, -1.0),
        (negative, q, 1.0),
    ):
        for index in indices:
            grad = gradients.setdefault(index, [0.0] * dimensions)
            for axis in range(dimensions):
                grad[axis] += sign * factor * vector[axis] / len(indices)
    for index, grad in gradients.items():
        for axis in range(dimensions):
            value = weights[index][axis] - learning_rate * (grad[axis] + l2 * weights[index][axis])
            if not math.isfinite(value) or abs(value) > 1e6:
                raise DataValidationError("centroid training coordinate diverged")
            weights[index][axis] = value
    return _softplus_negative(margin)


def _indices(terms: tuple[str, ...], lookup: dict[str, int]) -> tuple[int, ...]:
    result = tuple(lookup[term] for term in terms if term in lookup)
    if not result:
        raise DataValidationError("validation/holdout entity has no train-vocabulary keyphrases")
    return result


def _validation_map(
    rows: tuple[CentroidTriplet, ...],
    documents: dict[str, tuple[int, ...]],
    reviewers: dict[str, tuple[int, ...]],
    weights: list[list[float]],
    dimensions: int,
) -> float:
    by_document: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        by_document[row.document_id][row.positive_expert_id] = True
        by_document[row.document_id][row.negative_expert_id] = False
    ap_values: list[float] = []
    for document_id in sorted(by_document):
        q = _centroid(documents[document_id], weights, dimensions)
        labels = by_document[document_id]
        ranked = sorted(
            labels,
            key=lambda expert_id: (
                -_dot(q, _centroid(reviewers[expert_id], weights, dimensions)),
                expert_id,
            ),
        )
        hits = 0
        precision_sum = 0.0
        for rank, expert_id in enumerate(ranked, 1):
            if labels[expert_id]:
                hits += 1
                precision_sum += hits / rank
        ap_values.append(precision_sum / sum(labels.values()))
    return math.fsum(ap_values) / len(ap_values)


@dataclass(frozen=True, slots=True)
class CentroidModel:
    config: CentroidConfig
    vocabulary: tuple[str, ...]
    vectors: tuple[tuple[float, ...], ...]
    best_epoch: int
    validation_map: float
    history: tuple[tuple[int, float, float], ...]
    source_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.config, CentroidConfig)
            or not self.vocabulary
            or len(self.vocabulary) > _MAX_TERMS
            or any(not _safe_term(term) for term in self.vocabulary)
            or tuple(sorted(set(self.vocabulary))) != self.vocabulary
            or len(self.vectors) != len(self.vocabulary)
            or any(
                len(row) != self.config.dimensions
                or any(not _finite_real(value) or abs(value) > 1e6 for value in row)
                for row in self.vectors
            )
            or type(self.best_epoch) is not int
            or not 1 <= self.best_epoch <= self.config.epochs
            or not _finite_real(self.validation_map)
            or not 0 <= self.validation_map <= 1
            or len(self.history) != self.config.epochs
            or any(
                type(epoch) is not int
                or epoch != index
                or not _finite_real(loss)
                or loss < 0
                or not _finite_real(metric)
                or not 0 <= metric <= 1
                for index, (epoch, loss, metric) in enumerate(self.history, 1)
            )
            or not self.source_sha256
            or set(self.source_sha256)
            not in (
                {
                    "documents",
                    "experts",
                    "keyphrase-manifest.json",
                    "keyphrases.jsonl",
                    "train",
                    "validation",
                },
                {
                    "documents",
                    "experts",
                    "keyphrase-manifest.json",
                    "keyphrases.jsonl",
                    "train",
                    "validation",
                    "conflicts",
                },
            )
            or any(
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in self.source_sha256.values()
            )
        ):
            raise DataValidationError(
                "centroid model has invalid dimensions, coordinates or provenance"
            )
        object.__setattr__(self, "source_sha256", MappingProxyType(dict(self.source_sha256)))

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "config": self.config.as_dict(),
            "vocabulary": list(self.vocabulary),
            "vectors": [list(row) for row in self.vectors],
            "best_epoch": self.best_epoch,
            "validation_map": self.validation_map,
            "history": [
                {"epoch": epoch, "train_loss": loss, "validation_map": metric}
                for epoch, loss, metric in self.history
            ],
            "source_sha256": dict(self.source_sha256),
        }

    def to_bytes(self) -> bytes:
        payload = self.payload()
        raw = _json(payload)
        if len(raw) > _MAX_MODEL:
            raise DataValidationError("centroid checkpoint exceeds byte limit")
        envelope = _json({"payload": payload, "sha256": _sha(raw)})
        if len(envelope) > _MAX_MODEL:
            raise DataValidationError("centroid checkpoint exceeds byte limit")
        return envelope


def load_centroid_model(raw: bytes) -> CentroidModel:
    """Validate a versioned checkpoint and its canonical payload digest."""
    if type(raw) is not bytes or len(raw) > _MAX_MODEL:
        raise DataValidationError("centroid checkpoint exceeds byte limit")
    try:
        envelope = load_json_text(_decode(raw, "centroid checkpoint"))
    except (ValueError, RecursionError) as error:
        raise DataValidationError("invalid centroid checkpoint JSON") from error
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"payload", "sha256"}
        or not isinstance(envelope["payload"], dict)
    ):
        raise DataValidationError("centroid checkpoint envelope is invalid")
    payload = envelope["payload"]
    if (
        envelope["sha256"] != _sha(_json(payload))
        or set(payload)
        != {
            "schema_version",
            "protocol",
            "config",
            "vocabulary",
            "vectors",
            "best_epoch",
            "validation_map",
            "history",
            "source_sha256",
        }
        or payload["schema_version"] != 1
        or payload["protocol"] != PROTOCOL
    ):
        raise DataValidationError("centroid checkpoint digest or protocol is invalid")
    try:
        config = CentroidConfig(**payload["config"])
        history = tuple(
            (row["epoch"], row["train_loss"], row["validation_map"]) for row in payload["history"]
        )
        model = CentroidModel(
            config,
            tuple(payload["vocabulary"]),
            tuple(tuple(row) for row in payload["vectors"]),
            payload["best_epoch"],
            payload["validation_map"],
            history,
            dict(payload["source_sha256"]),
        )
    except (TypeError, KeyError, ValueError, AttributeError) as error:
        raise DataValidationError("centroid checkpoint fields are invalid") from error
    if model.to_bytes() != raw:
        raise DataValidationError("centroid checkpoint is not canonical")
    return model


def train_keyphrase_centroid(
    inputs: CentroidInputs, *, config: CentroidConfig | None = None
) -> CentroidModel:
    """Train on train triplets only; choose first highest-MAP validation epoch."""
    if not isinstance(inputs, CentroidInputs):
        raise DataValidationError("centroid training requires validated inputs")
    _assert_captured_inputs(inputs)
    chosen = config or CentroidConfig()
    train_docs = {row.document_id for row in inputs.train}
    train_reviewers = {
        reviewer
        for row in inputs.train
        for reviewer in (row.positive_expert_id, row.negative_expert_id)
    }
    vocabulary = tuple(
        sorted(
            {term for document_id in train_docs for term in inputs.submissions[document_id]}
            | {term for reviewer_id in train_reviewers for term in inputs.reviewers[reviewer_id]}
        )
    )
    if len(vocabulary) > _MAX_TERMS or len(vocabulary) * chosen.dimensions > 100_000:
        raise DataValidationError("centroid vocabulary/dimension resource bound exceeded")
    lookup = {term: index for index, term in enumerate(vocabulary)}
    used_docs = (
        train_docs | {row.document_id for row in inputs.validation} | set(inputs.holdout_ids)
    )
    used_reviewers = (
        train_reviewers
        | {
            reviewer
            for row in inputs.validation
            for reviewer in (row.positive_expert_id, row.negative_expert_id)
        }
        | set(inputs.reviewers)
    )
    document_indices = {key: _indices(inputs.submissions[key], lookup) for key in used_docs}
    reviewer_indices = {key: _indices(inputs.reviewers[key], lookup) for key in used_reviewers}
    train_work = (
        sum(
            len(document_indices[row.document_id])
            + len(reviewer_indices[row.positive_expert_id])
            + len(reviewer_indices[row.negative_expert_id])
            + 5
            for row in inputs.train
        )
        * chosen.dimensions
        * chosen.epochs
    )
    validation_work = (
        sum(
            len(document_indices[row.document_id])
            + len(reviewer_indices[row.positive_expert_id])
            + len(reviewer_indices[row.negative_expert_id])
            + 5
            for row in inputs.validation
        )
        * chosen.dimensions
        * chosen.epochs
    )
    score_work = sum(
        (
            len(document_indices[document_id])
            + len(reviewer_indices[reviewer_id])
            + chosen.dimensions
        )
        * chosen.dimensions
        for document_id in inputs.holdout_ids
        for reviewer_id in inputs.reviewers
    )
    if train_work + validation_work + score_work > chosen.max_work:
        raise DataValidationError("centroid coordinate work exceeds max_work")
    # Reproducible model initialization does not generate credentials or secrets.
    rng = random.Random(chosen.seed)  # nosec B311
    weights = [[rng.uniform(-0.1, 0.1) for _ in range(chosen.dimensions)] for _ in vocabulary]
    best_map = -1.0
    best_epoch = 0
    best_vectors: tuple[tuple[float, ...], ...] = ()
    history: list[tuple[int, float, float]] = []
    for epoch in range(1, chosen.epochs + 1):
        losses = [
            _update(
                weights,
                document_indices[row.document_id],
                reviewer_indices[row.positive_expert_id],
                reviewer_indices[row.negative_expert_id],
                dimensions=chosen.dimensions,
                learning_rate=chosen.learning_rate,
                l2=chosen.l2,
            )
            for row in inputs.train
        ]
        metric = _validation_map(
            inputs.validation, document_indices, reviewer_indices, weights, chosen.dimensions
        )
        loss = math.fsum(losses) / len(losses)
        if not math.isfinite(loss):
            raise DataValidationError("centroid training loss is non-finite")
        history.append((epoch, loss, metric))
        if metric > best_map + 1e-12:
            best_map, best_epoch = metric, epoch
            best_vectors = tuple(tuple(row) for row in weights)
    return CentroidModel(
        chosen, vocabulary, best_vectors, best_epoch, best_map, tuple(history), inputs.hashes
    )


def score_keyphrase_centroid(inputs: CentroidInputs, model: CentroidModel) -> tuple[Affinity, ...]:
    """Score only unseen holdout submissions for the existing assignment API."""
    if not isinstance(inputs, CentroidInputs) or not isinstance(model, CentroidModel):
        raise DataValidationError("centroid scoring requires validated inputs and model")
    _assert_captured_inputs(inputs)
    if model.source_sha256 != inputs.hashes:
        raise DataValidationError("centroid checkpoint source hashes do not match current inputs")
    if len(inputs.holdout_ids) * len(inputs.experts) > _MAX_PAIRS:
        raise DataValidationError("centroid affinity matrix exceeds pair limit")
    lookup = {term: index for index, term in enumerate(model.vocabulary)}
    document_indices = {
        document_id: _indices(inputs.submissions[document_id], lookup)
        for document_id in inputs.holdout_ids
    }
    reviewer_indices = {
        expert.id: _indices(inputs.reviewers[expert.id], lookup) for expert in inputs.experts
    }
    work = sum(
        (
            len(document_indices[document_id])
            + len(reviewer_indices[expert.id])
            + model.config.dimensions
        )
        * model.config.dimensions
        for document_id in inputs.holdout_ids
        for expert in inputs.experts
    )
    if work > model.config.max_work:
        raise DataValidationError("centroid inference coordinate work exceeds max_work")
    reviewer_centroids = {
        expert.id: _centroid(reviewer_indices[expert.id], model.vectors, model.config.dimensions)
        for expert in sorted(inputs.experts, key=lambda value: value.id)
    }
    result = []
    for document_id in inputs.holdout_ids:
        query = _centroid(
            document_indices[document_id],
            model.vectors,
            model.config.dimensions,
        )
        for expert_id, centroid in reviewer_centroids.items():
            result.append(
                Affinity(document_id, expert_id, round(_sigmoid(_dot(query, centroid)), 6))
            )
    return tuple(result)


def verify_keyphrase_centroid(inputs: CentroidInputs, checkpoint: bytes) -> bool:
    """Replay the exact captured train/validation snapshots and compare bytes."""
    model = load_centroid_model(checkpoint)
    return train_keyphrase_centroid(inputs, config=model.config).to_bytes() == checkpoint


def _document_row(item: Document) -> dict[str, object]:
    return {
        "id": item.id,
        "title": item.title,
        "abstract": item.abstract,
        "topics": list(item.topics),
        "keywords": list(item.keywords),
        "required_experts": item.required_experts,
        "metadata": dict(item.metadata),
    }


def _expert_row(item: Expert) -> dict[str, object]:
    return {
        "id": item.id,
        "name": item.name,
        "summary": item.summary,
        "topics": list(item.topics),
        "keywords": list(item.keywords),
        "publications": [
            {
                "id": publication.id,
                "title": publication.title,
                "abstract": publication.abstract,
                "year": publication.year,
            }
            for publication in item.publications
        ],
        "capacity": item.capacity,
        "institution": item.institution,
        "regions": list(item.regions),
        "seniority": item.seniority,
        "bids": dict(item.bids),
        "metadata": dict(item.metadata),
    }


def _affinity_bytes(scores: tuple[Affinity, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("document_id", "expert_id", "score"))
    for item in scores:
        writer.writerow((item.document_id, item.expert_id, f"{item.score:.6f}"))
    return stream.getvalue().encode("utf-8", "strict")


def write_keyphrase_centroid_run(
    inputs: CentroidInputs,
    model: CentroidModel,
    directory: str | Path,
) -> dict[str, object]:
    """Replay and publish match-ready holdout artifacts without replacing a target."""
    destination = Path(directory)
    if destination.exists() or destination.is_symlink():
        raise DataValidationError("centroid destination already exists")
    if not verify_keyphrase_centroid(inputs, model.to_bytes()):
        raise DataValidationError("centroid model does not replay from exact sources")
    for name, path in inputs.paths.items():
        if _read(path, len(inputs.raw[name]), name) != inputs.raw[name]:
            raise DataValidationError(f"centroid source changed after loading: {name}")
    # Published normalized rows must still describe the captured source bytes,
    # even if a caller constructed a replacement CentroidInputs object.
    if (
        inputs.documents
        != load_documents_text(_decode(inputs.raw["documents"], "documents"), max_records=10_000)
        or inputs.experts
        != load_experts_text(_decode(inputs.raw["experts"], "experts"), max_records=10_000)
        or inputs.conflicts
        != (
            load_conflicts_text(_decode(inputs.raw["conflicts"], "conflicts"), max_records=10_000)
            if "conflicts" in inputs.raw
            else ()
        )
    ):
        raise DataValidationError("centroid normalized input objects changed after loading")
    scores = score_keyphrase_centroid(inputs, model)
    holdout = set(inputs.holdout_ids)
    artifacts = {
        "model.json": model.to_bytes(),
        "documents.json": _json(
            [
                _document_row(item)
                for item in sorted(inputs.documents, key=lambda value: value.id)
                if item.id in holdout
            ]
        ),
        "experts.json": _json(
            [_expert_row(item) for item in sorted(inputs.experts, key=lambda value: value.id)]
        ),
        "conflicts.json": _json(
            [
                {
                    "document_id": item.document_id,
                    "expert_id": item.expert_id,
                    "reason": item.reason,
                }
                for item in sorted(
                    inputs.conflicts, key=lambda value: (value.document_id, value.expert_id)
                )
                if item.document_id in holdout
            ]
        ),
        "affinities.csv": _affinity_bytes(scores),
    }
    if sum(map(len, artifacts.values())) > 32 * 1024 * 1024:
        raise DataValidationError("centroid run exceeds output byte limit")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "input_sha256": inputs.hashes,
        "files": {
            name: {"bytes": len(raw), "sha256": _sha(raw)} for name, raw in artifacts.items()
        },
        "counts": {
            "train_triplets": len(inputs.train),
            "validation_triplets": len(inputs.validation),
            "holdout_documents": len(holdout),
            "reviewers": len(inputs.experts),
            "affinities": len(scores),
        },
        "selected_epoch": model.best_epoch,
        "validation_map": model.validation_map,
    }
    manifest_raw = _json(manifest)
    if len(manifest_raw) > 64 * 1024:
        raise DataValidationError("centroid manifest exceeds output byte limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as stage:
        staging = Path(stage)
        for name, raw in artifacts.items():
            (staging / name).write_bytes(raw)
        (staging / "manifest.json").write_bytes(manifest_raw)
        _install_directory_no_replace(staging, destination)
    return manifest
