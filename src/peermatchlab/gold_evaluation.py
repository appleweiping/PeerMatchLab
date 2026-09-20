"""Bounded, deterministic offline ranking evaluation against judged expertise pairs.

Only judged document/expert pairs enter a rank. Unjudged affinities are ignored,
not silently labeled irrelevant. Missing judged scores follow every scored pair.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from peermatchlab.models import DataValidationError, _identifier_is_valid

MAX_INPUT_FILE_BYTES = 16 * 1024 * 1024
MAX_ROW_BYTES = 64 * 1024
MAX_ROWS = 100_000
MAX_DOCUMENTS = 10_000
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_ID_LENGTH = 512
MAX_GRADE = 1_000
MAX_K = 1_000
MAX_K_VALUES = 20
_CANONICAL_HEADER = ("document_id", "expert_id", "relevance")
_AFFINITY_HEADER = ("document_id", "expert_id", "score")
_PUBLIC_HEADER = (
    "ParticipantID",
    *(cell for index in range(1, 11) for cell in (f"Paper{index}", f"Expertise{index}")),
)
_INTEGER = re.compile(r"[0-9]+\Z", re.ASCII)


@dataclass(frozen=True, slots=True)
class GoldJudgment:
    """One graded, explicitly judged document/expert pair."""

    document_id: str
    expert_id: str
    relevance: int

    def __post_init__(self) -> None:
        _validate_id(self.document_id, "document_id")
        _validate_id(self.expert_id, "expert_id")
        if (
            isinstance(self.relevance, bool)
            or not isinstance(self.relevance, int)
            or not 0 <= self.relevance <= MAX_GRADE
        ):
            raise DataValidationError(f"relevance must be an integer in [0, {MAX_GRADE}]")


@dataclass(frozen=True, slots=True)
class GoldScore:
    """One finite unit-interval affinity, including zero."""

    document_id: str
    expert_id: str
    score: float

    def __post_init__(self) -> None:
        _validate_id(self.document_id, "document_id")
        _validate_id(self.expert_id, "expert_id")
        if isinstance(self.score, bool) or not isinstance(self.score, (float, int)):
            raise DataValidationError("score must be a finite number in [0, 1]")
        try:
            number = float(self.score)
        except OverflowError as error:
            raise DataValidationError("score must be a finite number in [0, 1]") from error
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise DataValidationError("score must be a finite number in [0, 1]")
        object.__setattr__(self, "score", number)


@dataclass(frozen=True, slots=True)
class GoldEvaluationConfig:
    """Finite resource limits and positive-label semantics for one evaluation."""

    k_values: tuple[int, ...] = (1, 3, 5, 10)
    relevance_threshold: int = 1
    strict_coverage: bool = False
    max_input_file_bytes: int = MAX_INPUT_FILE_BYTES
    max_row_bytes: int = MAX_ROW_BYTES
    max_rows: int = MAX_ROWS
    max_documents: int = MAX_DOCUMENTS

    def __post_init__(self) -> None:
        for label, value, ceiling in (
            ("relevance_threshold", self.relevance_threshold, MAX_GRADE),
            ("max_input_file_bytes", self.max_input_file_bytes, MAX_INPUT_FILE_BYTES),
            ("max_row_bytes", self.max_row_bytes, MAX_ROW_BYTES),
            ("max_rows", self.max_rows, MAX_ROWS),
            ("max_documents", self.max_documents, MAX_DOCUMENTS),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
                raise DataValidationError(f"{label} must be an integer in [1, {ceiling}]")
        if not isinstance(self.strict_coverage, bool):
            raise DataValidationError("strict_coverage must be a boolean")
        if not isinstance(self.k_values, tuple) or not 1 <= len(self.k_values) <= MAX_K_VALUES:
            raise DataValidationError(f"k_values must contain 1 to {MAX_K_VALUES} values")
        if any(
            isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K
            for k in self.k_values
        ):
            raise DataValidationError(f"each k must be an integer in [1, {MAX_K}]")
        if len(self.k_values) != len(set(self.k_values)):
            raise DataValidationError("k_values must be unique")
        object.__setattr__(self, "k_values", tuple(sorted(self.k_values)))


def _validate_id(value: object, label: str) -> None:
    if not isinstance(value, str) or not _identifier_is_valid(value) or len(value) > MAX_ID_LENGTH:
        raise DataValidationError(
            f"{label} must be a non-empty printable identifier "
            f"of at most {MAX_ID_LENGTH} characters"
        )


def _integer_grade(value: str, line: int) -> int:
    if not _INTEGER.fullmatch(value) or len(value) > 4:
        raise DataValidationError(f"gold row {line} has an invalid relevance grade")
    grade = int(value)
    if grade > MAX_GRADE:
        raise DataValidationError(f"gold row {line} exceeds maximum relevance grade")
    return grade


def _read_bounded(path: str | Path, *, config: GoldEvaluationConfig) -> bytes:
    source = Path(path)
    with source.open("rb") as stream:
        data = stream.read(config.max_input_file_bytes + 1)
    if len(data) > config.max_input_file_bytes:
        raise DataValidationError("evaluation input exceeds max_input_file_bytes")
    if not data or b"\x00" in data:
        raise DataValidationError("evaluation input must be a non-empty text file without NUL")
    _check_physical_rows(data, max_row_bytes=config.max_row_bytes)
    return data


def _check_physical_rows(data: bytes, *, max_row_bytes: int) -> None:
    """Scan LF, CRLF, and CR boundaries without materializing physical rows."""

    length = 0
    for byte in data:
        if byte in (10, 13):
            length = 0
        else:
            length += 1
        if length > max_row_bytes:
            raise DataValidationError("evaluation input row exceeds max_row_bytes")


def _rows(
    data: bytes, *, label: str, expected_fields: int, config: GoldEvaluationConfig
) -> list[list[str]]:
    if len(data) > config.max_input_file_bytes:
        raise DataValidationError(f"{label} exceeds max_input_file_bytes")
    if not data or b"\x00" in data:
        raise DataValidationError(f"{label} must be non-empty text without NUL")
    _check_physical_rows(data, max_row_bytes=config.max_row_bytes)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeError as error:
        raise DataValidationError(f"{label} must be valid UTF-8") from error
    delimiter = "\t" if "\t" in text.partition("\n")[0] else ","
    if label == "affinity" and delimiter != ",":
        raise DataValidationError("affinity file must be comma-separated")
    parsed: list[list[str]] = []
    try:
        for line, row in enumerate(
            csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True),
            start=1,
        ):
            if not row:
                raise DataValidationError(f"{label} contains an empty row")
            if len(row) != expected_fields:
                raise DataValidationError(
                    f"{label} row {line} has {len(row)} fields; expected {expected_fields}"
                )
            if len(parsed) >= config.max_rows + 1:
                raise DataValidationError(f"{label} exceeds max_rows")
            parsed.append(row)
    except csv.Error as error:
        raise DataValidationError(f"{label} contains malformed CSV/TSV") from error
    if not parsed:
        raise DataValidationError(f"{label} contains no rows")
    return parsed


def load_gold_bytes(
    data: bytes, *, format: str, config: GoldEvaluationConfig | None = None
) -> tuple[GoldJudgment, ...]:
    """Parse canonical triples or the 10-paper public OpenReview gold shape."""

    active = config or GoldEvaluationConfig()
    if format not in {"triples", "openreview"}:
        raise DataValidationError("gold format must be 'triples' or 'openreview'")
    header = _CANONICAL_HEADER if format == "triples" else _PUBLIC_HEADER
    rows = _rows(data, label="gold", expected_fields=len(header), config=active)
    if tuple(rows[0]) != header:
        raise DataValidationError(f"gold header does not match {format} format")
    results: list[GoldJudgment] = []
    seen: set[tuple[str, str]] = set()
    participants: set[str] = set()
    for line, row in enumerate(rows[1:], start=2):
        candidates: Sequence[tuple[str, str, str]]
        if format == "triples":
            candidates = ((row[0], row[1], row[2]),)
        else:
            _validate_id(row[0], "ParticipantID")
            if row[0] in participants:
                raise DataValidationError("duplicate ParticipantID row")
            participants.add(row[0])
            candidates = tuple((row[index], row[0], row[index + 1]) for index in range(1, 21, 2))
            if all(not paper and not grade for paper, _expert, grade in candidates):
                raise DataValidationError("public gold participant has no judged paper")
        for document_id, expert_id, grade_text in candidates:
            if format == "openreview" and not document_id and not grade_text:
                continue
            judgment = GoldJudgment(document_id, expert_id, _integer_grade(grade_text, line))
            pair = (judgment.document_id, judgment.expert_id)
            if pair in seen:
                raise DataValidationError("gold contains a duplicate judged pair")
            seen.add(pair)
            results.append(judgment)
            if len(results) > active.max_rows:
                raise DataValidationError("gold judged pairs exceed max_rows")
    if not results:
        raise DataValidationError("gold must contain at least one judged pair")
    if len({item.document_id for item in results}) > active.max_documents:
        raise DataValidationError("gold exceeds max_documents")
    return tuple(sorted(results, key=lambda item: (item.document_id, item.expert_id)))


def load_affinity_bytes(
    data: bytes, *, config: GoldEvaluationConfig | None = None
) -> tuple[GoldScore, ...]:
    """Parse the existing sparse affinity CSV, with or without its canonical header."""

    active = config or GoldEvaluationConfig()
    rows = _rows(data, label="affinity", expected_fields=3, config=active)
    data_rows = rows[1:] if tuple(rows[0]) == _AFFINITY_HEADER else rows
    if len(data_rows) > active.max_rows:
        raise DataValidationError("affinity exceeds max_rows")
    first_line = 2 if data_rows is not rows else 1
    results: list[GoldScore] = []
    seen: set[tuple[str, str]] = set()
    for line, row in enumerate(data_rows, start=first_line):
        try:
            score = float(row[2])
        except (OverflowError, ValueError) as error:
            raise DataValidationError(f"affinity row {line} has an invalid score") from error
        item = GoldScore(row[0], row[1], score)
        pair = (item.document_id, item.expert_id)
        if pair in seen:
            raise DataValidationError("affinity contains a duplicate scored pair")
        seen.add(pair)
        results.append(item)
    if not results:
        raise DataValidationError("affinity must contain at least one scored pair")
    return tuple(sorted(results, key=lambda item: (item.document_id, item.expert_id)))


def evaluate_gold(
    judgments: Sequence[GoldJudgment],
    affinities: Sequence[GoldScore],
    *,
    config: GoldEvaluationConfig | None = None,
) -> dict[str, Any]:
    """Calculate macro and per-document P/R/Hits/AP at each requested cutoff."""

    active = config or GoldEvaluationConfig()
    if not judgments or any(not isinstance(item, GoldJudgment) for item in judgments):
        raise DataValidationError("judgments must contain GoldJudgment values")
    if any(not isinstance(item, GoldScore) for item in affinities):
        raise DataValidationError("affinities must contain GoldScore values")
    if len(judgments) > active.max_rows or len(affinities) > active.max_rows:
        raise DataValidationError("evaluation pairs exceed max_rows")
    gold: dict[str, dict[str, int]] = defaultdict(dict)
    for judgment in judgments:
        if judgment.expert_id in gold[judgment.document_id]:
            raise DataValidationError("duplicate judged pair")
        gold[judgment.document_id][judgment.expert_id] = judgment.relevance
    if len(gold) > active.max_documents:
        raise DataValidationError("evaluation exceeds max_documents")
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    ignored = 0
    for affinity in affinities:
        if affinity.expert_id in scores[affinity.document_id]:
            raise DataValidationError("duplicate scored pair")
        scores[affinity.document_id][affinity.expert_id] = affinity.score
        if affinity.expert_id not in gold.get(affinity.document_id, {}):
            ignored += 1
    documents: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    missing_scores = 0
    for document_id, grades in sorted(gold.items()):
        judged_scored = scores.get(document_id, {}).keys() & grades.keys()
        if not judged_scored:
            raise DataValidationError("every judged document needs at least one scored judged pair")
        missing = len(grades) - len(judged_scored)
        missing_scores += missing
        if active.strict_coverage and missing:
            raise DataValidationError("strict coverage rejects missing judged scores")
        positive_count = sum(grade >= active.relevance_threshold for grade in grades.values())
        if not positive_count:
            skipped.append({"document_id": document_id, "reason": "no_judged_positive"})
            continue
        ranked = sorted(
            grades,
            key=lambda expert_id: (
                expert_id not in judged_scored,
                -scores[document_id][expert_id] if expert_id in judged_scored else 0.0,
                expert_id,
            ),
        )
        metrics: dict[str, dict[str, float]] = {}
        for k in active.k_values:
            hits = 0
            precision_sum = 0.0
            for rank, expert_id in enumerate(ranked[:k], start=1):
                if grades[expert_id] >= active.relevance_threshold:
                    hits += 1
                    precision_sum += hits / rank
            metrics[str(k)] = {
                "precision": hits / k,
                "recall": hits / positive_count,
                "hits": float(hits > 0),
                "average_precision": precision_sum / positive_count,
            }
        documents.append(
            {
                "document_id": document_id,
                "judged_pairs": len(grades),
                "judged_positives": positive_count,
                "missing_scores": missing,
                "metrics": metrics,
            }
        )
    macro: dict[str, dict[str, float]] = {}
    for k in active.k_values:
        label = str(k)
        macro[label] = {
            metric: sum(item["metrics"][label][metric] for item in documents) / len(documents)
            if documents
            else 0.0
            for metric in ("precision", "recall", "hits", "average_precision")
        }
    return {
        "schema_version": 1,
        "evaluation_universe": "judged_pairs_only",
        "rank_semantics": "descending_score_then_expert_id; missing_scores_last_then_expert_id",
        "positive_relevance_threshold": active.relevance_threshold,
        "k_values": list(active.k_values),
        "strict_coverage": active.strict_coverage,
        "counts": {
            "judged_documents": len(gold),
            "evaluated_documents": len(documents),
            "skipped_documents": len(skipped),
            "judged_pairs": len(judgments),
            "scored_pairs": len(affinities),
            "ignored_unjudged_scores": ignored,
            "missing_judged_scores": missing_scores,
        },
        "macro": macro,
        "documents": documents,
        "skipped": skipped,
    }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)


def evaluate_gold_files(
    gold_path: str | Path,
    affinity_path: str | Path,
    output_path: str | Path,
    *,
    format: str = "triples",
    config: GoldEvaluationConfig | None = None,
) -> dict[str, Any]:
    """Validate bounded sources and atomically publish a provenance-bearing JSON report."""

    active = config or GoldEvaluationConfig()
    gold_source, affinity_source, destination = map(Path, (gold_path, affinity_path, output_path))
    if _same_file(destination, gold_source) or _same_file(destination, affinity_source):
        raise DataValidationError("evaluation output must not alias an input")
    gold_bytes = _read_bounded(gold_source, config=active)
    affinity_bytes = _read_bounded(affinity_source, config=active)
    report = evaluate_gold(
        load_gold_bytes(gold_bytes, format=format, config=active),
        load_affinity_bytes(affinity_bytes, config=active),
        config=active,
    )
    fingerprint = hashlib.sha256(_canonical_bytes(report)).hexdigest()
    report["result_fingerprint_sha256"] = fingerprint
    report["source"] = {
        "gold": {
            "format": format,
            "bytes": len(gold_bytes),
            "sha256": hashlib.sha256(gold_bytes).hexdigest(),
        },
        "affinities": {
            "bytes": len(affinity_bytes),
            "sha256": hashlib.sha256(affinity_bytes).hexdigest(),
        },
    }
    output = _canonical_bytes(report)
    if len(output) > MAX_REPORT_BYTES:
        raise DataValidationError("evaluation report exceeds output byte limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            if stream.write(output) != len(output):
                raise OSError("short write while publishing evaluation report")
            stream.flush()
            os.fsync(stream.fileno())
        # Re-check aliases after work but before the atomic replacement.
        if _same_file(destination, gold_source) or _same_file(destination, affinity_source):
            raise DataValidationError("evaluation output must not alias an input")
        if (
            _read_bounded(gold_source, config=active) != gold_bytes
            or _read_bounded(affinity_source, config=active) != affinity_bytes
        ):
            raise DataValidationError("evaluation input changed during report generation")
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return report
