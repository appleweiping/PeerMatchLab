"""Strict JSON and JSONL adapters for reproducible matching runs."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from peermatchlab.models import (
    Assignment,
    AssignmentDiagnostics,
    Conflict,
    DataValidationError,
    DemandDiagnostic,
    Document,
    Expert,
    FeasibilityStatus,
    MatchPlan,
    Publication,
    UnmetReason,
)

_MAX_JSON_NESTING = 200


def _validate_json_nesting(text: str) -> None:
    """Reject excessive nesting independently of the runtime JSON parser."""

    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_JSON_NESTING:
                raise DataValidationError("JSON nesting exceeds the supported depth")
        elif character in "]}" and depth:
            depth -= 1


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_json_text(text: str) -> object:
    """Parse strict JSON while rejecting duplicate fields and non-finite numbers."""

    _validate_json_nesting(text)
    try:
        return json.loads(
            text,
            parse_constant=_reject_non_finite_json,
            parse_float=_parse_finite_json_float,
            object_pairs_hook=_unique_json_object,
        )
    except RecursionError as error:
        raise DataValidationError("JSON nesting exceeds the supported depth") from error


def _record_limit(value: int | None, source: str) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > sys.maxsize - 1
    ):
        raise DataValidationError(f"{source} record limit must be a bounded positive integer")
    return int(value)


def _skip_json_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position] in " \t\r\n":
        position += 1
    return position


def _bounded_json_array_records(
    text: str, source: str, max_records: int
) -> list[Mapping[str, Any]]:
    _validate_json_nesting(text)
    decoder = json.JSONDecoder(
        parse_constant=_reject_non_finite_json,
        parse_float=_parse_finite_json_float,
        object_pairs_hook=_unique_json_object,
    )
    position = _skip_json_whitespace(text, 0) + 1
    rows: list[Mapping[str, Any]] = []
    position = _skip_json_whitespace(text, position)
    if position < len(text) and text[position] == "]":
        position = _skip_json_whitespace(text, position + 1)
        if position != len(text):
            raise DataValidationError(f"invalid JSON in {source}: trailing data")
        return rows
    while True:
        try:
            value, position = decoder.raw_decode(text, position)
        except RecursionError as error:
            raise DataValidationError("JSON nesting exceeds the supported depth") from error
        except json.JSONDecodeError as error:
            raise DataValidationError(f"invalid JSON in {source}: {error}") from error
        if not isinstance(value, dict):
            raise DataValidationError(f"{source} must contain an object, array, or JSONL objects")
        rows.append(value)
        if len(rows) > max_records:
            raise DataValidationError(f"{source} exceeds its configured record limit")
        position = _skip_json_whitespace(text, position)
        if position >= len(text):
            raise DataValidationError(f"invalid JSON in {source}: unterminated array")
        delimiter = text[position]
        position = _skip_json_whitespace(text, position + 1)
        if delimiter == "]":
            if position != len(text):
                raise DataValidationError(f"invalid JSON in {source}: trailing data")
            return rows
        if delimiter != ",":
            raise DataValidationError(f"invalid JSON in {source}: expected ',' or ']'")


def _records_from_text(
    text: str, source: str, *, max_records: int | None = None
) -> list[Mapping[str, Any]]:
    limit = _record_limit(max_records, source)
    position = _skip_json_whitespace(text, 0)
    if limit is not None and position < len(text) and text[position] == "[":
        return _bounded_json_array_records(text, source, limit)
    try:
        parsed = load_json_text(text)
    except json.JSONDecodeError:
        rows: list[Mapping[str, Any]] = []
        try:
            for line in StringIO(text):
                if not line.strip():
                    continue
                if limit is not None and len(rows) >= limit:
                    raise DataValidationError(f"{source} exceeds its configured record limit")
                value = load_json_text(line)
                if not isinstance(value, dict):
                    raise DataValidationError(
                        f"{source} must contain an object, array, or JSONL objects"
                    )
                rows.append(value)
        except json.JSONDecodeError as error:
            raise DataValidationError(f"invalid JSON in {source}: {error}") from error
        return rows
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
        raise DataValidationError(f"{source} must contain an object, array, or JSONL objects")
    if limit is not None and len(parsed) > limit:
        raise DataValidationError(f"{source} exceeds its configured record limit")
    return parsed


def _records(path: str | Path) -> list[Mapping[str, Any]]:
    file_path = Path(path)
    return _records_from_text(file_path.read_text(encoding="utf-8"), str(file_path))


def _reject_non_finite_json(value: str) -> None:
    raise DataValidationError(f"non-finite JSON number is not allowed: {value}")


def _parse_finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise DataValidationError(f"non-finite JSON number is not allowed: {value}")
    return result


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise DataValidationError(f"{field} must be a string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{field} must be an integer")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise DataValidationError(f"{field} must be a boolean")
    return value


def _optional_integer(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field)


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError(f"{field} must be a number")
    try:
        result = float(value)
    except OverflowError as error:
        raise DataValidationError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise DataValidationError(f"{field} must be finite")
    return result


def _tuple_of_strings(value: object, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DataValidationError(f"{field} must be an array of strings")
    return tuple(value)


def _documents_from_records(rows: Iterable[Mapping[str, Any]]) -> tuple[Document, ...]:
    result: list[Document] = []
    for row in rows:
        allowed = {"id", "title", "abstract", "topics", "keywords", "required_experts", "metadata"}
        unknown = set(row) - allowed
        if unknown:
            raise DataValidationError(f"unknown document fields: {sorted(unknown)}")
        try:
            result.append(
                Document(
                    id=_string(row["id"], "document id"),
                    title=_string(row["title"], "document title"),
                    abstract=_string(row.get("abstract", ""), "document abstract"),
                    topics=_tuple_of_strings(row.get("topics"), "topics"),
                    keywords=_tuple_of_strings(row.get("keywords"), "keywords"),
                    required_experts=_optional_integer(
                        row.get("required_experts"), "required_experts"
                    ),
                    metadata=_mapping(row.get("metadata"), "metadata"),
                )
            )
        except KeyError as error:
            raise DataValidationError(f"missing document field: {error.args[0]}") from error
    _unique_ids(result, "document")
    return tuple(result)


def load_documents(path: str | Path) -> tuple[Document, ...]:
    """Load and validate documents from JSON or JSONL."""

    return _documents_from_records(_records(path))


def load_documents_text(
    text: str, *, source: str = "document input", max_records: int | None = None
) -> tuple[Document, ...]:
    """Validate documents from an already bounded, decoded byte snapshot."""

    return _documents_from_records(_records_from_text(text, source, max_records=max_records))


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise DataValidationError(f"{field} must be an object")
    return value


def _required_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataValidationError(f"{field} must be an object")
    return value


def _experts_from_records(rows: Iterable[Mapping[str, Any]]) -> tuple[Expert, ...]:
    result: list[Expert] = []
    for row in rows:
        allowed = {
            "id",
            "name",
            "summary",
            "topics",
            "keywords",
            "publications",
            "capacity",
            "institution",
            "regions",
            "seniority",
            "bids",
            "metadata",
        }
        unknown = set(row) - allowed
        if unknown:
            raise DataValidationError(f"unknown expert fields: {sorted(unknown)}")
        publications_value = row.get("publications", [])
        if not isinstance(publications_value, list):
            raise DataValidationError("publications must be an array")
        publications: list[Publication] = []
        for publication in publications_value:
            if not isinstance(publication, dict):
                raise DataValidationError("each publication must be an object")
            unknown_publication = set(publication) - {"id", "title", "abstract", "year"}
            if unknown_publication:
                raise DataValidationError(
                    f"unknown publication fields: {sorted(unknown_publication)}"
                )
            try:
                publications.append(
                    Publication(
                        title=_string(publication["title"], "publication title"),
                        abstract=_string(publication.get("abstract", ""), "publication abstract"),
                        year=_optional_integer(publication.get("year"), "publication year"),
                        id=_optional_string(publication.get("id"), "publication id"),
                    )
                )
            except KeyError as error:
                raise DataValidationError("publication title is required") from error
        try:
            result.append(
                Expert(
                    id=_string(row["id"], "expert id"),
                    name=_string(row["name"], "expert name"),
                    summary=_string(row.get("summary", ""), "expert summary"),
                    topics=_tuple_of_strings(row.get("topics"), "topics"),
                    keywords=_tuple_of_strings(row.get("keywords"), "keywords"),
                    publications=tuple(publications),
                    capacity=_integer(row.get("capacity", 3), "capacity"),
                    institution=_optional_string(row.get("institution"), "institution"),
                    regions=_tuple_of_strings(row.get("regions"), "regions"),
                    seniority=_number(row.get("seniority", 0.5), "seniority"),
                    bids={
                        _string(key, "bid document id"): _number(value, "bid value")
                        for key, value in _mapping(row.get("bids"), "bids").items()
                    },
                    metadata=_mapping(row.get("metadata"), "metadata"),
                )
            )
        except KeyError as error:
            raise DataValidationError(f"missing expert field: {error.args[0]}") from error
    _unique_ids(result, "expert")
    return tuple(result)


def load_experts(path: str | Path) -> tuple[Expert, ...]:
    """Load and validate experts with nested publications."""

    return _experts_from_records(_records(path))


def load_experts_text(
    text: str, *, source: str = "expert input", max_records: int | None = None
) -> tuple[Expert, ...]:
    """Validate experts from an already bounded, decoded byte snapshot."""

    return _experts_from_records(_records_from_text(text, source, max_records=max_records))


def load_conflicts(path: str | Path | None) -> tuple[Conflict, ...]:
    """Load hard exclusions; ``None`` is an empty conflict set."""

    if path is None:
        return ()
    return _conflicts_from_records(_records(path))


def load_conflicts_text(
    text: str, *, source: str = "conflict input", max_records: int | None = None
) -> tuple[Conflict, ...]:
    """Validate conflicts from an already bounded, decoded byte snapshot."""

    return _conflicts_from_records(_records_from_text(text, source, max_records=max_records))


def _conflicts_from_records(rows: Iterable[Mapping[str, Any]]) -> tuple[Conflict, ...]:
    result: list[Conflict] = []
    for row in rows:
        unknown = set(row) - {"document_id", "expert_id", "reason"}
        if unknown:
            raise DataValidationError(f"unknown conflict fields: {sorted(unknown)}")
        try:
            result.append(
                Conflict(
                    document_id=_string(row["document_id"], "conflict document_id"),
                    expert_id=_string(row["expert_id"], "conflict expert_id"),
                    reason=_string(row.get("reason", "declared conflict"), "conflict reason"),
                )
            )
        except KeyError as error:
            raise DataValidationError(f"missing conflict field: {error.args[0]}") from error
    return tuple(result)


def _unique_ids(values: Iterable[Document | Expert], label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value.id in seen:
            raise DataValidationError(f"duplicate {label} id: {value.id}")
        seen.add(value.id)


def plan_to_dict(plan: MatchPlan) -> dict[str, object]:
    """Serialize a plan while preserving deterministic assignment order."""

    result: dict[str, object] = {
        "strategy": plan.strategy,
        "total_score": plan.total_score,
        "unmet": dict(plan.unmet),
        "assignments": [
            {
                "document_id": item.document_id,
                "expert_id": item.expert_id,
                "score": item.score,
                "rank": item.rank,
                "components": dict(item.components),
            }
            for item in plan.assignments
        ],
    }
    if plan.diagnostics is not None:
        result["diagnostics"] = {
            "status": plan.diagnostics.status.value,
            "certified": plan.diagnostics.certified,
            "requested": plan.diagnostics.requested,
            "assigned": plan.diagnostics.assigned,
            "unmet": plan.diagnostics.unmet,
            "documents": [
                {
                    "document_id": item.document_id,
                    "requested": item.requested,
                    "assigned": item.assigned,
                    "unmet": item.unmet,
                    "reason_codes": [reason.value for reason in item.reason_codes],
                    "evidence": dict(item.evidence),
                    "saturated_experts": list(item.saturated_experts),
                }
                for item in plan.diagnostics.documents
            ],
        }
    return result


def _diagnostics_from_dict(value: object) -> AssignmentDiagnostics | None:
    if value is None:
        return None
    data = _required_mapping(value, "diagnostics")
    allowed = {"status", "certified", "requested", "assigned", "unmet", "documents"}
    unknown = set(data) - allowed
    if unknown:
        raise DataValidationError(f"unknown diagnostic fields: {sorted(unknown)}")
    rows = data.get("documents")
    if not isinstance(rows, list):
        raise DataValidationError("diagnostic documents must be an array")
    documents: list[DemandDiagnostic] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DataValidationError(f"diagnostic document {index} must be an object")
        row_allowed = {
            "document_id",
            "requested",
            "assigned",
            "unmet",
            "reason_codes",
            "evidence",
            "saturated_experts",
        }
        row_unknown = set(row) - row_allowed
        if row_unknown:
            raise DataValidationError(f"unknown document diagnostic fields: {sorted(row_unknown)}")
        evidence = _required_mapping(row.get("evidence"), "diagnostic evidence")
        try:
            documents.append(
                DemandDiagnostic(
                    document_id=_string(row["document_id"], "diagnostic document_id"),
                    requested=_integer(row["requested"], "diagnostic requested"),
                    assigned=_integer(row["assigned"], "diagnostic assigned"),
                    unmet=_integer(row["unmet"], "diagnostic unmet"),
                    reason_codes=tuple(
                        UnmetReason(reason)
                        for reason in _tuple_of_strings(
                            row.get("reason_codes"), "diagnostic reason_codes"
                        )
                    ),
                    evidence={
                        _string(key, "diagnostic evidence name"): _integer(
                            count, "diagnostic evidence count"
                        )
                        for key, count in evidence.items()
                    },
                    saturated_experts=_tuple_of_strings(
                        row.get("saturated_experts"), "saturated_experts"
                    ),
                )
            )
        except KeyError as error:
            raise DataValidationError(
                f"missing document diagnostic field: {error.args[0]}"
            ) from error
        except ValueError as error:
            raise DataValidationError("diagnostic contains an unknown reason code") from error
    try:
        result = AssignmentDiagnostics(
            status=FeasibilityStatus(_string(data["status"], "diagnostic status")),
            requested=_integer(data["requested"], "diagnostic requested"),
            assigned=_integer(data["assigned"], "diagnostic assigned"),
            unmet=_integer(data["unmet"], "diagnostic unmet"),
            documents=tuple(documents),
        )
    except KeyError as error:
        raise DataValidationError(f"missing diagnostic field: {error.args[0]}") from error
    except ValueError as error:
        raise DataValidationError("diagnostic status is not supported") from error
    certified = _boolean(data.get("certified"), "diagnostic certified")
    if certified is not result.certified:
        raise DataValidationError("diagnostic certified flag does not match its status")
    return result


def plan_from_dict(value: Mapping[str, Any]) -> MatchPlan:
    """Deserialize a plan for independent auditing."""

    if not isinstance(value, Mapping):
        raise DataValidationError("plan must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise DataValidationError("plan field names must be strings")
    allowed = {"assignments", "unmet", "strategy", "total_score", "diagnostics", "audit"}
    unknown = set(value) - allowed
    if unknown:
        raise DataValidationError(f"unknown plan fields: {sorted(unknown)}")
    rows = value.get("assignments", [])
    if not isinstance(rows, list):
        raise DataValidationError("assignments must be an array")
    assignments: list[Assignment] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DataValidationError(f"assignment {index} must be an object")
        if any(not isinstance(key, str) for key in row):
            raise DataValidationError("assignment field names must be strings")
        unknown_assignment = set(row) - {
            "document_id",
            "expert_id",
            "score",
            "rank",
            "components",
        }
        if unknown_assignment:
            raise DataValidationError(f"unknown assignment fields: {sorted(unknown_assignment)}")
        components = _required_mapping(row.get("components", {}), "assignment components")
        try:
            assignments.append(
                Assignment(
                    document_id=_string(row["document_id"], "assignment document_id"),
                    expert_id=_string(row["expert_id"], "assignment expert_id"),
                    score=_number(row["score"], "assignment score"),
                    rank=_integer(row["rank"], "assignment rank"),
                    components={
                        _string(key, "component name"): _number(item, "component value")
                        for key, item in components.items()
                    },
                )
            )
        except KeyError as error:
            raise DataValidationError(f"missing assignment field: {error.args[0]}") from error
    unmet_value = _required_mapping(value.get("unmet", {}), "unmet")
    unmet = {
        _string(key, "unmet document id"): _integer(item, "unmet count")
        for key, item in unmet_value.items()
    }
    strategy = _string(value.get("strategy", "unknown"), "plan strategy")
    default_total = sum(item.score for item in assignments)
    return MatchPlan(
        assignments=tuple(assignments),
        unmet=unmet,
        strategy=strategy,
        total_score=_number(value.get("total_score", default_total), "plan total_score"),
        diagnostics=_diagnostics_from_dict(value.get("diagnostics")),
    )


def write_json(path: str | Path, value: object) -> None:
    """Write stable, human-readable UTF-8 JSON with a trailing newline."""

    try:
        serialized = json.dumps(
            value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
    except RecursionError as error:
        raise DataValidationError("JSON value exceeds the supported nesting depth") from error
    Path(path).write_text(serialized + "\n", encoding="utf-8", newline="\n")
