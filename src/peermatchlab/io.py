"""Strict JSON and JSONL adapters for reproducible matching runs."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from peermatchlab.models import (
    Assignment,
    Conflict,
    DataValidationError,
    Document,
    Expert,
    MatchPlan,
    Publication,
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_json_text(text: str) -> object:
    """Parse strict JSON while rejecting duplicate fields and non-finite numbers."""

    return json.loads(
        text,
        parse_constant=_reject_non_finite_json,
        parse_float=_parse_finite_json_float,
        object_pairs_hook=_unique_json_object,
    )


def _records(path: str | Path) -> list[Mapping[str, Any]]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    try:
        parsed = load_json_text(text)
    except json.JSONDecodeError:
        try:
            parsed = [load_json_text(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise DataValidationError(f"invalid JSON in {file_path}: {error}") from error
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
        raise DataValidationError(f"{file_path} must contain an object, array, or JSONL objects")
    return parsed


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


def load_documents(path: str | Path) -> tuple[Document, ...]:
    """Load and validate documents from JSON or JSONL."""

    result: list[Document] = []
    for row in _records(path):
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


def load_experts(path: str | Path) -> tuple[Expert, ...]:
    """Load and validate experts with nested publications."""

    result: list[Expert] = []
    for row in _records(path):
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
            unknown_publication = set(publication) - {"title", "abstract", "year"}
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


def load_conflicts(path: str | Path | None) -> tuple[Conflict, ...]:
    """Load hard exclusions; ``None`` is an empty conflict set."""

    if path is None:
        return ()
    result: list[Conflict] = []
    for row in _records(path):
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

    return {
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


def plan_from_dict(value: Mapping[str, Any]) -> MatchPlan:
    """Deserialize a plan for independent auditing."""

    if not isinstance(value, Mapping):
        raise DataValidationError("plan must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise DataValidationError("plan field names must be strings")
    allowed = {"assignments", "unmet", "strategy", "total_score", "audit"}
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
    )


def write_json(path: str | Path, value: object) -> None:
    """Write stable, human-readable UTF-8 JSON with a trailing newline."""

    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
