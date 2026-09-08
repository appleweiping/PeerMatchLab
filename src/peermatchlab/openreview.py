"""Loss-aware import of local OpenReview exports."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from itertools import islice
from pathlib import Path
from typing import Any

from peermatchlab.io import load_json_text
from peermatchlab.models import DataValidationError, Document, Expert

_MAX_LOCAL_RECORDS = 100_000
_MAX_LOCAL_FILE_BYTES = 64 * 1024 * 1024
_MAX_LOCAL_LINE_BYTES = 8 * 1024 * 1024


def _bounded_values(
    values: Iterable[Any], limit: int, label: str, *, maximum: int = _MAX_LOCAL_RECORDS
) -> tuple[Any, ...]:
    normalized_limit = _validate_limit(limit, f"{label} limit", maximum)
    try:
        iterator = iter(values)
    except TypeError as error:
        raise DataValidationError(f"{label} must be iterable") from error
    items = tuple(islice(iterator, normalized_limit + 1))
    if len(items) > normalized_limit:
        raise DataValidationError(f"{label} exceeds its configured limit")
    return items


def _validate_limit(value: int, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise DataValidationError(f"{name} must be an integer between 1 and {maximum}")
    return int(value)


def _bounded_utf8_lines(
    path: str | Path, *, max_bytes: int, max_line_bytes: int, label: str
) -> Iterator[tuple[int, str]]:
    max_bytes = _validate_limit(max_bytes, "max_input_file_bytes", _MAX_LOCAL_FILE_BYTES)
    max_line_bytes = _validate_limit(max_line_bytes, "max_line_bytes", _MAX_LOCAL_LINE_BYTES)
    total = 0
    line_number = 0
    with Path(path).open("rb") as stream:
        while True:
            remaining = max_bytes - total
            raw = stream.readline(min(max_line_bytes + 1, remaining + 1))
            if not raw:
                return
            total += len(raw)
            if total > max_bytes:
                raise DataValidationError(f"{label} exceeds max_input_file_bytes")
            line_number += 1
            if len(raw) > max_line_bytes:
                raise DataValidationError(f"{label} line {line_number} exceeds max_line_bytes")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise DataValidationError(f"{label} line {line_number} must be UTF-8") from error
            yield line_number, text.removesuffix("\n").removesuffix("\r")


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataValidationError(f"OpenReview {field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise DataValidationError(f"OpenReview {field} keys must be strings")
    return value


def _value(value: object, field: str) -> object:
    if isinstance(value, Mapping):
        if "value" not in value:
            raise DataValidationError(f"OpenReview {field} object must contain value")
        return value["value"]
    return value


def _text(value: object, field: str, *, required: bool = True) -> str:
    unwrapped = _value(value, field)
    if not isinstance(unwrapped, str):
        raise DataValidationError(f"OpenReview {field} must be a string")
    if required and not unwrapped.strip():
        raise DataValidationError(f"OpenReview {field} must not be empty")
    return unwrapped


def _labels(value: object, field: str, *, allow_scalar: bool = False) -> tuple[str, ...]:
    unwrapped = _value(value, field)
    if unwrapped is None:
        return ()
    if allow_scalar and isinstance(unwrapped, str):
        return (unwrapped,)
    if not isinstance(unwrapped, (list, tuple)) or any(
        not isinstance(item, str) for item in unwrapped
    ):
        raise DataValidationError(f"OpenReview {field} must be an array of strings")
    return tuple(unwrapped)


def openreview_submissions_from_records(
    records: Iterable[object], *, max_records: int = _MAX_LOCAL_RECORDS
) -> tuple[Document, ...]:
    """Convert API-v1 or API-v2-shaped submission-note objects."""

    documents: list[Document] = []
    seen: set[str] = set()
    for record_number, record in enumerate(
        _bounded_values(records, max_records, "OpenReview submission records"), start=1
    ):
        try:
            note = _mapping(record, "submission note")
            note_id = _text(note.get("id"), "submission id")
            content = _mapping(note.get("content"), "submission content")
            title = _text(content.get("title"), "submission title")
            abstract = _text(content.get("abstract", ""), "submission abstract", required=False)
            keywords = _labels(content.get("keywords"), "submission keywords")
            subject_areas = tuple(
                dict.fromkeys(
                    (
                        *_labels(
                            content.get("subject_areas"),
                            "submission subject_areas",
                            allow_scalar=True,
                        ),
                        *_labels(
                            content.get("subject_area"),
                            "submission subject_area",
                            allow_scalar=True,
                        ),
                    )
                )
            )
        except DataValidationError as error:
            raise DataValidationError(
                f"invalid OpenReview submission record {record_number}: {error}"
            ) from error
        if note_id in seen:
            raise DataValidationError(f"duplicate OpenReview submission id: {note_id}")
        seen.add(note_id)
        metadata: dict[str, Any] = {"adapter": "openreview-note-v1-v2"}
        for source_field, output_field in (
            ("forum", "openreview_forum"),
            ("invitation", "openreview_invitation"),
            ("venueid", "openreview_venue"),
        ):
            raw = note.get(source_field)
            if raw is not None:
                metadata[output_field] = _text(raw, source_field, required=False)
        raw_invitations = note.get("invitations")
        if raw_invitations is not None:
            metadata["openreview_invitations"] = _labels(
                raw_invitations, "submission invitations", allow_scalar=True
            )
        content_venue = content.get("venueid")
        if content_venue is not None:
            metadata["openreview_venue"] = _text(
                content_venue, "submission content venueid", required=False
            )
        documents.append(
            Document(
                id=note_id,
                title=title,
                abstract=abstract,
                topics=subject_areas,
                keywords=keywords,
                metadata=metadata,
            )
        )
    if not documents:
        raise DataValidationError("OpenReview submission records contain no notes")
    return tuple(documents)


def load_openreview_submissions(
    path: str | Path,
    *,
    max_records: int = _MAX_LOCAL_RECORDS,
    max_input_file_bytes: int = _MAX_LOCAL_FILE_BYTES,
    max_line_bytes: int = _MAX_LOCAL_LINE_BYTES,
) -> tuple[Document, ...]:
    """Load API-v1 or API-v2-shaped submission notes from local JSONL."""

    max_records = _validate_limit(max_records, "max_records", _MAX_LOCAL_RECORDS)
    records: list[object] = []
    for line_number, line in _bounded_utf8_lines(
        path,
        max_bytes=max_input_file_bytes,
        max_line_bytes=max_line_bytes,
        label="OpenReview submission file",
    ):
        if not line.strip():
            continue
        if len(records) >= max_records:
            raise DataValidationError("OpenReview submission records exceed their configured limit")
        try:
            records.append(load_json_text(line))
        except (DataValidationError, json.JSONDecodeError) as error:
            raise DataValidationError(
                f"invalid OpenReview submission on line {line_number}: {error}"
            ) from error
    return openreview_submissions_from_records(records, max_records=max_records)


def reviewer_ids_to_experts(
    reviewer_ids: Iterable[str],
    *,
    capacity: int,
    max_reviewers: int = _MAX_LOCAL_RECORDS,
) -> tuple[Expert, ...]:
    """Create capacity-bearing expert shells for an external affinity matrix."""

    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
        raise DataValidationError("reviewer capacity must be a non-negative integer")
    normalized = _bounded_values(reviewer_ids, max_reviewers, "reviewer identifiers")
    if any(
        not isinstance(reviewer_id, str)
        or reviewer_id != reviewer_id.strip()
        or not reviewer_id
        or not reviewer_id.isprintable()
        for reviewer_id in normalized
    ):
        raise DataValidationError(
            "reviewer identifiers must be printable strings without surrounding whitespace"
        )
    if len(normalized) != len(set(normalized)):
        raise DataValidationError("reviewer id file contains duplicates")
    if not normalized:
        raise DataValidationError("reviewer id file contains no identifiers")
    return tuple(
        Expert(
            id=reviewer_id,
            name=reviewer_id,
            capacity=capacity,
            metadata={"adapter": "openreview-reviewer-id"},
        )
        for reviewer_id in normalized
    )


def load_reviewer_ids(
    path: str | Path,
    *,
    capacity: int,
    max_reviewers: int = _MAX_LOCAL_RECORDS,
    max_input_file_bytes: int = _MAX_LOCAL_FILE_BYTES,
    max_line_bytes: int = _MAX_LOCAL_LINE_BYTES,
) -> tuple[Expert, ...]:
    """Load reviewer IDs and create capacity-bearing expert shells."""

    max_reviewers = _validate_limit(max_reviewers, "max_reviewers", _MAX_LOCAL_RECORDS)
    reviewer_ids: list[str] = []
    for _line_number, line in _bounded_utf8_lines(
        path,
        max_bytes=max_input_file_bytes,
        max_line_bytes=max_line_bytes,
        label="reviewer id file",
    ):
        if not line:
            continue
        if len(reviewer_ids) >= max_reviewers:
            raise DataValidationError("reviewer identifiers exceed their configured limit")
        reviewer_ids.append(line)
    return reviewer_ids_to_experts(reviewer_ids, capacity=capacity, max_reviewers=max_reviewers)
