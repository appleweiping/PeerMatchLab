"""Loss-aware import of local OpenReview exports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from peermatchlab.io import load_json_text
from peermatchlab.models import DataValidationError, Document, Expert


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
    if not isinstance(unwrapped, list) or any(not isinstance(item, str) for item in unwrapped):
        raise DataValidationError(f"OpenReview {field} must be an array of strings")
    return tuple(unwrapped)


def load_openreview_submissions(path: str | Path) -> tuple[Document, ...]:
    """Load API-v1 or API-v2-shaped submission notes from local JSONL."""

    source = Path(path)
    documents: list[Document] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            note = _mapping(load_json_text(line), "submission note")
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
                f"invalid OpenReview submission on line {line_number}: {error}"
            ) from error
        if note_id in seen:
            raise DataValidationError(f"duplicate OpenReview submission id: {note_id}")
        seen.add(note_id)
        metadata: dict[str, str] = {"adapter": "openreview-note-v1-v2"}
        for source_field, output_field in (
            ("forum", "openreview_forum"),
            ("invitation", "openreview_invitation"),
            ("venueid", "openreview_venue"),
        ):
            raw = note.get(source_field)
            if raw is not None:
                metadata[output_field] = _text(raw, source_field, required=False)
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
        raise DataValidationError("OpenReview submission file contains no notes")
    return tuple(documents)


def load_reviewer_ids(path: str | Path, *, capacity: int) -> tuple[Expert, ...]:
    """Create capacity-bearing expert shells for an external affinity matrix."""

    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
        raise DataValidationError("reviewer capacity must be a non-negative integer")
    reviewer_ids = [
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(reviewer_ids) != len(set(reviewer_ids)):
        raise DataValidationError("reviewer id file contains duplicates")
    if not reviewer_ids:
        raise DataValidationError("reviewer id file contains no identifiers")
    return tuple(
        Expert(
            id=reviewer_id,
            name=reviewer_id,
            capacity=capacity,
            metadata={"adapter": "openreview-reviewer-id"},
        )
        for reviewer_id in reviewer_ids
    )
