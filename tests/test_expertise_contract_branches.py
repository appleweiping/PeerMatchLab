"""Public expertise-model invariants, including malformed persisted-state edges."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import peermatchlab.expertise as expertise_module
from peermatchlab.expertise import (
    EvidenceDocument,
    ExpertiseConfig,
    ExpertiseScore,
    TermContribution,
)
from peermatchlab.models import DataValidationError


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stopwords": 2}, "stopwords must be an array"),
        ({"stopwords": ("too-long",), "max_token_characters": 2}, "bounded"),
        ({"stopwords": ("word",), "max_document_bytes": 2}, "max_document_bytes"),
        ({"stopwords": ("\ud800",)}, "valid Unicode"),
        ({"stopwords": (" ",)}, "bounded"),
        ({"bm25_k1": True}, "bm25_k1"),
        ({"bm25_b": 2.0}, "bm25_b"),
        ({"minimum_output_score": -1.0}, "minimum_output_score"),
    ],
)
def test_config_rejects_malformed_numeric_and_text_controls(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        ExpertiseConfig(**kwargs)  # type: ignore[arg-type]


def test_config_rejects_wrong_mapping_and_file_encodings(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="object with string keys"):
        ExpertiseConfig.from_mapping([])  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="JSON array"):
        ExpertiseConfig.from_mapping({"stopwords": ()})
    invalid = tmp_path / "bad.json"
    invalid.write_bytes(b"\xff")
    with pytest.raises(DataValidationError, match="UTF-8"):
        ExpertiseConfig.from_json(invalid)
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(DataValidationError, match="JSON object"):
        ExpertiseConfig.from_json(invalid)
    invalid.write_text('{"max_input_file_bytes":1}', encoding="utf-8")
    with pytest.raises(DataValidationError, match="max_input_file_bytes"):
        ExpertiseConfig.from_json(invalid)
    with pytest.raises(DataValidationError, match="hard limit"):
        expertise_module._read_bounded_utf8(invalid, max_bytes=1, label="config")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "wrong"}, "kind"),
        ({"fields": []}, "fields must map"),
        ({"fields": {"title": 1}}, "fields must map"),
        ({"tokens": ("",)}, "tokens"),
        ({"year": True}, "year"),
        ({"year": 2300}, "year"),
        ({"id": " bad"}, "identifiers"),
    ],
)
def test_evidence_document_rejects_invalid_public_state(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, Any] = {
        "id": "paper",
        "owner_id": "paper",
        "kind": "submission",
        "fields": {"title": "graph"},
        "tokens": ("graph",),
    }
    values.update(kwargs)
    with pytest.raises(DataValidationError, match=message):
        EvidenceDocument(**values)


@pytest.mark.parametrize(
    "evidence_id",
    [
        "publication:r:",
        "publication:r:position:0",
        "publication:r:position:01",
        "publication:r:other:x",
        "profile:other",
    ],
)
def test_reviewer_evidence_id_namespace_rejects_ambiguous_suffixes(evidence_id: str) -> None:
    assert not expertise_module._reviewer_evidence_id_is_valid(evidence_id, "r")


def test_evidence_config_validation_detects_profile_and_field_mismatch() -> None:
    profile = EvidenceDocument("profile:r", "r", "profile", {"summary": "graph"}, ("graph",))
    with pytest.raises(DataValidationError, match="include_profile"):
        expertise_module._validate_evidence_for_config(
            profile, ExpertiseConfig(include_profile=False)
        )
    wrong_field = EvidenceDocument("p", "p", "submission", {"other": "graph"}, ("graph",))
    with pytest.raises(DataValidationError, match="excluded"):
        expertise_module._validate_evidence_for_config(wrong_field, ExpertiseConfig())
    untrimmed = EvidenceDocument("p", "p", "submission", {"title": " graph "}, ("graph",))
    with pytest.raises(DataValidationError, match="normalized"):
        expertise_module._validate_evidence_for_config(untrimmed, ExpertiseConfig())
    wrong_tokens = EvidenceDocument("p", "p", "submission", {"title": "graph"}, ("other",))
    with pytest.raises(DataValidationError, match="tokens disagree"):
        expertise_module._validate_evidence_for_config(wrong_tokens, ExpertiseConfig())
    thin_profile = EvidenceDocument("profile:r", "r", "profile", {"summary": "graph"}, ("graph",))
    with pytest.raises(DataValidationError, match="minimum_evidence_tokens"):
        expertise_module._validate_evidence_for_config(
            thin_profile, ExpertiseConfig(minimum_evidence_tokens=2)
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"score": float("nan")}, "score must be finite"),
        ({"raw_score": -1.0}, "raw expertise score"),
        ({"model": "other"}, "model or aggregation"),
        ({"evidence_count": True}, "evidence_count"),
        ({"contributions": ("not-a-contribution",)}, "term contributions"),
        ({"evidence_count": 1, "contributions": (TermContribution("x", 1.0),)}, "add to raw_score"),
        ({"selected_evidence_id": " bad"}, "selected_evidence_id"),
    ],
)
def test_score_invariant_rejects_malformed_explanations(
    change: dict[str, object], message: str
) -> None:
    values: dict[str, Any] = {
        "document_id": "p",
        "expert_id": "r",
        "score": 0.0,
        "raw_score": 0.0,
        "model": "tfidf",
        "aggregation": "average",
        "evidence_count": 0,
        "selected_evidence_id": None,
        "contributions": (),
    }
    values.update(change)
    with pytest.raises(DataValidationError, match=message):
        ExpertiseScore(**values)
