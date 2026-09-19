"""Adversarial provenance and strict snapshot-boundary branch tests."""

from __future__ import annotations

import ctypes
import errno
import json
from pathlib import Path
from typing import Any

import pytest

import peermatchlab.expertise_io as io_module
from peermatchlab.expertise import ExpertiseConfig
from peermatchlab.expertise_io import (
    ExpertiseConfigSource,
    LocalExpertiseInputs,
    load_local_domain_expertise_inputs,
)
from peermatchlab.models import DataValidationError


def _source(tmp_path: Path) -> LocalExpertiseInputs:
    documents = tmp_path / "documents.json"
    experts = tmp_path / "experts.json"
    documents.write_text(json.dumps([{"id": "p", "title": "Graph search"}]), encoding="utf-8")
    experts.write_text(
        json.dumps([{"id": "r", "name": "R", "summary": "Graph search"}]), encoding="utf-8"
    )
    return load_local_domain_expertise_inputs(documents, experts)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("adapter", "other", "adapter is not supported"),
        ("documents", (), "Document objects"),
        ("experts", (), "Expert objects"),
        ("source_files", [], "source_files must map"),
        ("source_files", {"documents": Path("x")}, "source_files do not match"),
        ("source_bytes", [], "source_bytes must map"),
        ("source_bytes", {"documents": b"[]"}, "source_bytes do not match"),
        ("source_bytes", {"documents": b"[]", "experts": b"[]"}, "do not match"),
    ],
)
def test_captured_source_boundary_rejects_untrusted_cache(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    source = _source(tmp_path)
    kwargs: dict[str, Any] = {
        "documents": source.documents,
        "experts": source.experts,
        "source_files": source.source_files,
        "source_bytes": source.source_bytes,
        "adapter": source.adapter,
        "adapter_parameters": source.adapter_parameters,
    }
    kwargs[field] = value
    with pytest.raises(DataValidationError, match=message):
        LocalExpertiseInputs(**kwargs)


def test_captured_source_detects_byte_ceiling_and_domain_mismatch(tmp_path: Path) -> None:
    source = _source(tmp_path)
    values = dict(source.adapter_parameters)
    values["max_input_file_bytes"] = 1
    with pytest.raises(DataValidationError, match="source bytes exceed"):
        LocalExpertiseInputs(
            source.documents,
            source.experts,
            source.source_files,
            source.source_bytes,
            source.adapter,
            values,
        )
    changed = dict(source.source_bytes)
    changed["documents"] = b'[{"id":"other","title":"Graph search"}]'
    with pytest.raises(DataValidationError, match="do not match the immutable source bytes"):
        LocalExpertiseInputs(
            source.documents,
            source.experts,
            source.source_files,
            changed,
            source.adapter,
            source.adapter_parameters,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda values: values.pop("max_reviewers"), "adapter contract"),
        (lambda values: values.update(max_reviewers=True), "max_reviewers"),
        (lambda values: values.update(max_reviewers=0), "max_reviewers"),
        (lambda values: values.update(max_reviewers="many"), "max_reviewers"),
        (lambda values: values.update(reviewer_capacity=-1), "adapter contract"),
    ],
)
def test_source_parameter_contract_rejects_invalid_limits(change: Any, message: str) -> None:
    values = dict(io_module._source_parameters(ExpertiseConfig()))
    change(values)
    with pytest.raises(DataValidationError, match=message):
        io_module._validate_source_parameters("peermatchlab-local-domain-v1", values)


def test_openreview_capacity_parameter_is_not_a_boolean() -> None:
    values = dict(io_module._source_parameters(ExpertiseConfig(), reviewer_capacity=1))
    values["reviewer_capacity"] = True
    with pytest.raises(DataValidationError, match="reviewer_capacity"):
        io_module._validate_source_parameters("openreview-local-expertise-snapshot-v1", values)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\n", "contains no records"),
        (b"[]\n", "JSON object"),
        (b"{bad}\n", "invalid"),
        (b"{}\n{}\n", "record limit"),
        (b"\xff", "UTF-8"),
    ],
)
def test_jsonl_reader_rejects_empty_malformed_and_excess_records(
    payload: bytes, message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        io_module._jsonl_from_bytes(payload, label="test snapshot", max_records=1)
    assert io_module._jsonl_from_bytes(b"\n", label="empty", max_records=1, allow_empty=True) == ()


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ({"id": "r", "content": {}}, "must contain names"),
        ({"id": "r", "content": {"names": [{}]}}, "fullname"),
        ({"id": "r", "content": {"names": [{"fullname": "R", "preferred": 1}]}}, "preferred"),
        ({"id": "r", "content": {"names": [{"fullname": "R"}], "keywords": [1]}}, "keywords"),
        ({"id": "r", "content": {"names": [{"fullname": "R"}], "bio": 3}}, "bio"),
    ],
)
def test_offline_profile_contract_rejects_invalid_evidence(
    profile: dict[str, object], message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        io_module._profile_to_parts(profile)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"reviewer_id": "r", "note": {}}, "publication id"),
        (
            {"reviewer_id": "r", "note": {"id": "p", "content": {"title": "T", "year": True}}},
            "year",
        ),
        (
            {"reviewer_id": "r", "note": {"id": "p", "content": {"title": "T", "year": 2300}}},
            "year",
        ),
        ({"reviewer_id": "r", "note": {"id": "p", "content": {"title": ""}}}, "title"),
        (
            {"reviewer_id": "r", "note": {"id": "p", "content": {"title": "T"}}, "extra": 1},
            "exactly",
        ),
    ],
)
def test_offline_publication_contract_rejects_invalid_evidence(
    row: dict[str, object], message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        io_module._publication(row)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"path": "config.json"}, "path must be a Path"),
        ({"raw_bytes": "{}"}, "must contain bytes"),
        ({"config": object()}, "ExpertiseConfig"),
        ({"raw_bytes": b"\xff"}, "UTF-8"),
        ({"raw_bytes": b"[]"}, "JSON object"),
        ({"raw_bytes": b'{"model":"bm25"}'}, "do not match"),
    ],
)
def test_config_provenance_rederives_only_valid_captured_bytes(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, Any] = {
        "path": Path("config.json"),
        "raw_bytes": b"{}",
        "config": ExpertiseConfig(),
    }
    values.update(kwargs)
    with pytest.raises(DataValidationError, match=message):
        ExpertiseConfigSource(**values)


@pytest.mark.parametrize("limit", [True, 0, "bad"])
def test_local_loader_rejects_invalid_optional_byte_limit(tmp_path: Path, limit: object) -> None:
    source = _source(tmp_path)
    with pytest.raises(DataValidationError, match="max_input_file_bytes"):
        load_local_domain_expertise_inputs(
            source.source_files["documents"],
            source.source_files["experts"],
            max_input_file_bytes=limit,  # type: ignore[arg-type]
        )


def test_local_loader_rejects_conflicting_config_byte_limit(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with pytest.raises(DataValidationError, match="must match"):
        load_local_domain_expertise_inputs(
            source.source_files["documents"],
            source.source_files["experts"],
            config=ExpertiseConfig(max_input_file_bytes=100),
            max_input_file_bytes=101,
        )


def test_local_loader_accepts_matching_explicit_byte_limit(tmp_path: Path) -> None:
    source = _source(tmp_path)
    loaded = load_local_domain_expertise_inputs(
        source.source_files["documents"],
        source.source_files["experts"],
        max_input_file_bytes=1000,
    )
    assert loaded.documents == source.documents


def test_config_source_rejects_bytes_longer_than_its_own_declared_limit() -> None:
    payload = b'{"max_input_file_bytes":1}'
    with pytest.raises(DataValidationError, match="input byte limit"):
        ExpertiseConfigSource(Path("config.json"), payload, ExpertiseConfig(max_input_file_bytes=1))


def test_file_and_mapping_helpers_fail_on_wrong_container_types(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="positive integer"):
        io_module._read_bounded(tmp_path / "not-read", max_bytes=True)
    with pytest.raises(DataValidationError, match="adapter_parameters"):
        io_module._validate_source_parameters("peermatchlab-local-domain-v1", [])  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="must be an object"):
        io_module._mapping([], "profile")
    with pytest.raises(DataValidationError, match="must contain value"):
        io_module._unwrap({"wrong": 1}, "title")
    config_file = tmp_path / "config.json"
    config_file.write_text("[]", encoding="utf-8")
    with pytest.raises(DataValidationError, match="JSON object"):
        io_module.load_expertise_config_source(config_file)


def test_atomic_rename_error_translation_is_explicit(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    io_module._raise_rename_error(0, destination)
    ctypes.set_errno(errno.EEXIST)
    with pytest.raises(FileExistsError):
        io_module._raise_rename_error(-1, destination)
    ctypes.set_errno(errno.EACCES)
    with pytest.raises(OSError):
        io_module._raise_rename_error(-1, destination)
