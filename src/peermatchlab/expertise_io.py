"""Strict local snapshot adapters and atomic expertise-run persistence."""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from peermatchlab.expertise import (
    _MAX_MODEL_BYTES,
    ExpertiseConfig,
    ExpertiseModel,
    ExpertiseRun,
    build_expertise_corpus,
)
from peermatchlab.io import load_documents_text, load_experts_text, load_json_text, write_json
from peermatchlab.models import DataValidationError, Document, Expert, Publication
from peermatchlab.openreview import openreview_submissions_from_records

_SOURCE_LIMIT_NAMES = frozenset(
    {
        "max_input_file_bytes",
        "max_submissions",
        "max_reviewers",
        "max_publications_per_reviewer",
        "max_total_publications",
    }
)
_SOURCE_LIMIT_MAXIMUMS = {
    "max_input_file_bytes": _MAX_MODEL_BYTES,
    "max_submissions": 1_000_000,
    "max_reviewers": 1_000_000,
    "max_publications_per_reviewer": 1_000_000,
    "max_total_publications": 10_000_000,
}


def _bounded_items(values: Iterable[Any], limit: int, label: str) -> tuple[Any, ...]:
    try:
        iterator = iter(values)
    except TypeError as error:
        raise DataValidationError(f"{label} must be iterable") from error
    items: list[Any] = []
    for item in iterator:
        if len(items) >= limit:
            raise DataValidationError(f"{label} exceeds its configured limit")
        items.append(item)
    return tuple(items)


def _source_parameters(
    config: ExpertiseConfig, *, reviewer_capacity: int | None = None
) -> Mapping[str, object]:
    parameters: dict[str, object] = {
        name: getattr(config, name) for name in sorted(_SOURCE_LIMIT_NAMES)
    }
    if reviewer_capacity is not None:
        parameters["reviewer_capacity"] = reviewer_capacity
    return MappingProxyType(parameters)


def _parameter_int(parameters: Mapping[str, object], name: str) -> int:
    return cast(int, parameters[name])


@dataclass(frozen=True, slots=True)
class LocalExpertiseInputs:
    """Domain inputs and byte-level provenance from one local source."""

    documents: tuple[Document, ...]
    experts: tuple[Expert, ...]
    source_files: Mapping[str, Path]
    source_bytes: Mapping[str, bytes] = field(repr=False)
    adapter: str
    adapter_parameters: Mapping[str, object]
    source_records: Mapping[str, Mapping[str, object]] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, str) or self.adapter not in {
            "peermatchlab-local-domain-v1",
            "openreview-local-expertise-snapshot-v1",
        }:
            raise DataValidationError("local expertise input adapter is not supported")
        parameters = _validate_source_parameters(self.adapter, self.adapter_parameters)
        documents = _bounded_items(
            self.documents,
            _parameter_int(parameters, "max_submissions"),
            "local submission inputs",
        )
        experts = _bounded_items(
            self.experts,
            _parameter_int(parameters, "max_reviewers"),
            "local reviewer inputs",
        )
        if not documents or any(not isinstance(item, Document) for item in documents):
            raise DataValidationError("local expertise inputs require Document objects")
        if not experts or any(not isinstance(item, Expert) for item in experts):
            raise DataValidationError("local expertise inputs require Expert objects")
        expected_names = (
            {"documents", "experts"}
            if self.adapter == "peermatchlab-local-domain-v1"
            else {"submissions.jsonl", "profiles.jsonl", "reviewer-publications.jsonl"}
        )
        if not isinstance(self.source_files, Mapping):
            raise DataValidationError("source_files must map names to Path objects")
        files = dict(
            _bounded_items(self.source_files.items(), len(expected_names) + 1, "source files")
        )
        if set(files) != expected_names or any(
            not isinstance(path, Path) for path in files.values()
        ):
            raise DataValidationError("source_files do not match the adapter contract")
        if not isinstance(self.source_bytes, Mapping):
            raise DataValidationError("source_bytes must map source names to immutable bytes")
        raw_values = dict(
            _bounded_items(self.source_bytes.items(), len(expected_names) + 1, "source bytes")
        )
        if set(raw_values) != expected_names or any(
            not isinstance(data, bytes) for data in raw_values.values()
        ):
            raise DataValidationError("source_bytes do not match the adapter contract")
        byte_limit = _parameter_int(parameters, "max_input_file_bytes")
        if any(len(data) > byte_limit for data in raw_values.values()):
            raise DataValidationError("source bytes exceed max_input_file_bytes")
        frozen_bytes = MappingProxyType({name: bytes(data) for name, data in raw_values.items()})
        derived_documents, derived_experts = _derive_local_expertise_inputs(
            self.adapter, frozen_bytes, parameters
        )
        if documents != derived_documents or experts != derived_experts:
            raise DataValidationError(
                "local domain objects do not match the immutable source bytes"
            )
        records = {
            name: MappingProxyType(_bytes_record(data)) for name, data in frozen_bytes.items()
        }
        object.__setattr__(self, "documents", derived_documents)
        object.__setattr__(self, "experts", derived_experts)
        object.__setattr__(self, "source_files", MappingProxyType(files))
        object.__setattr__(self, "source_bytes", frozen_bytes)
        object.__setattr__(self, "adapter_parameters", parameters)
        object.__setattr__(
            self,
            "source_records",
            MappingProxyType(records),
        )

    def rederive(self) -> tuple[tuple[Document, ...], tuple[Expert, ...]]:
        """Rebuild domain inputs from the captured bytes without trusting cached objects."""

        return _derive_local_expertise_inputs(
            self.adapter, self.source_bytes, self.adapter_parameters
        )


@dataclass(frozen=True, slots=True)
class ExpertiseConfigSource:
    """One bounded configuration read and its immutable byte provenance."""

    path: Path
    raw_bytes: bytes = field(repr=False)
    config: ExpertiseConfig
    byte_count: int = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise DataValidationError("expertise configuration source path must be a Path")
        if not isinstance(self.raw_bytes, bytes):
            raise DataValidationError("expertise configuration source must contain bytes")
        if not isinstance(self.config, ExpertiseConfig):
            raise DataValidationError(
                "expertise configuration source must contain an ExpertiseConfig"
            )
        raw_bytes = bytes(self.raw_bytes)
        object.__setattr__(self, "raw_bytes", raw_bytes)
        if len(raw_bytes) > _MAX_MODEL_BYTES:
            raise DataValidationError("expertise configuration exceeds its input byte limit")
        parsed_config = self.rederive()
        if len(raw_bytes) > parsed_config.max_input_file_bytes:
            raise DataValidationError("expertise configuration exceeds its input byte limit")
        if parsed_config != self.config:
            raise DataValidationError(
                "expertise configuration source bytes do not match the parsed configuration"
            )
        object.__setattr__(self, "config", parsed_config)
        object.__setattr__(self, "byte_count", len(raw_bytes))
        object.__setattr__(self, "sha256", hashlib.sha256(raw_bytes).hexdigest())

    def rederive(self) -> ExpertiseConfig:
        """Parse the captured bytes without trusting the cached config object."""

        try:
            text = self.raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DataValidationError("expertise configuration must be UTF-8") from error
        value = load_json_text(text)
        if not isinstance(value, Mapping):
            raise DataValidationError("expertise configuration must be a JSON object")
        return ExpertiseConfig.from_mapping(value)

    def as_record(self) -> dict[str, object]:
        """Return the exact byte record written to a run manifest."""

        return _bytes_record(self.raw_bytes)


def _read_bounded(
    path: Path, *, max_bytes: int, label: str = "local expertise snapshot file"
) -> bytes:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise DataValidationError("max_input_file_bytes must be a positive integer")
    try:
        with path.open("rb") as stream:
            data = stream.read(max_bytes + 1)
    except FileNotFoundError as error:
        raise DataValidationError(f"missing {label}: {path.name}") from error
    if len(data) > max_bytes:
        raise DataValidationError(f"{path.name} exceeds max_input_file_bytes")
    return data


def _decode_utf8(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataValidationError(f"{label} must be UTF-8") from error


def load_expertise_config_source(path: str | Path) -> ExpertiseConfigSource:
    """Read, parse, and hash an expertise configuration from one byte snapshot."""

    source = Path(path).resolve()
    data = _read_bounded(source, max_bytes=_MAX_MODEL_BYTES, label="expertise configuration file")
    text = _decode_utf8(data, "expertise configuration")
    value = load_json_text(text)
    if not isinstance(value, Mapping):
        raise DataValidationError("expertise configuration must be a JSON object")
    config = ExpertiseConfig.from_mapping(value)
    return ExpertiseConfigSource(path=source, raw_bytes=data, config=config)


def _bytes_record(data: bytes) -> dict[str, object]:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _validate_source_parameters(adapter: str, value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DataValidationError("adapter_parameters must be an object")
    expected = set(_SOURCE_LIMIT_NAMES)
    if adapter == "openreview-local-expertise-snapshot-v1":
        expected.add("reviewer_capacity")
    parameters = dict(_bounded_items(value.items(), len(expected) + 1, "adapter parameters"))
    if set(parameters) != expected:
        raise DataValidationError("adapter_parameters do not match the adapter contract")
    for name in _SOURCE_LIMIT_NAMES:
        number = parameters[name]
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or not 1 <= number <= _SOURCE_LIMIT_MAXIMUMS[name]
        ):
            raise DataValidationError(
                f"adapter parameter {name} must be between 1 and {_SOURCE_LIMIT_MAXIMUMS[name]}"
            )
    if "reviewer_capacity" in parameters:
        capacity = parameters["reviewer_capacity"]
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise DataValidationError("adapter reviewer_capacity must be non-negative")
    return MappingProxyType(parameters)


def _jsonl_from_bytes(
    data: bytes, *, label: str, max_records: int, allow_empty: bool = False
) -> tuple[Mapping[str, Any], ...]:
    text = _decode_utf8(data, label)
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if len(rows) >= max_records:
            raise DataValidationError(f"{label} exceeds its configured record limit")
        try:
            value = load_json_text(line)
        except (DataValidationError, json.JSONDecodeError) as error:
            raise DataValidationError(f"invalid {label} on line {line_number}: {error}") from error
        if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
            raise DataValidationError(f"{label} line {line_number} must be a JSON object")
        rows.append(value)
    if not rows and not allow_empty:
        raise DataValidationError(f"{label} contains no records")
    return tuple(rows)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DataValidationError(f"OpenReview {field} must be an object")
    return value


def _unwrap(value: object, field: str) -> object:
    if isinstance(value, Mapping):
        if "value" not in value:
            raise DataValidationError(f"OpenReview {field} object must contain value")
        return value["value"]
    return value


def _text(value: object, field: str, *, required: bool = False) -> str:
    unwrapped = _unwrap(value, field)
    if unwrapped is None and not required:
        return ""
    if not isinstance(unwrapped, str):
        raise DataValidationError(f"OpenReview {field} must be a string")
    if required and not unwrapped.strip():
        raise DataValidationError(f"OpenReview {field} must not be empty")
    return unwrapped


def _labels(value: object, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    unwrapped = _unwrap(value, field)
    if isinstance(unwrapped, str):
        return (unwrapped,) if unwrapped.strip() else ()
    if not isinstance(unwrapped, list) or any(not isinstance(item, str) for item in unwrapped):
        raise DataValidationError(f"OpenReview {field} must be a string or array of strings")
    return tuple(item for item in unwrapped if item.strip())


def _profile_name(content: Mapping[str, Any], reviewer_id: str) -> str:
    raw_names = _unwrap(content.get("names"), "profile names")
    if not isinstance(raw_names, list) or not raw_names:
        raise DataValidationError(f"OpenReview profile {reviewer_id!r} must contain names")
    candidates: list[tuple[bool, str]] = []
    for raw_name in raw_names:
        name = _mapping(raw_name, "profile name")
        fullname = _text(name.get("fullname"), "profile fullname", required=True)
        preferred = name.get("preferred", False)
        if not isinstance(preferred, bool):
            raise DataValidationError("OpenReview profile preferred flag must be a boolean")
        candidates.append((preferred, fullname))
    return next((name for preferred, name in candidates if preferred), candidates[0][1])


def _profile_to_parts(
    record: Mapping[str, Any],
) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...]]:
    reviewer_id = _text(record.get("id"), "profile id", required=True)
    content = _mapping(record.get("content"), "profile content")
    topics = tuple(
        dict.fromkeys(
            (
                *_labels(content.get("expertise"), "profile expertise"),
                *_labels(content.get("research_interests"), "profile research_interests"),
            )
        )
    )
    return (
        reviewer_id,
        _profile_name(content, reviewer_id),
        _text(content.get("bio"), "profile bio"),
        topics,
        _labels(content.get("keywords"), "profile keywords"),
    )


def _publication(record: Mapping[str, Any]) -> tuple[str, str, Publication]:
    if set(record) != {"reviewer_id", "note"}:
        raise DataValidationError(
            "reviewer-publication rows must contain exactly reviewer_id and note"
        )
    reviewer_id = _text(record.get("reviewer_id"), "reviewer_id", required=True)
    note = _mapping(record.get("note"), "publication note")
    note_id = _text(note.get("id"), "publication id", required=True)
    content = _mapping(note.get("content"), "publication content")
    raw_year = _unwrap(content.get("year"), "publication year")
    if raw_year is not None and (
        isinstance(raw_year, bool) or not isinstance(raw_year, int) or not 1800 <= raw_year <= 2200
    ):
        raise DataValidationError(
            "OpenReview publication year must be null or between 1800 and 2200"
        )
    return (
        reviewer_id,
        note_id,
        Publication(
            title=_text(content.get("title"), "publication title", required=True),
            abstract=_text(content.get("abstract"), "publication abstract"),
            year=raw_year,
            id=note_id,
        ),
    )


def _openreview_domain_from_rows(
    submission_rows: tuple[Mapping[str, Any], ...],
    profile_rows: tuple[Mapping[str, Any], ...],
    publication_rows: tuple[Mapping[str, Any], ...],
    parameters: Mapping[str, object],
) -> tuple[tuple[Document, ...], tuple[Expert, ...]]:
    max_submissions = _parameter_int(parameters, "max_submissions")
    max_reviewers = _parameter_int(parameters, "max_reviewers")
    max_per_reviewer = _parameter_int(parameters, "max_publications_per_reviewer")
    documents = openreview_submissions_from_records(submission_rows, max_records=max_submissions)
    profiles: dict[str, tuple[str, str, tuple[str, ...], tuple[str, ...]]] = {}
    for position, row in enumerate(profile_rows, start=1):
        try:
            reviewer_id, name, summary, topics, keywords = _profile_to_parts(row)
        except DataValidationError as error:
            raise DataValidationError(
                f"invalid OpenReview profile record {position}: {error}"
            ) from error
        if reviewer_id in profiles:
            raise DataValidationError(f"duplicate OpenReview profile id: {reviewer_id}")
        profiles[reviewer_id] = (name, summary, topics, keywords)
    if len(profiles) > max_reviewers:
        raise DataValidationError("profile snapshot exceeds its configured record limit")
    publications: dict[str, list[Publication]] = {reviewer_id: [] for reviewer_id in profiles}
    seen_publications: set[tuple[str, str]] = set()
    for position, row in enumerate(publication_rows, start=1):
        try:
            reviewer_id, note_id, item = _publication(row)
        except DataValidationError as error:
            raise DataValidationError(
                f"invalid reviewer-publication record {position}: {error}"
            ) from error
        if reviewer_id not in profiles:
            raise DataValidationError(
                f"reviewer-publication references unknown profile: {reviewer_id}"
            )
        pair = (reviewer_id, note_id)
        if pair in seen_publications:
            raise DataValidationError(f"duplicate reviewer-publication association: {pair}")
        seen_publications.add(pair)
        if len(publications[reviewer_id]) >= max_per_reviewer:
            raise DataValidationError(
                f"reviewer {reviewer_id!r} exceeds max_publications_per_reviewer"
            )
        publications[reviewer_id].append(item)
    capacity = _parameter_int(parameters, "reviewer_capacity")
    experts = tuple(
        Expert(
            id=reviewer_id,
            name=parts[0],
            summary=parts[1],
            topics=parts[2],
            keywords=parts[3],
            publications=tuple(
                sorted(
                    publications[reviewer_id],
                    key=lambda item: (item.id or "", item.title, item.year or 0),
                )
            ),
            capacity=capacity,
            metadata={"adapter": "openreview-local-expertise-snapshot-v1"},
        )
        for reviewer_id, parts in sorted(profiles.items())
    )
    return documents, experts


def _derive_local_expertise_inputs(
    adapter: str,
    source_bytes: Mapping[str, bytes],
    parameters: Mapping[str, object],
) -> tuple[tuple[Document, ...], tuple[Expert, ...]]:
    if adapter == "peermatchlab-local-domain-v1":
        documents = load_documents_text(
            _decode_utf8(source_bytes["documents"], "document input"),
            source="captured documents",
            max_records=_parameter_int(parameters, "max_submissions"),
        )
        experts = load_experts_text(
            _decode_utf8(source_bytes["experts"], "expert input"),
            source="captured experts",
            max_records=_parameter_int(parameters, "max_reviewers"),
        )
    else:
        submissions = _jsonl_from_bytes(
            source_bytes["submissions.jsonl"],
            label="submission snapshot",
            max_records=_parameter_int(parameters, "max_submissions"),
        )
        profiles = _jsonl_from_bytes(
            source_bytes["profiles.jsonl"],
            label="profile snapshot",
            max_records=_parameter_int(parameters, "max_reviewers"),
        )
        publications = _jsonl_from_bytes(
            source_bytes["reviewer-publications.jsonl"],
            label="reviewer-publication snapshot",
            max_records=_parameter_int(parameters, "max_total_publications"),
            allow_empty=True,
        )
        return _openreview_domain_from_rows(submissions, profiles, publications, parameters)

    documents = _bounded_items(
        documents, _parameter_int(parameters, "max_submissions"), "captured submissions"
    )
    experts = _bounded_items(
        experts, _parameter_int(parameters, "max_reviewers"), "captured reviewers"
    )
    total_publications = 0
    for expert in experts:
        publications = _bounded_items(
            expert.publications,
            _parameter_int(parameters, "max_publications_per_reviewer"),
            f"reviewer {expert.id!r} publications",
        )
        total_publications += len(publications)
        if total_publications > _parameter_int(parameters, "max_total_publications"):
            raise DataValidationError("captured publications exceed max_total_publications")
    return documents, experts


def load_openreview_expertise_snapshot(
    directory: str | Path,
    *,
    config: ExpertiseConfig | None = None,
    reviewer_capacity: int = 1,
) -> LocalExpertiseInputs:
    """Load an offline, explicitly joined OpenReview-shaped expertise snapshot.

    The directory contract is ``submissions.jsonl``, ``profiles.jsonl``, and
    ``reviewer-publications.jsonl``.  Each publication row carries an explicit
    reviewer ID plus its OpenReview-shaped note; authorship is never inferred.
    """

    active = config or ExpertiseConfig()
    if (
        isinstance(reviewer_capacity, bool)
        or not isinstance(reviewer_capacity, int)
        or reviewer_capacity < 0
    ):
        raise DataValidationError("reviewer_capacity must be a non-negative integer")
    root = Path(directory)
    source_files = {
        "submissions.jsonl": root / "submissions.jsonl",
        "profiles.jsonl": root / "profiles.jsonl",
        "reviewer-publications.jsonl": root / "reviewer-publications.jsonl",
    }
    source_bytes = {
        name: _read_bounded(path, max_bytes=active.max_input_file_bytes)
        for name, path in source_files.items()
    }
    parameters = _source_parameters(active, reviewer_capacity=reviewer_capacity)
    documents, experts = _derive_local_expertise_inputs(
        "openreview-local-expertise-snapshot-v1", source_bytes, parameters
    )
    return LocalExpertiseInputs(
        documents=documents,
        experts=experts,
        source_files=source_files,
        source_bytes=source_bytes,
        adapter="openreview-local-expertise-snapshot-v1",
        adapter_parameters=parameters,
    )


def local_domain_inputs(
    documents: tuple[Document, ...],
    experts: tuple[Expert, ...],
    *,
    document_path: str | Path,
    expert_path: str | Path,
    max_input_file_bytes: int | None = None,
    config: ExpertiseConfig | None = None,
) -> LocalExpertiseInputs:
    """Attach stable provenance to already validated PeerMatchLab fixtures."""

    loaded = load_local_domain_expertise_inputs(
        document_path,
        expert_path,
        max_input_file_bytes=max_input_file_bytes,
        config=config,
    )
    if loaded.documents != documents or loaded.experts != experts:
        raise DataValidationError("local domain objects do not match their source files")
    return loaded


def load_local_domain_expertise_inputs(
    document_path: str | Path,
    expert_path: str | Path,
    *,
    max_input_file_bytes: int | None = None,
    config: ExpertiseConfig | None = None,
) -> LocalExpertiseInputs:
    """Read normalized domain inputs once with a pre-parse byte ceiling."""

    active = config or ExpertiseConfig()
    if max_input_file_bytes is not None:
        if (
            isinstance(max_input_file_bytes, bool)
            or not isinstance(max_input_file_bytes, int)
            or max_input_file_bytes < 1
        ):
            raise DataValidationError("max_input_file_bytes must be a positive integer")
        if config is not None and max_input_file_bytes != config.max_input_file_bytes:
            raise DataValidationError(
                "max_input_file_bytes must match the supplied expertise configuration"
            )
        if config is None:
            active = ExpertiseConfig(max_input_file_bytes=max_input_file_bytes)
    paths = {"documents": Path(document_path), "experts": Path(expert_path)}
    data = {
        name: _read_bounded(path, max_bytes=active.max_input_file_bytes)
        for name, path in paths.items()
    }
    parameters = _source_parameters(active)
    documents, experts = _derive_local_expertise_inputs(
        "peermatchlab-local-domain-v1", data, parameters
    )
    return LocalExpertiseInputs(
        documents=documents,
        experts=experts,
        source_files=paths,
        source_bytes=data,
        adapter="peermatchlab-local-domain-v1",
        adapter_parameters=parameters,
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    try:
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                json.dump(
                    row,
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                stream.write("\n")
    except RecursionError as error:
        raise DataValidationError("JSON nesting exceeds the supported depth") from error


def _raise_rename_error(result: int, destination: Path) -> None:
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _install_directory_no_replace(staging: Path, destination: Path) -> None:
    """Atomically install a directory without replacing any concurrent target."""

    if sys.platform == "linux":
        library = ctypes.CDLL(None, use_errno=True)
        try:
            rename = library.renameat2
        except AttributeError as error:
            raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable") from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        ctypes.set_errno(0)
        _raise_rename_error(
            rename(
                -100,
                os.fsencode(staging),
                -100,
                os.fsencode(destination),
                1,
            ),
            destination,
        )
        return
    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        try:
            rename = library.renamex_np
        except AttributeError as error:
            raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable") from error
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        ctypes.set_errno(0)
        _raise_rename_error(
            rename(os.fsencode(staging), os.fsencode(destination), 0x00000004), destination
        )
        return
    if os.name == "nt":
        os.rename(staging, destination)
        return
    raise OSError(errno.ENOTSUP, "atomic no-replace directory installation is unavailable")


def write_expertise_run(
    run: ExpertiseRun,
    directory: str | Path,
    *,
    source: LocalExpertiseInputs,
    config_source: ExpertiseConfigSource | None = None,
) -> Mapping[str, Any]:
    """Atomically create a self-describing expertise artifact directory."""

    destination = Path(directory)
    if destination.exists():
        raise DataValidationError(f"expertise destination already exists: {destination}")
    derived_documents, derived_experts = source.rederive()
    if derived_documents != source.documents or derived_experts != source.experts:
        raise DataValidationError(
            "local expertise inputs changed after their source bytes were validated"
        )
    expected_corpus = build_expertise_corpus(
        derived_documents, derived_experts, config=run.model.config
    )
    if expected_corpus != run.corpus:
        raise DataValidationError("expertise source content does not match the generated run")
    expected_model = ExpertiseModel.fit(expected_corpus, config=run.model.config)
    if expected_model.to_dict() != run.model.to_dict():
        raise DataValidationError("expertise model does not match the generated corpus")
    if expected_model.score(expected_corpus.submissions) != run.scores:
        raise DataValidationError("expertise scores do not match the fitted model")
    published_config = run.model.config
    if config_source is not None:
        published_config = config_source.rederive()
        if published_config != run.model.config:
            raise DataValidationError("expertise configuration does not match the generated run")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name or 'expertise'}-", dir=destination.parent)
    )
    installed = False
    try:
        write_json(
            staging / "documents.json",
            [
                {
                    "id": item.id,
                    "title": item.title,
                    "abstract": item.abstract,
                    "topics": list(item.topics),
                    "keywords": list(item.keywords),
                    "required_experts": item.required_experts,
                    "metadata": dict(item.metadata),
                }
                for item in derived_documents
            ],
        )
        write_json(
            staging / "experts.json",
            [
                {
                    "id": item.id,
                    "name": item.name,
                    "summary": item.summary,
                    "topics": list(item.topics),
                    "keywords": list(item.keywords),
                    "publications": [
                        {
                            "title": publication.title,
                            "abstract": publication.abstract,
                            "year": publication.year,
                            "id": publication.id,
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
                for item in derived_experts
            ],
        )
        _write_jsonl(
            staging / "submission-documents.jsonl",
            (item.as_dict() for item in run.corpus.submissions),
        )
        _write_jsonl(
            staging / "reviewer-documents.jsonl",
            (
                item.as_dict()
                for _reviewer_id, evidence in sorted(run.corpus.reviewer_evidence.items())
                for item in evidence
            ),
        )
        emitted = tuple(
            item
            for item in run.scores
            if item.score > 0.0 and item.score >= run.model.config.minimum_output_score
        )
        with (staging / "affinities.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("document_id", "expert_id", "score"))
            for item in emitted:
                writer.writerow((item.document_id, item.expert_id, repr(item.score)))
        _write_jsonl(staging / "explanations.jsonl", (item.as_dict() for item in emitted))
        run.model.save(staging / "model.json")
        if (staging / "model.json").stat().st_size > min(
            run.model.config.max_model_file_bytes, _MAX_MODEL_BYTES
        ):
            raise DataValidationError("generated expertise model exceeds its load-time byte limit")
        replayed_model = ExpertiseModel.load(staging / "model.json")
        if (
            replayed_model.to_dict() != run.model.to_dict()
            or replayed_model.score(run.corpus.submissions) != run.scores
        ):
            raise DataValidationError("generated expertise model does not replay exactly")
        for name, path in source.source_files.items():
            current_source = _read_bounded(
                path,
                max_bytes=run.model.config.max_input_file_bytes,
                label="expertise source file",
            )
            if current_source != source.source_bytes[name]:
                raise DataValidationError(
                    f"expertise source file changed after it was loaded: {name}"
                )
        input_files = {
            name: _bytes_record(data) for name, data in sorted(source.source_bytes.items())
        }
        if config_source is not None:
            current_config = _read_bounded(
                config_source.path,
                max_bytes=min(run.model.config.max_input_file_bytes, _MAX_MODEL_BYTES),
                label="expertise configuration file",
            )
            if current_config != config_source.raw_bytes:
                raise DataValidationError("expertise configuration changed after it was loaded")
            input_files["expertise-config"] = config_source.as_record()
        derived_names = (
            "documents.json",
            "experts.json",
            "submission-documents.jsonl",
            "reviewer-documents.jsonl",
            "affinities.csv",
            "explanations.jsonl",
            "model.json",
        )
        derived_files = {name: _file_record(staging / name) for name in derived_names}
        from peermatchlab import __version__

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "adapter": "peermatchlab-expertise-run-v1",
            "generator": {"package": "peermatchlab", "version": __version__},
            "source": {
                "adapter": source.adapter,
                "parameters": dict(source.adapter_parameters),
                "files": input_files,
            },
            "configuration": published_config.as_dict(),
            "records": {
                "submissions": len(run.corpus.submissions),
                "reviewers": len(run.corpus.reviewer_evidence),
                "reviewer_evidence": sum(
                    len(items) for items in run.corpus.reviewer_evidence.values()
                ),
                "candidate_pairs": len(run.scores),
                "emitted_pairs": len(emitted),
            },
            "filters": dict(run.corpus.filters),
            "statistics": {
                "index_documents": run.model.document_count,
                "average_document_length": run.model.average_document_length,
                "vocabulary_terms": len(run.model.inverse_document_frequency),
            },
            "score_semantics": {
                "tfidf": "smoothed IDF, sublinear term frequency, cosine similarity",
                "bm25": "Robertson IDF and binary query terms; raw/(1+raw) affinity normalization",
                "emission_rule": (
                    "emit exactly when score > 0 and score >= minimum_output_score; "
                    "all other pairs are omitted from affinities.csv and explanations.jsonl"
                ),
            },
            "files": derived_files,
        }
        write_json(staging / "manifest.json", manifest)
        try:
            _install_directory_no_replace(staging, destination)
        except FileExistsError as error:
            raise DataValidationError(
                f"expertise destination already exists: {destination}"
            ) from error
        installed = True
        return MappingProxyType(manifest)
    finally:
        if not installed:
            shutil.rmtree(staging, ignore_errors=True)
