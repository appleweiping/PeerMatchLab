"""Bounded SPECTER-family embedding interchange and reviewer scoring.

This module does not implement or distribute a neural encoder.  The provider
protocol is the executable boundary at which a licensed encoder can be wired
in; the bundled provider reads the frozen JSONL format emitted by the pinned
OpenReview Expertise predictor.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Literal, Protocol, cast

from peermatchlab.expertise_io import LocalExpertiseInputs, _install_directory_no_replace
from peermatchlab.io import load_json_text
from peermatchlab.models import DataValidationError, Document, Expert, _identifier_is_valid

_DIMENSIONS = 768
_MAX_FILE_BYTES = 64 * 1024 * 1024
_MAX_ROWS = 500_000
_MAX_CANDIDATE_PAIRS = 1_000_000
_MAX_GENERATED_FILE_BYTES = 256 * 1024 * 1024
_MAX_GENERATED_TOTAL_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """One submission or reviewer publication at the encoder boundary."""

    paper_id: str
    title: str
    abstract: str


class EmbeddingProvider(Protocol):
    """Pluggable encoder/interchange contract, with no implicit network access."""

    @property
    def provenance(self) -> Mapping[str, object]: ...

    def embed(
        self, kind: Literal["submissions", "publications"], requests: Sequence[EmbeddingRequest]
    ) -> Mapping[str, tuple[float, ...]]: ...


def embedding_requests(
    documents: Iterable[Document], experts: Iterable[Expert]
) -> tuple[
    tuple[EmbeddingRequest, ...], tuple[EmbeddingRequest, ...], Mapping[str, tuple[str, ...]]
]:
    """Derive the exact paper IDs and reviewer-publication associations."""

    submissions = tuple(
        EmbeddingRequest(item.id, item.title, item.abstract)
        for item in sorted(documents, key=lambda item: item.id)
    )
    publications: dict[str, EmbeddingRequest] = {}
    by_reviewer: dict[str, tuple[str, ...]] = {}
    association_count = 0
    for expert in sorted(experts, key=lambda item: item.id):
        paper_ids: list[str] = []
        for position, item in enumerate(expert.publications):
            association_count += 1
            if association_count > _MAX_ROWS:
                raise DataValidationError("reviewer-publication associations exceed row limit")
            paper_id = item.id or f"publication:{expert.id}:position:{position}"
            request = EmbeddingRequest(paper_id, item.title, item.abstract)
            previous = publications.get(paper_id)
            if previous is not None and previous != request:
                raise DataValidationError("shared publication ID has inconsistent text")
            publications[paper_id] = request
            paper_ids.append(paper_id)
        by_reviewer[expert.id] = tuple(dict.fromkeys(paper_ids))
    return (
        submissions,
        tuple(publications[key] for key in sorted(publications)),
        MappingProxyType(by_reviewer),
    )


def request_sha256(requests: Sequence[EmbeddingRequest]) -> str:
    """Hash canonical encoder inputs, binding a vector fixture to paper text."""

    rows = [
        {"paper_id": item.paper_id, "title": item.title, "abstract": item.abstract}
        for item in requests
    ]
    data = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(_MAX_FILE_BYTES + 1)
    if len(data) > _MAX_FILE_BYTES:
        raise DataValidationError(f"embedding file exceeds {_MAX_FILE_BYTES} bytes: {path.name}")
    return data


def _file_record(data: bytes) -> dict[str, object]:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _write_captured_bytes(path: Path, data: bytes) -> None:
    """Copy captured input bytes exactly or abort the staged publication."""

    with path.open("wb") as stream:
        if stream.write(data) != len(data):
            raise DataValidationError("short write while copying embedding input")


class _OutputBudget:
    def __init__(self) -> None:
        self.remaining = _MAX_GENERATED_TOTAL_BYTES

    def consume(self, count: int) -> None:
        if count > self.remaining:
            raise DataValidationError("generated embedding output exceeds total byte limit")
        self.remaining -= count


class _BoundedTextWriter:
    """Hash UTF-8 output while enforcing file and run ceilings before each write."""

    def __init__(self, path: Path, budget: _OutputBudget) -> None:
        self._stream = path.open("wb")
        self._budget = budget
        self._hash = hashlib.sha256()
        self._bytes = 0

    def __enter__(self) -> _BoundedTextWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stream.close()

    def write(self, value: str) -> int:
        data = value.encode("utf-8")
        if self._bytes + len(data) > _MAX_GENERATED_FILE_BYTES:
            raise DataValidationError("generated embedding file exceeds output byte limit")
        self._budget.consume(len(data))
        if self._stream.write(data) != len(data):
            raise DataValidationError("short write while publishing embedding output")
        self._hash.update(data)
        self._bytes += len(data)
        return len(value)

    def record(self) -> dict[str, object]:
        return {"bytes": self._bytes, "sha256": self._hash.hexdigest()}


def _write_json_array(
    path: Path, rows: Iterable[Mapping[str, Any]], budget: _OutputBudget
) -> dict[str, object]:
    with _BoundedTextWriter(path, budget) as stream:
        stream.write("[\n")
        first = True
        for row in rows:
            if not first:
                stream.write(",\n")
            try:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
            except RecursionError as error:
                raise DataValidationError("generated JSON exceeds nesting limit") from error
            first = False
        stream.write("\n]\n")
    return stream.record()


def _write_json_object(path: Path, value: Mapping[str, object], budget: _OutputBudget) -> None:
    with _BoundedTextWriter(path, budget) as stream:
        try:
            encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, allow_nan=False)
            for chunk in encoder.iterencode(value):
                stream.write(chunk)
        except RecursionError as error:
            raise DataValidationError("generated JSON exceeds nesting limit") from error
        stream.write("\n")


def _coordinate_is_valid(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except OverflowError:
        return False
    return math.isfinite(number) and abs(number) <= 1e12


def _parse_jsonl(data: bytes, *, kind: str) -> Mapping[str, tuple[float, ...]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataValidationError(f"{kind} embeddings must be UTF-8") from error
    result: dict[str, tuple[float, ...]] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise DataValidationError(f"{kind} embedding line {number} is blank")
        if len(result) >= _MAX_ROWS:
            raise DataValidationError(f"{kind} embeddings exceed row limit")
        try:
            row = load_json_text(line)
        except (ValueError, RecursionError) as error:
            raise DataValidationError(f"invalid {kind} embedding line {number}") from error
        if not isinstance(row, Mapping) or set(row) != {"paper_id", "embedding"}:
            raise DataValidationError(f"{kind} embedding line {number} has invalid schema")
        paper_id = row["paper_id"]
        values = row["embedding"]
        if not _identifier_is_valid(paper_id) or paper_id in result:
            raise DataValidationError(f"{kind} embedding paper IDs must be unique and valid")
        if not isinstance(values, list) or len(values) not in {0, _DIMENSIONS}:
            raise DataValidationError(f"{kind} embedding must have 768 dimensions or be empty")
        if any(not _coordinate_is_valid(value) for value in values):
            raise DataValidationError(f"{kind} embedding contains invalid coordinates")
        result[str(paper_id)] = tuple(float(value) for value in values)
    return MappingProxyType(result)


class FrozenJsonlEmbeddingProvider:
    """Read hashed, comparator-shaped vector files from a local directory."""

    def __init__(self, directory: str | Path) -> None:
        root = Path(directory)
        names = ("manifest.json", "submissions.jsonl", "publications.jsonl")
        data = {name: _read_bytes(root / name) for name in names}
        try:
            manifest = load_json_text(data["manifest.json"].decode("utf-8"))
        except (UnicodeError, ValueError, RecursionError) as error:
            raise DataValidationError("invalid embedding manifest") from error
        if not isinstance(manifest, Mapping) or set(manifest) != {
            "schema_version",
            "adapter",
            "encoder",
            "requests",
            "files",
        }:
            raise DataValidationError("embedding manifest has invalid schema")
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise DataValidationError("unsupported embedding manifest schema version")
        if manifest["adapter"] != "openreview-specter-family-jsonl-v1":
            raise DataValidationError("unsupported embedding adapter")
        encoder = manifest["encoder"]
        if not isinstance(encoder, Mapping) or set(encoder) != {
            "family",
            "model_id",
            "revision",
            "origin",
            "weights_sha256",
        }:
            raise DataValidationError("embedding encoder provenance is invalid")
        if type(encoder["family"]) is not str or encoder["family"] not in {
            "specter",
            "specter2",
            "scincl",
        }:
            raise DataValidationError("unsupported embedding family")
        if any(
            type(encoder[name]) is not str or not encoder[name] or len(encoder[name]) > 256
            for name in ("model_id", "revision")
        ):
            raise DataValidationError("embedding model identity is invalid")
        if type(encoder["origin"]) is not str or encoder["origin"] not in {
            "synthetic-fixture",
            "external-encoder",
        }:
            raise DataValidationError("embedding origin is invalid")
        digest = encoder["weights_sha256"]
        if digest is not None and (
            type(digest) is not str
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise DataValidationError("weights_sha256 must be a lowercase SHA-256 or null")
        if encoder["origin"] == "synthetic-fixture" and digest is not None:
            raise DataValidationError("synthetic fixtures must not claim model weights")
        if encoder["origin"] == "external-encoder" and digest is None:
            raise DataValidationError("external encoders must declare weights_sha256")
        requests = manifest["requests"]
        if (
            not isinstance(requests, Mapping)
            or set(requests) != {"submissions_sha256", "publications_sha256"}
            or any(
                type(value) is not str
                or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
                for value in requests.values()
            )
        ):
            raise DataValidationError("embedding request hashes are invalid")
        files = manifest["files"]
        if not isinstance(files, Mapping) or set(files) != {
            "submissions.jsonl",
            "publications.jsonl",
        }:
            raise DataValidationError("embedding file manifest is invalid")
        for name in ("submissions.jsonl", "publications.jsonl"):
            if files[name] != _file_record(data[name]):
                raise DataValidationError(f"embedding file hash or length mismatch: {name}")
        self._root = root
        self._bytes = MappingProxyType(data)
        self._manifest = cast(Mapping[str, object], _freeze_json(manifest))
        self._vectors = MappingProxyType(
            {
                kind: _parse_jsonl(data[f"{kind}.jsonl"], kind=kind)
                for kind in ("submissions", "publications")
            }
        )

    @property
    def provenance(self) -> Mapping[str, object]:
        # Never expose the internal nested manifest to a caller. The captured
        # bytes are immutable and are also checked against the on-disk source
        # immediately before publication.
        return cast(
            Mapping[str, object],
            load_json_text(self._bytes["manifest.json"].decode("utf-8")),
        )

    @property
    def source_bytes(self) -> Mapping[str, bytes]:
        return self._bytes

    def verify_unchanged(self) -> None:
        for name, data in self._bytes.items():
            if _read_bytes(self._root / name) != data:
                raise DataValidationError(f"embedding source changed after loading: {name}")

    def embed(
        self, kind: Literal["submissions", "publications"], requests: Sequence[EmbeddingRequest]
    ) -> Mapping[str, tuple[float, ...]]:
        if kind not in {"submissions", "publications"}:
            raise DataValidationError("embedding request kind is invalid")
        if len(requests) > _MAX_ROWS or len({item.paper_id for item in requests}) != len(requests):
            raise DataValidationError("embedding requests exceed limit or repeat paper IDs")
        if any(
            not isinstance(item, EmbeddingRequest)
            or not _identifier_is_valid(item.paper_id)
            or type(item.title) is not str
            or type(item.abstract) is not str
            for item in requests
        ):
            raise DataValidationError("embedding requests are invalid")
        expected = cast(Mapping[str, str], self._manifest["requests"])
        if request_sha256(requests) != expected[f"{kind}_sha256"]:
            raise DataValidationError(f"{kind} embedding request text hash mismatch")
        vectors = self._vectors[kind]
        if set(vectors) != {item.paper_id for item in requests}:
            raise DataValidationError(f"{kind} embedding IDs do not match requests")
        return vectors


@dataclass(frozen=True, slots=True)
class EmbeddingScore:
    document_id: str
    expert_id: str
    score: float
    evidence_count: int
    selected_publication_id: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "expert_id": self.expert_id,
            "score": self.score,
            "evidence_count": self.evidence_count,
            "selected_publication_id": self.selected_publication_id,
        }


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _cosine(
    left: Sequence[float], right: Sequence[float], left_norm: float, right_norm: float
) -> float:
    # An empty or all-zero vector contributes zero without constructing a
    # 768-element zero tuple for every missing publication.
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return math.fsum(
        (a / (left_norm + 1e-12)) * (b / (right_norm + 1e-12))
        for a, b in zip(left, right, strict=True)
    )


def _scale_cosine(value: float, low: float, high: float) -> float:
    return max(0.0, min(1.0, value)) if high == low else (value - low) / (high - low)


def score_embedding_expertise(
    documents: Iterable[Document],
    experts: Iterable[Expert],
    provider: EmbeddingProvider,
    *,
    aggregation: Literal["max", "average"] = "max",
    max_paper_pairs: int = 1_000_000,
    max_candidate_pairs: int = 1_000_000,
) -> tuple[EmbeddingScore, ...]:
    """L2 cosine, global min-max, reviewer aggregation, four-decimal scores.

    Empty publication vectors enter the global range as zero, then are
    excluded from reviewer aggregation, matching the pinned comparator. The
    paper matrix is scanned twice; only one submission row is retained.
    Candidate output is hard-capped at one million records because this API
    returns an in-memory tuple rather than a stream.
    """

    if aggregation not in {"max", "average"}:
        raise DataValidationError("embedding aggregation must be max or average")
    if type(max_paper_pairs) is not int or not 1 <= max_paper_pairs <= 10_000_000:
        raise DataValidationError("max_paper_pairs must be an integer in [1, 10000000]")
    if type(max_candidate_pairs) is not int or not 1 <= max_candidate_pairs <= _MAX_CANDIDATE_PAIRS:
        raise DataValidationError("max_candidate_pairs must be an integer in [1, 1000000]")
    docs = tuple(islice(documents, max_candidate_pairs + 1))
    reviewers = tuple(islice(experts, max_candidate_pairs + 1))
    if len(docs) > max_candidate_pairs or len(reviewers) > max_candidate_pairs:
        raise DataValidationError("embedding domain inputs exceed max_candidate_pairs")
    if (
        not docs
        or not reviewers
        or any(not isinstance(item, Document) for item in docs)
        or any(not isinstance(item, Expert) for item in reviewers)
    ):
        raise DataValidationError("embedding scoring requires documents and experts")
    if len({item.id for item in docs}) != len(docs) or len({item.id for item in reviewers}) != len(
        reviewers
    ):
        raise DataValidationError("embedding document and reviewer IDs must be unique")
    submissions, publications, associations = embedding_requests(docs, reviewers)
    if len(submissions) * len(publications) > max_paper_pairs:
        raise DataValidationError("embedding paper matrix exceeds max_paper_pairs")
    if len(submissions) * len(reviewers) > max_candidate_pairs:
        raise DataValidationError("embedding reviewer matrix exceeds max_candidate_pairs")
    sub_vectors = provider.embed("submissions", submissions)
    pub_vectors = provider.embed("publications", publications)
    if set(sub_vectors) != {item.paper_id for item in submissions} or set(pub_vectors) != {
        item.paper_id for item in publications
    }:
        raise DataValidationError("embedding provider returned unexpected paper IDs")
    for vector in (*sub_vectors.values(), *pub_vectors.values()):
        if (
            not isinstance(vector, (list, tuple))
            or len(vector) not in {0, _DIMENSIONS}
            or any(not _coordinate_is_valid(value) for value in vector)
        ):
            raise DataValidationError("embedding provider returned invalid coordinates")
    sub_norms = {key: math.hypot(*value) for key, value in sub_vectors.items()}
    pub_norms = {key: math.hypot(*value) for key, value in pub_vectors.items()}
    sorted_reviewers = sorted(reviewers, key=lambda item: item.id)
    eligible_by_reviewer = {
        reviewer.id: tuple(
            paper_id for paper_id in associations[reviewer.id] if pub_vectors[paper_id]
        )
        for reviewer in sorted_reviewers
    }
    low = math.inf
    high = -math.inf
    for submission in submissions:
        for publication in publications:
            value = _cosine(
                sub_vectors[submission.paper_id],
                pub_vectors[publication.paper_id],
                sub_norms[submission.paper_id],
                pub_norms[publication.paper_id],
            )
            low = min(low, value)
            high = max(high, value)
    if not publications:
        low = high = 0.0
    result: list[EmbeddingScore] = []
    for submission in submissions:
        scaled = {
            publication.paper_id: _scale_cosine(
                _cosine(
                    sub_vectors[submission.paper_id],
                    pub_vectors[publication.paper_id],
                    sub_norms[submission.paper_id],
                    pub_norms[publication.paper_id],
                ),
                low,
                high,
            )
            for publication in publications
        }
        for reviewer in sorted_reviewers:
            eligible = eligible_by_reviewer[reviewer.id]
            if not eligible:
                score = 0.0
                selected = None
            elif aggregation == "max":
                selected = max(eligible, key=lambda paper_id: scaled[paper_id])
                score = scaled[selected]
            else:
                selected = None
                score = math.fsum(scaled[paper_id] for paper_id in eligible) / len(eligible)
            result.append(
                EmbeddingScore(
                    submission.paper_id, reviewer.id, round(score, 4), len(eligible), selected
                )
            )
    return tuple(result)


def write_embedding_run(
    scores: Sequence[EmbeddingScore],
    directory: str | Path,
    *,
    source: LocalExpertiseInputs,
    provider: FrozenJsonlEmbeddingProvider,
    aggregation: Literal["max", "average"] = "max",
    max_paper_pairs: int = 1_000_000,
    max_candidate_pairs: int = 1_000_000,
) -> Mapping[str, object]:
    """Atomically publish independently replayed scores and exact input bytes."""

    destination = Path(directory)
    if destination.exists():
        raise DataValidationError(f"embedding destination already exists: {destination}")
    documents, experts = source.rederive()
    if documents != source.documents or experts != source.experts:
        raise DataValidationError("embedding source objects changed after loading")
    replayed = score_embedding_expertise(
        documents,
        experts,
        provider,
        aggregation=aggregation,
        max_paper_pairs=max_paper_pairs,
        max_candidate_pairs=max_candidate_pairs,
    )
    if tuple(scores) != replayed:
        raise DataValidationError("embedding scores do not match captured inputs")
    provider.verify_unchanged()
    # Use the immutable, verified byte capture for publication, never a
    # caller-held view of nested provenance metadata.
    frozen_embedding_manifest = load_json_text(
        provider.source_bytes["manifest.json"].decode("utf-8")
    )
    for name, path in source.source_files.items():
        if _read_bytes(path) != source.source_bytes[name]:
            raise DataValidationError(f"expertise source changed after loading: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    installed = False
    try:
        budget = _OutputBudget()
        generated_files: dict[str, dict[str, object]] = {}
        generated_files["documents.json"] = _write_json_array(
            staging / "documents.json",
            (
                {
                    "id": item.id,
                    "title": item.title,
                    "abstract": item.abstract,
                    "topics": list(item.topics),
                    "keywords": list(item.keywords),
                    "required_experts": item.required_experts,
                    "metadata": dict(item.metadata),
                }
                for item in documents
            ),
            budget,
        )
        generated_files["experts.json"] = _write_json_array(
            staging / "experts.json",
            (
                {
                    "id": item.id,
                    "name": item.name,
                    "summary": item.summary,
                    "topics": list(item.topics),
                    "keywords": list(item.keywords),
                    "publications": [
                        {
                            "id": pub.id,
                            "title": pub.title,
                            "abstract": pub.abstract,
                            "year": pub.year,
                        }
                        for pub in item.publications
                    ],
                    "capacity": item.capacity,
                    "institution": item.institution,
                    "regions": list(item.regions),
                    "seniority": item.seniority,
                    "bids": dict(item.bids),
                    "metadata": dict(item.metadata),
                }
                for item in experts
            ),
            budget,
        )
        input_files: dict[str, dict[str, object]] = {}
        for name, data in source.source_bytes.items():
            target = f"source-{name}"
            _write_captured_bytes(staging / target, data)
            input_files[target] = _file_record(data)
        for name, data in provider.source_bytes.items():
            target = f"embedding-{name}"
            _write_captured_bytes(staging / target, data)
            input_files[target] = _file_record(data)
        emitted = tuple(item for item in replayed if item.score > 0.0)
        with _BoundedTextWriter(staging / "affinities.csv", budget) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("document_id", "expert_id", "score"))
            for item in emitted:
                writer.writerow((item.document_id, item.expert_id, repr(item.score)))
        generated_files["affinities.csv"] = stream.record()
        with _BoundedTextWriter(staging / "scores.jsonl", budget) as stream:
            for item in replayed:
                stream.write(
                    json.dumps(item.as_dict(), sort_keys=True, separators=(",", ":")) + "\n"
                )
        generated_files["scores.jsonl"] = stream.record()
        from peermatchlab import __version__

        manifest: dict[str, object] = {
            "schema_version": 1,
            "adapter": "peermatchlab-embedding-expertise-run-v1",
            "generator": {"package": "peermatchlab", "version": __version__},
            "source": {"adapter": source.adapter, "parameters": dict(source.adapter_parameters)},
            "embedding": frozen_embedding_manifest,
            "scoring": {
                "aggregation": aggregation,
                "max_paper_pairs": max_paper_pairs,
                "max_candidate_pairs": max_candidate_pairs,
                "semantics": (
                    "L2 cosine; global min-max over all submission-publication pairs; "
                    "empty publication vectors excluded from reviewer aggregation; "
                    "round to four decimals"
                ),
            },
            "records": {
                "submissions": len(documents),
                "reviewers": len(experts),
                "candidate_pairs": len(replayed),
                "emitted_pairs": len(emitted),
            },
            "limits": {
                "max_generated_file_bytes": _MAX_GENERATED_FILE_BYTES,
                "max_generated_total_bytes": _MAX_GENERATED_TOTAL_BYTES,
            },
            "files": input_files | generated_files,
        }
        _write_json_object(staging / "manifest.json", manifest, budget)
        try:
            _install_directory_no_replace(staging, destination)
        except FileExistsError as error:
            raise DataValidationError(
                f"embedding destination already exists: {destination}"
            ) from error
        installed = True
        return MappingProxyType(manifest)
    finally:
        if not installed:
            shutil.rmtree(staging, ignore_errors=True)
