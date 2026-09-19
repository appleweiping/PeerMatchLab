"""Independent numeric and adversarial checks for frozen embedding interchange."""

from __future__ import annotations

import hashlib
import json
import shutil
import tracemalloc
from pathlib import Path
from typing import Literal

import pytest

from peermatchlab.cli import main
from peermatchlab.embedding import (
    EmbeddingRequest,
    FrozenJsonlEmbeddingProvider,
    _BoundedTextWriter,
    _OutputBudget,
    _parse_jsonl,
    _write_captured_bytes,
    embedding_requests,
    request_sha256,
    score_embedding_expertise,
    write_embedding_run,
)
from peermatchlab.expertise_io import load_openreview_expertise_snapshot
from peermatchlab.models import DataValidationError, Document, Expert, Publication

_ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOT = _ROOT / "examples" / "expertise" / "snapshot"
_VECTORS = _ROOT / "examples" / "expertise" / "embeddings"


def _vector(x: float, y: float) -> tuple[float, ...]:
    return (x, y) + (0.0,) * 766


class OracleProvider:
    def __init__(
        self, submissions: dict[str, tuple[float, ...]], publications: dict[str, tuple[float, ...]]
    ) -> None:
        self.vectors = {"submissions": submissions, "publications": publications}

    @property
    def provenance(self) -> dict[str, object]:
        return {"origin": "independent-test"}

    def embed(
        self, kind: Literal["submissions", "publications"], requests: tuple[EmbeddingRequest, ...]
    ) -> dict[str, tuple[float, ...]]:
        assert set(self.vectors[kind]) == {item.paper_id for item in requests}
        return self.vectors[kind]


def _domain() -> tuple[tuple[Document, ...], tuple[Expert, ...]]:
    return (
        (Document("s1", "One"), Document("s2", "Two")),
        (
            Expert(
                "r1",
                "Reviewer One",
                publications=(
                    Publication("A", id="a"),
                    Publication("B", id="b"),
                ),
            ),
            Expert("r2", "Reviewer Two", publications=(Publication("C", id="c"),)),
        ),
    )


def test_hand_computed_global_minmax_max_and_average() -> None:
    docs, experts = _domain()
    # Unit vectors make the cosine table exactly:
    #             a   b   c
    #       s1    1   0  -1
    #       s2    0   1   0
    # Global min/max -1/+1 yields 1, .5, 0 / .5, 1, .5.
    provider = OracleProvider(
        {"s1": _vector(1, 0), "s2": _vector(0, 1)},
        {"a": _vector(1, 0), "b": _vector(0, 1), "c": _vector(-1, 0)},
    )
    maximum = score_embedding_expertise(docs, experts, provider, aggregation="max")
    average = score_embedding_expertise(docs, experts, provider, aggregation="average")
    assert [(item.document_id, item.expert_id, item.score) for item in maximum] == [
        ("s1", "r1", 1.0),
        ("s1", "r2", 0.0),
        ("s2", "r1", 1.0),
        ("s2", "r2", 0.5),
    ]
    assert [item.score for item in average] == [0.75, 0.0, 0.75, 0.5]
    assert [item.selected_publication_id for item in maximum] == ["a", "c", "b", "c"]


def test_empty_publication_affects_global_range_but_not_reviewer_aggregation() -> None:
    docs = (Document("s", "Submission"),)
    experts = (
        Expert(
            "r1",
            "Reviewer",
            publications=(
                Publication("Valid", id="valid"),
                Publication("Missing", id="missing"),
            ),
        ),
        Expert("r2", "No publications"),
    )
    provider = OracleProvider({"s": _vector(1, 0)}, {"valid": _vector(1, 0), "missing": ()})
    scores = score_embedding_expertise(docs, experts, provider, aggregation="average")
    assert [(item.score, item.evidence_count) for item in scores] == [(1.0, 1), (0.0, 0)]


def test_constant_cosine_clamps_without_dividing_by_zero() -> None:
    docs = (Document("s", "Submission"),)
    experts = (Expert("r", "Reviewer", publications=(Publication("P", id="p"),)),)
    positive = OracleProvider({"s": _vector(1, 0)}, {"p": _vector(1, 0)})
    negative = OracleProvider({"s": _vector(1, 0)}, {"p": _vector(-1, 0)})
    assert score_embedding_expertise(docs, experts, positive)[0].score == 1.0
    assert score_embedding_expertise(docs, experts, negative)[0].score == 0.0


def test_frozen_fixture_binds_text_and_is_synthetic() -> None:
    source = load_openreview_expertise_snapshot(_SNAPSHOT)
    provider = FrozenJsonlEmbeddingProvider(_VECTORS)
    docs, pubs, _ = embedding_requests(source.documents, source.experts)
    assert len(provider.embed("submissions", docs)) == 1
    assert len(provider.embed("publications", pubs)) == 2
    assert provider.provenance["encoder"]["origin"] == "synthetic-fixture"
    assert provider.provenance["encoder"]["weights_sha256"] is None
    assert request_sha256(docs) == provider.provenance["requests"]["submissions_sha256"]
    with pytest.raises(DataValidationError, match="text hash mismatch"):
        provider.embed("submissions", (EmbeddingRequest("paper-1", "Wrong title", ""),))


def test_nested_provenance_mutation_cannot_forge_published_origin(tmp_path: Path) -> None:
    source = load_openreview_expertise_snapshot(_SNAPSHOT)
    provider = FrozenJsonlEmbeddingProvider(_VECTORS)
    caller_view = provider.provenance
    caller_view["encoder"]["origin"] = "external-encoder"
    caller_view["encoder"]["weights_sha256"] = "a" * 64
    assert provider.provenance["encoder"]["origin"] == "synthetic-fixture"
    scores = score_embedding_expertise(source.documents, source.experts, provider)
    write_embedding_run(scores, tmp_path / "run", source=source, provider=provider)
    published = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert published["embedding"]["encoder"] == {
        "family": "specter2",
        "model_id": "synthetic-specter2-interchange-oracle",
        "revision": "fixture-v1",
        "origin": "synthetic-fixture",
        "weights_sha256": None,
    }


def test_manifest_changed_on_disk_before_publication_is_rejected(tmp_path: Path) -> None:
    source = load_openreview_expertise_snapshot(_SNAPSHOT)
    fixture = tmp_path / "fixture"
    shutil.copytree(_VECTORS, fixture)
    provider = FrozenJsonlEmbeddingProvider(fixture)
    scores = score_embedding_expertise(source.documents, source.experts, provider)
    manifest_path = fixture / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["encoder"]["origin"] = "external-encoder"
    manifest["encoder"]["weights_sha256"] = "a" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataValidationError, match="changed after loading"):
        write_embedding_run(scores, tmp_path / "run", source=source, provider=provider)
    assert not (tmp_path / "run").exists()


def test_generated_output_over_input_ceiling_streams_hash(tmp_path: Path) -> None:
    # Regression: generated score output may validly exceed the 64 MiB INPUT cap.
    path = tmp_path / "generated.jsonl"
    chunk = "x" * (1024 * 1024)
    expected = hashlib.sha256()
    with _BoundedTextWriter(path, _OutputBudget()) as stream:
        for _ in range(65):
            stream.write(chunk)
            expected.update(chunk.encode("ascii"))
    assert stream.record() == {"bytes": 65 * 1024 * 1024, "sha256": expected.hexdigest()}
    assert path.stat().st_size > 64 * 1024 * 1024


def test_short_binary_writes_fail_before_hash_or_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ShortWriter:
        def __enter__(self) -> ShortWriter:
            return self

        def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
            return None

        def write(self, data: bytes) -> int:
            return len(data) - 1

        def close(self) -> None:
            return None

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", lambda _path, _mode: ShortWriter())
        with (
            _BoundedTextWriter(tmp_path / "generated.jsonl", _OutputBudget()) as stream,
            pytest.raises(DataValidationError, match=r"short write.*output"),
        ):
            stream.write("abc")
        assert stream.record() == {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
        with pytest.raises(DataValidationError, match=r"short write.*input"):
            _write_captured_bytes(tmp_path / "source.json", b"abc")


def test_published_embedding_manifest_matches_every_file(tmp_path: Path) -> None:
    source = load_openreview_expertise_snapshot(_SNAPSHOT)
    provider = FrozenJsonlEmbeddingProvider(_VECTORS)
    scores = score_embedding_expertise(source.documents, source.experts, provider)
    destination = tmp_path / "run"
    manifest = write_embedding_run(scores, destination, source=source, provider=provider)
    files = manifest["files"]
    assert isinstance(files, dict)
    for name, expected in files.items():
        assert isinstance(name, str) and isinstance(expected, dict)
        data = (destination / name).read_bytes()
        assert expected == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.mark.parametrize(
    ("limit_name", "message"),
    [
        ("_MAX_GENERATED_FILE_BYTES", "file exceeds output byte limit"),
        ("_MAX_GENERATED_TOTAL_BYTES", "total byte limit"),
    ],
)
def test_output_cap_failure_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit_name: str, message: str
) -> None:
    import peermatchlab.embedding as embedding_module

    source = load_openreview_expertise_snapshot(_SNAPSHOT)
    provider = FrozenJsonlEmbeddingProvider(_VECTORS)
    scores = score_embedding_expertise(source.documents, source.experts, provider)
    monkeypatch.setattr(embedding_module, limit_name, 100)
    destination = tmp_path / "run"
    with pytest.raises(DataValidationError, match=message):
        write_embedding_run(scores, destination, source=source, provider=provider)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_fixture_hash_and_schema_rejections(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(_VECTORS, fixture)
    with (fixture / "publications.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(DataValidationError, match="hash or length mismatch"):
        FrozenJsonlEmbeddingProvider(fixture)
    shutil.copy2(_VECTORS / "publications.jsonl", fixture / "publications.jsonl")
    path = fixture / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["encoder"]["family"] = []
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataValidationError, match="family"):
        FrozenJsonlEmbeddingProvider(fixture)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda m: m.update(schema_version=True), "schema version"),
        (lambda m: m.update(adapter="wrong"), "adapter"),
        (lambda m: m["encoder"].update(origin="wrong"), "origin"),
        (lambda m: m["encoder"].update(weights_sha256="a" * 64), "synthetic fixtures"),
        (lambda m: m["encoder"].update(origin="external-encoder"), "weights_sha256"),
        (lambda m: m["requests"].update(submissions_sha256="wrong"), "request hashes"),
        (lambda m: m.update(files={}), "file manifest"),
    ],
)
def test_manifest_adversarial_cases(tmp_path: Path, mutation: object, message: str) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(_VECTORS, fixture)
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    mutation(manifest)
    (fixture / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataValidationError, match=message):
        FrozenJsonlEmbeddingProvider(fixture)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"\n", "blank"),
        (b'{"paper_id":"p","embedding":[1]}\n', "768 dimensions"),
        (b'{"paper_id":"p","embedding":[]}\n{"paper_id":"p","embedding":[]}\n', "unique"),
        (b'{"paper_id":"p","embedding":[],"other":1}\n', "schema"),
        (b"\xff", "UTF-8"),
    ],
)
def test_jsonl_rejects_malformed_rows(data: bytes, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        _parse_jsonl(data, kind="publications")


def test_provider_rejects_missing_ids_and_bad_kind() -> None:
    provider = FrozenJsonlEmbeddingProvider(_VECTORS)
    with pytest.raises(DataValidationError, match="kind"):
        provider.embed("other", ())  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="repeat"):
        provider.embed("publications", (EmbeddingRequest("x", "X", ""),) * 2)


def test_scoring_rejects_bad_limits_and_ids() -> None:
    docs, experts = _domain()
    provider = OracleProvider(
        {"s1": _vector(1, 0), "s2": _vector(0, 1)},
        {"a": _vector(1, 0), "b": _vector(0, 1), "c": _vector(-1, 0)},
    )
    with pytest.raises(DataValidationError, match="aggregation"):
        score_embedding_expertise(docs, experts, provider, aggregation="sum")  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="max_candidate_pairs"):
        score_embedding_expertise(docs, experts, provider, max_candidate_pairs=True)
    with pytest.raises(DataValidationError, match="max_candidate_pairs"):
        score_embedding_expertise(docs, experts, provider, max_candidate_pairs=1_000_001)
    with pytest.raises(DataValidationError, match="unique"):
        score_embedding_expertise((docs[0], docs[0]), experts, provider)
    with pytest.raises(DataValidationError, match="max_candidate_pairs"):
        score_embedding_expertise(docs, experts, provider, max_candidate_pairs=1)


def test_many_empty_publications_have_bounded_peak_memory() -> None:
    documents = tuple(Document(f"s{i}", "Submission") for i in range(500))
    publications = tuple(Publication(f"P{i}", id=f"p{i}") for i in range(200))
    experts = (Expert("r", "Reviewer", publications=publications),)
    provider = OracleProvider(
        {item.id: () for item in documents},
        {item.id: () for item in publications if item.id is not None},
    )
    tracemalloc.start()
    try:
        scores = score_embedding_expertise(documents, experts, provider, max_paper_pairs=100_000)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(scores) == 500
    assert all(item.score == 0.0 for item in scores)
    assert peak < 10_000_000


def test_thousands_of_empty_vectors_do_not_allocate_zero_tuples() -> None:
    documents = (Document("s", "Submission"),)
    publications = tuple(Publication(f"P{i}", id=f"p{i}") for i in range(5_000))
    experts = (Expert("r", "Reviewer", publications=publications),)
    provider = OracleProvider({"s": ()}, {f"p{i}": () for i in range(5_000)})
    tracemalloc.start()
    try:
        scores = score_embedding_expertise(documents, experts, provider)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(scores) == 1 and scores[0].score == 0.0
    assert peak < 12_000_000


@pytest.mark.parametrize(
    "bad", [[], [True] * 768, [float("nan")] * 768, [10**400] * 768, [1.0] * 767]
)
def test_provider_coordinate_validation(bad: list[object]) -> None:
    docs = (Document("s", "Submission"),)
    experts = (Expert("r", "Reviewer", publications=(Publication("P", id="p"),)),)
    provider = OracleProvider({"s": _vector(1, 0)}, {"p": tuple(bad)})  # type: ignore[arg-type]
    if bad == []:
        assert score_embedding_expertise(docs, experts, provider)[0].score == 0.0
    else:
        with pytest.raises(DataValidationError, match="invalid coordinates"):
            score_embedding_expertise(docs, experts, provider)


def test_bounds_and_replay_tamper(tmp_path: Path) -> None:
    source = load_openreview_expertise_snapshot(_SNAPSHOT)
    provider = FrozenJsonlEmbeddingProvider(_VECTORS)
    with pytest.raises(DataValidationError, match="max_paper_pairs"):
        score_embedding_expertise(source.documents, source.experts, provider, max_paper_pairs=1)
    scores = score_embedding_expertise(source.documents, source.experts, provider)
    with pytest.raises(DataValidationError, match="do not match"):
        write_embedding_run(scores[:1], tmp_path / "bad", source=source, provider=provider)
    manifest = write_embedding_run(scores, tmp_path / "good", source=source, provider=provider)
    assert manifest["records"]["emitted_pairs"] == 1
    assert (tmp_path / "good" / "affinities.csv").read_text(encoding="utf-8").splitlines() == [
        "document_id,expert_id,score",
        "paper-1,~Ada_Reviewer1,1.0",
    ]


def test_cli_embedding_to_match_smoke(tmp_path: Path) -> None:
    output = tmp_path / "embedding-run"
    assert (
        main(
            [
                "expertise-embedding",
                "--snapshot",
                str(_SNAPSHOT),
                "--embeddings",
                str(_VECTORS),
                "--directory",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter"] == "peermatchlab-embedding-expertise-run-v1"
    assert manifest["records"]["candidate_pairs"] == 2
    assert (
        main(
            [
                "match-affinity",
                "--documents",
                str(output / "documents.json"),
                "--experts",
                str(output / "experts.json"),
                "--affinities",
                str(output / "affinities.csv"),
                "--config",
                str(_ROOT / "examples" / "expertise" / "match-config.json"),
                "--output",
                str(tmp_path / "plan.json"),
            ]
        )
        == 0
    )
