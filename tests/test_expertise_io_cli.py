from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from peermatchlab.cli import main
from peermatchlab.expertise import ExpertiseConfig, ExpertiseModel, generate_expertise
from peermatchlab.expertise_io import (
    LocalExpertiseInputs,
    load_expertise_config_source,
    load_local_domain_expertise_inputs,
    load_openreview_expertise_snapshot,
    local_domain_inputs,
    write_expertise_run,
)
from peermatchlab.io import load_documents, load_experts, load_json_text
from peermatchlab.models import DataValidationError, Document, Expert, Publication


def _line(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _snapshot(root: Path) -> Path:
    root.mkdir()
    _line(
        root / "submissions.jsonl",
        {
            "id": "paper-1",
            "forum": "forum-1",
            "content": {
                "title": {"value": "Graph retrieval"},
                "abstract": {"value": "Sparse neural ranking"},
                "keywords": {"value": ["IR"]},
            },
        },
    )
    _line(
        root / "profiles.jsonl",
        {
            "id": "~Ada_Lovelace1",
            "content": {
                "names": {
                    "value": [
                        {"fullname": "A. Lovelace"},
                        {"fullname": "Ada Lovelace", "preferred": True},
                    ]
                },
                "bio": {"value": "Sparse retrieval researcher"},
                "expertise": {"value": ["information retrieval"]},
                "research_interests": "graph ranking",
                "keywords": {"value": ["BM25"]},
            },
        },
    )
    _line(
        root / "reviewer-publications.jsonl",
        {
            "reviewer_id": "~Ada_Lovelace1",
            "note": {
                "id": "publication-1",
                "content": {
                    "title": {"value": "Sparse graph ranking"},
                    "abstract": "Retrieval systems",
                    "year": {"value": 2024},
                },
            },
        },
    )
    (root / "manifest.json").write_text('{"source":"fixture"}\n', encoding="utf-8")
    return root


def test_openreview_snapshot_builds_profile_and_publication_evidence(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "snapshot")
    inputs = load_openreview_expertise_snapshot(snapshot, reviewer_capacity=4)
    assert inputs.adapter == "openreview-local-expertise-snapshot-v1"
    assert inputs.documents[0].title == "Graph retrieval"
    expert = inputs.experts[0]
    assert expert.id == "~Ada_Lovelace1"
    assert expert.name == "Ada Lovelace"
    assert expert.summary == "Sparse retrieval researcher"
    assert expert.topics == ("information retrieval", "graph ranking")
    assert expert.keywords == ("BM25",)
    assert expert.capacity == 4
    assert expert.publications == (
        Publication("Sparse graph ranking", "Retrieval systems", 2024, "publication-1"),
    )
    run = generate_expertise(
        inputs.documents,
        inputs.experts,
        config=ExpertiseConfig(aggregation="max", include_profile=False),
    )
    assert run.scores[0].selected_evidence_id.endswith(":publication-1")
    assert set(inputs.source_files) == {
        "submissions.jsonl",
        "profiles.jsonl",
        "reviewer-publications.jsonl",
    }


@pytest.mark.parametrize(
    ("filename", "value", "message"),
    [
        ("profiles.jsonl", {"id": "r", "content": {}}, "must contain names"),
        (
            "reviewer-publications.jsonl",
            {
                "reviewer_id": "missing",
                "note": {"id": "p", "content": {"title": "Evidence"}},
            },
            "unknown profile",
        ),
        (
            "reviewer-publications.jsonl",
            {"reviewer_id": "~Ada_Lovelace1", "note": {"id": "p", "content": {}}},
            "publication title",
        ),
    ],
)
def test_snapshot_rejects_missing_or_unjoined_evidence(
    tmp_path: Path, filename: str, value: object, message: str
) -> None:
    snapshot = _snapshot(tmp_path / "snapshot")
    _line(snapshot / filename, value)
    with pytest.raises(DataValidationError, match=message):
        load_openreview_expertise_snapshot(snapshot)


def test_snapshot_rejects_duplicate_json_and_byte_or_record_limits(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "snapshot")
    (snapshot / "profiles.jsonl").write_text(
        '{"id":"r","id":"r2","content":{}}\n', encoding="utf-8"
    )
    with pytest.raises(DataValidationError, match="duplicate JSON field"):
        load_openreview_expertise_snapshot(snapshot)

    snapshot = _snapshot(tmp_path / "snapshot-two")
    with pytest.raises(DataValidationError, match="max_input_file_bytes"):
        load_openreview_expertise_snapshot(
            snapshot, config=ExpertiseConfig(max_input_file_bytes=10)
        )


def test_snapshot_publication_record_limit(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "snapshot")
    row = (snapshot / "reviewer-publications.jsonl").read_text(encoding="utf-8")
    (snapshot / "reviewer-publications.jsonl").write_text(row + row, encoding="utf-8")
    with pytest.raises(DataValidationError, match="record limit"):
        load_openreview_expertise_snapshot(
            snapshot, config=ExpertiseConfig(max_total_publications=1)
        )


def test_atomic_run_artifacts_have_replayable_model_and_byte_provenance(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    _line(document_path, {"id": "p", "title": "graph ranking"})
    _line(
        expert_path,
        {
            "id": "r",
            "name": "R",
            "summary": "graph",
            "publications": [{"title": "ranking systems", "year": 2024}],
        },
    )
    documents = load_documents(document_path)
    experts = load_experts(expert_path)
    source = local_domain_inputs(
        documents,
        experts,
        document_path=document_path,
        expert_path=expert_path,
    )
    run = generate_expertise(documents, experts)
    destination = tmp_path / "run"
    manifest = write_expertise_run(run, destination, source=source)

    assert set(path.name for path in destination.iterdir()) == {
        "affinities.csv",
        "documents.json",
        "experts.json",
        "explanations.jsonl",
        "manifest.json",
        "model.json",
        "reviewer-documents.jsonl",
        "submission-documents.jsonl",
    }
    for name, record in manifest["files"].items():  # type: ignore[union-attr]
        data = (destination / name).read_bytes()
        assert record == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    for name, record in manifest["source"]["files"].items():  # type: ignore[index,union-attr]
        source_file = {"documents": document_path, "experts": expert_path}[name]
        data = source_file.read_bytes()
        assert record == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    persisted = ExpertiseModel.load(destination / "model.json")
    assert persisted.score(run.corpus.submissions) == run.scores
    with (destination / "affinities.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["document_id", "expert_id", "score"]
    assert rows[1][:2] == ["p", "r"]
    explanation = load_json_text((destination / "explanations.jsonl").read_text(encoding="utf-8"))
    assert isinstance(explanation, dict)
    assert explanation["document_id"] == "p"


def test_publication_id_namespace_collision_cannot_create_unloadable_artifact(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    _line(document_path, {"id": "p", "title": "graph"})
    _line(
        expert_path,
        {
            "id": "r",
            "name": "R",
            "publications": [
                {"id": "2", "title": "graph systems"},
                {"title": "neural retrieval"},
            ],
        },
    )
    documents = load_documents(document_path)
    experts = load_experts(expert_path)
    run = generate_expertise(
        documents,
        experts,
        config=ExpertiseConfig(aggregation="max", include_profile=False),
    )
    destination = tmp_path / "run"
    write_expertise_run(
        run,
        destination,
        source=local_domain_inputs(
            documents,
            experts,
            document_path=document_path,
            expert_path=expert_path,
        ),
    )
    reloaded = ExpertiseModel.load(destination / "model.json")
    assert [item.id for item in reloaded.reviewer_evidence["r"]] == [
        "publication:r:id:2",
        "publication:r:position:2",
    ]


def test_configuration_provenance_uses_the_bytes_that_were_parsed(tmp_path: Path) -> None:
    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    config_path = tmp_path / "config.json"
    _line(document_path, {"id": "p", "title": "graph"})
    _line(expert_path, {"id": "r", "name": "R", "summary": "graph"})
    original_config = b'{"model":"tfidf"}\n'
    config_path.write_bytes(original_config)
    config_source = load_expertise_config_source(config_path)
    object.__setattr__(config_source, "byte_count", 0)
    object.__setattr__(config_source, "sha256", "0" * 64)
    documents = load_documents(document_path)
    experts = load_experts(expert_path)
    source = local_domain_inputs(
        documents,
        experts,
        document_path=document_path,
        expert_path=expert_path,
    )
    run = generate_expertise(documents, experts, config=config_source.config)
    destination = tmp_path / "run"
    manifest = write_expertise_run(
        run,
        destination,
        source=source,
        config_source=config_source,
    )
    assert manifest["source"]["files"]["expertise-config"] == {  # type: ignore[index]
        "bytes": len(original_config),
        "sha256": hashlib.sha256(original_config).hexdigest(),
    }

    config_path.write_text('{\n  "model": "tfidf"\n}\n', encoding="utf-8")
    changed_destination = tmp_path / "changed-run"
    with pytest.raises(DataValidationError, match="configuration changed"):
        write_expertise_run(
            run,
            changed_destination,
            source=source,
            config_source=config_source,
        )
    assert not changed_destination.exists()


def test_run_writer_reparses_config_bytes_instead_of_trusting_mutated_cached_config(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    config_path = tmp_path / "config.json"
    _line(document_path, {"id": "p", "title": "graph"})
    _line(expert_path, {"id": "r", "name": "R", "summary": "graph"})
    config_path.write_bytes(b'{"minimum_output_score":0.0}\n')
    config_source = load_expertise_config_source(config_path)
    source = load_local_domain_expertise_inputs(
        document_path, expert_path, config=config_source.config
    )

    object.__setattr__(config_source.config, "minimum_output_score", 0.75)
    run = generate_expertise(source.documents, source.experts, config=config_source.config)
    destination = tmp_path / "forged-config-run"
    with pytest.raises(DataValidationError, match="configuration does not match"):
        write_expertise_run(
            run,
            destination,
            source=source,
            config_source=config_source,
        )
    assert config_source.rederive().minimum_output_score == 0.0
    assert not destination.exists()


def test_normalized_local_json_stops_at_max_submissions_plus_one(tmp_path: Path) -> None:
    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    document_path.write_text(
        '[{"id":"p1","title":"One"},'
        '{"id":"p2","title":"Two"},'
        '{"id":"never-parsed","id":"duplicate","title":"Three"}]',
        encoding="utf-8",
    )
    _line(expert_path, {"id": "r", "name": "R"})

    with pytest.raises(DataValidationError, match="record limit"):
        load_local_domain_expertise_inputs(
            document_path,
            expert_path,
            config=ExpertiseConfig(max_submissions=1),
        )


def test_local_domain_provenance_rejects_objects_from_different_files(tmp_path: Path) -> None:
    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    _line(document_path, {"id": "p", "title": "source title"})
    _line(expert_path, {"id": "r", "name": "R"})
    with pytest.raises(DataValidationError, match="do not match"):
        local_domain_inputs(
            (Document("p", "different title"),),
            load_experts(expert_path),
            document_path=document_path,
            expert_path=expert_path,
        )


def test_local_input_constructor_closes_bytes_objects_adapter_and_empty_source(
    tmp_path: Path,
) -> None:
    a_documents = tmp_path / "a-documents.json"
    a_experts = tmp_path / "a-experts.json"
    b_documents = tmp_path / "b-documents.json"
    b_experts = tmp_path / "b-experts.json"
    _line(a_documents, {"id": "a", "title": "Graph"})
    _line(a_experts, {"id": "ra", "name": "A"})
    _line(b_documents, {"id": "b", "title": "Biology"})
    _line(b_experts, {"id": "rb", "name": "B"})
    source_a = load_local_domain_expertise_inputs(a_documents, a_experts)
    source_b = load_local_domain_expertise_inputs(b_documents, b_experts)

    with pytest.raises(DataValidationError, match="immutable source bytes"):
        LocalExpertiseInputs(
            source_a.documents,
            source_a.experts,
            source_b.source_files,
            source_b.source_bytes,
            source_b.adapter,
            source_b.adapter_parameters,
        )
    with pytest.raises(DataValidationError, match="adapter contract"):
        LocalExpertiseInputs(
            source_a.documents,
            source_a.experts,
            {},
            {},
            source_a.adapter,
            source_a.adapter_parameters,
        )

    assert isinstance(source_a.documents, tuple)
    assert isinstance(source_a.experts, tuple)
    with pytest.raises(TypeError):
        source_a.source_bytes["documents"] = b"forged"  # type: ignore[index]
    with pytest.raises(TypeError):
        source_a.adapter_parameters["max_submissions"] = 99  # type: ignore[index]


def test_writer_rederives_from_bytes_and_never_trusts_cached_digest(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    _line(document_path, {"id": "p", "title": "Graph"})
    _line(expert_path, {"id": "r", "name": "R", "summary": "Graph"})
    source = load_local_domain_expertise_inputs(document_path, expert_path)
    run = generate_expertise(source.documents, source.experts)
    object.__setattr__(
        source,
        "source_records",
        {name: {"bytes": 0, "sha256": "0" * 64} for name in source.source_bytes},
    )
    destination = tmp_path / "digest-run"
    manifest = write_expertise_run(run, destination, source=source)
    assert manifest["source"]["files"]["documents"] == {  # type: ignore[index]
        "bytes": len(document_path.read_bytes()),
        "sha256": hashlib.sha256(document_path.read_bytes()).hexdigest(),
    }

    compromised = load_local_domain_expertise_inputs(document_path, expert_path)
    object.__setattr__(compromised, "documents", (Document("x", "Forged"),))
    with pytest.raises(DataValidationError, match="changed after"):
        write_expertise_run(run, tmp_path / "compromised", source=compromised)


def test_local_input_constructor_bounds_public_iterables(tmp_path: Path) -> None:
    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    _line(document_path, {"id": "p", "title": "Graph"})
    _line(expert_path, {"id": "r", "name": "R"})
    source = load_local_domain_expertise_inputs(document_path, expert_path)
    parameters = dict(source.adapter_parameters)
    parameters["max_submissions"] = 1

    def documents() -> object:
        yield source.documents[0]
        yield Document("extra", "Extra")
        raise AssertionError("local document iterator was over-consumed")

    with pytest.raises(DataValidationError, match="configured limit"):
        LocalExpertiseInputs(
            documents(),  # type: ignore[arg-type]
            source.experts,
            source.source_files,
            source.source_bytes,
            source.adapter,
            parameters,
        )


def test_sparse_output_omits_zero_and_below_threshold_pairs(tmp_path: Path) -> None:
    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    _line(document_path, {"id": "p", "title": "graph"})
    _line(
        expert_path,
        [
            {"id": "partial", "name": "Partial", "summary": "graph systems"},
            {"id": "zero", "name": "Zero", "summary": "biology"},
        ],
    )
    documents = load_documents(document_path)
    experts = load_experts(expert_path)
    config = ExpertiseConfig(minimum_output_score=0.9)
    run = generate_expertise(documents, experts, config=config)
    destination = tmp_path / "sparse"
    manifest = write_expertise_run(
        run,
        destination,
        source=local_domain_inputs(
            documents,
            experts,
            document_path=document_path,
            expert_path=expert_path,
        ),
    )
    assert (destination / "affinities.csv").read_text(encoding="utf-8") == (
        "document_id,expert_id,score\n"
    )
    assert (destination / "explanations.jsonl").read_text(encoding="utf-8") == ""
    assert manifest["records"]["candidate_pairs"] == 2  # type: ignore[index]
    assert manifest["records"]["emitted_pairs"] == 0  # type: ignore[index]
    assert "minimum_output_score" in manifest["score_semantics"]["emission_rule"]  # type: ignore[index]


def test_run_writer_refuses_existing_destination_and_cleans_failed_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = (Document("p", "graph"),)
    experts = (Expert("r", "R", summary="graph"),)
    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    _line(document_path, {"id": "p", "title": "graph"})
    _line(expert_path, {"id": "r", "name": "R", "summary": "graph"})
    source = local_domain_inputs(
        documents,
        experts,
        document_path=document_path,
        expert_path=expert_path,
    )
    run = generate_expertise(documents, experts)
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(DataValidationError, match="already exists"):
        write_expertise_run(run, destination, source=source)

    document_path.write_text("[]", encoding="utf-8")
    changed = tmp_path / "changed"
    with pytest.raises(DataValidationError, match="source file changed"):
        write_expertise_run(run, changed, source=source)
    assert not changed.exists()
    _line(document_path, {"id": "p", "title": "graph"})

    import peermatchlab.expertise_io as expertise_io

    def fail(_path: object, _value: object) -> None:
        raise OSError("injected failure")

    monkeypatch.setattr(expertise_io, "write_json", fail)
    failed = tmp_path / "failed"
    with pytest.raises(OSError, match="injected"):
        write_expertise_run(run, failed, source=source)
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed-*"))


def test_run_writer_loses_install_race_without_replacing_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import peermatchlab.expertise_io as expertise_io

    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    _line(document_path, {"id": "p", "title": "graph"})
    _line(expert_path, {"id": "r", "name": "R", "summary": "graph"})
    source = load_local_domain_expertise_inputs(document_path, expert_path)
    run = generate_expertise(source.documents, source.experts)
    destination = tmp_path / "raced"
    original_install = expertise_io._install_directory_no_replace

    def create_competitor_then_install(staging: Path, target: Path) -> None:
        target.mkdir()
        original_install(staging, target)

    monkeypatch.setattr(
        expertise_io, "_install_directory_no_replace", create_competitor_then_install
    )
    with pytest.raises(DataValidationError, match="already exists"):
        write_expertise_run(run, destination, source=source)
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert not list(tmp_path.glob(".raced-*"))


def test_normalized_local_inputs_parse_each_bounded_byte_snapshot_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import peermatchlab.expertise_io as expertise_io

    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    _line(document_path, {"id": "p", "title": "graph"})
    _line(expert_path, {"id": "r", "name": "R"})
    original_read = expertise_io._read_bounded
    calls: list[Path] = []

    def counted_read(path: Path, *, max_bytes: int, label: str = "snapshot") -> bytes:
        calls.append(path)
        return original_read(path, max_bytes=max_bytes, label=label)

    monkeypatch.setattr(expertise_io, "_read_bounded", counted_read)
    source = load_local_domain_expertise_inputs(document_path, expert_path)
    assert source.documents[0].id == "p"
    assert source.experts[0].id == "r"
    assert calls == [document_path, expert_path]


def test_cli_turns_deep_json_recursion_into_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "deep.json"
    config_path.write_text("[" * 2_000 + "0" + "]" * 2_000, encoding="utf-8")
    assert (
        main(
            [
                "expertise",
                "--documents",
                str(tmp_path / "unused-documents.json"),
                "--experts",
                str(tmp_path / "unused-experts.json"),
                "--config",
                str(config_path),
                "--directory",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert "nesting exceeds" in capsys.readouterr().err


def test_run_writer_reloads_model_before_installing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    _line(document_path, {"id": "p", "title": "graph"})
    _line(expert_path, {"id": "r", "name": "R", "summary": "graph"})
    documents = load_documents(document_path)
    experts = load_experts(expert_path)
    run = generate_expertise(documents, experts)
    source = local_domain_inputs(
        documents,
        experts,
        document_path=document_path,
        expert_path=expert_path,
    )
    original_save = ExpertiseModel.save

    def save_corrupt_model(model: ExpertiseModel, path: str | Path) -> None:
        original_save(model, path)
        target = Path(path)
        state = load_json_text(target.read_text(encoding="utf-8"))
        assert isinstance(state, dict)
        statistics = state["statistics"]
        assert isinstance(statistics, dict)
        statistics["document_count"] = 99
        target.write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.setattr(ExpertiseModel, "save", save_corrupt_model)
    destination = tmp_path / "corrupt-run"
    with pytest.raises(DataValidationError, match="document_count"):
        write_expertise_run(run, destination, source=source)
    assert not destination.exists()
    assert not list(tmp_path.glob(".corrupt-run-*"))


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_run_writer_cleans_stage_for_base_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    import peermatchlab.expertise_io as expertise_io

    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    _line(document_path, {"id": "p", "title": "Graph"})
    _line(expert_path, {"id": "r", "name": "R"})
    source = load_local_domain_expertise_inputs(document_path, expert_path)
    run = generate_expertise(source.documents, source.experts)

    def interrupt(_path: object, _value: object) -> None:
        raise interruption()

    monkeypatch.setattr(expertise_io, "write_json", interrupt)
    destination = tmp_path / "interrupted"
    with pytest.raises(interruption):
        write_expertise_run(run, destination, source=source)
    assert not destination.exists()
    assert not list(tmp_path.glob(".interrupted-*"))


def test_successful_install_does_not_remove_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import peermatchlab.expertise_io as expertise_io

    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    _line(document_path, {"id": "p", "title": "Graph"})
    _line(expert_path, {"id": "r", "name": "R", "summary": "Graph"})
    source = load_local_domain_expertise_inputs(document_path, expert_path)
    run = generate_expertise(source.documents, source.experts)

    def unexpected_cleanup(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("successful installation attempted cleanup")

    monkeypatch.setattr(expertise_io.shutil, "rmtree", unexpected_cleanup)
    destination = tmp_path / "installed"
    write_expertise_run(run, destination, source=source)
    assert (destination / "manifest.json").is_file()


def test_cli_generates_from_normalized_fixtures_and_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document_path = tmp_path / "documents.json"
    expert_path = tmp_path / "experts.json"
    config_path = tmp_path / "config.json"
    _line(document_path, {"id": "p", "title": "graph ranking"})
    _line(expert_path, {"id": "r", "name": "R", "summary": "graph ranking"})
    _line(config_path, {"model": "bm25", "aggregation": "aggregate"})
    destination = tmp_path / "domain-run"
    assert (
        main(
            [
                "expertise",
                "--documents",
                str(document_path),
                "--experts",
                str(expert_path),
                "--config",
                str(config_path),
                "--directory",
                str(destination),
            ]
        )
        == 0
    )
    assert "generated 1 sparse affinities" in capsys.readouterr().out
    manifest = load_json_text((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator"]["version"] == "0.7.0"  # type: ignore[index]
    assert "expertise-config" in manifest["source"]["files"]  # type: ignore[index,operator]

    snapshot = _snapshot(tmp_path / "snapshot")
    snapshot_destination = tmp_path / "snapshot-run"
    assert (
        main(
            [
                "expertise",
                "--snapshot",
                str(snapshot),
                "--reviewer-capacity",
                "3",
                "--directory",
                str(snapshot_destination),
            ]
        )
        == 0
    )
    assert (snapshot_destination / "model.json").is_file()


@pytest.mark.parametrize(
    "arguments",
    [
        ["expertise", "--documents", "missing", "--directory", "out"],
        [
            "expertise",
            "--snapshot",
            "missing",
            "--experts",
            "experts.json",
            "--directory",
            "out",
        ],
        [
            "expertise",
            "--documents",
            "documents.json",
            "--experts",
            "experts.json",
            "--reviewer-capacity",
            "2",
            "--directory",
            "out",
        ],
    ],
)
def test_cli_reports_invalid_source_combinations(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(arguments) == 2
    assert "error:" in capsys.readouterr().err
