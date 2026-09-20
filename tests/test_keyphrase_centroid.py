from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import peermatchlab.keyphrase_centroid as centroid
from peermatchlab.affinity import load_affinities_csv
from peermatchlab.cli import main
from peermatchlab.io import load_json_text
from peermatchlab.keyphrase_centroid import (
    CentroidConfig,
    _update,
    _validation_map,
    load_centroid_inputs,
    load_centroid_model,
    score_keyphrase_centroid,
    train_keyphrase_centroid,
    verify_keyphrase_centroid,
    write_keyphrase_centroid_run,
)
from peermatchlab.keyphrases import write_keyphrase_run
from peermatchlab.models import DataValidationError
from peermatchlab.pipeline import run_affinity_matching

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "centroid"


def _fixture(tmp_path: Path, *, train: bytes | None = None, validation: bytes | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    documents = tmp_path / "documents.json"
    experts = tmp_path / "experts.json"
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    documents.write_bytes((EXAMPLE / "documents.json").read_bytes())
    experts.write_bytes((EXAMPLE / "experts.json").read_bytes())
    train_path.write_bytes(train if train is not None else (EXAMPLE / "train.jsonl").read_bytes())
    validation_path.write_bytes(
        validation if validation is not None else (EXAMPLE / "validation.jsonl").read_bytes()
    )
    keyphrases = tmp_path / "keyphrases"
    write_keyphrase_run(
        keyphrases, documents_source=documents.read_bytes(), experts_source=experts.read_bytes()
    )
    return keyphrases, documents, experts, train_path, validation_path


def _inputs(paths: tuple[Path, Path, Path, Path, Path], *, conflicts: Path | None = None):
    return load_centroid_inputs(*paths, conflicts_path=conflicts)


def _rewrite_phrases(paths: tuple[Path, Path, Path, Path, Path], change: str) -> None:
    records_path = paths[0] / "keyphrases.jsonl"
    manifest_path = paths[0] / "manifest.json"
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    selected = next(row for row in rows if row["kind"] == "profile")
    if change == "missing_field":
        selected.pop("token_count")
    elif change == "unknown_kind":
        selected["kind"] = "unknown"
    elif change == "wrong_evidence":
        selected["evidence_id"] = "not-owner"
    elif change == "bad_count":
        selected["token_count"] = True
    elif change == "bad_list":
        selected["keyphrases"] = "terms"
    elif change == "bad_phrase":
        selected["keyphrases"][0].pop("score")
    elif change == "bad_term":
        selected["keyphrases"][0]["term"] = "Capital"
    elif change == "bad_score":
        selected["keyphrases"][0]["score"] = -1
    elif change == "empty_profile":
        selected["keyphrases"] = []
    elif change == "missing_profile":
        rows.remove(selected)
    elif change == "duplicate_profile":
        rows.append(dict(selected))
    elif change == "too_many_terms":
        selected["keyphrases"] = [{"term": f"term{index}", "score": 0.01} for index in range(257)]
    else:
        raise AssertionError(change)
    raw = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows
    )
    records_path.write_bytes(raw)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"] = len(rows)
    manifest["keyphrases_bytes"] = len(raw)
    manifest["keyphrases_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_independent_one_step_pairwise_bce_gradient_oracle() -> None:
    # q=(1,0), p=(1,0), n=(0,1), margin=1. For BCEWithLogits(-margin),
    # d loss/d margin = -1/(1+e), so each coordinate moves by 0.1/(1+e).
    weights = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    loss = _update(weights, (0,), (1,), (2,), dimensions=2, learning_rate=0.1, l2=0.0)
    delta = 0.1 / (1.0 + math.e)
    assert loss == pytest.approx(math.log1p(math.exp(-1.0)))
    assert weights[0] == pytest.approx([1 + delta, -delta])
    assert weights[1] == pytest.approx([1 + delta, 0.0])
    assert weights[2] == pytest.approx([-delta, 1.0])


def test_independent_average_precision_tie_oracle() -> None:
    from peermatchlab.keyphrase_centroid import CentroidTriplet

    weights = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    rows = (
        CentroidTriplet("a", "positive", "negative"),
        CentroidTriplet("b", "positive", "negative"),
    )
    # a: positive first -> AP 1; b: negative first -> AP 1/2.
    assert _validation_map(
        rows, {"a": (0,), "b": (1,)}, {"positive": (2,), "negative": (1,)}, weights, 2
    ) == pytest.approx(0.75)


def test_exact_source_training_checkpoint_and_match_pipeline(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    inputs = _inputs(paths)
    config = CentroidConfig(dimensions=2, epochs=3, seed=7)
    model = train_keyphrase_centroid(inputs, config=config)
    assert model.best_epoch == 1 and model.validation_map == 1.0
    assert len(model.history) == 3
    assert model.to_bytes() == train_keyphrase_centroid(inputs, config=config).to_bytes()
    assert load_centroid_model(model.to_bytes()) == model
    assert verify_keyphrase_centroid(inputs, model.to_bytes())
    scores = score_keyphrase_centroid(inputs, model)
    assert [(item.document_id, item.expert_id) for item in scores] == [
        ("d-hold", "e-graph"),
        ("d-hold", "e-math"),
    ]
    assert scores[1].score > scores[0].score
    run = run_affinity_matching(
        tuple(item for item in inputs.documents if item.id == "d-hold"), inputs.experts, scores
    )
    assert run.plan.assignments[0].expert_id == "e-math"
    output = tmp_path / "run"
    manifest = write_keyphrase_centroid_run(inputs, model, output)
    assert manifest["counts"]["affinities"] == 2
    assert load_affinities_csv(output / "affinities.csv") == scores
    assert [item["id"] for item in load_json_text((output / "documents.json").read_text())] == [
        "d-hold"
    ]
    assert load_centroid_model((output / "model.json").read_bytes()) == model
    for name, record in manifest["files"].items():
        raw = (output / name).read_bytes()
        assert record == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    with pytest.raises(DataValidationError, match="already exists"):
        write_keyphrase_centroid_run(inputs, model, output)


def test_loaded_source_snapshot_and_model_provenance_are_immutable(tmp_path: Path) -> None:
    inputs = _inputs(_fixture(tmp_path))
    original_hashes = inputs.hashes
    with pytest.raises(TypeError):
        inputs.submissions["d-train"] = ("forged",)  # type: ignore[index]
    with pytest.raises(TypeError):
        inputs.raw["train"] = b"forged"  # type: ignore[index]
    assert inputs.hashes == original_hashes
    model = train_keyphrase_centroid(inputs, config=CentroidConfig(dimensions=2, epochs=1))
    with pytest.raises(TypeError):
        model.source_sha256["train"] = "0" * 64  # type: ignore[index]
    assert verify_keyphrase_centroid(inputs, model.to_bytes())


def test_writer_rejects_replaced_normalized_rows_not_in_source_bytes(tmp_path: Path) -> None:
    inputs = _inputs(_fixture(tmp_path))
    model = train_keyphrase_centroid(inputs, config=CentroidConfig(dimensions=2, epochs=1))
    forged = replace(
        inputs,
        documents=(replace(inputs.documents[0], title="Changed title"), *inputs.documents[1:]),
    )
    with pytest.raises(DataValidationError, match="normalized input objects changed"):
        write_keyphrase_centroid_run(forged, model, tmp_path / "forged-run")
    assert not (tmp_path / "forged-run").exists()


def test_extreme_untrusted_numeric_fields_fail_as_validation_errors(tmp_path: Path) -> None:
    huge = 10**1000
    with pytest.raises(DataValidationError, match="learning_rate"):
        CentroidConfig(learning_rate=huge)
    with pytest.raises(DataValidationError, match="l2"):
        CentroidConfig(l2=huge)
    model = train_keyphrase_centroid(
        _inputs(_fixture(tmp_path)), config=CentroidConfig(dimensions=2, epochs=1)
    )
    for mutation in (
        {"vectors": ((huge, *model.vectors[0][1:]), *model.vectors[1:])},
        {"validation_map": huge},
        {"history": ((1, huge, 1.0),)},
        {"history": ((1, 0.1, huge),)},
    ):
        with pytest.raises(DataValidationError, match="invalid dimensions"):
            replace(model, **mutation)


def test_validation_labels_do_not_enter_weight_updates(tmp_path: Path) -> None:
    baseline = _inputs(_fixture(tmp_path / "baseline"))
    reversed_validation = (
        (EXAMPLE / "validation.jsonl")
        .read_bytes()
        .replace(
            b'"positive_expert_id":"e-math","negative_expert_id":"e-graph"',
            b'"positive_expert_id":"e-graph","negative_expert_id":"e-math"',
        )
    )
    changed = _inputs(_fixture(tmp_path / "changed", validation=reversed_validation))
    config = CentroidConfig(dimensions=2, epochs=1, seed=11)
    a = train_keyphrase_centroid(baseline, config=config)
    b = train_keyphrase_centroid(changed, config=config)
    assert a.vocabulary == b.vocabulary and a.vectors == b.vectors
    assert a.history[0][1] == b.history[0][1]
    assert a.validation_map != b.validation_map
    assert a.source_sha256["validation"] != b.source_sha256["validation"]


@pytest.mark.parametrize(
    "train,validation,reason",
    [
        (
            b'{"document_id":"d-train","positive_expert_id":"e-math","negative_expert_id":"e-math"}\n',
            None,
            "distinct",
        ),
        ((EXAMPLE / "train.jsonl").read_bytes() * 2, None, "repeats"),
        (
            (EXAMPLE / "train.jsonl").read_bytes()
            + b'{"document_id":"d-train","positive_expert_id":"e-graph",'
            b'"negative_expert_id":"e-math"}\n',
            None,
            "contradictory",
        ),
        (None, (EXAMPLE / "train.jsonl").read_bytes(), "overlap"),
        (
            b'{"document_id":"missing","positive_expert_id":"e-math","negative_expert_id":"e-graph"}\n',
            None,
            "missing",
        ),
        (
            b'{"document_id":"d-train","positive_expert_id":"missing","negative_expert_id":"e-graph"}\n',
            None,
            "missing",
        ),
        (b"{}\n", None, "schema"),
        (b'{"document_id":"x","document_id":"y"}\n', None, "invalid train"),
        (b"\xff", None, "UTF-8"),
    ],
)
def test_invalid_triplets_fail_closed(
    tmp_path: Path, train: bytes | None, validation: bytes | None, reason: str
) -> None:
    with pytest.raises(DataValidationError, match=reason):
        _inputs(_fixture(tmp_path, train=train, validation=validation))


def test_hash_conflict_empty_evidence_and_resource_guards(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths[1].write_bytes(paths[1].read_bytes() + b" ")
    with pytest.raises(DataValidationError, match="hashes"):
        _inputs(paths)
    paths[1].write_bytes((EXAMPLE / "documents.json").read_bytes())
    conflict = tmp_path / "conflicts.json"
    conflict.write_text('[{"document_id":"d-train","expert_id":"e-math"}]', encoding="utf-8")
    with pytest.raises(DataValidationError, match="positive"):
        _inputs(paths, conflicts=conflict)
    inputs = _inputs(paths)
    with pytest.raises(DataValidationError, match="max_work"):
        train_keyphrase_centroid(inputs, config=CentroidConfig(max_work=1))
    for kwargs in ({"dimensions": 0}, {"epochs": 51}, {"learning_rate": 0}, {"seed": -1}):
        with pytest.raises(DataValidationError):
            CentroidConfig(**kwargs)
    model = train_keyphrase_centroid(inputs, config=CentroidConfig(dimensions=2, epochs=1))
    tampered = bytearray(model.to_bytes())
    tampered[-10] = ord("f") if tampered[-10] != ord("f") else ord("e")
    with pytest.raises(DataValidationError):
        load_centroid_model(bytes(tampered))
    with pytest.raises(DataValidationError, match="hashes"):
        changed = (
            (EXAMPLE / "validation.jsonl")
            .read_bytes()
            .replace(
                b'"positive_expert_id":"e-math","negative_expert_id":"e-graph"',
                b'"positive_expert_id":"e-graph","negative_expert_id":"e-math"',
            )
        )
        score_keyphrase_centroid(_inputs(_fixture(tmp_path / "other", validation=changed)), model)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("missing_field", "schema"),
        ("unknown_kind", "identity"),
        ("wrong_evidence", "inconsistent"),
        ("bad_count", "token_count"),
        ("bad_list", "list"),
        ("bad_phrase", "term schema"),
        ("bad_term", "term or score"),
        ("bad_score", "term or score"),
        ("empty_profile", "missing"),
        ("missing_profile", "IDs do not match"),
        ("duplicate_profile", "duplicate"),
        ("too_many_terms", "per-entity"),
    ],
)
def test_forged_but_hash_consistent_keyphrase_records_rejected(
    tmp_path: Path, change: str, reason: str
) -> None:
    paths = _fixture(tmp_path)
    _rewrite_phrases(paths, change)
    with pytest.raises(DataValidationError, match=reason):
        _inputs(paths)


def test_manifest_labels_partition_and_source_limits(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = paths[0] / "manifest.json"
    original = manifest.read_bytes()
    value = json.loads(original)
    value["schema_version"] = 2
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DataValidationError, match="unsupported"):
        _inputs(paths)
    manifest.write_bytes(original)
    value = json.loads(original)
    value["records"] = 500
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DataValidationError, match="record count mismatch"):
        _inputs(paths)
    manifest.write_bytes(original)
    paths[3].write_bytes(b"")
    with pytest.raises(DataValidationError, match="empty"):
        _inputs(paths)
    paths[3].write_bytes(b"0\n")
    with pytest.raises(DataValidationError, match="object"):
        _inputs(paths)
    paths[3].write_bytes(b" " * (2 * 1024 * 1024 + 1))
    with pytest.raises(DataValidationError, match="exceeds"):
        _inputs(paths)
    paths[3].write_bytes((EXAMPLE / "train.jsonl").read_bytes())
    with pytest.raises(DataValidationError, match="distinct"):
        load_centroid_inputs(paths[0], paths[1], paths[2], paths[3], paths[3])
    conflicts = tmp_path / "conflicts.json"
    conflicts.write_text('[{"document_id":"missing","expert_id":"e-math"}]', encoding="utf-8")
    with pytest.raises(DataValidationError, match="unknown"):
        _inputs(paths, conflicts=conflicts)
    conflicts.write_text(
        '[{"document_id":"d-hold","expert_id":"e-math"},'
        '{"document_id":"d-hold","expert_id":"e-math"}]',
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="duplicate conflict"):
        _inputs(paths, conflicts=conflicts)


def test_checkpoint_shape_and_publication_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    inputs = _inputs(paths)
    model = train_keyphrase_centroid(inputs, config=CentroidConfig(dimensions=2, epochs=1))
    envelope = json.loads(model.to_bytes())
    envelope["payload"]["vectors"][0] = [True, False]
    envelope["sha256"] = hashlib.sha256(centroid._json(envelope["payload"])).hexdigest()
    with pytest.raises(DataValidationError, match="fields are invalid"):
        load_centroid_model(centroid._json(envelope))
    with pytest.raises(DataValidationError, match="byte limit"):
        load_centroid_model(b" " * (8 * 1024 * 1024 + 1))
    with pytest.raises(DataValidationError, match="source changed"):
        paths[3].write_bytes(paths[3].read_bytes().replace(b"d-train", b"D-train"))
        write_keyphrase_centroid_run(inputs, model, tmp_path / "not-published")
    assert not (tmp_path / "not-published").exists()
    monkeypatch.setattr(centroid, "_MAX_TERMS", 1)
    with pytest.raises(DataValidationError, match="vocabulary"):
        train_keyphrase_centroid(inputs, config=CentroidConfig(dimensions=2, epochs=1))


def test_cli_create_only_and_complete_match_smoke(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path)
    directory = tmp_path / "cli-run"
    args = [
        "expertise-centroid",
        "--keyphrases",
        str(paths[0]),
        "--documents",
        str(paths[1]),
        "--experts",
        str(paths[2]),
        "--train",
        str(paths[3]),
        "--validation",
        str(paths[4]),
        "--dimensions",
        "2",
        "--epochs",
        "2",
        "--directory",
        str(directory),
    ]
    assert main(args) == 0
    assert "generated 2 holdout affinities" in capsys.readouterr().out
    assert main(args) == 2
    assert "already exists" in capsys.readouterr().err
    assert (
        main(
            [
                "match-affinity",
                "--documents",
                str(directory / "documents.json"),
                "--experts",
                str(directory / "experts.json"),
                "--conflicts",
                str(directory / "conflicts.json"),
                "--affinities",
                str(directory / "affinities.csv"),
                "--config",
                str(EXAMPLE.parent / "expertise" / "match-config.json"),
                "--output",
                str(tmp_path / "assignment.json"),
            ]
        )
        == 0
    )
    assert (
        json.loads((tmp_path / "assignment.json").read_text())["assignments"][0]["expert_id"]
        == "e-math"
    )


def test_concurrent_create_only_run_has_one_winner(tmp_path: Path) -> None:
    inputs = _inputs(_fixture(tmp_path))
    model = train_keyphrase_centroid(inputs, config=CentroidConfig(dimensions=2, epochs=1))
    output = tmp_path / "race"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(write_keyphrase_centroid_run, inputs, model, output) for _ in range(2)
        ]
        outcomes = []
        for future in futures:
            try:
                future.result()
                outcomes.append("success")
            except (DataValidationError, OSError):
                outcomes.append("exists")
    assert sorted(outcomes) == ["exists", "success"]
    assert load_centroid_model((output / "model.json").read_bytes()) == model
