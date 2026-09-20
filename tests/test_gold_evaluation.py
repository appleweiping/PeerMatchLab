"""Independent hand oracles and adversarial boundaries for gold evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

import peermatchlab.gold_evaluation as gold_module
from peermatchlab.cli import main
from peermatchlab.gold_evaluation import (
    GoldEvaluationConfig,
    GoldJudgment,
    GoldScore,
    evaluate_gold,
    evaluate_gold_files,
    load_affinity_bytes,
    load_gold_bytes,
)
from peermatchlab.models import DataValidationError


def _csv(path: Path, rows: list[list[str]], *, delimiter: str = ",") -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
        writer.writerows(rows)


def _gold_rows() -> list[list[str]]:
    return [
        ["document_id", "expert_id", "relevance"],
        ["d1", "e1", "1"],
        ["d1", "e2", "0"],
        ["d1", "e3", "1"],
        ["d2", "a", "0"],
        ["d2", "b", "1"],
        ["d2", "c", "0"],
        ["d3", "z", "0"],
    ]


def _score_rows() -> list[list[str]]:
    return [
        ["document_id", "expert_id", "score"],
        ["d1", "e2", "0.8"],
        ["d2", "b", "0.8"],
        ["d1", "e1", "0.8"],
        ["d2", "a", "0.9"],
        ["d2", "c", "0.7"],
        ["d3", "z", "0.5"],
        ["unjudged-document", "x", "0.99"],
        ["d1", "unjudged-expert", "0.99"],
    ]


def test_two_document_hand_oracle_tie_missing_and_skip() -> None:
    judgments = [GoldJudgment(row[0], row[1], int(row[2])) for row in _gold_rows()[1:]]
    scores = [GoldScore(row[0], row[1], float(row[2])) for row in _score_rows()[1:]]
    report = evaluate_gold(judgments, scores, config=GoldEvaluationConfig(k_values=(1, 2, 3)))

    assert report["counts"] == {
        "judged_documents": 3,
        "evaluated_documents": 2,
        "skipped_documents": 1,
        "judged_pairs": 7,
        "scored_pairs": 8,
        "ignored_unjudged_scores": 2,
        "missing_judged_scores": 1,
    }
    assert report["skipped"] == [{"document_id": "d3", "reason": "no_judged_positive"}]
    first, second = report["documents"]
    assert (first["document_id"], second["document_id"]) == ("d1", "d2")
    assert first["metrics"]["1"] == {
        "precision": 1.0,
        "recall": 0.5,
        "hits": 1.0,
        "average_precision": 0.5,
    }
    assert first["metrics"]["2"] == {
        "precision": 0.5,
        "recall": 0.5,
        "hits": 1.0,
        "average_precision": 0.5,
    }
    assert first["metrics"]["3"]["average_precision"] == pytest.approx(5 / 6)
    assert second["metrics"]["1"] == {
        "precision": 0.0,
        "recall": 0.0,
        "hits": 0.0,
        "average_precision": 0.0,
    }
    assert second["metrics"]["2"] == {
        "precision": 0.5,
        "recall": 1.0,
        "hits": 1.0,
        "average_precision": 0.5,
    }
    assert report["macro"]["1"] == {
        "precision": 0.5,
        "recall": 0.25,
        "hits": 0.5,
        "average_precision": 0.25,
    }
    assert report["macro"]["2"] == {
        "precision": 0.5,
        "recall": 0.75,
        "hits": 1.0,
        "average_precision": 0.5,
    }


def test_public_shape_transposes_independently_authored_triples_and_hashes(tmp_path: Path) -> None:
    triples = tmp_path / "triples.csv"
    _csv(
        triples,
        [
            ["document_id", "expert_id", "relevance"],
            ["p1", "r1", "3"],
            ["p2", "r1", "0"],
            ["p1", "r2", "1"],
        ],
    )
    public = tmp_path / "evaluations.csv"
    header = ["ParticipantID"] + [
        field for i in range(1, 11) for field in (f"Paper{i}", f"Expertise{i}")
    ]
    _csv(
        public,
        [header, ["r1", "p1", "3", "p2", "0", *([""] * 16)], ["r2", "p1", "1", *([""] * 18)]],
        delimiter="\t",
    )
    affinity = tmp_path / "affinity.csv"
    _csv(affinity, [["p1", "r1", "0.7"], ["p1", "r2", "0.8"], ["p2", "r1", "0.4"]])
    triples_report = evaluate_gold_files(
        triples, affinity, tmp_path / "triples.json", config=GoldEvaluationConfig(k_values=(1, 2))
    )
    public_report = evaluate_gold_files(
        public,
        affinity,
        tmp_path / "public.json",
        format="openreview",
        config=GoldEvaluationConfig(k_values=(1, 2)),
    )

    assert triples_report["result_fingerprint_sha256"] == public_report["result_fingerprint_sha256"]
    assert triples_report["documents"] == public_report["documents"]
    assert (
        triples_report["source"]["gold"]["sha256"]
        == hashlib.sha256(triples.read_bytes()).hexdigest()
    )
    assert (
        public_report["source"]["gold"]["sha256"] == hashlib.sha256(public.read_bytes()).hexdigest()
    )
    assert triples_report["source"]["gold"]["sha256"] != public_report["source"]["gold"]["sha256"]
    assert [
        item.document_id for item in load_gold_bytes(public.read_bytes(), format="openreview")
    ] == ["p1", "p1", "p2"]


def test_cli_is_repeatable_and_permutation_changes_only_raw_provenance(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    affinity = tmp_path / "affinities.csv"
    _csv(gold, _gold_rows())
    _csv(affinity, _score_rows())
    first, same, permuted = (
        tmp_path / name for name in ("first.json", "same.json", "permuted.json")
    )
    args = [
        "evaluate-gold",
        "--gold",
        str(gold),
        "--affinities",
        str(affinity),
        "--k",
        "2",
        "--k",
        "1",
    ]
    assert main([*args, "--output", str(first)]) == 0
    assert main([*args, "--output", str(same)]) == 0
    assert first.read_bytes() == same.read_bytes()
    _csv(gold, [*_gold_rows()[:1], *_gold_rows()[1:][::-1]])
    _csv(affinity, [*_score_rows()[:1], *_score_rows()[1:][::-1]])
    assert main([*args, "--output", str(permuted)]) == 0
    before, after = json.loads(first.read_bytes()), json.loads(permuted.read_bytes())
    assert before["result_fingerprint_sha256"] == after["result_fingerprint_sha256"]
    assert before["macro"] == after["macro"]
    assert before["source"] != after["source"]
    assert before["k_values"] == [1, 2]


@pytest.mark.parametrize(
    ("payload", "format", "error"),
    [
        (b"document_id,expert_id,relevance\nd,e,1001\n", "triples", "maximum relevance"),
        (b"document_id,expert_id,relevance\nd,e,1.5\n", "triples", "invalid relevance"),
        (b"document_id,expert_id,relevance\nd,e,1\nd,e,1\n", "triples", "duplicate"),
        (b"document_id,expert_id,relevance\n,e,1\n", "triples", "document_id"),
        (b"document_id,expert_id,relevance\nd, e,1\n", "triples", "expert_id"),
        (b"document_id,expert_id,relevance\nd,e,1\n\xff", "triples", "UTF-8"),
        (b'document_id,expert_id,relevance\nd,"e,1\n', "triples", "malformed"),
        (b"document_id,expert_id,relevance\n", "triples", "at least one"),
    ],
)
def test_malformed_gold_rejected(payload: bytes, format: str, error: str) -> None:
    with pytest.raises(DataValidationError, match=error):
        load_gold_bytes(payload, format=format)


@pytest.mark.parametrize("score", ["nan", "inf", "-0.1", "1.1", "bad", "1e1000"])
def test_malformed_affinity_score_rejected(score: str) -> None:
    with pytest.raises(DataValidationError):
        load_affinity_bytes(f"document_id,expert_id,score\nd,e,{score}\n".encode())


def test_affinity_duplicates_missing_ids_and_malformed_rows() -> None:
    for payload in (
        b"d,e,0.3\nd,e,0.4\n",
        b"d,,0.3\n",
        b"d,e,0.3,extra\n",
        b"d,e,0.3\n\x00",
        b"d,e,0.3\n\xff",
    ):
        with pytest.raises(DataValidationError):
            load_affinity_bytes(payload)


def test_oversized_file_row_count_documents_and_k_are_bounded(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    affinity = tmp_path / "affinity.csv"
    gold.write_bytes(b"document_id,expert_id,relevance\nd,e,1\n")
    affinity.write_bytes(b"d,e,0.5\n")
    with pytest.raises(DataValidationError, match="max_input_file_bytes"):
        evaluate_gold_files(
            gold,
            affinity,
            tmp_path / "out.json",
            config=GoldEvaluationConfig(max_input_file_bytes=10),
        )
    with pytest.raises(DataValidationError, match="max_row_bytes"):
        evaluate_gold_files(
            gold, affinity, tmp_path / "out.json", config=GoldEvaluationConfig(max_row_bytes=8)
        )
    gold.write_bytes(b"document_id,expert_id,relevance\nd,e,1\nd,f,0\n")
    with pytest.raises(DataValidationError, match="max_rows"):
        evaluate_gold_files(
            gold, affinity, tmp_path / "out.json", config=GoldEvaluationConfig(max_rows=1)
        )
    with pytest.raises(DataValidationError, match="max_documents"):
        load_gold_bytes(
            b"document_id,expert_id,relevance\na,e,1\nb,e,1\n",
            format="triples",
            config=GoldEvaluationConfig(max_documents=1),
        )
    for value in (0, 1001, True):
        with pytest.raises(DataValidationError, match="each k"):
            GoldEvaluationConfig(k_values=(value,))  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="unique"):
        GoldEvaluationConfig(k_values=(1, 1))
    assert not (tmp_path / "out.json").exists()


def test_zero_positive_and_missing_coverage_are_explicit() -> None:
    judgments = [GoldJudgment("d", "a", 0), GoldJudgment("d", "b", 0)]
    scores = [GoldScore("d", "a", 0.5)]
    report = evaluate_gold(judgments, scores, config=GoldEvaluationConfig(k_values=(1,)))
    assert report["counts"]["skipped_documents"] == 1
    assert report["counts"]["missing_judged_scores"] == 1
    assert report["macro"]["1"]["precision"] == 0.0
    with pytest.raises(DataValidationError, match="strict coverage"):
        evaluate_gold(judgments, scores, config=GoldEvaluationConfig(strict_coverage=True))
    with pytest.raises(DataValidationError, match="at least one scored judged pair"):
        evaluate_gold(judgments, [GoldScore("d", "unjudged", 0.5)])


def test_output_alias_and_atomic_failure_preserve_existing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold.csv"
    affinity = tmp_path / "affinity.csv"
    _csv(gold, [["document_id", "expert_id", "relevance"], ["d", "e", "1"]])
    _csv(affinity, [["document_id", "expert_id", "score"], ["d", "e", "0.8"]])
    alias = tmp_path / "alias.json"
    os.link(gold, alias)
    with pytest.raises(DataValidationError, match="alias"):
        evaluate_gold_files(gold, affinity, alias)
    assert gold.read_bytes() == alias.read_bytes()
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"previous report")

    def fail_replace(_source: str, _destination: Path) -> None:
        raise OSError("simulated atomic install failure")

    monkeypatch.setattr("peermatchlab.gold_evaluation.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        evaluate_gold_files(gold, affinity, existing)
    assert existing.read_bytes() == b"previous report"
    assert not list(tmp_path.glob(".existing.json-*.tmp"))


def test_short_write_never_installs_partial_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold.csv"
    affinity = tmp_path / "affinity.csv"
    _csv(gold, [["document_id", "expert_id", "relevance"], ["d", "e", "1"]])
    _csv(affinity, [["document_id", "expert_id", "score"], ["d", "e", "0.8"]])
    output = tmp_path / "existing.json"
    output.write_bytes(b"previous report")
    original_fdopen = os.fdopen

    class ShortWriteFile:
        def __init__(self, handle: int, mode: str) -> None:
            self.stream = original_fdopen(handle, mode)

        def __enter__(self) -> ShortWriteFile:
            return self

        def __exit__(self, *_exc: object) -> None:
            self.stream.close()

        def write(self, data: bytes) -> int:
            return self.stream.write(data[:-1])

    monkeypatch.setattr("peermatchlab.gold_evaluation.os.fdopen", ShortWriteFile)
    with pytest.raises(OSError, match="short write"):
        evaluate_gold_files(gold, affinity, output)
    assert output.read_bytes() == b"previous report"
    assert not list(tmp_path.glob(".existing.json-*.tmp"))


def test_input_mutation_before_atomic_replace_leaves_existing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold.csv"
    affinity = tmp_path / "affinity.csv"
    _csv(gold, [["document_id", "expert_id", "relevance"], ["d", "e", "1"]])
    _csv(affinity, [["document_id", "expert_id", "score"], ["d", "e", "0.8"]])
    output = tmp_path / "existing.json"
    output.write_bytes(b"previous report")
    original_read = gold_module._read_bounded
    gold_reads = 0

    def mutate_before_recheck(path: str | Path, *, config: GoldEvaluationConfig) -> bytes:
        nonlocal gold_reads
        if Path(path) == gold:
            gold_reads += 1
            if gold_reads == 2:
                gold.write_bytes(b"document_id,expert_id,relevance\nd,e,0\n")
        return original_read(path, config=config)

    monkeypatch.setattr(gold_module, "_read_bounded", mutate_before_recheck)
    with pytest.raises(DataValidationError, match="changed during report generation"):
        evaluate_gold_files(gold, affinity, output)
    assert output.read_bytes() == b"previous report"
    assert not list(tmp_path.glob(".existing.json-*.tmp"))


def test_output_symlink_to_input_is_rejected_without_mutation(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    affinity = tmp_path / "affinity.csv"
    _csv(gold, [["document_id", "expert_id", "relevance"], ["d", "e", "1"]])
    _csv(affinity, [["document_id", "expert_id", "score"], ["d", "e", "0.8"]])
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(affinity)
    except OSError:
        pytest.skip("file symlinks are unavailable on this runner")
    original = affinity.read_bytes()
    with pytest.raises(DataValidationError, match="alias"):
        evaluate_gold_files(gold, affinity, alias)
    assert affinity.read_bytes() == original
    assert alias.is_symlink()


def test_public_shape_rejects_duplicate_reviewer_and_incomplete_slots() -> None:
    header = "\t".join(
        ["ParticipantID", *(cell for i in range(1, 11) for cell in (f"Paper{i}", f"Expertise{i}"))]
    )
    row = "\t".join(["r", "p", "1", *([""] * 18)])
    with pytest.raises(DataValidationError, match="duplicate ParticipantID"):
        load_gold_bytes(f"{header}\n{row}\n{row}\n".encode(), format="openreview")
    incomplete = "\t".join(["r", "p", "", *([""] * 18)])
    with pytest.raises(DataValidationError, match="invalid relevance"):
        load_gold_bytes(f"{header}\n{incomplete}\n".encode(), format="openreview")
    empty = "\t".join(["r", *([""] * 20)])
    with pytest.raises(DataValidationError, match="no judged paper"):
        load_gold_bytes(f"{header}\n{empty}\n".encode(), format="openreview")


def test_headerless_affinity_and_public_transposition_have_pair_limits() -> None:
    with pytest.raises(DataValidationError, match="max_rows"):
        load_affinity_bytes(b"d,e,0.1\nd,f,0.2\n", config=GoldEvaluationConfig(max_rows=1))
    header = "\t".join(
        ["ParticipantID", *(cell for i in range(1, 11) for cell in (f"Paper{i}", f"Expertise{i}"))]
    )
    row = "\t".join(["r", "p1", "1", "p2", "0", *([""] * 16)])
    with pytest.raises(DataValidationError, match="judged pairs exceed max_rows"):
        load_gold_bytes(
            f"{header}\n{row}\n".encode(),
            format="openreview",
            config=GoldEvaluationConfig(max_rows=1),
        )


def test_csv_row_limit_stops_parsing_before_unbounded_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_reader = csv.reader
    consumed = 0

    def counted_reader(*args: object, **kwargs: object) -> object:
        nonlocal consumed
        for row in original_reader(*args, **kwargs):  # type: ignore[arg-type]
            consumed += 1
            if consumed > 3:
                raise AssertionError("parser read beyond bounded lookahead")
            yield row

    monkeypatch.setattr(gold_module.csv, "reader", counted_reader)
    with pytest.raises(DataValidationError, match="max_rows"):
        load_affinity_bytes(b"d,e,0.1\n" * 1_000_000, config=GoldEvaluationConfig(max_rows=1))
    assert consumed == 3


def test_wide_malformed_csv_row_rejected_before_retaining_later_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_reader = csv.reader
    consumed = 0

    def counted_reader(*args: object, **kwargs: object) -> object:
        nonlocal consumed
        for row in original_reader(*args, **kwargs):  # type: ignore[arg-type]
            consumed += 1
            if consumed > 2:
                raise AssertionError("parser retained malformed wide rows")
            yield row

    monkeypatch.setattr(gold_module.csv, "reader", counted_reader)
    wide_row = b",".join([b"x"] * 100) + b"\n"
    with pytest.raises(DataValidationError, match="fields; expected 3"):
        load_gold_bytes(
            b"document_id,expert_id,relevance\n" + wide_row * 1_000,
            format="triples",
        )
    assert consumed == 2


def test_cr_only_csv_rows_are_supported_and_bounded() -> None:
    config = GoldEvaluationConfig(max_row_bytes=64)
    gold = load_gold_bytes(
        b"document_id,expert_id,relevance\rd,e,1\r", format="triples", config=config
    )
    scores = load_affinity_bytes(b"d,e,0.8\r", config=config)
    assert gold == (GoldJudgment("d", "e", 1),)
    assert scores == (GoldScore("d", "e", 0.8),)
    with pytest.raises(DataValidationError, match="max_row_bytes"):
        load_affinity_bytes(b"d," + b"x" * 65 + b",0.8\r", config=config)


def test_missing_score_ties_follow_expert_id_not_gold_input_order() -> None:
    scores = [GoldScore("d", "c", 0.9)]
    judgments = [GoldJudgment("d", "b", 1), GoldJudgment("d", "c", 0), GoldJudgment("d", "a", 0)]
    config = GoldEvaluationConfig(k_values=(1, 2, 3))
    first = evaluate_gold(judgments, scores, config=config)
    permuted = evaluate_gold(list(reversed(judgments)), scores, config=config)
    assert first == permuted
    assert first["documents"][0]["metrics"]["2"]["hits"] == 0.0
    assert first["documents"][0]["metrics"]["3"]["hits"] == 1.0
    assert first["counts"]["missing_judged_scores"] == 2


def test_report_excludes_unjudged_private_identifiers_and_input_paths(tmp_path: Path) -> None:
    gold = tmp_path / "private-gold.csv"
    affinity = tmp_path / "private-affinity.csv"
    _csv(gold, [["document_id", "expert_id", "relevance"], ["d", "e", "1"]])
    _csv(
        affinity,
        [
            ["document_id", "expert_id", "score"],
            ["d", "e", "0.7"],
            ["SECRET_DOCUMENT", "SECRET_EXPERT", "0.9"],
        ],
    )
    output = tmp_path / "report.json"
    evaluate_gold_files(gold, affinity, output)
    text = output.read_text(encoding="utf-8")
    assert "SECRET_DOCUMENT" not in text
    assert "SECRET_EXPERT" not in text
    assert str(gold) not in text
    assert str(affinity) not in text
    assert '"ignored_unjudged_scores":1' in text


def test_invalid_api_collection_and_config_boundaries() -> None:
    with pytest.raises(DataValidationError, match="GoldJudgment"):
        evaluate_gold([object()], [])  # type: ignore[list-item]
    with pytest.raises(DataValidationError, match="GoldScore"):
        evaluate_gold([GoldJudgment("d", "e", 1)], [object()])  # type: ignore[list-item]
    with pytest.raises(DataValidationError, match="duplicate judged"):
        evaluate_gold(
            [GoldJudgment("d", "e", 1), GoldJudgment("d", "e", 1)], [GoldScore("d", "e", 1.0)]
        )
    with pytest.raises(DataValidationError, match="duplicate scored"):
        evaluate_gold(
            [GoldJudgment("d", "e", 1)], [GoldScore("d", "e", 1.0), GoldScore("d", "e", 0.5)]
        )
    with pytest.raises(DataValidationError, match="relevance_threshold"):
        GoldEvaluationConfig(relevance_threshold=1001)
    with pytest.raises(DataValidationError, match="strict_coverage"):
        GoldEvaluationConfig(strict_coverage=1)  # type: ignore[arg-type]
