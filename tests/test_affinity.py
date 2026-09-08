from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from peermatchlab.affinity import Affinity, AffinityScorer, load_affinities_csv
from peermatchlab.cli import main
from peermatchlab.config import MatchConfig
from peermatchlab.io import load_documents, load_experts
from peermatchlab.models import Conflict, DataValidationError, Document, Expert
from peermatchlab.pipeline import run_affinity_matching


def _csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def test_load_affinities_csv_preserves_sparse_rows(tmp_path: Path) -> None:
    path = tmp_path / "affinities.csv"
    _csv(path, [["document_id", "expert_id", "score"], ["d", "e", "0.75"]])

    assert load_affinities_csv(path) == (Affinity("d", "e", 0.75),)


def test_load_affinities_csv_accepts_openreview_headerless_rows(tmp_path: Path) -> None:
    path = tmp_path / "openreview-affinities.csv"
    _csv(path, [["paper-id", "~Reviewer1", "0.82"], ["paper-id", "~Reviewer2", "0.71"]])

    assert load_affinities_csv(path) == (
        Affinity("paper-id", "~Reviewer1", 0.82),
        Affinity("paper-id", "~Reviewer2", 0.71),
    )


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [["expert_id", "document_id", "score"]],
        [["document_id", "expert_id", "score"], ["d", "e"]],
        [["document_id", "expert_id", "score"], ["d", "e", "bad"]],
        [
            ["document_id", "expert_id", "score"],
            ["d", "e", "0.2"],
            ["d", "e", "0.3"],
        ],
    ],
)
def test_load_affinities_csv_rejects_malformed_input(
    tmp_path: Path, rows: list[list[object]]
) -> None:
    path = tmp_path / "affinities.csv"
    _csv(path, rows)
    with pytest.raises(DataValidationError):
        load_affinities_csv(path)


@pytest.mark.parametrize(
    "args",
    [
        ("", "e", 0.5),
        ("d", "", 0.5),
        ("d", "e", True),
        ("d", "e", float("nan")),
        ("d", "e", -0.1),
        ("d", "e", 1.1),
        ("d", "e", 10**1_000),
    ],
)
def test_affinity_rejects_invalid_values(args: tuple[object, object, object]) -> None:
    with pytest.raises(DataValidationError):
        Affinity(*args)  # type: ignore[arg-type]


def test_affinity_scorer_keeps_missing_pairs_absent_and_conflicts_ineligible() -> None:
    documents = [Document("d1", "One"), Document("d2", "Two")]
    experts = [Expert("e1", "One"), Expert("e2", "Two")]
    scorer = AffinityScorer(
        documents,
        experts,
        [Affinity("d1", "e1", 0.9), Affinity("d2", "e2", 0.8)],
        conflicts=[Conflict("d1", "e1")],
    )

    assert len(scorer.matrix()) == 2
    assert not scorer.matrix()[0].eligible
    assert scorer.matrix()[1].content == 0.0
    assert scorer.matrix()[1].affinity == 0.8
    assert scorer.matrix()[1].component_map()["affinity"] == 0.8


def test_affinity_scorer_rejects_unknown_and_duplicate_pairs() -> None:
    documents = [Document("d", "Document")]
    experts = [Expert("e", "Expert")]
    with pytest.raises(DataValidationError, match="unknown pair"):
        AffinityScorer(documents, experts, [Affinity("missing", "e", 0.5)])
    with pytest.raises(DataValidationError, match="duplicate affinity"):
        AffinityScorer(
            documents,
            experts,
            [Affinity("d", "e", 0.5), Affinity("d", "e", 0.6)],
        )
    with pytest.raises(DataValidationError, match="unknown document/expert"):
        AffinityScorer(
            documents,
            experts,
            [Affinity("d", "e", 0.5)],
            conflicts=[Conflict("missing", "e")],
        )
    with pytest.raises(DataValidationError, match="Affinity objects"):
        AffinityScorer(documents, experts, [object()])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("documents", "experts", "conflicts", "message"),
    [
        ([], [Expert("e", "E")], [], "at least one document"),
        ([Document("d", "D")], [], [], "at least one expert"),
        ([object()], [Expert("e", "E")], [], "Document objects"),
        ([Document("d", "D")], [object()], [], "Expert objects"),
        ([Document("d", "D")], [Expert("e", "E")], [object()], "Conflict objects"),
    ],
)
def test_affinity_scorer_validates_collection_boundaries(
    documents: list[object], experts: list[object], conflicts: list[object], message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        AffinityScorer(  # type: ignore[arg-type]
            documents,
            experts,
            [],
            conflicts=conflicts,
        )


def test_external_affinity_pipeline_uses_constraints() -> None:
    documents = [Document("d1", "One"), Document("d2", "Two")]
    experts = [Expert("e1", "One", capacity=1), Expert("e2", "Two", capacity=1)]
    run = run_affinity_matching(
        documents,
        experts,
        [
            Affinity("d1", "e1", 0.9),
            Affinity("d1", "e2", 0.8),
            Affinity("d2", "e1", 0.85),
            Affinity("d2", "e2", 0.1),
        ],
        config=MatchConfig(reviewers_per_document=1),
    )

    assert {(item.document_id, item.expert_id) for item in run.plan.assignments} == {
        ("d1", "e2"),
        ("d2", "e1"),
    }
    assert run.audit.safe


def test_match_affinity_cli_writes_plan_and_report(tmp_path: Path) -> None:
    documents = load_documents("examples/documents.json")
    experts = load_experts("examples/experts.json")
    affinity_path = tmp_path / "affinities.csv"
    rows: list[list[object]] = [["document_id", "expert_id", "score"]]
    for document_index, document in enumerate(documents):
        for expert_index, expert in enumerate(experts):
            rows.append(
                [document.id, expert.id, str(0.9 - document_index * 0.1 - expert_index * 0.01)]
            )
    _csv(affinity_path, rows)
    output = tmp_path / "plan.json"
    html = tmp_path / "plan.html"

    code = main(
        [
            "match-affinity",
            "--documents",
            "examples/documents.json",
            "--experts",
            "examples/experts.json",
            "--affinities",
            str(affinity_path),
            "--output",
            str(output),
            "--html",
            str(html),
        ]
    )

    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["assignments"]
    assert "externally supplied affinity" in html.read_text(encoding="utf-8")


def test_match_affinity_cli_refuses_output_hard_linked_to_affinity_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    affinities = tmp_path / "affinities.csv"
    _csv(affinities, [["document_id", "expert_id", "score"], ["p", "r", "0.9"]])
    output = tmp_path / "plan.json"
    os.link(affinities, output)
    original = affinities.read_bytes()

    code = main(
        [
            "match-affinity",
            "--documents",
            "examples/documents.json",
            "--experts",
            "examples/experts.json",
            "--affinities",
            str(affinities),
            "--output",
            str(output),
        ]
    )

    assert code == 2
    assert affinities.read_bytes() == original
    assert output.read_bytes() == original
    assert "output path must not refer to the affinities input" in capsys.readouterr().err
