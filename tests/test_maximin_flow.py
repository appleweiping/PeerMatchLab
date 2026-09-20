"""Independent small-graph oracle for the bottleneck edge-flow objective."""

from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from peermatchlab.affinity import Affinity
from peermatchlab.assignment import AssignmentEngine
from peermatchlab.cli import main
from peermatchlab.config import MatchConfig
from peermatchlab.models import Conflict, DataValidationError, Document, Expert, MatchScore
from peermatchlab.pipeline import run_affinity_matching


@dataclass
class _Scores:
    documents: dict[str, Document]
    experts: dict[str, Expert]
    scores: tuple[MatchScore, ...]

    def matrix(self) -> tuple[MatchScore, ...]:
        return self.scores


def _score(document: str, expert: str, value: float, *, eligible: bool = True) -> MatchScore:
    return MatchScore(document, expert, value, value, 0, 0, 0, 0, eligible=eligible)


def _objective(selected: tuple[MatchScore, ...]) -> tuple[int, float, int]:
    return (
        len(selected),
        min((item.total for item in selected), default=0.0),
        sum(round(item.total * 1_000_000) for item in selected),
    )


def _oracle(
    scorer: _Scores,
    *,
    default_demand: int,
    minimum_score: float,
    diverse: bool,
    senior_floor: int,
) -> tuple[int, float, int]:
    """Enumerate *all* edge subsets, without invoking solver flow or threshold code."""

    edges = tuple(item for item in scorer.scores if item.eligible and item.total >= minimum_score)
    best = (0, 0.0, 0)
    for mask in itertools.product((False, True), repeat=len(edges)):
        selected = tuple(item for item, enabled in zip(edges, mask, strict=True) if enabled)
        by_document = {key: [] for key in scorer.documents}
        by_expert = dict.fromkeys(scorer.experts, 0)
        for item in selected:
            by_document[item.document_id].append(item.expert_id)
            by_expert[item.expert_id] += 1
        if any(by_expert[key] > item.capacity for key, item in scorer.experts.items()):
            continue
        feasible = True
        for key, item in scorer.documents.items():
            assigned = by_document[key]
            demand = item.required_experts if item.required_experts is not None else default_demand
            if len(assigned) > demand or len(assigned) != len(set(assigned)):
                feasible = False
                break
            if sum(scorer.experts[expert].seniority < 0.75 for expert in assigned) > demand - min(
                senior_floor, demand
            ):
                feasible = False
                break
            affiliations = [
                scorer.experts[expert].institution
                for expert in assigned
                if scorer.experts[expert].institution is not None
            ]
            if diverse and len(affiliations) != len(set(affiliations)):
                feasible = False
                break
        if feasible:
            best = max(best, _objective(selected))
    return best


def _selected(scorer: _Scores, plan_pairs: set[tuple[str, str]]) -> tuple[MatchScore, ...]:
    return tuple(item for item in scorer.scores if (item.document_id, item.expert_id) in plan_pairs)


def test_maximin_flow_improves_bottleneck_without_losing_coverage() -> None:
    documents = [Document("d1", "One"), Document("d2", "Two")]
    experts = [Expert("e1", "One", capacity=1), Expert("e2", "Two", capacity=1)]
    affinities = [
        Affinity("d1", "e1", 0.95),
        Affinity("d1", "e2", 0.75),
        Affinity("d2", "e1", 0.75),
        Affinity("d2", "e2", 0.70),
    ]
    optimal = run_affinity_matching(
        documents, experts, affinities, config=MatchConfig(reviewers_per_document=1)
    )
    fair = run_affinity_matching(
        documents,
        experts,
        affinities,
        config=MatchConfig(strategy="maximin-flow", reviewers_per_document=1),
    )
    assert sorted(item.score for item in optimal.plan.assignments) == [0.70, 0.95]
    assert sorted(item.score for item in fair.plan.assignments) == [0.75, 0.75]
    assert fair.plan.strategy == "maximin-flow"
    assert fair.plan.diagnostics is not None and fair.plan.diagnostics.certified
    assert fair.audit.safe


def test_bottleneck_differs_from_document_sum_maximin_for_multiple_reviewers() -> None:
    scorer = _Scores(
        {key: Document(key, "D", required_experts=2) for key in ("d1", "d2")},
        {key: Expert(key, key, capacity=1) for key in ("a", "b", "c", "d")},
        tuple(
            _score(document, expert, value)
            for document, values in (
                ("d1", (0.2, 0.2, 0.7, 0.0)),
                ("d2", (0.0, 0.1, 0.7, 0.8)),
            )
            for expert, value in zip(("a", "b", "c", "d"), values, strict=True)
        ),
    )
    engine = AssignmentEngine(scorer)
    paper_sum = engine.assign(strategy="maximin")
    edge_floor = engine.assign(strategy="maximin-flow")
    assert {(item.document_id, item.expert_id) for item in paper_sum.assignments} == {
        ("d1", "a"),
        ("d1", "c"),
        ("d2", "b"),
        ("d2", "d"),
    }
    assert {(item.document_id, item.expert_id) for item in edge_floor.assignments} == {
        ("d1", "a"),
        ("d1", "b"),
        ("d2", "c"),
        ("d2", "d"),
    }
    assert min(item.score for item in paper_sum.assignments) == 0.1
    assert min(item.score for item in edge_floor.assignments) == 0.2
    assert min(
        sum(item.score for item in paper_sum.assignments if item.document_id == document)
        for document in scorer.documents
    ) == pytest.approx(0.9)
    assert min(
        sum(item.score for item in edge_floor.assignments if item.document_id == document)
        for document in scorer.documents
    ) == pytest.approx(0.4)


@pytest.mark.parametrize("seed", range(30))
def test_maximin_flow_matches_independent_global_edge_subset_oracle(seed: int) -> None:
    rng = random.Random(seed)
    documents = {f"d{i}": Document(f"d{i}", "D", required_experts=1 + i % 2) for i in range(3)}
    experts = {
        f"e{i}": Expert(
            f"e{i}",
            "E",
            capacity=rng.randrange(3),
            institution=("A" if i < 2 else None),
            seniority=(0.9 if i % 2 else 0.1),
        )
        for i in range(4)
    }
    scores = tuple(
        _score(document, expert, rng.randrange(11) / 10, eligible=rng.choice((True, True, False)))
        for document in documents
        for expert in experts
        if rng.choice((True, False))
    )
    scorer = _Scores(documents, experts, scores)
    diverse = bool(seed % 2)
    senior_floor = 1 if not diverse and seed % 3 == 0 else 0
    minimum_score = 0.2 if seed % 5 == 0 else 0.0
    engine = AssignmentEngine(scorer)
    plan = engine.assign(
        strategy="maximin-flow",
        reviewers_per_document=1,
        minimum_score=minimum_score,
        require_distinct_institutions=diverse,
        minimum_senior_reviewers=senior_floor,
    )
    pairs = {(item.document_id, item.expert_id) for item in plan.assignments}
    assert _objective(_selected(scorer, pairs)) == _oracle(
        scorer,
        default_demand=1,
        minimum_score=minimum_score,
        diverse=diverse,
        senior_floor=senior_floor,
    )
    assert len(plan.assignments) == len(pairs)
    assert plan.diagnostics is not None and plan.diagnostics.certified
    assert plan == engine.assign(
        strategy="maximin-flow",
        reviewers_per_document=1,
        minimum_score=minimum_score,
        require_distinct_institutions=diverse,
        minimum_senior_reviewers=senior_floor,
    )


def test_maximin_flow_preserves_eligible_conflicts_and_unmet() -> None:
    run = run_affinity_matching(
        [Document("d1", "D"), Document("d2", "D")],
        [Expert("e", "E", capacity=1)],
        [Affinity("d1", "e", 0.8), Affinity("d2", "e", 0.9)],
        conflicts=[Conflict("d2", "e")],
        config=MatchConfig(strategy="maximin-flow", reviewers_per_document=1),
    )
    assert [(item.document_id, item.expert_id) for item in run.plan.assignments] == [("d1", "e")]
    assert run.plan.unmet == {"d2": 1}
    assert run.audit.safe


def test_partial_coverage_keeps_cardinality_but_may_change_unmet_document() -> None:
    scorer = _Scores(
        {key: Document(key, "D") for key in ("d1", "d2")},
        {"e": Expert("e", "E", capacity=1)},
        (_score("d1", "e", 0.1), _score("d2", "e", 0.9)),
    )
    plan = AssignmentEngine(scorer).assign(strategy="maximin-flow", reviewers_per_document=1)
    assert [(item.document_id, item.expert_id) for item in plan.assignments] == [("d2", "e")]
    assert plan.unmet == {"d1": 1}
    assert plan.diagnostics is not None and plan.diagnostics.certified


def test_equal_score_ties_are_reproducible_under_score_input_permutation() -> None:
    documents = {key: Document(key, "D") for key in ("d1", "d2")}
    experts = {key: Expert(key, "E", capacity=1) for key in ("a", "b")}
    scores = tuple(_score(document, expert, 0.5) for document in documents for expert in experts)
    forward = AssignmentEngine(_Scores(documents, experts, scores)).assign(
        strategy="maximin-flow", reviewers_per_document=1
    )
    reversed_input = AssignmentEngine(_Scores(documents, experts, scores[::-1])).assign(
        strategy="maximin-flow", reviewers_per_document=1
    )
    assert forward == reversed_input


def test_raw_bottleneck_threshold_distinguishes_sub_micro_scores() -> None:
    scorer = _Scores(
        {key: Document(key, "D") for key in ("d1", "d2")},
        {key: Expert(key, "E", capacity=1) for key in ("a", "b")},
        (
            _score("d1", "a", 0.5000001),
            _score("d1", "b", 0.5000002),
            _score("d2", "a", 0.5000002),
            _score("d2", "b", 0.5000001),
        ),
    )
    plan = AssignmentEngine(scorer).assign(strategy="maximin-flow", reviewers_per_document=1)
    assert {(item.document_id, item.expert_id) for item in plan.assignments} == {
        ("d1", "b"),
        ("d2", "a"),
    }


def test_no_eligible_pair_has_zero_floor_and_certified_shortfall() -> None:
    scorer = _Scores(
        {"d": Document("d", "D")},
        {"e": Expert("e", "E", capacity=1)},
        (_score("d", "e", 0.9, eligible=False),),
    )
    plan = AssignmentEngine(scorer).assign(strategy="maximin-flow")
    assert plan.assignments == ()
    assert plan.unmet == {"d": 2}
    assert plan.diagnostics is not None and plan.diagnostics.certified


def test_maximin_flow_rejects_competing_load_objective_and_duplicates() -> None:
    with pytest.raises(DataValidationError, match="does not support"):
        MatchConfig(strategy="maximin-flow", load_balance_penalty=0.1)
    scorer = _Scores(
        {"d": Document("d", "D")},
        {"e": Expert("e", "E")},
        (_score("d", "e", 0.5), _score("d", "e", 0.8)),
    )
    with pytest.raises(ValueError, match="one eligible score"):
        AssignmentEngine(scorer).assign(strategy="maximin-flow")
    with pytest.raises(ValueError, match="does not support"):
        AssignmentEngine(scorer).assign(strategy="maximin-flow", load_balance_penalty=0.1)


@pytest.mark.parametrize("dimension", ("documents", "experts", "slots", "capacity"))
def test_dimension_limits_precede_scoring(dimension: str) -> None:
    class ExplodingScorer:
        def __init__(self) -> None:
            self.documents = {"d": Document("d", "D")}
            self.experts = {"e": Expert("e", "E", capacity=1)}
            if dimension == "documents":
                self.documents = {f"d{i}": Document(f"d{i}", "D") for i in range(25)}
            elif dimension == "experts":
                self.experts = {f"e{i}": Expert(f"e{i}", "E") for i in range(49)}
            elif dimension == "slots":
                self.documents = {"d": Document("d", "D", required_experts=65)}
            else:
                self.experts = {"e": Expert("e", "E", capacity=257)}

        def matrix(self) -> tuple[MatchScore, ...]:
            raise AssertionError("must not score oversized graph")

    with pytest.raises(ValueError, match="maximin-flow supports at most"):
        AssignmentEngine(ExplodingScorer()).assign(strategy="maximin-flow")


def test_pair_limit_and_larger_than_exhaustive_solver() -> None:
    scorer = _Scores(
        {f"d{i}": Document(f"d{i}", "D") for i in range(7)},
        {f"e{i}": Expert(f"e{i}", "E", capacity=1) for i in range(9)},
        tuple(
            _score(f"d{i}", f"e{j}", ((i + j) % 10 + 1) / 10) for i in range(7) for j in range(9)
        ),
    )
    plan = AssignmentEngine(scorer).assign(strategy="maximin-flow", reviewers_per_document=1)
    assert len(plan.assignments) == 7
    with pytest.raises(ValueError, match="maximin supports at most"):
        AssignmentEngine(scorer).assign(strategy="maximin", reviewers_per_document=1)
    scorer.scores = scorer.scores * 5
    with pytest.raises(ValueError, match="256 eligible pairs"):
        AssignmentEngine(scorer).assign(strategy="maximin-flow", reviewers_per_document=1)


def test_distinct_thresholds_require_at_most_ten_flow_solves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = _Scores(
        {f"d{i}": Document(f"d{i}", "D") for i in range(16)},
        {f"e{i}": Expert(f"e{i}", "E", capacity=1) for i in range(16)},
        tuple(
            _score(f"d{i}", f"e{j}", (i * 16 + j + 1) / 256) for i in range(16) for j in range(16)
        ),
    )
    engine = AssignmentEngine(scorer)
    original = engine._optimal
    calls = 0

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        assert calls <= 10
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_optimal", counted)
    plan = engine.assign(strategy="maximin-flow", reviewers_per_document=1)
    assert len(plan.assignments) == 16
    assert 1 < calls <= 10


def test_cli_affinity_smoke(tmp_path: Path) -> None:
    documents = tmp_path / "documents.json"
    experts = tmp_path / "experts.json"
    affinities = tmp_path / "affinities.csv"
    config = tmp_path / "config.json"
    output = tmp_path / "plan.json"
    documents.write_text(json.dumps([{"id": "d1", "title": "One"}, {"id": "d2", "title": "Two"}]))
    experts.write_text(
        json.dumps(
            [{"id": "e1", "name": "One", "capacity": 1}, {"id": "e2", "name": "Two", "capacity": 1}]
        )
    )
    affinities.write_text(
        "document_id,expert_id,score\nd1,e1,0.95\nd1,e2,0.75\nd2,e1,0.75\nd2,e2,0.70\n"
    )
    config.write_text(json.dumps({"strategy": "maximin-flow", "reviewers_per_document": 1}))
    assert (
        main(
            [
                "match-affinity",
                "--documents",
                str(documents),
                "--experts",
                str(experts),
                "--affinities",
                str(affinities),
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported["strategy"] == "maximin-flow"
    assert sorted(item["score"] for item in exported["assignments"]) == [0.75, 0.75]


def test_synthetic_benchmark_is_audited_and_shows_objective_tradeoff() -> None:
    script = Path(__file__).parents[1] / "scripts" / "benchmark_maximin_flow.py"
    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True
    )
    report = json.loads(completed.stdout)
    assert report["fixture"] == "deterministic-synthetic-v1"
    assert (report["documents"], report["experts"], report["eligible_pairs"]) == (12, 12, 144)
    assert report["optimal"]["assigned"] == report["maximin-flow"]["assigned"] == 12
    assert report["optimal"]["minimum_selected_edge_score"] == 0.7
    assert report["maximin-flow"]["minimum_selected_edge_score"] == 0.75
    assert report["optimal"]["total_score"] > report["maximin-flow"]["total_score"]
    assert report["optimal"]["audit_safe"] and report["maximin-flow"]["audit_safe"]
