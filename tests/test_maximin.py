"""Independent small-instance oracle for the document-side fairness objective."""

from __future__ import annotations

import itertools
import json
import math
import random
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


def _score(document_id: str, expert_id: str, value: float, *, eligible: bool = True) -> MatchScore:
    return MatchScore(document_id, expert_id, value, value, 0, 0, 0, 0, eligible=eligible)


def _objective(
    pairs: tuple[MatchScore, ...], documents: dict[str, Document]
) -> tuple[int, float, float]:
    totals = [
        math.fsum(score.total for score in pairs if score.document_id == document_id)
        for document_id in sorted(documents)
    ]
    return len(pairs), min(totals), math.fsum(totals)


def _oracle(
    scorer: _Scores,
    demand: int,
    minimum: float = 0.0,
    *,
    diverse: bool = False,
    senior_floor: int = 0,
    senior_threshold: float = 0.75,
) -> tuple[int, float, float]:
    """Enumerate the *global* edge power set, independently of solver's per-paper subsets."""

    edges = tuple(score for score in scorer.scores if score.eligible and score.total >= minimum)
    best = (0, 0.0, 0.0)
    for mask in itertools.product((False, True), repeat=len(edges)):
        selected = tuple(score for score, enabled in zip(edges, mask, strict=True) if enabled)
        document_counts = {document_id: 0 for document_id in scorer.documents}
        expert_counts = {expert_id: 0 for expert_id in scorer.experts}
        groups: dict[str, set[str]] = {document_id: set() for document_id in scorer.documents}
        valid = True
        for score in selected:
            document_counts[score.document_id] += 1
            expert_counts[score.expert_id] += 1
            affiliation = scorer.experts[score.expert_id].institution
            if diverse and affiliation is not None:
                if affiliation in groups[score.document_id]:
                    valid = False
                groups[score.document_id].add(affiliation)
        for document_id, document in scorer.documents.items():
            requested = document.required_experts or demand
            free = requested - min(senior_floor, requested)
            juniors = sum(
                scorer.experts[score.expert_id].seniority < senior_threshold
                for score in selected
                if score.document_id == document_id
            )
            if document_counts[document_id] > requested or juniors > free:
                valid = False
        if any(expert_counts[key] > expert.capacity for key, expert in scorer.experts.items()):
            valid = False
        if valid:
            best = max(best, _objective(selected, scorer.documents))
    return best


def _selected_scores(scorer: _Scores, pairs: set[tuple[str, str]]) -> tuple[MatchScore, ...]:
    return tuple(score for score in scorer.scores if (score.document_id, score.expert_id) in pairs)


def test_maximin_improves_weakest_document_at_total_score_cost() -> None:
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
        config=MatchConfig(strategy="maximin", reviewers_per_document=1),
    )

    assert optimal.plan.total_score == pytest.approx(1.65)
    assert fair.plan.total_score == pytest.approx(1.50)
    assert {item.document_id: item.score for item in fair.plan.assignments} == {
        "d1": 0.75,
        "d2": 0.75,
    }
    assert fair.plan.strategy == "maximin"
    assert fair.plan.diagnostics is not None and fair.plan.diagnostics.certified
    assert fair.audit.safe


def test_maximin_prioritizes_cardinality_and_preserves_conflicts() -> None:
    documents = [Document("d1", "One"), Document("d2", "Two")]
    experts = [Expert("e1", "One", capacity=1), Expert("e2", "Two", capacity=1)]
    run = run_affinity_matching(
        documents,
        experts,
        [Affinity("d1", "e1", 1.0), Affinity("d1", "e2", 0.9), Affinity("d2", "e2", 0.01)],
        conflicts=[Conflict("d1", "e2")],
        config=MatchConfig(strategy="maximin", reviewers_per_document=1),
    )
    assert {(item.document_id, item.expert_id) for item in run.plan.assignments} == {
        ("d1", "e1"),
        ("d2", "e2"),
    }
    assert run.audit.safe


def test_maximin_respects_sparse_scores_threshold_and_unmet_diagnostics() -> None:
    scorer = _Scores(
        documents={"d1": Document("d1", "One"), "d2": Document("d2", "Two")},
        experts={"e": Expert("e", "Only", capacity=1)},
        scores=(_score("d1", "e", 0.8), _score("d2", "e", 0.9, eligible=False)),
    )
    engine = AssignmentEngine(scorer)
    plan = engine.assign(strategy="maximin", reviewers_per_document=1, minimum_score=0.5)
    assert [(item.document_id, item.expert_id) for item in plan.assignments] == [("d1", "e")]
    assert plan.unmet == {"d2": 1}
    assert plan.diagnostics is not None and plan.diagnostics.certified
    assert engine.assign(strategy="maximin", minimum_score=1.0).assignments == ()


def test_maximin_diversity_treats_unknown_institutions_separately() -> None:
    scorer = _Scores(
        documents={"d": Document("d", "One", required_experts=2)},
        experts={
            "a": Expert("a", "A", capacity=1, institution="Same"),
            "b": Expert("b", "B", capacity=1, institution="Same"),
            "c": Expert("c", "C", capacity=1),
        },
        scores=(_score("d", "a", 1.0), _score("d", "b", 0.9), _score("d", "c", 0.1)),
    )
    plan = AssignmentEngine(scorer).assign(strategy="maximin", require_distinct_institutions=True)
    assert {item.expert_id for item in plan.assignments} == {"a", "c"}


def test_maximin_senior_reservation_allows_only_free_junior_slots() -> None:
    scorer = _Scores(
        documents={"d": Document("d", "One", required_experts=2)},
        experts={
            "junior": Expert("junior", "Junior", seniority=0.1),
            "senior": Expert("senior", "Senior", seniority=0.9),
        },
        scores=(_score("d", "junior", 0.9), _score("d", "senior", 0.1)),
    )
    engine = AssignmentEngine(scorer)
    assert len(engine.assign(strategy="maximin", minimum_senior_reviewers=1).assignments) == 2
    plan = engine.assign(strategy="maximin", minimum_senior_reviewers=2)
    assert [item.expert_id for item in plan.assignments] == ["senior"]
    assert plan.unmet == {"d": 1}


def test_maximin_resolves_exact_ties_by_stable_pairs() -> None:
    scorer = _Scores(
        documents={"d1": Document("d1", "One"), "d2": Document("d2", "Two")},
        experts={"b": Expert("b", "B", capacity=1), "a": Expert("a", "A", capacity=1)},
        scores=tuple(
            _score(document_id, expert_id, 0.5)
            for document_id in ("d2", "d1")
            for expert_id in ("b", "a")
        ),
    )
    engine = AssignmentEngine(scorer)
    first = engine.assign(strategy="maximin", reviewers_per_document=1)
    assert first == engine.assign(strategy="maximin", reviewers_per_document=1)
    assert {(item.document_id, item.expert_id) for item in first.assignments} == {
        ("d1", "a"),
        ("d2", "b"),
    }


@pytest.mark.parametrize("seed", range(25))
def test_maximin_matches_independent_global_subset_oracle(seed: int) -> None:
    randomizer = random.Random(seed)
    documents = {f"d{i}": Document(f"d{i}", "D", required_experts=1 + i % 2) for i in range(3)}
    experts = {
        f"e{i}": Expert(
            f"e{i}",
            "E",
            capacity=randomizer.randrange(3),
            institution=("A" if i < 2 else None),
            seniority=(0.1 if i % 2 else 0.9),
        )
        for i in range(4)
    }
    scores = tuple(
        _score(
            document_id,
            expert_id,
            randomizer.randrange(11) / 10,
            eligible=randomizer.choice((True, True, False)),
        )
        for document_id in documents
        for expert_id in experts
        if randomizer.choice((True, False))
    )
    scorer = _Scores(documents, experts, scores)
    diverse = bool(seed % 2)
    senior_floor = 1 if not diverse and seed % 3 == 0 else 0
    plan = AssignmentEngine(scorer).assign(
        strategy="maximin",
        reviewers_per_document=1,
        require_distinct_institutions=diverse,
        minimum_senior_reviewers=senior_floor,
    )
    selected = _selected_scores(
        scorer, {(item.document_id, item.expert_id) for item in plan.assignments}
    )
    assert _objective(selected, documents) == _oracle(
        scorer, 1, diverse=diverse, senior_floor=senior_floor
    )


@pytest.mark.parametrize("count_type", ("documents", "experts", "pairs"))
def test_maximin_rejects_instances_outside_explicit_exact_limit(count_type: str) -> None:
    doc_count = 7 if count_type == "documents" else 3
    expert_count = 9 if count_type == "experts" else 6
    scorer = _Scores(
        {f"d{i}": Document(f"d{i}", "D") for i in range(doc_count)},
        {f"e{i}": Expert(f"e{i}", "E") for i in range(expert_count)},
        tuple(_score(f"d{i}", f"e{j}", 0.5) for i in range(doc_count) for j in range(expert_count)),
    )
    with pytest.raises(ValueError, match="maximin supports at most"):
        AssignmentEngine(scorer).assign(strategy="maximin")


def test_maximin_rejects_competing_load_penalty() -> None:
    with pytest.raises(DataValidationError, match="does not support"):
        MatchConfig(strategy="maximin", load_balance_penalty=0.1)
    scorer = _Scores({"d": Document("d", "D")}, {"e": Expert("e", "E")}, ())
    with pytest.raises(ValueError, match="does not support"):
        AssignmentEngine(scorer).assign(strategy="maximin", load_balance_penalty=0.1)

    class ExplodingScorer:
        def __init__(self) -> None:
            self.documents = {"d": Document("d", "D")}
            self.experts = {"e": Expert("e", "E")}

        def matrix(self) -> tuple[MatchScore, ...]:
            raise AssertionError("score matrix must not be generated")

    with pytest.raises(ValueError, match="does not support"):
        AssignmentEngine(ExplodingScorer()).assign(strategy="maximin", load_balance_penalty=0.1)


def test_maximin_rejects_duplicate_eligible_pair_rows() -> None:
    scorer = _Scores(
        {"d": Document("d", "D")},
        {"e": Expert("e", "E")},
        (_score("d", "e", 0.8), _score("d", "e", 0.9)),
    )
    with pytest.raises(ValueError, match="one eligible score"):
        AssignmentEngine(scorer).assign(strategy="maximin")


def test_maximin_limit_counts_only_admissible_pairs() -> None:
    scorer = _Scores(
        {f"d{i}": Document(f"d{i}", "D") for i in range(3)},
        {f"e{i}": Expert(f"e{i}", "E") for i in range(6)},
        tuple(
            _score(f"d{i}", f"e{j}", 0.8 if i == 0 and j == 0 else 0.2)
            for i in range(3)
            for j in range(6)
        ),
    )
    plan = AssignmentEngine(scorer).assign(strategy="maximin", minimum_score=0.5)
    assert len(plan.assignments) == 1


@pytest.mark.parametrize("dimension", ("documents", "experts"))
def test_maximin_dimension_limits_precede_score_matrix_generation(dimension: str) -> None:
    class ExplodingScorer:
        def __init__(self) -> None:
            self.documents = {
                f"d{i}": Document(f"d{i}", "D") for i in range(7 if dimension == "documents" else 1)
            }
            self.experts = {
                f"e{i}": Expert(f"e{i}", "E") for i in range(9 if dimension == "experts" else 1)
            }

        def matrix(self) -> tuple[MatchScore, ...]:
            raise AssertionError("score matrix must not be generated")

    with pytest.raises(ValueError, match="maximin supports at most"):
        AssignmentEngine(ExplodingScorer()).assign(strategy="maximin")


def test_maximin_cli_config_exports_standard_plan(tmp_path: Path) -> None:
    documents = tmp_path / "documents.json"
    experts = tmp_path / "experts.json"
    affinities = tmp_path / "affinities.csv"
    config = tmp_path / "config.json"
    output = tmp_path / "plan.json"
    documents.write_text(
        json.dumps([{"id": "d1", "title": "One"}, {"id": "d2", "title": "Two"}]),
        encoding="utf-8",
    )
    experts.write_text(
        json.dumps(
            [
                {"id": "e1", "name": "One", "capacity": 1},
                {"id": "e2", "name": "Two", "capacity": 1},
            ]
        ),
        encoding="utf-8",
    )
    affinities.write_text(
        "document_id,expert_id,score\nd1,e1,0.95\nd1,e2,0.75\nd2,e1,0.75\nd2,e2,0.70\n",
        encoding="utf-8",
    )
    config.write_text(json.dumps({"strategy": "maximin", "reviewers_per_document": 1}))

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
    assert exported["strategy"] == "maximin"
    assert {item["score"] for item in exported["assignments"]} == {0.75}
