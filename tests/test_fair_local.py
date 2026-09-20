"""Independent feasibility oracle and limits for bounded local fairness."""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pytest

from peermatchlab import fair_local as fair_local_module
from peermatchlab.affinity import Affinity
from peermatchlab.assignment import AssignmentEngine
from peermatchlab.cli import main
from peermatchlab.config import MatchConfig
from peermatchlab.models import DataValidationError, Document, Expert, MatchPlan, MatchScore
from peermatchlab.pipeline import run_affinity_matching


@dataclass
class _Scorer:
    documents: dict[str, Document]
    experts: dict[str, Expert]
    scores: tuple[MatchScore, ...]

    def matrix(self) -> tuple[MatchScore, ...]:
        return self.scores


def _score(document: str, expert: str, value: float, *, eligible: bool = True) -> MatchScore:
    return MatchScore(document, expert, value, value, 0, 0, 0, 0, eligible=eligible)


def _vector(plan: MatchPlan, documents: dict[str, Document]) -> tuple[float, ...]:
    return tuple(
        sorted(
            math.fsum(item.score for item in plan.assignments if item.document_id == document)
            for document in documents
        )
    )


def _assert_valid(
    plan: MatchPlan,
    scorer: _Scorer,
    *,
    diverse: bool,
    senior_floor: int,
) -> None:
    permitted = {
        (edge.document_id, edge.expert_id): edge for edge in scorer.scores if edge.eligible
    }
    pairs = [(item.document_id, item.expert_id) for item in plan.assignments]
    assert len(pairs) == len(set(pairs))
    assert all(pair in permitted for pair in pairs)
    assert all(
        item.score == permitted[(item.document_id, item.expert_id)].total
        for item in plan.assignments
    )
    for document_id, document in scorer.documents.items():
        selected = [expert for current, expert in pairs if current == document_id]
        demand = document.required_experts or 1
        assert len(selected) <= demand
        assert sum(scorer.experts[key].seniority < 0.75 for key in selected) <= demand - min(
            senior_floor, demand
        )
        known = [
            scorer.experts[key].institution
            for key in selected
            if scorer.experts[key].institution is not None
        ]
        assert not diverse or len(known) == len(set(known))
        assert plan.unmet.get(document_id, 0) == demand - len(selected)
    for expert_id, expert in scorer.experts.items():
        assert sum(key == expert_id for _, key in pairs) <= expert.capacity


def _oracle_cardinality(
    scorer: _Scorer,
    *,
    diverse: bool,
    senior_floor: int,
    senior_threshold: float = 0.75,
) -> int:
    """Check every subset of independently declared edges, not solver moves."""

    edges = tuple(score for score in scorer.scores if score.eligible)
    best = 0
    for enabled in itertools.product((False, True), repeat=len(edges)):
        picked = [edge for edge, use in zip(edges, enabled, strict=True) if use]
        by_document = {key: [] for key in scorer.documents}
        by_expert = {key: 0 for key in scorer.experts}
        for edge in picked:
            by_document[edge.document_id].append(edge.expert_id)
            by_expert[edge.expert_id] += 1
        if any(by_expert[key] > expert.capacity for key, expert in scorer.experts.items()):
            continue
        valid = True
        for document_id, document in scorer.documents.items():
            selected = by_document[document_id]
            demand = document.required_experts or 1
            if len(selected) > demand:
                valid = False
                break
            free = demand - min(senior_floor, demand)
            if sum(scorer.experts[key].seniority < senior_threshold for key in selected) > free:
                valid = False
                break
            affiliations = [
                scorer.experts[key].institution
                for key in selected
                if scorer.experts[key].institution is not None
            ]
            if diverse and len(affiliations) != len(set(affiliations)):
                valid = False
                break
        if valid:
            best = max(best, len(picked))
    return best


def test_fair_local_improves_weakest_document_without_losing_cardinality() -> None:
    documents = [Document("d1", "One"), Document("d2", "Two")]
    experts = [Expert("e1", "One", capacity=1), Expert("e2", "Two", capacity=1)]
    affinities = [
        Affinity("d1", "e1", 0.95),
        Affinity("d1", "e2", 0.75),
        Affinity("d2", "e1", 0.75),
        Affinity("d2", "e2", 0.70),
    ]
    baseline = run_affinity_matching(
        documents, experts, affinities, config=MatchConfig(reviewers_per_document=1)
    )
    fair = run_affinity_matching(
        documents,
        experts,
        affinities,
        config=MatchConfig(strategy="fair-local", reviewers_per_document=1),
    )
    assert _vector(baseline.plan, {item.id: item for item in documents}) == (0.70, 0.95)
    assert _vector(fair.plan, {item.id: item for item in documents}) == (0.75, 0.75)
    assert len(fair.plan.assignments) == len(baseline.plan.assignments) == 2
    assert fair.plan.strategy == "fair-local"
    assert fair.plan.diagnostics is not None and fair.plan.diagnostics.certified
    assert fair.audit.safe


def test_fair_local_transfer_improves_unfilled_document_and_preserves_certificate() -> None:
    documents = {
        "d1": Document("d1", "One", required_experts=2),
        "d2": Document("d2", "Two", required_experts=1),
    }
    experts = {
        "e1": Expert("e1", "One", capacity=1),
        "e2": Expert("e2", "Two", capacity=1),
    }
    scorer = _Scorer(
        documents,
        experts,
        (_score("d1", "e1", 1), _score("d1", "e2", 0.9), _score("d2", "e2", 0.8)),
    )
    engine = AssignmentEngine(scorer)
    baseline = engine.assign(reviewers_per_document=1)
    fair = engine.assign(strategy="fair-local", reviewers_per_document=1)
    assert _vector(baseline, documents) == (0.0, 1.9)
    assert _vector(fair, documents) == (0.8, 1.0)
    assert len(fair.assignments) == len(baseline.assignments) == 2
    assert fair.unmet == {"d1": 1}
    assert fair.diagnostics is not None and fair.diagnostics.certified
    _assert_valid(fair, scorer, diverse=False, senior_floor=0)


def test_fair_local_is_not_misrepresented_as_global_maximin() -> None:
    scorer = _Scorer(
        {f"d{i}": Document(f"d{i}", "D") for i in range(3)},
        {f"e{i}": Expert(f"e{i}", "E", capacity=1) for i in range(3)},
        (
            _score("d0", "e0", 1.0),
            _score("d0", "e1", 0.7),
            _score("d1", "e1", 1.0),
            _score("d1", "e2", 0.7),
            _score("d2", "e2", 0.4),
            _score("d2", "e0", 0.7),
        ),
    )
    engine = AssignmentEngine(scorer)
    local = engine.assign(strategy="fair-local", reviewers_per_document=1)
    exact = engine.assign(strategy="maximin", reviewers_per_document=1)
    assert _vector(local, scorer.documents) == (0.4, 1.0, 1.0)
    assert _vector(exact, scorer.documents) == (0.7, 0.7, 0.7)


@pytest.mark.parametrize("seed", range(20))
def test_fixed_pair_bitmask_order_matches_sorted_pair_lists_for_multi_edge_moves(seed: int) -> None:
    """Independent combinatorial check of the exact tie-order identity."""

    rng = random.Random(seed)
    universe = sorted((f"d{i}", f"e{j}") for i in range(4) for j in range(4))
    weights = {pair: 1 << (len(universe) - index - 1) for index, pair in enumerate(universe)}
    for _ in range(50):
        size = rng.randrange(4, 13)
        before = set(rng.sample(universe, size))
        count = rng.randrange(1, min(3, size, len(universe) - size) + 1)
        removed = set(rng.sample(sorted(before), count))
        added = set(rng.sample(sorted(set(universe) - before), count))
        after = (before - removed) | added
        before_mask = sum(weights[pair] for pair in before)
        after_mask = before_mask
        for pair in removed | added:
            after_mask ^= weights[pair]
        assert (after_mask > before_mask) == (tuple(sorted(after)) < tuple(sorted(before)))
        assert (after_mask < before_mask) == (tuple(sorted(after)) > tuple(sorted(before)))


@pytest.mark.parametrize("seed", range(20))
def test_fair_local_matches_independent_cardinality_oracle_and_never_regresses(seed: int) -> None:
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
        _score(document, expert, rng.randrange(11) / 10, eligible=rng.choice((True, False)))
        for document in documents
        for expert in experts
        if rng.choice((True, False))
    )
    scorer = _Scorer(documents, experts, scores)
    diverse = bool(seed % 2)
    senior_floor = 1 if not diverse and seed % 3 == 0 else 0
    engine = AssignmentEngine(scorer)
    baseline = engine.assign(
        reviewers_per_document=1,
        require_distinct_institutions=diverse,
        minimum_senior_reviewers=senior_floor,
    )
    fair = engine.assign(
        strategy="fair-local",
        reviewers_per_document=1,
        require_distinct_institutions=diverse,
        minimum_senior_reviewers=senior_floor,
    )
    assert (
        len(fair.assignments)
        == len(baseline.assignments)
        == _oracle_cardinality(scorer, diverse=diverse, senior_floor=senior_floor)
    )
    assert _vector(fair, documents) >= _vector(baseline, documents)
    _assert_valid(fair, scorer, diverse=diverse, senior_floor=senior_floor)
    assert fair == engine.assign(
        strategy="fair-local",
        reviewers_per_document=1,
        require_distinct_institutions=diverse,
        minimum_senior_reviewers=senior_floor,
    )


def test_fair_local_exceeds_exact_search_dimensions_and_respects_budget() -> None:
    scorer = _Scorer(
        {f"d{i}": Document(f"d{i}", "D") for i in range(12)},
        {f"e{i}": Expert(f"e{i}", "E", capacity=1) for i in range(15)},
        tuple(
            _score(f"d{i}", f"e{j}", ((i + j) % 10 + 1) / 10)
            for i in range(12)
            for j in range(15)
            if (i + j) % 3 == 0
        ),
    )
    engine = AssignmentEngine(scorer)
    baseline = engine.assign(reviewers_per_document=1)
    fair = engine.assign(strategy="fair-local", reviewers_per_document=1, fair_local_max_checks=3)
    assert len(fair.assignments) == len(baseline.assignments)
    assert _vector(fair, scorer.documents) >= _vector(baseline, scorer.documents)
    assert fair.diagnostics is not None and fair.diagnostics.certified


def test_fair_local_does_not_consume_more_than_the_candidate_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = _Scorer(
        {"d1": Document("d1", "D"), "d2": Document("d2", "D")},
        {"e1": Expert("e1", "E", capacity=1), "e2": Expert("e2", "E", capacity=1)},
        tuple(
            _score(document, expert, 0.5) for document in ("d1", "d2") for expert in ("e1", "e2")
        ),
    )
    original = fair_local_module._neighbors
    generated = 0

    def counted(*args: object, **kwargs: object):
        nonlocal generated
        for move in original(*args, **kwargs):
            generated += 1
            if generated > 3:
                raise AssertionError("candidate budget exceeded")
            yield move

    monkeypatch.setattr(fair_local_module, "_neighbors", counted)
    plan = AssignmentEngine(scorer).assign(
        strategy="fair-local", reviewers_per_document=1, fair_local_max_checks=3
    )
    assert generated == 3
    assert len(plan.assignments) == 2


def test_fair_local_rejects_excess_requested_slots_before_scoring() -> None:
    class ExplodingScorer:
        def __init__(self) -> None:
            self.documents = {"d": Document("d", "D", required_experts=513)}
            self.experts = {"e": Expert("e", "E", capacity=1)}

        def matrix(self) -> tuple[MatchScore, ...]:
            raise AssertionError("must not score outside declared limits")

    with pytest.raises(ValueError, match="512 requested slots"):
        AssignmentEngine(ExplodingScorer()).assign(strategy="fair-local")


def test_fair_local_rejects_excess_eligible_pairs() -> None:
    scorer = _Scorer(
        {f"d{i}": Document(f"d{i}", "D") for i in range(65)},
        {f"e{i}": Expert(f"e{i}", "E", capacity=1) for i in range(64)},
        tuple(_score(f"d{i}", f"e{j}", 0.5) for i in range(65) for j in range(64)),
    )
    with pytest.raises(ValueError, match="4096 eligible pairs"):
        AssignmentEngine(scorer).assign(strategy="fair-local")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fair_local_max_steps", True),
        ("fair_local_max_steps", 0),
        ("fair_local_max_steps", 129),
        ("fair_local_max_checks", False),
        ("fair_local_max_checks", 0),
        ("fair_local_max_checks", 2_000_001),
    ],
)
def test_fair_local_rejects_invalid_resource_controls(field: str, value: object) -> None:
    with pytest.raises(DataValidationError, match=field):
        MatchConfig.from_mapping({"strategy": "fair-local", field: value})
    scorer = _Scorer({"d": Document("d", "D")}, {"e": Expert("e", "E")}, ())
    with pytest.raises(ValueError, match=field):
        AssignmentEngine(scorer).assign(strategy="fair-local", **{field: value})


def test_fair_local_rejects_competing_load_objective_and_duplicate_pairs() -> None:
    with pytest.raises(DataValidationError, match="does not support"):
        MatchConfig(strategy="fair-local", load_balance_penalty=0.1)
    scorer = _Scorer(
        {"d": Document("d", "D")},
        {"e": Expert("e", "E")},
        (_score("d", "e", 0.8), _score("d", "e", 0.9)),
    )
    with pytest.raises(ValueError, match="one eligible score"):
        AssignmentEngine(scorer).assign(strategy="fair-local")
    with pytest.raises(ValueError, match="does not support"):
        AssignmentEngine(scorer).assign(strategy="fair-local", load_balance_penalty=0.1)


def test_fair_local_controls_are_not_silently_ignored_by_other_strategies() -> None:
    with pytest.raises(DataValidationError, match="require strategy"):
        MatchConfig(strategy="optimal", fair_local_max_checks=2)
    scorer = _Scorer({"d": Document("d", "D")}, {"e": Expert("e", "E")}, ())
    with pytest.raises(ValueError, match="require strategy"):
        AssignmentEngine(scorer).assign(strategy="optimal", fair_local_max_steps=2)


def test_fair_local_dimension_limits_precede_score_generation() -> None:
    class ExplodingScorer:
        def __init__(self) -> None:
            self.documents = {f"d{i}": Document(f"d{i}", "D") for i in range(129)}
            self.experts = {"e": Expert("e", "E")}

        def matrix(self) -> tuple[MatchScore, ...]:
            raise AssertionError("must not score outside declared dimensions")

    with pytest.raises(ValueError, match="at most 128 documents"):
        AssignmentEngine(ExplodingScorer()).assign(strategy="fair-local")


def test_fair_local_cli_smoke(tmp_path: Path) -> None:
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
    config.write_text(json.dumps({"strategy": "fair-local", "reviewers_per_document": 1}))
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
    assert exported["strategy"] == "fair-local"
    assert {item["score"] for item in exported["assignments"]} == {0.75}
