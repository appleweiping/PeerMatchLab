from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import pytest

from peermatchlab.assignment import AssignmentEngine, AssignmentStrategy, _FlowNetwork
from peermatchlab.audit import audit_plan, gini
from peermatchlab.models import (
    Assignment,
    Conflict,
    Document,
    Expert,
    MatchPlan,
    MatchScore,
)
from peermatchlab.scoring import MatchScorer


def test_flow_network_sends_requested_units() -> None:
    network = _FlowNetwork(4)
    network.add_edge(0, 1, 1, 0)
    network.add_edge(1, 3, 1, -5)
    network.add_edge(0, 2, 1, 0)
    network.add_edge(2, 3, 1, -3)
    assert network.min_cost_flow(0, 3, 2) == (2, -8)


def test_flow_network_stops_when_sink_unreachable() -> None:
    network = _FlowNetwork(3)
    network.add_edge(0, 1, 1, 0)
    assert network.min_cost_flow(0, 2, 1) == (0, 0)


def test_optimal_assignment_respects_capacity(documents, experts) -> None:
    plan = AssignmentEngine(MatchScorer(documents, experts)).assign(reviewers_per_document=2)
    counts: dict[str, int] = {}
    for assignment in plan.assignments:
        counts[assignment.expert_id] = counts.get(assignment.expert_id, 0) + 1
    assert all(counts.get(expert.id, 0) <= expert.capacity for expert in experts)


def test_assignment_respects_conflict(documents, experts) -> None:
    scorer = MatchScorer(documents, experts, conflicts=[Conflict("paper-1", "expert-a")])
    plan = AssignmentEngine(scorer).assign(reviewers_per_document=1)
    assert ("paper-1", "expert-a") not in {
        (item.document_id, item.expert_id) for item in plan.assignments
    }


def test_document_specific_demand_overrides_default(experts) -> None:
    document = Document("d", "Machine learning", required_experts=1)
    plan = AssignmentEngine(MatchScorer([document], experts)).assign(reviewers_per_document=3)
    assert len(plan.for_document("d")) == 1


def test_minimum_score_can_create_unmet_demand(documents, experts) -> None:
    plan = AssignmentEngine(MatchScorer(documents, experts)).assign(
        reviewers_per_document=1, minimum_score=1.1
    )
    assert plan.assignments == ()
    assert plan.unmet == {"paper-1": 1, "paper-2": 1}


def test_greedy_is_deterministic(documents, experts) -> None:
    engine = AssignmentEngine(MatchScorer(documents, experts))
    first = engine.assign(strategy="greedy", reviewers_per_document=1)
    second = engine.assign(strategy="greedy", reviewers_per_document=1)
    assert first == second


def test_unknown_strategy_is_rejected(documents, experts) -> None:
    with pytest.raises(ValueError):
        AssignmentEngine(MatchScorer(documents, experts)).assign(strategy="random")


def test_non_positive_reviewer_count_is_rejected(documents, experts) -> None:
    with pytest.raises(ValueError):
        AssignmentEngine(MatchScorer(documents, experts)).assign(reviewers_per_document=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reviewers_per_document": True},
        {"minimum_score": float("nan")},
        {"minimum_score": 10**1_000},
        {"minimum_score": True},
        {"require_distinct_institutions": 1},
        {"load_balance_penalty": float("nan")},
        {"load_balance_penalty": -0.1},
        {"load_balance_penalty": 1.1},
    ],
)
def test_assignment_rejects_invalid_runtime_controls(documents, experts, kwargs) -> None:
    with pytest.raises(ValueError):
        AssignmentEngine(MatchScorer(documents, experts)).assign(**kwargs)


def test_diverse_assignment_avoids_duplicate_institutions(documents, experts) -> None:
    plan = AssignmentEngine(MatchScorer(documents[:1], experts)).assign(
        reviewers_per_document=2, require_distinct_institutions=True
    )
    institutions = [
        {expert.id: expert for expert in experts}[item.expert_id].institution
        for item in plan.assignments
    ]
    assert len(institutions) == len(set(institutions))
    assert plan.strategy == "optimal-diverse"


def test_greedy_diverse_strategy_remains_explicit(documents, experts) -> None:
    plan = AssignmentEngine(MatchScorer(documents[:1], experts)).assign(
        strategy="greedy",
        reviewers_per_document=2,
        require_distinct_institutions=True,
    )

    assert plan.strategy == "greedy-diverse"


@pytest.mark.parametrize("strategy", ["optimal", "greedy"])
def test_diversity_does_not_group_experts_with_unknown_institutions(strategy: str) -> None:
    scorer = _FixedScorer(
        documents={"d": Document("d", "D")},
        experts={
            "e1": Expert("e1", "E1", capacity=1),
            "e2": Expert("e2", "E2", capacity=1),
        },
        scores=(_score("d", "e1", 0.9), _score("d", "e2", 0.8)),
    )

    plan = AssignmentEngine(scorer).assign(  # type: ignore[arg-type]
        strategy=strategy,
        reviewers_per_document=2,
        require_distinct_institutions=True,
    )

    assert {item.expert_id for item in plan.assignments} == {"e1", "e2"}
    assert plan.unmet == {}


@dataclass
class _FixedScorer:
    documents: dict[str, Document]
    experts: dict[str, Expert]
    scores: tuple[MatchScore, ...]

    def matrix(self) -> tuple[MatchScore, ...]:
        return self.scores


def _score(document: str, expert: str, value: float) -> MatchScore:
    return MatchScore(document, expert, value, value, 0, 0, 0, 0)


def test_optimal_can_beat_round_robin_greedy() -> None:
    scorer = _FixedScorer(
        documents={"d1": Document("d1", "D1"), "d2": Document("d2", "D2")},
        experts={
            "e1": Expert("e1", "E1", capacity=1),
            "e2": Expert("e2", "E2", capacity=1),
        },
        scores=(
            _score("d1", "e1", 0.90),
            _score("d1", "e2", 0.80),
            _score("d2", "e1", 0.85),
            _score("d2", "e2", 0.10),
        ),
    )
    engine = AssignmentEngine(scorer)  # type: ignore[arg-type]
    optimal = engine.assign(strategy=AssignmentStrategy.OPTIMAL, reviewers_per_document=1)
    greedy = engine.assign(strategy=AssignmentStrategy.GREEDY, reviewers_per_document=1)
    assert optimal.total_score > greedy.total_score
    assert {(item.document_id, item.expert_id) for item in optimal.assignments} == {
        ("d1", "e2"),
        ("d2", "e1"),
    }


def test_optimal_diverse_can_beat_diverse_greedy() -> None:
    scorer = _FixedScorer(
        documents={"d1": Document("d1", "D1"), "d2": Document("d2", "D2")},
        experts={
            "a1": Expert("a1", "A1", capacity=1, institution="A"),
            "a2": Expert("a2", "A2", capacity=1, institution="A"),
            "b1": Expert("b1", "B1", capacity=1, institution="B"),
            "b2": Expert("b2", "B2", capacity=1, institution="B"),
        },
        scores=(
            _score("d1", "a1", 0.90),
            _score("d1", "a2", 0.80),
            _score("d1", "b1", 0.90),
            _score("d1", "b2", 0.10),
            _score("d2", "a1", 0.85),
            _score("d2", "a2", 0.10),
            _score("d2", "b1", 0.85),
            _score("d2", "b2", 0.80),
        ),
    )
    engine = AssignmentEngine(scorer)  # type: ignore[arg-type]

    optimal = engine.assign(
        strategy="optimal", reviewers_per_document=2, require_distinct_institutions=True
    )
    greedy = engine.assign(
        strategy="greedy", reviewers_per_document=2, require_distinct_institutions=True
    )

    assert optimal.total_score > greedy.total_score
    assert optimal.strategy == "optimal-diverse"
    for document_id in scorer.documents:
        institutions = {
            scorer.experts[item.expert_id].institution for item in optimal.for_document(document_id)
        }
        assert len(institutions) == 2


def test_optimal_diverse_matches_exhaustive_small_instances() -> None:
    generator = random.Random(31)
    document_ids = ("d1", "d2", "d3")
    expert_ids = ("e1", "e2", "e3", "e4")
    institutions = {"e1": "A", "e2": "A", "e3": "B", "e4": "C"}
    choices = tuple(
        subset
        for size in range(3)
        for subset in itertools.combinations(expert_ids, size)
        if len({institutions[expert_id] for expert_id in subset}) == len(subset)
    )

    for _ in range(30):
        capacities = {expert_id: generator.randint(1, 2) for expert_id in expert_ids}
        score_values = {
            (document_id, expert_id): generator.randint(0, 100) / 100
            for document_id in document_ids
            for expert_id in expert_ids
        }
        scorer = _FixedScorer(
            documents={key: Document(key, key) for key in document_ids},
            experts={
                key: Expert(
                    key,
                    key,
                    capacity=capacities[key],
                    institution=institutions[key],
                )
                for key in expert_ids
            },
            scores=tuple(
                _score(document_id, expert_id, score_values[(document_id, expert_id)])
                for document_id in document_ids
                for expert_id in expert_ids
            ),
        )
        plan = AssignmentEngine(scorer).assign(  # type: ignore[arg-type]
            reviewers_per_document=2,
            require_distinct_institutions=True,
        )

        exhaustive: list[tuple[int, float]] = []
        for allocation in itertools.product(choices, repeat=len(document_ids)):
            loads = {
                expert_id: sum(expert_id in selected for selected in allocation)
                for expert_id in expert_ids
            }
            if any(loads[key] > capacities[key] for key in expert_ids):
                continue
            exhaustive.append(
                (
                    sum(len(selected) for selected in allocation),
                    sum(
                        score_values[(document_id, expert_id)]
                        for document_id, selected in zip(document_ids, allocation, strict=True)
                        for expert_id in selected
                    ),
                )
            )
        expected_count, expected_score = max(exhaustive)
        assert len(plan.assignments) == expected_count
        assert plan.total_score == pytest.approx(expected_score)


def test_optimal_score_dominates_tie_breaker_for_large_expert_pool() -> None:
    experts = {
        f"e{index:04d}": Expert(f"e{index:04d}", f"E{index}", capacity=1) for index in range(1_002)
    }
    scorer = _FixedScorer(
        documents={"d": Document("d", "D")},
        experts=experts,
        scores=(
            _score("d", "e0000", 0.500000),
            _score("d", "e1001", 0.500001),
        ),
    )

    plan = AssignmentEngine(scorer).assign(reviewers_per_document=1)  # type: ignore[arg-type]

    assert [(item.expert_id, item.score) for item in plan.assignments] == [("e1001", 0.500001)]


@pytest.mark.parametrize("strategy", ["optimal", "greedy"])
def test_load_balance_penalty_spreads_equal_score_work(strategy: str) -> None:
    scorer = _FixedScorer(
        documents={f"d{index}": Document(f"d{index}", f"D{index}") for index in range(4)},
        experts={
            "e1": Expert("e1", "E1", capacity=4),
            "e2": Expert("e2", "E2", capacity=4),
        },
        scores=tuple(
            _score(document_id, expert_id, 0.8)
            for document_id in ("d0", "d1", "d2", "d3")
            for expert_id in ("e1", "e2")
        ),
    )

    plan = AssignmentEngine(scorer).assign(  # type: ignore[arg-type]
        strategy=strategy,
        reviewers_per_document=1,
        load_balance_penalty=0.1,
    )

    loads = {
        expert_id: sum(item.expert_id == expert_id for item in plan.assignments)
        for expert_id in scorer.experts
    }
    assert loads == {"e1": 2, "e2": 2}
    assert plan.strategy == f"{strategy}-balanced"


def test_optimal_load_balance_uses_convex_marginal_objective() -> None:
    scorer = _FixedScorer(
        documents={f"d{index}": Document(f"d{index}", f"D{index}") for index in range(3)},
        experts={
            "e1": Expert("e1", "E1", capacity=3),
            "e2": Expert("e2", "E2", capacity=3),
        },
        scores=tuple(
            _score(document_id, expert_id, 0.9 if expert_id == "e1" else 0.75)
            for document_id in ("d0", "d1", "d2")
            for expert_id in ("e1", "e2")
        ),
    )

    unbalanced = AssignmentEngine(scorer).assign(  # type: ignore[arg-type]
        reviewers_per_document=1,
        load_balance_penalty=0.0,
    )
    balanced = AssignmentEngine(scorer).assign(  # type: ignore[arg-type]
        reviewers_per_document=1,
        load_balance_penalty=0.2,
    )

    assert [item.expert_id for item in unbalanced.assignments] == ["e1", "e1", "e1"]
    assert sorted(item.expert_id for item in balanced.assignments) == ["e1", "e1", "e2"]
    # The selected load (2, 1) maximizes raw affinity minus
    # penalty * sum(load * (load - 1) / 2): 2.55 - 0.2 > 2.70 - 0.6.
    assert balanced.total_score == pytest.approx(2.55)


def test_gini_zero_for_equal_workload() -> None:
    assert gini([2, 2, 2]) == pytest.approx(0.0)


def test_gini_detects_unequal_workload() -> None:
    assert gini([0, 0, 3]) == pytest.approx(2 / 3)


def test_gini_handles_arbitrarily_large_integer_workloads() -> None:
    assert gini([0, 10**1_000]) == pytest.approx(0.5)


@pytest.mark.parametrize("values", [[-1, 2], [True, 1]])
def test_gini_rejects_invalid_workloads(values) -> None:
    with pytest.raises(ValueError, match="workloads"):
        gini(values)


def test_audit_detects_conflict_violation(documents, experts) -> None:
    plan = MatchPlan((Assignment("paper-1", "expert-a", 0.8, 1),), {}, "manual", 0.8)
    report = audit_plan(plan, documents, experts, [Conflict("paper-1", "expert-a")])
    assert report.conflict_violations == (("paper-1", "expert-a"),)
    assert not report.safe


def test_audit_detects_capacity_violation(documents, experts) -> None:
    plan = MatchPlan(
        (
            Assignment("paper-1", "expert-a", 0.8, 1),
            Assignment("paper-2", "expert-a", 0.7, 1),
        ),
        {},
        "manual",
        1.5,
    )
    report = audit_plan(plan, documents, experts)
    assert report.capacity_violations == ("expert-a",)


def test_audit_reports_duplicate_institutions(documents, experts) -> None:
    plan = MatchPlan(
        (
            Assignment("paper-1", "expert-b", 0.8, 1),
            Assignment("paper-1", "expert-c", 0.7, 2),
        ),
        {},
        "manual",
        1.5,
    )
    report = audit_plan(plan, documents, experts)
    assert report.institution_duplicates == {"paper-1": ("West Lab",)}
    assert report.safe

    strict_report = audit_plan(plan, documents, experts, require_distinct_institutions=True)
    assert strict_report.institution_duplicates == {"paper-1": ("West Lab",)}
    assert strict_report.institution_diversity_required
    assert not strict_report.safe


def test_audit_empty_inputs_are_fully_covered() -> None:
    report = audit_plan(MatchPlan((), {}, "empty", 0), [], [], default_demand=1)
    assert report.document_coverage == 1.0
    assert report.demand_coverage == 1.0


def test_audit_rejects_unknown_references_as_unsafe(documents, experts) -> None:
    plan = MatchPlan(
        (
            Assignment("missing-document", "expert-a", 0.8, 1),
            Assignment("paper-1", "missing-expert", 0.7, 2),
        ),
        {},
        "external",
        1.5,
    )

    report = audit_plan(plan, documents, experts)

    assert report.unknown_documents == ("missing-document",)
    assert report.unknown_experts == ("missing-expert",)
    assert report.demand_coverage == 0.0
    assert not report.safe


def test_audit_rejects_duplicate_and_excess_assignments() -> None:
    documents = [Document("d", "D", required_experts=1)]
    experts = [Expert("e1", "E1", capacity=3), Expert("e2", "E2", capacity=3)]
    plan = MatchPlan(
        (
            Assignment("d", "e1", 0.9, 1),
            Assignment("d", "e1", 0.8, 2),
            Assignment("d", "e2", 0.7, 3),
        ),
        {},
        "external",
        2.4,
    )

    report = audit_plan(plan, documents, experts)

    assert report.duplicate_assignments == (("d", "e1"),)
    assert report.demand_violations == ("d",)
    assert report.demand_coverage == 1.0
    assert not report.safe


@pytest.mark.parametrize("default_demand", [0, True, 1.5])
def test_audit_requires_positive_integer_default_demand(documents, experts, default_demand) -> None:
    with pytest.raises(ValueError, match="default_demand"):
        audit_plan(
            MatchPlan((), {}, "external", 0),
            documents,
            experts,
            default_demand=default_demand,
        )


def test_audit_requires_boolean_diversity_control(documents, experts) -> None:
    with pytest.raises(ValueError, match="require_distinct_institutions"):
        audit_plan(
            MatchPlan((), {}, "external", 0),
            documents,
            experts,
            require_distinct_institutions=1,  # type: ignore[arg-type]
        )


def test_audit_rejects_duplicate_input_identifiers(documents, experts) -> None:
    with pytest.raises(ValueError, match="document identifiers"):
        audit_plan(
            MatchPlan((), {}, "external", 0),
            [documents[0], documents[0]],
            experts,
        )
    with pytest.raises(ValueError, match="expert identifiers"):
        audit_plan(
            MatchPlan((), {}, "external", 0),
            documents,
            [experts[0], experts[0]],
        )
