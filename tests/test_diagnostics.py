from __future__ import annotations

import itertools
import json
import random
from copy import deepcopy
from dataclasses import dataclass

import pytest

from peermatchlab.assignment import AssignmentEngine
from peermatchlab.cli import main
from peermatchlab.io import plan_from_dict, plan_to_dict
from peermatchlab.models import (
    AssignmentDiagnostics,
    DataValidationError,
    DemandDiagnostic,
    Document,
    Expert,
    FeasibilityStatus,
    MatchPlan,
    MatchScore,
    UnmetReason,
)


@dataclass
class _FixedScorer:
    documents: dict[str, Document]
    experts: dict[str, Expert]
    scores: tuple[MatchScore, ...]
    conflict_pairs: frozenset[tuple[str, str]] = frozenset()

    def matrix(self) -> tuple[MatchScore, ...]:
        return self.scores


def _score(document_id: str, expert_id: str, value: float, *, eligible: bool = True) -> MatchScore:
    reasons = () if eligible else ("excluded: declared hard conflict",)
    return MatchScore(
        document_id,
        expert_id,
        value,
        value,
        0.0,
        0.0,
        0.0,
        0.0,
        eligible=eligible,
        reasons=reasons,
    )


def test_diagnostics_separate_filter_evidence() -> None:
    scorer = _FixedScorer(
        documents={"d": Document("d", "D", required_experts=3)},
        experts={
            "conflict": Expert("conflict", "Conflict", capacity=1),
            "zero": Expert("zero", "Zero", capacity=0),
            "low": Expert("low", "Low", capacity=1),
            "good": Expert("good", "Good", capacity=1),
        },
        scores=(
            _score("d", "conflict", 0.9, eligible=False),
            _score("d", "zero", 0.9),
            _score("d", "low", 0.2),
            _score("d", "good", 0.9),
        ),
        conflict_pairs=frozenset({("d", "conflict")}),
    )

    plan = AssignmentEngine(scorer).assign(minimum_score=0.5)  # type: ignore[arg-type]
    assert plan.diagnostics is not None
    diagnostic = plan.diagnostics.for_document("d")
    assert diagnostic is not None
    assert diagnostic.reason_codes == (
        UnmetReason.HARD_CONFLICT,
        UnmetReason.ZERO_CAPACITY,
        UnmetReason.MINIMUM_SCORE,
        UnmetReason.CANDIDATE_SCARCITY,
    )
    assert diagnostic.evidence["hard_conflict_pairs"] == 1
    assert diagnostic.evidence["zero_capacity_pairs"] == 1
    assert diagnostic.evidence["below_minimum_score_pairs"] == 1
    assert diagnostic.evidence["admissible_pairs"] == 1
    assert (diagnostic.requested, diagnostic.assigned, diagnostic.unmet) == (3, 1, 2)
    assert plan.diagnostics.status is FeasibilityStatus.INFEASIBLE
    assert plan.diagnostics.certified


def test_optimal_diagnostics_expose_global_capacity_coupling() -> None:
    scorer = _FixedScorer(
        documents={"d1": Document("d1", "D1"), "d2": Document("d2", "D2")},
        experts={"e": Expert("e", "E", capacity=1)},
        scores=(_score("d1", "e", 0.9), _score("d2", "e", 0.8)),
    )

    plan = AssignmentEngine(scorer).assign(reviewers_per_document=1)  # type: ignore[arg-type]
    assert plan.diagnostics is not None
    diagnostic = next(item for item in plan.diagnostics.documents if item.unmet)
    assert diagnostic.saturated_experts == ("e",)
    assert UnmetReason.EXPERT_CAPACITY in diagnostic.reason_codes
    assert UnmetReason.GLOBAL_CAPACITY_COUPLING in diagnostic.reason_codes
    assert diagnostic.evidence["admissible_pairs"] == 1


def test_diagnostics_identify_institution_gate() -> None:
    scorer = _FixedScorer(
        documents={"d": Document("d", "D")},
        experts={
            "e1": Expert("e1", "E1", capacity=1, institution="A"),
            "e2": Expert("e2", "E2", capacity=1, institution="A"),
        },
        scores=(_score("d", "e1", 0.9), _score("d", "e2", 0.8)),
    )

    plan = AssignmentEngine(scorer).assign(  # type: ignore[arg-type]
        reviewers_per_document=2, require_distinct_institutions=True
    )
    assert plan.diagnostics is not None
    diagnostic = plan.diagnostics.for_document("d")
    assert diagnostic is not None
    assert UnmetReason.INSTITUTION_DIVERSITY in diagnostic.reason_codes
    assert diagnostic.evidence["institution_groups"] == 1
    assert diagnostic.evidence["institution_blocked_pairs"] == 1


def test_diagnostics_identify_unfilled_senior_reservation() -> None:
    scorer = _FixedScorer(
        documents={"d": Document("d", "D")},
        experts={
            "j1": Expert("j1", "J1", capacity=1, seniority=0.2),
            "j2": Expert("j2", "J2", capacity=1, seniority=0.3),
        },
        scores=(_score("d", "j1", 0.9), _score("d", "j2", 0.8)),
    )

    plan = AssignmentEngine(scorer).assign(  # type: ignore[arg-type]
        reviewers_per_document=2,
        minimum_senior_reviewers=1,
        senior_threshold=0.75,
    )
    assert plan.diagnostics is not None
    diagnostic = plan.diagnostics.for_document("d")
    assert diagnostic is not None
    assert UnmetReason.SENIORITY_FLOOR in diagnostic.reason_codes
    assert diagnostic.evidence["senior_admissible_pairs"] == 0
    assert diagnostic.evidence["senior_assigned"] == 0


def test_sparse_matrix_shortage_is_explicit() -> None:
    scorer = _FixedScorer(
        documents={"d": Document("d", "D")},
        experts={
            "e1": Expert("e1", "E1", capacity=1),
            "e2": Expert("e2", "E2", capacity=1),
        },
        scores=(_score("d", "e1", 0.9),),
    )

    plan = AssignmentEngine(scorer).assign(reviewers_per_document=2)  # type: ignore[arg-type]
    assert plan.diagnostics is not None
    diagnostic = plan.diagnostics.for_document("d")
    assert diagnostic is not None
    assert UnmetReason.SPARSE_SCORE_MATRIX in diagnostic.reason_codes
    assert diagnostic.evidence["unscored_experts"] == 1


def test_sparse_matrix_reports_conflict_without_affinity_row() -> None:
    scorer = _FixedScorer(
        documents={"d": Document("d", "D")},
        experts={
            "conflict": Expert("conflict", "Conflict", capacity=1),
            "good": Expert("good", "Good", capacity=1),
        },
        scores=(_score("d", "good", 0.9),),
        conflict_pairs=frozenset({("d", "conflict")}),
    )

    plan = AssignmentEngine(scorer).assign(reviewers_per_document=2)  # type: ignore[arg-type]
    assert plan.diagnostics is not None
    diagnostic = plan.diagnostics.for_document("d")
    assert diagnostic is not None
    assert UnmetReason.HARD_CONFLICT in diagnostic.reason_codes
    assert UnmetReason.SPARSE_SCORE_MATRIX in diagnostic.reason_codes
    assert diagnostic.evidence["hard_conflict_pairs"] == 1
    assert diagnostic.evidence["unscored_experts"] == 1


def test_greedy_shortage_is_not_misreported_as_proven_infeasibility() -> None:
    scorer = _FixedScorer(
        documents={"d1": Document("d1", "D1"), "d2": Document("d2", "D2")},
        experts={
            "e1": Expert("e1", "E1", capacity=1),
            "e2": Expert("e2", "E2", capacity=1),
        },
        scores=(
            _score("d1", "e1", 0.9),
            _score("d1", "e2", 0.8),
            _score("d2", "e1", 0.7),
        ),
    )
    engine = AssignmentEngine(scorer)  # type: ignore[arg-type]

    greedy = engine.assign(strategy="greedy", reviewers_per_document=1)
    optimal = engine.assign(strategy="optimal", reviewers_per_document=1)

    assert len(optimal.assignments) == 2
    assert greedy.diagnostics is not None
    assert greedy.diagnostics.status is FeasibilityStatus.NOT_CERTIFIED
    assert not greedy.diagnostics.certified
    diagnostic = next(item for item in greedy.diagnostics.documents if item.unmet)
    assert UnmetReason.GREEDY_NOT_CERTIFIED in diagnostic.reason_codes


def _exhaustive_maximum(
    scorer: _FixedScorer, *, demand: int, minimum_score: float, diverse: bool
) -> int:
    choices: list[tuple[tuple[str, ...], ...]] = []
    for document_id in sorted(scorer.documents):
        candidates = tuple(
            score.expert_id
            for score in scorer.scores
            if score.document_id == document_id
            and score.eligible
            and score.total >= minimum_score
            and scorer.experts[score.expert_id].capacity > 0
        )
        document_choices = []
        for size in range(min(demand, len(candidates)) + 1):
            for selected in itertools.combinations(candidates, size):
                groups = [
                    scorer.experts[expert_id].institution or f"unknown:{expert_id}"
                    for expert_id in selected
                ]
                if not diverse or len(groups) == len(set(groups)):
                    document_choices.append(selected)
        choices.append(tuple(document_choices))

    maximum = 0
    for allocation in itertools.product(*choices):
        loads = {
            expert_id: sum(expert_id in selected for selected in allocation)
            for expert_id in scorer.experts
        }
        if any(loads[key] > scorer.experts[key].capacity for key in loads):
            continue
        maximum = max(maximum, sum(len(selected) for selected in allocation))
    return maximum


@pytest.mark.parametrize("diverse", [False, True])
def test_optimal_feasibility_diagnostics_match_independent_enumeration(diverse: bool) -> None:
    """An independent subset oracle checks both maximum flow and its diagnosis."""

    generator = random.Random(912 if diverse else 411)
    document_ids = ("d0", "d1", "d2")
    expert_ids = ("e0", "e1", "e2")
    demand = 2
    minimum_score = 0.4
    for _ in range(60):
        experts = {
            expert_id: Expert(
                expert_id,
                expert_id,
                capacity=generator.randint(0, 2),
                institution=("A", "A", "B")[index],
            )
            for index, expert_id in enumerate(expert_ids)
        }
        scores = tuple(
            _score(
                document_id,
                expert_id,
                generator.randint(0, 10) / 10,
                eligible=generator.random() >= 0.25,
            )
            for document_id in document_ids
            for expert_id in expert_ids
        )
        scorer = _FixedScorer(
            documents={key: Document(key, key) for key in document_ids},
            experts=experts,
            scores=scores,
        )

        plan = AssignmentEngine(scorer).assign(  # type: ignore[arg-type]
            reviewers_per_document=demand,
            minimum_score=minimum_score,
            require_distinct_institutions=diverse,
        )
        expected = _exhaustive_maximum(
            scorer, demand=demand, minimum_score=minimum_score, diverse=diverse
        )

        assert len(plan.assignments) == expected
        assert plan.diagnostics is not None
        assert plan.diagnostics.assigned == expected
        assert plan.diagnostics.unmet == demand * len(document_ids) - expected
        assert plan.diagnostics.status is (
            FeasibilityStatus.SATISFIED
            if expected == demand * len(document_ids)
            else FeasibilityStatus.INFEASIBLE
        )
        assert plan.diagnostics.certified
        for diagnostic in plan.diagnostics.documents:
            assert diagnostic.assigned == len(plan.for_document(diagnostic.document_id))
            assert diagnostic.assigned + diagnostic.unmet == diagnostic.requested


def test_cli_exports_and_summarizes_certified_diagnostics(tmp_path, capsys) -> None:
    documents = tmp_path / "documents.json"
    experts = tmp_path / "experts.json"
    config = tmp_path / "config.json"
    output = tmp_path / "plan.json"
    documents.write_text('[{"id":"d","title":"D"}]', encoding="utf-8")
    experts.write_text('[{"id":"e","name":"E","capacity":1}]', encoding="utf-8")
    config.write_text('{"reviewers_per_document":2}', encoding="utf-8")

    code = main(
        [
            "match",
            "--documents",
            str(documents),
            "--experts",
            str(experts),
            "--config",
            str(config),
            "--output",
            str(output),
        ]
    )

    exported = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert exported["diagnostics"]["status"] == "infeasible"
    assert exported["diagnostics"]["certified"] is True
    assert exported["diagnostics"]["unmet"] == 1
    assert "status=infeasible, unmet=1, certified=true" in capsys.readouterr().out


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DemandDiagnostic("", 1, 0, 1),
        lambda: DemandDiagnostic("d", True, 0, 1),
        lambda: DemandDiagnostic("d", 0, 0, 0),
        lambda: DemandDiagnostic("d", 2, 0, 1),
        lambda: DemandDiagnostic("d", 1, 0, 1, ("unknown",)),
        lambda: DemandDiagnostic(
            "d", 1, 0, 1, (UnmetReason.ZERO_CAPACITY, UnmetReason.ZERO_CAPACITY)
        ),
        lambda: DemandDiagnostic("d", 1, 1, 0, (UnmetReason.ZERO_CAPACITY,)),
        lambda: DemandDiagnostic("d", 1, 0, 1, evidence=[]),
        lambda: DemandDiagnostic("d", 1, 0, 1, evidence={"": 1}),
        lambda: DemandDiagnostic("d", 1, 0, 1, evidence={"count": True}),
        lambda: DemandDiagnostic("d", 1, 0, 1, saturated_experts=("",)),
        lambda: DemandDiagnostic("d", 1, 0, 1, saturated_experts=("e", "e")),
    ],
)
def test_demand_diagnostic_rejects_invalid_values(factory) -> None:
    with pytest.raises(DataValidationError):
        factory()


def _unmet_document(document_id: str = "d") -> DemandDiagnostic:
    return DemandDiagnostic(document_id, 1, 0, 1, (UnmetReason.CANDIDATE_SCARCITY,))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AssignmentDiagnostics("unknown", 1, 0, 1, (_unmet_document(),)),
        lambda: AssignmentDiagnostics(FeasibilityStatus.INFEASIBLE, True, 0, 1, ()),
        lambda: AssignmentDiagnostics(FeasibilityStatus.INFEASIBLE, 1, -1, 2, ()),
        lambda: AssignmentDiagnostics(FeasibilityStatus.INFEASIBLE, 2, 0, 1, ()),
        lambda: AssignmentDiagnostics(FeasibilityStatus.INFEASIBLE, 1, 0, 1, ("bad",)),
        lambda: AssignmentDiagnostics(
            FeasibilityStatus.INFEASIBLE, 2, 0, 2, (_unmet_document(), _unmet_document())
        ),
        lambda: AssignmentDiagnostics(FeasibilityStatus.INFEASIBLE, 2, 0, 2, (_unmet_document(),)),
        lambda: AssignmentDiagnostics(FeasibilityStatus.INFEASIBLE, 1, 1, 0, (_unmet_document(),)),
        lambda: AssignmentDiagnostics(
            FeasibilityStatus.INFEASIBLE,
            2,
            0,
            2,
            (_unmet_document("d1"), DemandDiagnostic("d2", 1, 1, 0)),
        ),
        lambda: AssignmentDiagnostics(FeasibilityStatus.SATISFIED, 1, 0, 1, (_unmet_document(),)),
        lambda: AssignmentDiagnostics(
            FeasibilityStatus.INFEASIBLE, 1, 1, 0, (DemandDiagnostic("d", 1, 1, 0),)
        ),
    ],
)
def test_assignment_diagnostics_reject_inconsistent_values(factory) -> None:
    with pytest.raises(DataValidationError):
        factory()


def test_match_plan_rejects_diagnostics_that_disagree_with_the_plan() -> None:
    diagnostics = AssignmentDiagnostics(
        FeasibilityStatus.INFEASIBLE, 1, 0, 1, (_unmet_document("other"),)
    )
    with pytest.raises(DataValidationError, match="cover"):
        MatchPlan((), {"d": 1}, "optimal", 0.0, diagnostics)

    diagnostics = AssignmentDiagnostics(
        FeasibilityStatus.INFEASIBLE, 2, 1, 1, (DemandDiagnostic("d", 2, 1, 1),)
    )
    with pytest.raises(DataValidationError, match="match assignments"):
        MatchPlan((), {"d": 1}, "optimal", 0.0, diagnostics)

    with pytest.raises(DataValidationError, match="AssignmentDiagnostics"):
        MatchPlan((), {}, "external", 0.0, "invalid")


def test_diagnostics_json_loader_is_strict() -> None:
    scorer = _FixedScorer(
        documents={"d": Document("d", "D")},
        experts={"e": Expert("e", "E", capacity=1)},
        scores=(_score("d", "e", 0.9),),
    )
    raw = plan_to_dict(
        AssignmentEngine(scorer).assign(reviewers_per_document=2)  # type: ignore[arg-type]
    )
    assert plan_from_dict(raw).diagnostics is not None

    mutations = []
    for path, value in (
        (("diagnostics",), []),
        (("diagnostics", "unexpected"), True),
        (("diagnostics", "documents"), {}),
        (("diagnostics", "documents", 0), "bad"),
        (("diagnostics", "documents", 0, "unexpected"), True),
        (("diagnostics", "documents", 0, "evidence"), None),
        (("diagnostics", "documents", 0, "document_id"), None),
        (("diagnostics", "documents", 0, "reason_codes"), ["unknown"]),
        (("diagnostics", "status"), "unknown"),
        (("diagnostics", "certified"), "yes"),
        (("diagnostics", "certified"), False),
    ):
        candidate = deepcopy(raw)
        target = candidate
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        mutations.append(candidate)

    for candidate in mutations:
        with pytest.raises(DataValidationError):
            plan_from_dict(candidate)


def test_legacy_plan_without_diagnostics_remains_supported() -> None:
    plan = plan_from_dict(
        {
            "assignments": [{"document_id": "d", "expert_id": "e", "score": 0.5, "rank": 1}],
            "strategy": "external",
        }
    )
    assert plan.diagnostics is None
    assert "diagnostics" not in plan_to_dict(plan)
