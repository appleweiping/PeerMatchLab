"""High-level orchestration for applications and the command line."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from peermatchlab.affinity import Affinity, AffinityScorer
from peermatchlab.assignment import AssignmentEngine, AssignmentStrategy
from peermatchlab.audit import AuditReport, audit_plan
from peermatchlab.config import MatchConfig
from peermatchlab.models import Conflict, Document, Expert, MatchPlan, MatchScore
from peermatchlab.scoring import MatchScorer, ScoreWeights


@dataclass(frozen=True, slots=True)
class MatchRun:
    """All important products of one reproducible matching run."""

    plan: MatchPlan
    audit: AuditReport
    scores: tuple[MatchScore, ...]


def run_matching(
    documents: Iterable[Document],
    experts: Iterable[Expert],
    *,
    conflicts: Iterable[Conflict] = (),
    config: MatchConfig | None = None,
) -> MatchRun:
    """Score, assign, and independently audit a complete matching run."""

    documents_tuple = tuple(documents)
    experts_tuple = tuple(experts)
    conflicts_tuple = tuple(conflicts)
    selected = config or MatchConfig()
    scorer = MatchScorer(
        documents_tuple,
        experts_tuple,
        conflicts=conflicts_tuple,
        weights=ScoreWeights.from_mapping(selected.weights),
        current_year=selected.current_year,
        publication_half_life=selected.publication_half_life,
    )
    plan = AssignmentEngine(scorer).assign(
        strategy=AssignmentStrategy(selected.strategy),
        reviewers_per_document=selected.reviewers_per_document,
        minimum_score=selected.minimum_score,
        require_distinct_institutions=selected.require_distinct_institutions,
        load_balance_penalty=selected.load_balance_penalty,
        minimum_senior_reviewers=selected.minimum_senior_reviewers,
        senior_threshold=selected.senior_threshold,
        fair_local_max_steps=selected.fair_local_max_steps,
        fair_local_max_checks=selected.fair_local_max_checks,
    )
    audit = audit_plan(
        plan,
        documents_tuple,
        experts_tuple,
        conflicts_tuple,
        default_demand=selected.reviewers_per_document,
        require_distinct_institutions=selected.require_distinct_institutions,
    )
    return MatchRun(plan=plan, audit=audit, scores=scorer.matrix())


def run_affinity_matching(
    documents: Iterable[Document],
    experts: Iterable[Expert],
    affinities: Iterable[Affinity],
    *,
    conflicts: Iterable[Conflict] = (),
    config: MatchConfig | None = None,
) -> MatchRun:
    """Assign a sparse external affinity matrix under the same hard constraints."""

    documents_tuple = tuple(documents)
    experts_tuple = tuple(experts)
    conflicts_tuple = tuple(conflicts)
    selected = config or MatchConfig()
    scorer = AffinityScorer(
        documents_tuple,
        experts_tuple,
        affinities,
        conflicts=conflicts_tuple,
    )
    plan = AssignmentEngine(scorer).assign(
        strategy=AssignmentStrategy(selected.strategy),
        reviewers_per_document=selected.reviewers_per_document,
        minimum_score=selected.minimum_score,
        require_distinct_institutions=selected.require_distinct_institutions,
        load_balance_penalty=selected.load_balance_penalty,
        minimum_senior_reviewers=selected.minimum_senior_reviewers,
        senior_threshold=selected.senior_threshold,
        fair_local_max_steps=selected.fair_local_max_steps,
        fair_local_max_checks=selected.fair_local_max_checks,
    )
    audit = audit_plan(
        plan,
        documents_tuple,
        experts_tuple,
        conflicts_tuple,
        default_demand=selected.reviewers_per_document,
        require_distinct_institutions=selected.require_distinct_institutions,
    )
    return MatchRun(plan=plan, audit=audit, scores=scorer.matrix())
