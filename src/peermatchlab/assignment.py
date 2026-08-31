"""Capacity-constrained assignment strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from peermatchlab.models import Assignment, MatchPlan, MatchScore
from peermatchlab.scoring import MatchScorer


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


class AssignmentStrategy(StrEnum):
    """Supported global selection strategies."""

    OPTIMAL = "optimal"
    GREEDY = "greedy"


@dataclass(slots=True)
class _Edge:
    to: int
    reverse: int
    capacity: int
    cost: int


class _FlowNetwork:
    """A minimal integral min-cost-flow solver for bipartite assignment."""

    def __init__(self, size: int) -> None:
        self.graph: list[list[_Edge]] = [[] for _ in range(size)]

    def add_edge(self, source: int, target: int, capacity: int, cost: int) -> _Edge:
        forward = _Edge(target, len(self.graph[target]), capacity, cost)
        reverse = _Edge(source, len(self.graph[source]), 0, -cost)
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return forward

    def min_cost_flow(self, source: int, sink: int, requested: int) -> tuple[int, int]:
        """Send up to ``requested`` units using Bellman-Ford augmenting paths."""

        sent = 0
        total_cost = 0
        node_count = len(self.graph)
        infinity = 10**30
        while sent < requested:
            distance = [infinity] * node_count
            previous_node = [-1] * node_count
            previous_edge = [-1] * node_count
            distance[source] = 0
            # The residual network is small and may contain negative reverse edges.
            for _ in range(node_count - 1):
                changed = False
                for node, edges in enumerate(self.graph):
                    if distance[node] == infinity:
                        continue
                    for index, edge in enumerate(edges):
                        candidate = distance[node] + edge.cost
                        if edge.capacity > 0 and candidate < distance[edge.to]:
                            distance[edge.to] = candidate
                            previous_node[edge.to] = node
                            previous_edge[edge.to] = index
                            changed = True
                if not changed:
                    break
            if distance[sink] == infinity:
                break
            amount = requested - sent
            cursor = sink
            while cursor != source:
                node = previous_node[cursor]
                if node < 0:
                    amount = 0
                    break
                edge = self.graph[node][previous_edge[cursor]]
                amount = min(amount, edge.capacity)
                cursor = node
            if amount == 0:
                break
            cursor = sink
            while cursor != source:
                node = previous_node[cursor]
                edge = self.graph[node][previous_edge[cursor]]
                edge.capacity -= amount
                self.graph[cursor][edge.reverse].capacity += amount
                cursor = node
            sent += amount
            total_cost += amount * distance[sink]
        return sent, total_cost


class AssignmentEngine:
    """Assign scored experts while respecting hard exclusions and capacities."""

    def __init__(self, scorer: MatchScorer) -> None:
        self.scorer = scorer

    def assign(
        self,
        *,
        strategy: AssignmentStrategy | str = AssignmentStrategy.OPTIMAL,
        reviewers_per_document: int = 2,
        minimum_score: float = 0.0,
        require_distinct_institutions: bool = False,
    ) -> MatchPlan:
        """Create a deterministic assignment plan.

        Document-level ``required_experts`` values override the run-wide default.
        Institution diversity is handled by the greedy strategy because it is a
        per-document group constraint rather than a simple flow capacity.
        """

        if isinstance(reviewers_per_document, bool) or not isinstance(reviewers_per_document, int):
            raise ValueError("reviewers_per_document must be an integer")
        if reviewers_per_document < 1:
            raise ValueError("reviewers_per_document must be positive")
        if not _finite_number(minimum_score):
            raise ValueError("minimum_score must be a finite number")
        if not isinstance(require_distinct_institutions, bool):
            raise ValueError("require_distinct_institutions must be a boolean")
        selected = AssignmentStrategy(strategy)
        scores = tuple(score for score in self.scorer.matrix() if score.eligible)
        if require_distinct_institutions:
            return self._greedy(
                scores,
                reviewers_per_document,
                minimum_score,
                require_distinct_institutions=True,
                label="greedy-diverse",
            )
        if selected is AssignmentStrategy.GREEDY:
            return self._greedy(scores, reviewers_per_document, minimum_score)
        return self._optimal(scores, reviewers_per_document, minimum_score)

    def _demand(self, document_id: str, default: int) -> int:
        requested = self.scorer.documents[document_id].required_experts
        return requested if requested is not None else default

    def _greedy(
        self,
        scores: tuple[MatchScore, ...],
        reviewers_per_document: int,
        minimum_score: float,
        *,
        require_distinct_institutions: bool = False,
        label: str = "greedy",
    ) -> MatchPlan:
        by_document: dict[str, list[MatchScore]] = {key: [] for key in self.scorer.documents}
        for score in scores:
            if score.total >= minimum_score:
                by_document[score.document_id].append(score)
        for candidates in by_document.values():
            candidates.sort(key=lambda item: (-item.total, item.expert_id))

        remaining = {key: expert.capacity for key, expert in self.scorer.experts.items()}
        chosen: dict[str, list[MatchScore]] = {key: [] for key in self.scorer.documents}
        maximum_rounds = max(
            self._demand(document_id, reviewers_per_document)
            for document_id in self.scorer.documents
        )
        for round_index in range(maximum_rounds):
            for document_id in sorted(self.scorer.documents):
                if round_index >= self._demand(document_id, reviewers_per_document):
                    continue
                used_experts = {item.expert_id for item in chosen[document_id]}
                used_institutions = {
                    self.scorer.experts[item.expert_id].institution
                    for item in chosen[document_id]
                    if self.scorer.experts[item.expert_id].institution is not None
                }
                for candidate in by_document[document_id]:
                    expert = self.scorer.experts[candidate.expert_id]
                    if candidate.expert_id in used_experts or remaining[candidate.expert_id] == 0:
                        continue
                    if (
                        require_distinct_institutions
                        and expert.institution is not None
                        and expert.institution in used_institutions
                    ):
                        continue
                    chosen[document_id].append(candidate)
                    remaining[candidate.expert_id] -= 1
                    break
        return self._to_plan(chosen, reviewers_per_document, label)

    def _optimal(
        self,
        scores: tuple[MatchScore, ...],
        reviewers_per_document: int,
        minimum_score: float,
    ) -> MatchPlan:
        document_ids = sorted(self.scorer.documents)
        expert_ids = sorted(self.scorer.experts)
        source = 0
        document_offset = 1
        expert_offset = document_offset + len(document_ids)
        sink = expert_offset + len(expert_ids)
        network = _FlowNetwork(sink + 1)
        document_nodes = {key: document_offset + index for index, key in enumerate(document_ids)}
        expert_nodes = {key: expert_offset + index for index, key in enumerate(expert_ids)}
        requested = 0
        for document_id in document_ids:
            demand = self._demand(document_id, reviewers_per_document)
            network.add_edge(source, document_nodes[document_id], demand, 0)
            requested += demand
        for expert_id in expert_ids:
            network.add_edge(
                expert_nodes[expert_id], sink, self.scorer.experts[expert_id].capacity, 0
            )
        score_lookup: dict[tuple[str, str], MatchScore] = {}
        selection_edges: dict[tuple[str, str], _Edge] = {}
        # Keep the score objective lexicographically ahead of the aggregate tie
        # breaker.  A fixed multiplier would let expert indexes change the chosen
        # score once the expert pool (or number of assignments) became large enough.
        maximum_tie_total = requested * max(0, len(expert_ids) - 1)
        score_multiplier = maximum_tie_total + 1
        for score in scores:
            if score.total < minimum_score:
                continue
            key = (score.document_id, score.expert_id)
            score_lookup[key] = score
            # A stable expert-index tie breaker only resolves rounded-score ties.
            tie_breaker = expert_ids.index(score.expert_id)
            cost = -round(score.total * 1_000_000) * score_multiplier + tie_breaker
            selection_edges[key] = network.add_edge(
                document_nodes[score.document_id], expert_nodes[score.expert_id], 1, cost
            )
        network.min_cost_flow(source, sink, requested)
        chosen: dict[str, list[MatchScore]] = {key: [] for key in document_ids}
        for key, edge in selection_edges.items():
            if edge.capacity == 0:
                chosen[key[0]].append(score_lookup[key])
        return self._to_plan(chosen, reviewers_per_document, "optimal")

    def _to_plan(
        self,
        chosen: dict[str, list[MatchScore]],
        reviewers_per_document: int,
        strategy: str,
    ) -> MatchPlan:
        assignments: list[Assignment] = []
        unmet: dict[str, int] = {}
        for document_id in sorted(self.scorer.documents):
            ranked = sorted(chosen[document_id], key=lambda item: (-item.total, item.expert_id))
            demand = self._demand(document_id, reviewers_per_document)
            unmet_count = max(0, demand - len(ranked))
            if unmet_count:
                unmet[document_id] = unmet_count
            assignments.extend(
                Assignment(
                    document_id=document_id,
                    expert_id=score.expert_id,
                    score=score.total,
                    rank=rank,
                    components=score.component_map(),
                )
                for rank, score in enumerate(ranked, start=1)
            )
        return MatchPlan(
            assignments=tuple(assignments),
            unmet=unmet,
            strategy=strategy,
            total_score=sum(item.score for item in assignments),
        )
