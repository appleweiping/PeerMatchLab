"""Capacity-constrained assignment strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
from typing import Protocol

from peermatchlab.fair_local import improve_fair_local
from peermatchlab.models import (
    Assignment,
    AssignmentDiagnostics,
    DemandDiagnostic,
    Document,
    Expert,
    FeasibilityStatus,
    MatchPlan,
    MatchScore,
    UnmetReason,
)


class AssignmentScorer(Protocol):
    """Minimum score source required by the assignment engine."""

    documents: dict[str, Document]
    experts: dict[str, Expert]

    def matrix(self) -> tuple[MatchScore, ...]: ...


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
    MINMAX = "minmax"
    MAXIMIN = "maximin"
    FAIR_LOCAL = "fair-local"


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

    def __init__(self, scorer: AssignmentScorer) -> None:
        self.scorer = scorer

    def assign(
        self,
        *,
        strategy: AssignmentStrategy | str = AssignmentStrategy.OPTIMAL,
        reviewers_per_document: int = 2,
        minimum_score: float = 0.0,
        require_distinct_institutions: bool = False,
        load_balance_penalty: float = 0.0,
        minimum_senior_reviewers: int = 0,
        senior_threshold: float = 0.75,
        fair_local_max_steps: int = 32,
        fair_local_max_checks: int = 200_000,
    ) -> MatchPlan:
        """Create a deterministic assignment plan.

        Document-level ``required_experts`` values override the run-wide default.
        For optimal assignment, institution diversity is represented by one
        capacity-one node per document and institution. Experts without a known
        institution receive individual group nodes, so missing metadata does not
        create a false shared affiliation.

        ``minimum_senior_reviewers`` reserves that many of each document's slots
        for experts whose ``seniority`` is at least ``senior_threshold``. The
        reservation is a hard constraint, not a preference: a slot it cannot fill
        is reported as unmet rather than given to a junior expert.
        """

        if isinstance(reviewers_per_document, bool) or not isinstance(reviewers_per_document, int):
            raise ValueError("reviewers_per_document must be an integer")
        if reviewers_per_document < 1:
            raise ValueError("reviewers_per_document must be positive")
        if not _finite_number(minimum_score):
            raise ValueError("minimum_score must be a finite number")
        if not isinstance(require_distinct_institutions, bool):
            raise ValueError("require_distinct_institutions must be a boolean")
        if not _finite_number(load_balance_penalty) or not 0 <= load_balance_penalty <= 1:
            raise ValueError("load_balance_penalty must be a finite number between 0 and 1")
        if isinstance(minimum_senior_reviewers, bool) or not isinstance(
            minimum_senior_reviewers, int
        ):
            raise ValueError("minimum_senior_reviewers must be an integer")
        if minimum_senior_reviewers < 0:
            raise ValueError("minimum_senior_reviewers must not be negative")
        if not _finite_number(senior_threshold) or not 0 <= senior_threshold <= 1:
            raise ValueError("senior_threshold must be a finite number between 0 and 1")
        if isinstance(fair_local_max_steps, bool) or not isinstance(fair_local_max_steps, int):
            raise ValueError("fair_local_max_steps must be an integer")
        if not 1 <= fair_local_max_steps <= 128:
            raise ValueError("fair_local_max_steps must be between 1 and 128")
        if isinstance(fair_local_max_checks, bool) or not isinstance(fair_local_max_checks, int):
            raise ValueError("fair_local_max_checks must be an integer")
        if not 1 <= fair_local_max_checks <= 2_000_000:
            raise ValueError("fair_local_max_checks must be between 1 and 2000000")
        if minimum_senior_reviewers and require_distinct_institutions:
            # The two constraints cannot both be expressed exactly in one
            # min-cost flow. Reserving senior slots works by giving those units
            # their own arcs from the source, which may reach only senior pair
            # nodes; institution diversity needs a capacity-one gate between the
            # document and those pairs, and a unit passing through that gate no
            # longer carries which side of the split it came from. A network
            # that merged them would satisfy one constraint and quietly relax
            # the other, so the combination is refused rather than approximated.
            raise ValueError(
                "minimum_senior_reviewers cannot be combined with "
                "require_distinct_institutions: the two constraints are not "
                "jointly expressible in this flow network"
            )
        selected = AssignmentStrategy(strategy)
        if selected is not AssignmentStrategy.FAIR_LOCAL and (
            fair_local_max_steps != 32 or fair_local_max_checks != 200_000
        ):
            raise ValueError("fair-local resource controls require strategy='fair-local'")
        if selected is AssignmentStrategy.MAXIMIN and (
            len(self.scorer.documents) > 6 or len(self.scorer.experts) > 8
        ):
            raise ValueError(
                "maximin supports at most 6 documents, 8 experts, and 16 eligible pairs"
            )
        if selected is AssignmentStrategy.MAXIMIN and load_balance_penalty:
            raise ValueError("maximin does not support load_balance_penalty")
        if selected is AssignmentStrategy.FAIR_LOCAL and load_balance_penalty:
            raise ValueError("fair-local does not support load_balance_penalty")
        if selected is AssignmentStrategy.FAIR_LOCAL and (
            len(self.scorer.documents) > 128 or len(self.scorer.experts) > 256
        ):
            raise ValueError(
                "fair-local supports at most 128 documents, 256 experts, and 4096 eligible pairs"
            )
        if selected is AssignmentStrategy.FAIR_LOCAL and (
            sum(self._demand(key, reviewers_per_document) for key in self.scorer.documents) > 512
            or sum(expert.capacity for expert in self.scorer.experts.values()) > 4096
        ):
            raise ValueError(
                "fair-local supports at most 512 requested slots and 4096 total expert capacity"
            )
        all_scores = self.scorer.matrix()
        scores = tuple(score for score in all_scores if score.eligible)
        if selected is AssignmentStrategy.FAIR_LOCAL:
            return self._fair_local(
                scores,
                all_scores,
                reviewers_per_document,
                minimum_score,
                require_distinct_institutions=require_distinct_institutions,
                minimum_senior_reviewers=minimum_senior_reviewers,
                senior_threshold=float(senior_threshold),
                max_steps=fair_local_max_steps,
                max_checks=fair_local_max_checks,
            )
        if selected is AssignmentStrategy.MAXIMIN:
            return self._maximin(
                scores,
                all_scores,
                reviewers_per_document,
                minimum_score,
                require_distinct_institutions=require_distinct_institutions,
                minimum_senior_reviewers=minimum_senior_reviewers,
                senior_threshold=float(senior_threshold),
            )
        if selected is AssignmentStrategy.MINMAX:
            return self._minmax(
                scores,
                all_scores,
                reviewers_per_document,
                minimum_score,
                require_distinct_institutions=require_distinct_institutions,
                load_balance_penalty=float(load_balance_penalty),
                minimum_senior_reviewers=minimum_senior_reviewers,
                senior_threshold=float(senior_threshold),
            )
        if selected is AssignmentStrategy.GREEDY:
            return self._greedy(
                scores,
                all_scores,
                reviewers_per_document,
                minimum_score,
                require_distinct_institutions=require_distinct_institutions,
                load_balance_penalty=float(load_balance_penalty),
                minimum_senior_reviewers=minimum_senior_reviewers,
                senior_threshold=float(senior_threshold),
                label=self._strategy_label(
                    "greedy",
                    require_distinct_institutions,
                    load_balance_penalty > 0,
                    minimum_senior_reviewers > 0,
                ),
            )
        return self._optimal(
            scores,
            all_scores,
            reviewers_per_document,
            minimum_score,
            require_distinct_institutions=require_distinct_institutions,
            load_balance_penalty=float(load_balance_penalty),
            minimum_senior_reviewers=minimum_senior_reviewers,
            senior_threshold=float(senior_threshold),
        )

    @staticmethod
    def _strategy_label(strategy: str, diverse: bool, balanced: bool, senior: bool = False) -> str:
        qualifiers = [
            name
            for name, enabled in (
                ("diverse", diverse),
                ("balanced", balanced),
                ("senior", senior),
            )
            if enabled
        ]
        return "-".join((strategy, *qualifiers))

    def _demand(self, document_id: str, default: int) -> int:
        requested = self.scorer.documents[document_id].required_experts
        return requested if requested is not None else default

    def _greedy(
        self,
        scores: tuple[MatchScore, ...],
        all_scores: tuple[MatchScore, ...],
        reviewers_per_document: int,
        minimum_score: float,
        *,
        require_distinct_institutions: bool = False,
        load_balance_penalty: float = 0.0,
        minimum_senior_reviewers: int = 0,
        senior_threshold: float = 0.75,
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
                # Rounds still to run for this document, counting the current
                # one. While at least that many senior slots remain unfilled,
                # every remaining round is reserved, so only senior experts are
                # considered -- the baseline honours the same hard floor the
                # solver does, rather than merely preferring senior experts.
                demand = self._demand(document_id, reviewers_per_document)
                reserved = min(minimum_senior_reviewers, demand)
                seniors_chosen = sum(
                    1
                    for item in chosen[document_id]
                    if self.scorer.experts[item.expert_id].seniority >= senior_threshold
                )
                rounds_left = demand - round_index
                senior_only = reserved - seniors_chosen >= rounds_left
                available_candidates: list[MatchScore] = []
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
                    if senior_only and expert.seniority < senior_threshold:
                        continue
                    available_candidates.append(candidate)
                if available_candidates:
                    candidate = min(
                        available_candidates,
                        key=lambda item: (
                            -(
                                item.total
                                - load_balance_penalty
                                * (
                                    self.scorer.experts[item.expert_id].capacity
                                    - remaining[item.expert_id]
                                )
                            ),
                            -item.total,
                            item.expert_id,
                        ),
                    )
                    chosen[document_id].append(candidate)
                    remaining[candidate.expert_id] -= 1
        return self._to_plan(
            chosen,
            all_scores,
            reviewers_per_document,
            minimum_score,
            label,
            require_distinct_institutions=require_distinct_institutions,
            minimum_senior_reviewers=minimum_senior_reviewers,
            senior_threshold=senior_threshold,
            maximum_cardinality_certified=False,
        )

    def _optimal(
        self,
        scores: tuple[MatchScore, ...],
        all_scores: tuple[MatchScore, ...],
        reviewers_per_document: int,
        minimum_score: float,
        *,
        require_distinct_institutions: bool = False,
        load_balance_penalty: float = 0.0,
        minimum_senior_reviewers: int = 0,
        senior_threshold: float = 0.75,
        capacity_overrides: dict[str, int] | None = None,
        strategy_label: str | None = None,
    ) -> MatchPlan:
        document_ids = sorted(self.scorer.documents)
        expert_ids = sorted(self.scorer.experts)
        candidate_scores = tuple(score for score in scores if score.total >= minimum_score)
        if minimum_senior_reviewers:
            return self._optimal_with_senior_floor(
                candidate_scores,
                all_scores,
                document_ids,
                expert_ids,
                reviewers_per_document,
                minimum_score,
                load_balance_penalty=load_balance_penalty,
                minimum_senior_reviewers=minimum_senior_reviewers,
                senior_threshold=senior_threshold,
                capacity_overrides=capacity_overrides,
                strategy_label=strategy_label,
            )

        def group_key(score: MatchScore) -> tuple[str, str, str]:
            institution = self.scorer.experts[score.expert_id].institution
            if institution is None:
                return (score.document_id, "expert", score.expert_id)
            return (score.document_id, "institution", institution)

        group_keys = (
            sorted({group_key(score) for score in candidate_scores})
            if require_distinct_institutions
            else []
        )
        source = 0
        document_offset = 1
        group_offset = document_offset + len(document_ids)
        expert_offset = group_offset + len(group_keys)
        sink = expert_offset + len(expert_ids)
        network = _FlowNetwork(sink + 1)
        document_nodes = {key: document_offset + index for index, key in enumerate(document_ids)}
        group_nodes = {key: group_offset + index for index, key in enumerate(group_keys)}
        expert_nodes = {key: expert_offset + index for index, key in enumerate(expert_ids)}
        requested = 0
        for document_id in document_ids:
            demand = self._demand(document_id, reviewers_per_document)
            network.add_edge(source, document_nodes[document_id], demand, 0)
            requested += demand
        for diversity_key, node in group_nodes.items():
            network.add_edge(document_nodes[diversity_key[0]], node, 1, 0)
        score_lookup: dict[tuple[str, str], MatchScore] = {}
        selection_edges: dict[tuple[str, str], _Edge] = {}
        # Keep the score objective lexicographically ahead of the aggregate tie
        # breaker.  A fixed multiplier would let expert indexes change the chosen
        # score once the expert pool (or number of assignments) became large enough.
        maximum_tie_total = requested * max(0, len(expert_ids) - 1)
        score_multiplier = maximum_tie_total + 1
        expert_indexes = {expert_id: index for index, expert_id in enumerate(expert_ids)}
        for expert_id in expert_ids:
            capacity = (
                capacity_overrides.get(expert_id, self.scorer.experts[expert_id].capacity)
                if capacity_overrides is not None
                else self.scorer.experts[expert_id].capacity
            )
            for slot in range(capacity):
                marginal_penalty = round(load_balance_penalty * slot * 1_000_000)
                network.add_edge(
                    expert_nodes[expert_id], sink, 1, marginal_penalty * score_multiplier
                )
        for score in candidate_scores:
            pair = (score.document_id, score.expert_id)
            score_lookup[pair] = score
            # A stable expert-index tie breaker only resolves rounded-score ties.
            tie_breaker = expert_indexes[score.expert_id]
            cost = -round(score.total * 1_000_000) * score_multiplier + tie_breaker
            from_node = (
                group_nodes[group_key(score)]
                if require_distinct_institutions
                else document_nodes[score.document_id]
            )
            selection_edges[pair] = network.add_edge(
                from_node, expert_nodes[score.expert_id], 1, cost
            )
        network.min_cost_flow(source, sink, requested)
        chosen: dict[str, list[MatchScore]] = {key: [] for key in document_ids}
        for pair, edge in selection_edges.items():
            if edge.capacity == 0:
                chosen[pair[0]].append(score_lookup[pair])
        return self._to_plan(
            chosen,
            all_scores,
            reviewers_per_document,
            minimum_score,
            strategy_label
            or self._strategy_label(
                "optimal", require_distinct_institutions, load_balance_penalty > 0
            ),
            require_distinct_institutions=require_distinct_institutions,
            minimum_senior_reviewers=0,
            senior_threshold=senior_threshold,
            maximum_cardinality_certified=True,
        )

    def _optimal_with_senior_floor(
        self,
        candidate_scores: tuple[MatchScore, ...],
        all_scores: tuple[MatchScore, ...],
        document_ids: list[str],
        expert_ids: list[str],
        reviewers_per_document: int,
        minimum_score: float,
        *,
        load_balance_penalty: float,
        minimum_senior_reviewers: int,
        senior_threshold: float,
        capacity_overrides: dict[str, int] | None = None,
        strategy_label: str | None = None,
    ) -> MatchPlan:
        """Solve with a hard floor on senior reviewers per document.

        The floor is enforced by splitting each document's demand at the source:
        reserved units leave through an arc that reaches only senior pair nodes,
        so they cannot be spent on a junior expert, while the remaining units
        reach every eligible pair. Both sides meet at one capacity-one node per
        document-expert pair, which is what keeps an expert from filling a
        reserved slot and a free slot for the same document.

        The objective is unchanged, so within the reservation the solver still
        maximizes total evidence. A reserved unit that no senior expert can
        absorb simply does not flow and is reported as unmet, the same way
        insufficient eligible capacity already is.
        """

        def is_senior(expert_id: str) -> bool:
            return self.scorer.experts[expert_id].seniority >= senior_threshold

        source = 0
        reserved_offset = 1
        free_offset = reserved_offset + len(document_ids)
        pair_offset = free_offset + len(document_ids)
        pairs = sorted({(score.document_id, score.expert_id) for score in candidate_scores})
        expert_offset = pair_offset + len(pairs)
        sink = expert_offset + len(expert_ids)
        network = _FlowNetwork(sink + 1)
        reserved_nodes = {key: reserved_offset + index for index, key in enumerate(document_ids)}
        free_nodes = {key: free_offset + index for index, key in enumerate(document_ids)}
        pair_nodes = {key: pair_offset + index for index, key in enumerate(pairs)}
        expert_nodes = {key: expert_offset + index for index, key in enumerate(expert_ids)}

        requested = 0
        for document_id in document_ids:
            demand = self._demand(document_id, reviewers_per_document)
            reserved = min(minimum_senior_reviewers, demand)
            network.add_edge(source, reserved_nodes[document_id], reserved, 0)
            network.add_edge(source, free_nodes[document_id], demand - reserved, 0)
            requested += demand

        maximum_tie_total = requested * max(0, len(expert_ids) - 1)
        score_multiplier = maximum_tie_total + 1
        expert_indexes = {expert_id: index for index, expert_id in enumerate(expert_ids)}
        for expert_id in expert_ids:
            capacity = (
                capacity_overrides.get(expert_id, self.scorer.experts[expert_id].capacity)
                if capacity_overrides is not None
                else self.scorer.experts[expert_id].capacity
            )
            for slot in range(capacity):
                marginal_penalty = round(load_balance_penalty * slot * 1_000_000)
                network.add_edge(
                    expert_nodes[expert_id], sink, 1, marginal_penalty * score_multiplier
                )

        score_lookup: dict[tuple[str, str], MatchScore] = {}
        selection_edges: dict[tuple[str, str], _Edge] = {}
        for score in candidate_scores:
            pair = (score.document_id, score.expert_id)
            score_lookup[pair] = score
            pair_node = pair_nodes[pair]
            tie_breaker = expert_indexes[score.expert_id]
            cost = -round(score.total * 1_000_000) * score_multiplier + tie_breaker
            network.add_edge(free_nodes[score.document_id], pair_node, 1, 0)
            if is_senior(score.expert_id):
                network.add_edge(reserved_nodes[score.document_id], pair_node, 1, 0)
            selection_edges[pair] = network.add_edge(
                pair_node, expert_nodes[score.expert_id], 1, cost
            )
        network.min_cost_flow(source, sink, requested)
        chosen: dict[str, list[MatchScore]] = {key: [] for key in document_ids}
        for pair, edge in selection_edges.items():
            if edge.capacity == 0:
                chosen[pair[0]].append(score_lookup[pair])
        return self._to_plan(
            chosen,
            all_scores,
            reviewers_per_document,
            minimum_score,
            strategy_label
            or self._strategy_label("optimal", False, load_balance_penalty > 0, True),
            require_distinct_institutions=False,
            minimum_senior_reviewers=minimum_senior_reviewers,
            senior_threshold=senior_threshold,
            maximum_cardinality_certified=True,
        )

    def _minmax(
        self,
        scores: tuple[MatchScore, ...],
        all_scores: tuple[MatchScore, ...],
        reviewers_per_document: int,
        minimum_score: float,
        *,
        require_distinct_institutions: bool,
        load_balance_penalty: float,
        minimum_senior_reviewers: int,
        senior_threshold: float,
    ) -> MatchPlan:
        """Minimize the largest reviewer load without sacrificing cardinality.

        The feasibility predicate is monotone in a per-expert cap.  We first
        obtain the maximum cardinality available under the declared capacities,
        then binary-search the smallest cap that still reaches that cardinality.
        A final min-cost flow under those clipped capacities maximizes evidence
        while preserving the proven min-max load bound.
        """

        expert_ids = sorted(self.scorer.experts)
        capacities = {
            expert_id: self.scorer.experts[expert_id].capacity for expert_id in expert_ids
        }
        baseline = self._optimal(
            scores,
            all_scores,
            reviewers_per_document,
            minimum_score,
            require_distinct_institutions=require_distinct_institutions,
            load_balance_penalty=load_balance_penalty,
            minimum_senior_reviewers=minimum_senior_reviewers,
            senior_threshold=senior_threshold,
            strategy_label="minmax",
        )
        target = len(baseline.assignments)
        if target == 0 or not expert_ids:
            return baseline

        low = 0
        high = max(capacities.values(), default=0)
        best = capacities
        while low <= high:
            cap = (low + high) // 2
            clipped = {expert_id: min(capacity, cap) for expert_id, capacity in capacities.items()}
            candidate = self._optimal(
                scores,
                all_scores,
                reviewers_per_document,
                minimum_score,
                require_distinct_institutions=require_distinct_institutions,
                load_balance_penalty=load_balance_penalty,
                minimum_senior_reviewers=minimum_senior_reviewers,
                senior_threshold=senior_threshold,
                capacity_overrides=clipped,
                strategy_label="minmax",
            )
            if len(candidate.assignments) >= target:
                best = clipped
                high = cap - 1
            else:
                low = cap + 1

        return self._optimal(
            scores,
            all_scores,
            reviewers_per_document,
            minimum_score,
            require_distinct_institutions=require_distinct_institutions,
            load_balance_penalty=load_balance_penalty,
            minimum_senior_reviewers=minimum_senior_reviewers,
            senior_threshold=senior_threshold,
            capacity_overrides=best,
            strategy_label="minmax",
        )

    def _maximin(
        self,
        scores: tuple[MatchScore, ...],
        all_scores: tuple[MatchScore, ...],
        reviewers_per_document: int,
        minimum_score: float,
        *,
        require_distinct_institutions: bool,
        minimum_senior_reviewers: int,
        senior_threshold: float,
    ) -> MatchPlan:
        """Exactly maximize the weakest document's total score on small instances.

        Cardinality precedes fairness: no slot is left unfilled merely to raise
        the minimum. Only then do weakest-document score, total score, and a
        stable pair ordering break ties. This exhaustive solver is deliberately
        bounded rather than pretending to be a scalable FairFlow replacement.
        """

        document_ids = sorted(self.scorer.documents)
        expert_ids = sorted(self.scorer.experts)
        admissible = tuple(score for score in scores if score.total >= minimum_score)
        if len(document_ids) > 6 or len(expert_ids) > 8 or len(admissible) > 16:
            raise ValueError(
                "maximin supports at most 6 documents, 8 experts, and 16 eligible pairs"
            )
        pairs = {(score.document_id, score.expert_id) for score in admissible}
        if len(pairs) != len(admissible):
            raise ValueError("maximin requires at most one eligible score per document-expert pair")

        options: list[tuple[tuple[MatchScore, ...], ...]] = []
        for document_id in document_ids:
            candidates = sorted(
                (score for score in admissible if score.document_id == document_id),
                key=lambda score: score.expert_id,
            )
            demand = self._demand(document_id, reviewers_per_document)
            reserved = min(minimum_senior_reviewers, demand)
            free_slots = demand - reserved
            selections: list[tuple[MatchScore, ...]] = []
            for size in range(min(demand, len(candidates)) + 1):
                for selection in combinations(candidates, size):
                    if (
                        sum(
                            self.scorer.experts[score.expert_id].seniority < senior_threshold
                            for score in selection
                        )
                        > free_slots
                    ):
                        continue
                    if require_distinct_institutions:
                        known = [
                            self.scorer.experts[score.expert_id].institution
                            for score in selection
                            if self.scorer.experts[score.expert_id].institution is not None
                        ]
                        if len(known) != len(set(known)):
                            continue
                    selections.append(selection)
            options.append(tuple(selections))

        capacities = {
            expert_id: self.scorer.experts[expert_id].capacity for expert_id in expert_ids
        }
        loads = dict.fromkeys(expert_ids, 0)
        chosen: dict[str, tuple[MatchScore, ...]] = {}
        best_key: tuple[int, float, float] | None = None
        best_pairs: tuple[tuple[str, str], ...] | None = None
        best_chosen: dict[str, tuple[MatchScore, ...]] = {}

        def visit(index: int) -> None:
            nonlocal best_key, best_pairs, best_chosen
            if index == len(document_ids):
                selections = tuple(chosen[document_id] for document_id in document_ids)
                cardinality = sum(len(selection) for selection in selections)
                document_totals = tuple(
                    math.fsum(score.total for score in selection) for selection in selections
                )
                weakest = min(document_totals, default=0.0)
                total = math.fsum(document_totals)
                key = (cardinality, weakest, total)
                pairs = tuple(
                    (document_id, score.expert_id)
                    for document_id, selection in zip(document_ids, selections, strict=True)
                    for score in selection
                )
                if (
                    best_key is None
                    or key > best_key
                    or (key == best_key and best_pairs is not None and pairs < best_pairs)
                ):
                    best_key, best_pairs, best_chosen = key, pairs, dict(chosen)
                return
            document_id = document_ids[index]
            for selection in options[index]:
                if any(
                    loads[score.expert_id] >= capacities[score.expert_id] for score in selection
                ):
                    continue
                chosen[document_id] = selection
                for score in selection:
                    loads[score.expert_id] += 1
                visit(index + 1)
                for score in selection:
                    loads[score.expert_id] -= 1

        visit(0)
        return self._to_plan(
            {document_id: list(best_chosen.get(document_id, ())) for document_id in document_ids},
            all_scores,
            reviewers_per_document,
            minimum_score,
            "maximin",
            require_distinct_institutions=require_distinct_institutions,
            minimum_senior_reviewers=minimum_senior_reviewers,
            senior_threshold=senior_threshold,
            maximum_cardinality_certified=True,
        )

    def _fair_local(
        self,
        scores: tuple[MatchScore, ...],
        all_scores: tuple[MatchScore, ...],
        reviewers_per_document: int,
        minimum_score: float,
        *,
        require_distinct_institutions: bool,
        minimum_senior_reviewers: int,
        senior_threshold: float,
        max_steps: int,
        max_checks: int,
    ) -> MatchPlan:
        """Improve a maximum-cardinality flow plan using bounded local moves."""

        admissible_scores = tuple(score for score in scores if score.total >= minimum_score)
        if len(admissible_scores) > 4096:
            raise ValueError(
                "fair-local supports at most 128 documents, 256 experts, and 4096 eligible pairs"
            )
        admissible = {(score.document_id, score.expert_id): score for score in admissible_scores}
        if len(admissible) != len(admissible_scores):
            raise ValueError(
                "fair-local requires at most one eligible score per document-expert pair"
            )
        baseline = self._optimal(
            scores,
            all_scores,
            reviewers_per_document,
            minimum_score,
            require_distinct_institutions=require_distinct_institutions,
            load_balance_penalty=0.0,
            minimum_senior_reviewers=minimum_senior_reviewers,
            senior_threshold=senior_threshold,
            strategy_label="fair-local",
        )
        document_ids = tuple(sorted(self.scorer.documents))
        chosen = improve_fair_local(
            document_ids=document_ids,
            experts=self.scorer.experts,
            demands={key: self._demand(key, reviewers_per_document) for key in document_ids},
            admissible=admissible,
            baseline=baseline,
            require_distinct_institutions=require_distinct_institutions,
            minimum_senior_reviewers=minimum_senior_reviewers,
            senior_threshold=senior_threshold,
            max_steps=max_steps,
            max_checks=max_checks,
        )
        return self._to_plan(
            chosen,
            all_scores,
            reviewers_per_document,
            minimum_score,
            "fair-local",
            require_distinct_institutions=require_distinct_institutions,
            minimum_senior_reviewers=minimum_senior_reviewers,
            senior_threshold=senior_threshold,
            maximum_cardinality_certified=True,
        )

    def _to_plan(
        self,
        chosen: dict[str, list[MatchScore]],
        all_scores: tuple[MatchScore, ...],
        reviewers_per_document: int,
        minimum_score: float,
        strategy: str,
        *,
        require_distinct_institutions: bool,
        minimum_senior_reviewers: int,
        senior_threshold: float,
        maximum_cardinality_certified: bool,
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
        diagnostics = self._diagnostics(
            chosen,
            all_scores,
            reviewers_per_document,
            minimum_score,
            require_distinct_institutions=require_distinct_institutions,
            minimum_senior_reviewers=minimum_senior_reviewers,
            senior_threshold=senior_threshold,
            maximum_cardinality_certified=maximum_cardinality_certified,
        )
        return MatchPlan(
            assignments=tuple(assignments),
            unmet=unmet,
            strategy=strategy,
            total_score=sum(item.score for item in assignments),
            diagnostics=diagnostics,
        )

    def _diagnostics(
        self,
        chosen: dict[str, list[MatchScore]],
        all_scores: tuple[MatchScore, ...],
        reviewers_per_document: int,
        minimum_score: float,
        *,
        require_distinct_institutions: bool,
        minimum_senior_reviewers: int,
        senior_threshold: float,
        maximum_cardinality_certified: bool,
    ) -> AssignmentDiagnostics:
        """Explain shortages using observable constraint evidence.

        The optimal flow proves only that the complete run demand is
        infeasible, not that one particular document must be the one left
        short. Per-document codes are therefore deliberately overlapping
        evidence and never presented as a unique unsatisfiable core.
        """

        by_document: dict[str, list[MatchScore]] = {key: [] for key in self.scorer.documents}
        for score in all_scores:
            by_document[score.document_id].append(score)
        workload = {key: 0 for key in self.scorer.experts}
        for selected in chosen.values():
            for score in selected:
                workload[score.expert_id] += 1

        documents: list[DemandDiagnostic] = []
        total_requested = 0
        total_assigned = 0
        for document_id in sorted(self.scorer.documents):
            requested = self._demand(document_id, reviewers_per_document)
            selected = chosen[document_id]
            selected_ids = {score.expert_id for score in selected}
            assigned = len(selected)
            unmet = requested - assigned
            total_requested += requested
            total_assigned += assigned

            document_scores = by_document[document_id]
            scored_experts = {score.expert_id for score in document_scores}
            conflict_pairs: frozenset[tuple[str, str]] = getattr(
                self.scorer, "conflict_pairs", frozenset()
            )
            hard_conflicts = [
                pair
                for pair in conflict_pairs
                if pair[0] == document_id and pair[1] in self.scorer.experts
            ]
            zero_capacity = [
                score
                for score in document_scores
                if self.scorer.experts[score.expert_id].capacity == 0
            ]
            other_ineligible = [
                score
                for score in document_scores
                if not score.eligible
                and (score.document_id, score.expert_id) not in conflict_pairs
                and self.scorer.experts[score.expert_id].capacity > 0
            ]
            eligible = [
                score
                for score in document_scores
                if score.eligible and self.scorer.experts[score.expert_id].capacity > 0
            ]
            below_threshold = [score for score in eligible if score.total < minimum_score]
            admissible = [score for score in eligible if score.total >= minimum_score]
            senior_admissible = [
                score
                for score in admissible
                if self.scorer.experts[score.expert_id].seniority >= senior_threshold
            ]

            def institution_group(score: MatchScore) -> tuple[str, str]:
                institution = self.scorer.experts[score.expert_id].institution
                return (
                    ("institution", institution)
                    if institution is not None
                    else ("expert", score.expert_id)
                )

            institution_groups = {institution_group(score) for score in admissible}
            selected_groups = {institution_group(score) for score in selected}
            institution_blocked = [
                score
                for score in admissible
                if score.expert_id not in selected_ids
                and institution_group(score) in selected_groups
                and workload[score.expert_id] < self.scorer.experts[score.expert_id].capacity
            ]
            saturated_experts = tuple(
                sorted(
                    {
                        score.expert_id
                        for score in admissible
                        if score.expert_id not in selected_ids
                        and workload[score.expert_id]
                        >= self.scorer.experts[score.expert_id].capacity
                    }
                )
            )
            senior_assigned = sum(
                self.scorer.experts[score.expert_id].seniority >= senior_threshold
                for score in selected
            )
            reserved = min(minimum_senior_reviewers, requested)

            reasons: list[UnmetReason] = []
            if unmet:
                if hard_conflicts:
                    reasons.append(UnmetReason.HARD_CONFLICT)
                if other_ineligible:
                    reasons.append(UnmetReason.OTHER_INELIGIBLE)
                if zero_capacity:
                    reasons.append(UnmetReason.ZERO_CAPACITY)
                if below_threshold:
                    reasons.append(UnmetReason.MINIMUM_SCORE)
                if len(scored_experts) < len(self.scorer.experts):
                    reasons.append(UnmetReason.SPARSE_SCORE_MATRIX)
                if len(admissible) < requested:
                    reasons.append(UnmetReason.CANDIDATE_SCARCITY)
                if saturated_experts:
                    reasons.append(UnmetReason.EXPERT_CAPACITY)
                senior_shortfall = reserved > senior_assigned
                if senior_shortfall:
                    reasons.append(UnmetReason.SENIORITY_FLOOR)
                diversity_shortfall = require_distinct_institutions and (
                    len(institution_groups) < requested or bool(institution_blocked)
                )
                if diversity_shortfall:
                    reasons.append(UnmetReason.INSTITUTION_DIVERSITY)
                locally_large_enough = (
                    len(admissible) >= requested
                    and (not require_distinct_institutions or len(institution_groups) >= requested)
                    and (not reserved or len(senior_admissible) >= reserved)
                )
                if maximum_cardinality_certified and locally_large_enough:
                    reasons.append(UnmetReason.GLOBAL_CAPACITY_COUPLING)
                if not maximum_cardinality_certified:
                    reasons.append(UnmetReason.GREEDY_NOT_CERTIFIED)

            documents.append(
                DemandDiagnostic(
                    document_id=document_id,
                    requested=requested,
                    assigned=assigned,
                    unmet=unmet,
                    reason_codes=tuple(reasons),
                    evidence={
                        "total_experts": len(self.scorer.experts),
                        "scored_pairs": len(document_scores),
                        "unscored_experts": len(self.scorer.experts) - len(scored_experts),
                        "hard_conflict_pairs": len(hard_conflicts),
                        "other_ineligible_pairs": len(other_ineligible),
                        "zero_capacity_pairs": len(zero_capacity),
                        "eligible_pairs": len(eligible),
                        "below_minimum_score_pairs": len(below_threshold),
                        "admissible_pairs": len(admissible),
                        "senior_admissible_pairs": len(senior_admissible),
                        "senior_assigned": senior_assigned,
                        "institution_groups": len(institution_groups),
                        "institution_blocked_pairs": len(institution_blocked),
                        "saturated_admissible_experts": len(saturated_experts),
                    },
                    saturated_experts=saturated_experts,
                )
            )

        total_unmet = total_requested - total_assigned
        status = (
            FeasibilityStatus.SATISFIED
            if total_unmet == 0
            else (
                FeasibilityStatus.INFEASIBLE
                if maximum_cardinality_certified
                else FeasibilityStatus.NOT_CERTIFIED
            )
        )
        return AssignmentDiagnostics(
            status=status,
            requested=total_requested,
            assigned=total_assigned,
            unmet=total_unmet,
            documents=tuple(documents),
        )
