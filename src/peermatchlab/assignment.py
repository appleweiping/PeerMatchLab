"""Capacity-constrained assignment strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from peermatchlab.models import Assignment, Document, Expert, MatchPlan, MatchScore


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
        scores = tuple(score for score in self.scorer.matrix() if score.eligible)
        if selected is AssignmentStrategy.GREEDY:
            return self._greedy(
                scores,
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
        return self._to_plan(chosen, reviewers_per_document, label)

    def _optimal(
        self,
        scores: tuple[MatchScore, ...],
        reviewers_per_document: int,
        minimum_score: float,
        *,
        require_distinct_institutions: bool = False,
        load_balance_penalty: float = 0.0,
        minimum_senior_reviewers: int = 0,
        senior_threshold: float = 0.75,
    ) -> MatchPlan:
        document_ids = sorted(self.scorer.documents)
        expert_ids = sorted(self.scorer.experts)
        candidate_scores = tuple(score for score in scores if score.total >= minimum_score)
        if minimum_senior_reviewers:
            return self._optimal_with_senior_floor(
                candidate_scores,
                document_ids,
                expert_ids,
                reviewers_per_document,
                load_balance_penalty=load_balance_penalty,
                minimum_senior_reviewers=minimum_senior_reviewers,
                senior_threshold=senior_threshold,
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
            for slot in range(self.scorer.experts[expert_id].capacity):
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
            reviewers_per_document,
            self._strategy_label(
                "optimal", require_distinct_institutions, load_balance_penalty > 0
            ),
        )

    def _optimal_with_senior_floor(
        self,
        candidate_scores: tuple[MatchScore, ...],
        document_ids: list[str],
        expert_ids: list[str],
        reviewers_per_document: int,
        *,
        load_balance_penalty: float,
        minimum_senior_reviewers: int,
        senior_threshold: float,
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
            for slot in range(self.scorer.experts[expert_id].capacity):
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
            reviewers_per_document,
            self._strategy_label("optimal", False, load_balance_penalty > 0, True),
        )

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
