"""Bounded, deterministic local leximin improvement of a feasible assignment.

This is deliberately not an implementation of OpenReview's FairFlow algorithm.
The caller supplies a certified maximum-cardinality flow solution; every move
preserves its cardinality and is checked against the same hard constraints.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from itertools import combinations, islice

from peermatchlab.models import Expert, MatchPlan, MatchScore

Pair = tuple[str, str]
Move = tuple[tuple[Pair, ...], tuple[Pair, ...]]


def _neighbors(
    selected: set[Pair],
    by_document: Mapping[str, tuple[MatchScore, ...]],
    by_expert: Mapping[str, tuple[MatchScore, ...]],
    admissible: Mapping[Pair, MatchScore],
) -> Iterator[Move]:
    """Yield one-edge replacements, transfers, then two-document swaps."""

    pairs = sorted(selected)
    for document_id, old_expert in pairs:
        for score in by_document[document_id]:
            if score.expert_id != old_expert:
                yield (((document_id, old_expert),), ((document_id, score.expert_id),))
    for old_document, expert_id in pairs:
        for score in by_expert[expert_id]:
            if score.document_id != old_document:
                yield (((old_document, expert_id),), ((score.document_id, expert_id),))
    for (first_document, first_expert), (second_document, second_expert) in combinations(pairs, 2):
        if first_document == second_document or first_expert == second_expert:
            continue
        first_new = (first_document, second_expert)
        second_new = (second_document, first_expert)
        if first_new in admissible and second_new in admissible:
            yield (
                ((first_document, first_expert), (second_document, second_expert)),
                (first_new, second_new),
            )


def _valid_document(
    document_id: str,
    selected_experts: set[str],
    *,
    experts: Mapping[str, Expert],
    demands: Mapping[str, int],
    require_distinct_institutions: bool,
    minimum_senior_reviewers: int,
    senior_threshold: float,
) -> bool:
    demand = demands[document_id]
    if len(selected_experts) > demand:
        return False
    free_slots = demand - min(minimum_senior_reviewers, demand)
    if sum(experts[key].seniority < senior_threshold for key in selected_experts) > free_slots:
        return False
    if require_distinct_institutions:
        known = [
            experts[key].institution
            for key in selected_experts
            if experts[key].institution is not None
        ]
        if len(known) != len(set(known)):
            return False
    return True


def improve_fair_local(
    *,
    document_ids: tuple[str, ...],
    experts: Mapping[str, Expert],
    demands: Mapping[str, int],
    admissible: Mapping[Pair, MatchScore],
    baseline: MatchPlan,
    require_distinct_institutions: bool,
    minimum_senior_reviewers: int,
    senior_threshold: float,
    max_steps: int,
    max_checks: int,
) -> dict[str, list[MatchScore]]:
    """Improve the sorted document-score vector without reducing cardinality.

    The search is not globally optimal and a budget cutoff can stop before a
    local fixed point. Only a strict lexicographic improvement, or a stable
    pair-order tie, can be accepted, so neither cycling nor fairness regression
    is possible. All score comparisons use the original unrounded affinities.
    """

    by_document: dict[str, tuple[MatchScore, ...]] = {
        document_id: tuple(
            sorted(
                (score for score in admissible.values() if score.document_id == document_id),
                key=lambda score: score.expert_id,
            )
        )
        for document_id in document_ids
    }
    by_expert: dict[str, tuple[MatchScore, ...]] = {
        expert_id: tuple(
            sorted(
                (score for score in admissible.values() if score.expert_id == expert_id),
                key=lambda score: score.document_id,
            )
        )
        for expert_id in sorted(experts)
    }
    selected: set[Pair] = {
        (assignment.document_id, assignment.expert_id) for assignment in baseline.assignments
    }
    # For equal-sized pair sets, the lexicographically first sorted pair list
    # has the highest bit at their earliest difference. A fixed-width bitmask
    # compares that tie-break without sorting hundreds of pairs per candidate.
    pair_bits = {
        pair: 1 << (len(admissible) - index - 1) for index, pair in enumerate(sorted(admissible))
    }
    selected_mask = sum(pair_bits[pair] for pair in selected)
    chosen: dict[str, set[str]] = {document_id: set() for document_id in document_ids}
    loads = dict.fromkeys(experts, 0)
    for document_id, expert_id in selected:
        chosen[document_id].add(expert_id)
        loads[expert_id] += 1

    def total(document_id: str, expert_ids: set[str]) -> float:
        return math.fsum(admissible[(document_id, expert_id)].total for expert_id in expert_ids)

    totals = {document_id: total(document_id, chosen[document_id]) for document_id in document_ids}
    checks = 0
    for _ in range(max_steps):
        best_vector = tuple(sorted(totals.values()))
        best_mask = selected_mask
        best_move: Move | None = None
        for removed, added in islice(
            _neighbors(selected, by_document, by_expert, admissible), max_checks - checks
        ):
            checks += 1
            if any(pair in selected for pair in added) or any(
                pair not in selected for pair in removed
            ):
                continue
            # Copy only the affected documents. This also catches a candidate
            # that would duplicate an expert within one document.
            affected = {document_id for document_id, _ in (*removed, *added)}
            proposal = {document_id: set(chosen[document_id]) for document_id in affected}
            for document_id, expert_id in removed:
                proposal[document_id].remove(expert_id)
            for document_id, expert_id in added:
                proposal[document_id].add(expert_id)
            if sum(map(len, proposal.values())) != sum(len(chosen[key]) for key in affected):
                continue
            if any(
                not _valid_document(
                    document_id,
                    proposal[document_id],
                    experts=experts,
                    demands=demands,
                    require_distinct_institutions=require_distinct_institutions,
                    minimum_senior_reviewers=minimum_senior_reviewers,
                    senior_threshold=senior_threshold,
                )
                for document_id in affected
            ):
                continue
            proposed_loads = {expert_id: loads[expert_id] for _, expert_id in (*removed, *added)}
            for _, expert_id in removed:
                proposed_loads[expert_id] -= 1
            for _, expert_id in added:
                proposed_loads[expert_id] += 1
            if any(
                load > experts[expert_id].capacity for expert_id, load in proposed_loads.items()
            ):
                continue
            proposed_totals = dict(totals)
            for document_id in affected:
                proposed_totals[document_id] = total(document_id, proposal[document_id])
            vector = tuple(sorted(proposed_totals.values()))
            if vector < best_vector:
                continue
            candidate_mask = selected_mask
            for pair in (*removed, *added):
                candidate_mask ^= pair_bits[pair]
            if vector > best_vector or candidate_mask > best_mask:
                best_vector, best_mask, best_move = vector, candidate_mask, (removed, added)
        if best_move is None:
            break
        removed, added = best_move
        for document_id, expert_id in removed:
            selected.remove((document_id, expert_id))
            chosen[document_id].remove(expert_id)
            loads[expert_id] -= 1
        for document_id, expert_id in added:
            selected.add((document_id, expert_id))
            chosen[document_id].add(expert_id)
            loads[expert_id] += 1
        selected_mask = best_mask
        for document_id in {document_id for document_id, _ in (*removed, *added)}:
            totals[document_id] = total(document_id, chosen[document_id])
        if checks >= max_checks:
            break
    return {
        document_id: [
            admissible[(document_id, expert_id)] for expert_id in sorted(chosen[document_id])
        ]
        for document_id in document_ids
    }
