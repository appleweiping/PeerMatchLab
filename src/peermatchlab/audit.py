"""Post-assignment diagnostics for coverage, load balance, and safety."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from peermatchlab.models import Conflict, Document, Expert, MatchPlan


def gini(values: Iterable[int]) -> float:
    """Compute the Gini coefficient for a non-negative workload sequence."""

    ordered = sorted(values)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ordered):
        raise ValueError("workloads must be non-negative integers")
    if not ordered or sum(ordered) == 0:
        return 0.0
    count = len(ordered)
    total = sum(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    numerator = 2 * weighted - (count + 1) * total
    return float(Fraction(numerator, count * total))


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Serializable health indicators for a complete assignment plan."""

    document_coverage: float
    demand_coverage: float
    average_score: float
    minimum_score: float | None
    workload_gini: float
    workload: Mapping[str, int]
    conflict_violations: tuple[tuple[str, str], ...]
    capacity_violations: tuple[str, ...]
    institution_duplicates: Mapping[str, tuple[str, ...]]
    unknown_documents: tuple[str, ...] = ()
    unknown_experts: tuple[str, ...] = ()
    duplicate_assignments: tuple[tuple[str, str], ...] = ()
    demand_violations: tuple[str, ...] = ()

    @property
    def safe(self) -> bool:
        """Return true when hard conflicts and expert capacities are respected."""

        return not (
            self.conflict_violations
            or self.capacity_violations
            or self.unknown_documents
            or self.unknown_experts
            or self.duplicate_assignments
            or self.demand_violations
        )

    def as_dict(self) -> dict[str, object]:
        """Convert the report into JSON-compatible values."""

        return {
            "document_coverage": self.document_coverage,
            "demand_coverage": self.demand_coverage,
            "average_score": self.average_score,
            "minimum_score": self.minimum_score,
            "workload_gini": self.workload_gini,
            "workload": dict(self.workload),
            "conflict_violations": [list(value) for value in self.conflict_violations],
            "capacity_violations": list(self.capacity_violations),
            "institution_duplicates": {
                key: list(value) for key, value in self.institution_duplicates.items()
            },
            "unknown_documents": list(self.unknown_documents),
            "unknown_experts": list(self.unknown_experts),
            "duplicate_assignments": [list(value) for value in self.duplicate_assignments],
            "demand_violations": list(self.demand_violations),
            "safe": self.safe,
        }


def audit_plan(
    plan: MatchPlan,
    documents: Iterable[Document],
    experts: Iterable[Expert],
    conflicts: Iterable[Conflict] = (),
    *,
    default_demand: int = 2,
) -> AuditReport:
    """Audit a plan without trusting the assignment engine that created it."""

    if isinstance(default_demand, bool) or not isinstance(default_demand, int):
        raise ValueError("default_demand must be an integer")
    if default_demand < 1:
        raise ValueError("default_demand must be positive")
    document_items = tuple(documents)
    expert_items = tuple(experts)
    document_map = {item.id: item for item in document_items}
    expert_map = {item.id: item for item in expert_items}
    if len(document_map) != len(document_items):
        raise ValueError("document identifiers must be unique")
    if len(expert_map) != len(expert_items):
        raise ValueError("expert identifiers must be unique")

    conflict_pairs = {(item.document_id, item.expert_id) for item in conflicts}
    workload = {expert_id: 0 for expert_id in expert_map}
    by_document: dict[str, list[str]] = {document_id: [] for document_id in document_map}
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_assignments: set[tuple[str, str]] = set()
    unknown_documents: set[str] = set()
    unknown_experts: set[str] = set()
    valid_assignments = []
    for assignment in plan.assignments:
        pair = (assignment.document_id, assignment.expert_id)
        if pair in seen_pairs:
            duplicate_assignments.add(pair)
        else:
            seen_pairs.add(pair)
        if assignment.document_id not in document_map:
            unknown_documents.add(assignment.document_id)
        if assignment.expert_id not in expert_map:
            unknown_experts.add(assignment.expert_id)
        if (
            pair in duplicate_assignments
            or assignment.document_id not in document_map
            or assignment.expert_id not in expert_map
        ):
            continue
        valid_assignments.append(assignment)
        workload[assignment.expert_id] += 1
        by_document[assignment.document_id].append(assignment.expert_id)
    valid_pairs = {(item.document_id, item.expert_id) for item in valid_assignments}
    conflict_violations = tuple(sorted(conflict_pairs & valid_pairs))
    capacity_violations = tuple(
        sorted(
            expert_id
            for expert_id, count in workload.items()
            if count > expert_map[expert_id].capacity
        )
    )
    demands = {
        document.id: (
            document.required_experts if document.required_experts is not None else default_demand
        )
        for document in document_map.values()
    }
    demand = sum(demands.values())
    covered_documents = sum(bool(by_document.get(key)) for key in document_map)
    covered_demand = sum(
        min(len(by_document[document_id]), requested) for document_id, requested in demands.items()
    )
    demand_violations = tuple(
        sorted(
            document_id
            for document_id, requested in demands.items()
            if len(by_document[document_id]) > requested
        )
    )
    institutions: dict[str, tuple[str, ...]] = {}
    for document_id, expert_ids in by_document.items():
        values = [expert_map[key].institution for key in expert_ids if key in expert_map]
        duplicates = sorted(
            {value for value in values if value is not None and values.count(value) > 1}
        )
        if duplicates:
            institutions[document_id] = tuple(duplicates)
    scores = [item.score for item in valid_assignments]
    return AuditReport(
        document_coverage=covered_documents / len(document_map) if document_map else 1.0,
        demand_coverage=covered_demand / demand if demand else 1.0,
        average_score=statistics.fmean(scores) if scores else 0.0,
        minimum_score=min(scores) if scores else None,
        workload_gini=gini(workload.values()),
        workload=workload,
        conflict_violations=conflict_violations,
        capacity_violations=capacity_violations,
        institution_duplicates=institutions,
        unknown_documents=tuple(sorted(unknown_documents)),
        unknown_experts=tuple(sorted(unknown_experts)),
        duplicate_assignments=tuple(sorted(duplicate_assignments)),
        demand_violations=demand_violations,
    )
