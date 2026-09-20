"""Bounded, independent capacity/conflict scenario comparisons for local matching."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

from peermatchlab.config import MatchConfig
from peermatchlab.io import load_json_text
from peermatchlab.models import Conflict, DataValidationError, Document, Expert
from peermatchlab.pipeline import MatchRun, run_matching

MAX_PLAN_BYTES = 64 * 1024
MAX_DOCUMENTS = 32
MAX_EXPERTS = 64
MAX_SCENARIOS = 8
MAX_REQUESTED = 128
_SCENARIO_ID = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")
Pair = tuple[str, str]


def _pairs(value: object, label: str) -> tuple[Pair, ...]:
    if not isinstance(value, list) or len(value) > MAX_DOCUMENTS * MAX_EXPERTS:
        raise DataValidationError(f"{label} must be a bounded array")
    result: list[Pair] = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or any(not isinstance(item, str) or not item or item != item.strip() for item in row)
        ):
            raise DataValidationError(f"{label} must contain [document_id, expert_id] pairs")
        result.append((row[0], row[1]))
    if len(set(result)) != len(result):
        raise DataValidationError(f"{label} contains duplicate pairs")
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class WhatIfScenario:
    id: str
    capacity_overrides: tuple[tuple[str, int], ...]
    add_conflicts: tuple[Pair, ...]
    remove_conflicts: tuple[Pair, ...]

    def __post_init__(self) -> None:
        if type(self.id) is not str or _SCENARIO_ID.fullmatch(self.id) is None:
            raise DataValidationError("scenario id must be short lowercase ASCII")
        if (
            type(self.capacity_overrides) is not tuple
            or len(self.capacity_overrides) > MAX_EXPERTS
            or any(
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not row[0]
                or row[0] != row[0].strip()
                or type(row[1]) is not int
                or not 0 <= row[1] <= MAX_REQUESTED
                for row in self.capacity_overrides
            )
            or len({row[0] for row in self.capacity_overrides}) != len(self.capacity_overrides)
        ):
            raise DataValidationError("scenario capacity overrides are invalid")
        for label, pairs in (
            ("add_conflicts", self.add_conflicts),
            ("remove_conflicts", self.remove_conflicts),
        ):
            if (
                type(pairs) is not tuple
                or len(pairs) > MAX_DOCUMENTS * MAX_EXPERTS
                or any(
                    not isinstance(pair, tuple)
                    or len(pair) != 2
                    or any(
                        not isinstance(item, str) or not item or item != item.strip()
                        for item in pair
                    )
                    for pair in pairs
                )
                or len(set(pairs)) != len(pairs)
            ):
                raise DataValidationError(f"scenario {label} pairs are invalid")
        if set(self.add_conflicts) & set(self.remove_conflicts):
            raise DataValidationError("the same conflict cannot be added and removed")
        if not self.capacity_overrides and not self.add_conflicts and not self.remove_conflicts:
            raise DataValidationError("scenario must change capacity or conflicts")
        object.__setattr__(self, "capacity_overrides", tuple(sorted(self.capacity_overrides)))
        object.__setattr__(self, "add_conflicts", tuple(sorted(self.add_conflicts)))
        object.__setattr__(self, "remove_conflicts", tuple(sorted(self.remove_conflicts)))

    @classmethod
    def from_mapping(cls, value: object) -> WhatIfScenario:
        if not isinstance(value, dict) or set(value) != {
            "id",
            "capacity_overrides",
            "add_conflicts",
            "remove_conflicts",
        }:
            raise DataValidationError("scenario has missing or unknown fields")
        ident = value["id"]
        if type(ident) is not str or _SCENARIO_ID.fullmatch(ident) is None:
            raise DataValidationError("scenario id must be short lowercase ASCII")
        overrides = value["capacity_overrides"]
        if not isinstance(overrides, dict) or len(overrides) > MAX_EXPERTS:
            raise DataValidationError("capacity_overrides must be a bounded object")
        if any(
            not isinstance(key, str)
            or not key
            or key != key.strip()
            or type(count) is not int
            or not 0 <= count <= MAX_REQUESTED
            for key, count in overrides.items()
        ):
            raise DataValidationError("capacity override must map expert ids to bounded integers")
        added = _pairs(value["add_conflicts"], "add_conflicts")
        removed = _pairs(value["remove_conflicts"], "remove_conflicts")
        if set(added) & set(removed):
            raise DataValidationError("the same conflict cannot be added and removed")
        if not overrides and not added and not removed:
            raise DataValidationError("scenario must change capacity or conflicts")
        return cls(ident, tuple(sorted(overrides.items())), added, removed)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "capacity_overrides": dict(self.capacity_overrides),
            "add_conflicts": [list(pair) for pair in self.add_conflicts],
            "remove_conflicts": [list(pair) for pair in self.remove_conflicts],
        }


@dataclass(frozen=True, slots=True)
class WhatIfPlan:
    scenarios: tuple[WhatIfScenario, ...]

    def __post_init__(self) -> None:
        if (
            type(self.scenarios) is not tuple
            or not 1 <= len(self.scenarios) <= MAX_SCENARIOS
            or any(not isinstance(row, WhatIfScenario) for row in self.scenarios)
            or len({row.id for row in self.scenarios}) != len(self.scenarios)
        ):
            raise DataValidationError("what-if plan requires 1 to 8 unique scenarios")
        object.__setattr__(self, "scenarios", tuple(sorted(self.scenarios, key=lambda row: row.id)))

    @classmethod
    def from_bytes(cls, source: bytes) -> WhatIfPlan:
        if type(source) is not bytes or len(source) > MAX_PLAN_BYTES:
            raise DataValidationError("scenario plan must be a bounded immutable byte snapshot")
        try:
            raw = load_json_text(source.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise DataValidationError("scenario plan must be UTF-8") from error
        except json.JSONDecodeError as error:
            raise DataValidationError("scenario plan must be valid JSON") from error
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "scenarios"}:
            raise DataValidationError("scenario plan has missing or unknown fields")
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise DataValidationError("scenario plan requires schema_version 1")
        rows = raw["scenarios"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_SCENARIOS:
            raise DataValidationError("scenario plan requires 1 to 8 scenarios")
        scenarios = tuple(WhatIfScenario.from_mapping(row) for row in rows)
        if len({item.id for item in scenarios}) != len(scenarios):
            raise DataValidationError("scenario ids must be unique")
        return cls(tuple(sorted(scenarios, key=lambda item: item.id)))


def _summary(run: MatchRun) -> dict[str, Any]:
    diagnostics = run.plan.diagnostics
    if diagnostics is None:
        raise DataValidationError("generated plan lacks demand diagnostics")
    return {
        "assigned": diagnostics.assigned,
        "unmet": diagnostics.unmet,
        "status": diagnostics.status.value,
        "certified": diagnostics.certified,
        "total_score": run.plan.total_score,
        "assignments": sorted([[row.document_id, row.expert_id] for row in run.plan.assignments]),
        "workload": dict(sorted(run.audit.workload.items())),
        "audit": run.audit.as_dict(),
        "safe": run.audit.safe,
        "reasons": {
            item.document_id: [code.value for code in item.reason_codes]
            for item in diagnostics.documents
            if item.unmet
        },
    }


def _delta(base: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    original = {tuple(pair) for pair in base["assignments"]}
    changed = {tuple(pair) for pair in result["assignments"]}
    return {
        "assigned": result["assigned"] - base["assigned"],
        "unmet": result["unmet"] - base["unmet"],
        "total_score": result["total_score"] - base["total_score"],
        "added_assignments": [list(pair) for pair in sorted(changed - original)],
        "removed_assignments": [list(pair) for pair in sorted(original - changed)],
        "workload": {
            ident: result["workload"][ident] - count for ident, count in base["workload"].items()
        },
    }


def compare_what_if(
    documents: tuple[Document, ...],
    experts: tuple[Expert, ...],
    plan: WhatIfPlan,
    *,
    conflicts: tuple[Conflict, ...] = (),
    config: MatchConfig | None = None,
) -> dict[str, object]:
    """Compare non-cumulative scenarios against one independently audited baseline."""
    if not isinstance(plan, WhatIfPlan) or not isinstance(config, (MatchConfig, type(None))):
        raise DataValidationError("invalid what-if plan or match configuration")
    if (
        not 1 <= len(documents) <= MAX_DOCUMENTS
        or not 1 <= len(experts) <= MAX_EXPERTS
        or len(documents) * len(experts) > MAX_DOCUMENTS * MAX_EXPERTS
        or len(conflicts) > MAX_DOCUMENTS * MAX_EXPERTS
    ):
        raise DataValidationError("what-if domain exceeds document/expert/conflict limits")
    if (
        any(not isinstance(item, Document) for item in documents)
        or any(not isinstance(item, Expert) for item in experts)
        or any(not isinstance(item, Conflict) for item in conflicts)
    ):
        raise DataValidationError("what-if inputs have invalid domain objects")
    document_ids = {item.id for item in documents}
    expert_ids = {item.id for item in experts}
    if len(document_ids) != len(documents) or len(expert_ids) != len(experts):
        raise DataValidationError("what-if document and expert ids must be unique")
    selected = config or MatchConfig()
    demand = sum(item.required_experts or selected.reviewers_per_document for item in documents)
    if demand > MAX_REQUESTED:
        raise DataValidationError("what-if requested assignment count exceeds limit")
    baseline_pairs = {(item.document_id, item.expert_id) for item in conflicts}
    if any(d not in document_ids or e not in expert_ids for d, e in baseline_pairs):
        raise DataValidationError("baseline conflict references unknown document or expert")
    for scenario in plan.scenarios:
        if any(ident not in expert_ids for ident, _ in scenario.capacity_overrides):
            raise DataValidationError("capacity override references unknown expert")
        if any(
            d not in document_ids or e not in expert_ids
            for d, e in (*scenario.add_conflicts, *scenario.remove_conflicts)
        ):
            raise DataValidationError("scenario conflict references unknown document or expert")
        if set(scenario.add_conflicts) & baseline_pairs:
            raise DataValidationError("added conflict already exists in baseline")
        if set(scenario.remove_conflicts) - baseline_pairs:
            raise DataValidationError("removed conflict does not exist in baseline")
    baseline_run = run_matching(documents, experts, conflicts=conflicts, config=selected)
    baseline = _summary(baseline_run)
    if not baseline["safe"]:
        raise DataValidationError("baseline matching audit failed")
    scenarios: list[dict[str, object]] = []
    for scenario in plan.scenarios:
        capacities = dict(scenario.capacity_overrides)
        changed_experts = tuple(
            replace(item, capacity=capacities[item.id]) if item.id in capacities else item
            for item in experts
        )
        removed = set(scenario.remove_conflicts)
        changed_conflicts = tuple(
            item for item in conflicts if (item.document_id, item.expert_id) not in removed
        ) + tuple(Conflict(d, e, "what-if scenario") for d, e in scenario.add_conflicts)
        result = _summary(
            run_matching(documents, changed_experts, conflicts=changed_conflicts, config=selected)
        )
        if not result["safe"]:
            raise DataValidationError(f"scenario {scenario.id} matching audit failed")
        scenarios.append(
            {
                "id": scenario.id,
                "changes": scenario.to_dict(),
                "result": result,
                "delta": _delta(baseline, result),
            }
        )
    return {"schema_version": 1, "baseline": baseline, "scenarios": scenarios}
