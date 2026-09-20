"""Local deterministic synthetic benchmark; no external paper/reviewer data."""

from __future__ import annotations

import json
from time import perf_counter

from peermatchlab.affinity import Affinity
from peermatchlab.config import MatchConfig
from peermatchlab.models import Document, Expert
from peermatchlab.pipeline import run_affinity_matching


def main() -> None:
    documents = tuple(
        Document(f"paper-{index:02d}", f"Synthetic paper {index}") for index in range(12)
    )
    experts = tuple(
        Expert(f"reviewer-{index:02d}", f"Synthetic reviewer {index}", capacity=1)
        for index in range(12)
    )
    block = ((0.95, 0.75), (0.75, 0.70))
    affinities = tuple(
        Affinity(
            document.id,
            expert.id,
            block[paper % 2][reviewer % 2] if paper // 2 == reviewer // 2 else 0.05,
        )
        for paper, document in enumerate(documents)
        for reviewer, expert in enumerate(experts)
    )
    results: dict[str, object] = {
        "fixture": "deterministic-synthetic-v1",
        "documents": len(documents),
        "experts": len(experts),
        "eligible_pairs": len(affinities),
        "demand_per_document": 1,
    }
    for strategy in ("optimal", "maximin-flow"):
        started = perf_counter()
        run = run_affinity_matching(
            documents,
            experts,
            affinities,
            config=MatchConfig(strategy=strategy, reviewers_per_document=1),
        )
        elapsed = perf_counter() - started
        assert run.audit.safe
        results[strategy] = {
            "assigned": len(run.plan.assignments),
            "minimum_selected_edge_score": min(
                (assignment.score for assignment in run.plan.assignments), default=0.0
            ),
            "total_score": run.plan.total_score,
            "seconds_observed": elapsed,
            "audit_safe": run.audit.safe,
        }
    print(json.dumps(results, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
