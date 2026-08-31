"""Strict adapters for externally computed document-expert affinities."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from peermatchlab.models import Conflict, DataValidationError, Document, Expert, MatchScore


@dataclass(frozen=True, slots=True)
class Affinity:
    """One sparse, externally supplied affinity in the unit interval."""

    document_id: str
    expert_id: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise DataValidationError("affinity document_id must be a non-empty string")
        if not isinstance(self.expert_id, str) or not self.expert_id.strip():
            raise DataValidationError("affinity expert_id must be a non-empty string")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise DataValidationError("affinity score must be a finite number in [0, 1]")
        try:
            value = float(self.score)
        except OverflowError as error:
            raise DataValidationError("affinity score must be a finite number in [0, 1]") from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise DataValidationError("affinity score must be a finite number in [0, 1]")
        object.__setattr__(self, "score", value)


class AffinityScorer:
    """Expose a sparse external matrix through the assignment score contract."""

    def __init__(
        self,
        documents: Iterable[Document],
        experts: Iterable[Expert],
        affinities: Iterable[Affinity],
        *,
        conflicts: Iterable[Conflict] = (),
    ) -> None:
        document_items = tuple(documents)
        expert_items = tuple(experts)
        if any(not isinstance(item, Document) for item in document_items):
            raise DataValidationError("documents must contain Document objects")
        if any(not isinstance(item, Expert) for item in expert_items):
            raise DataValidationError("experts must contain Expert objects")
        if not document_items:
            raise DataValidationError("at least one document is required")
        if not expert_items:
            raise DataValidationError("at least one expert is required")
        self.documents = {item.id: item for item in document_items}
        self.experts = {item.id: item for item in expert_items}
        if len(self.documents) != len(document_items):
            raise DataValidationError("document identifiers must be unique")
        if len(self.experts) != len(expert_items):
            raise DataValidationError("expert identifiers must be unique")
        conflict_items = tuple(conflicts)
        if any(not isinstance(item, Conflict) for item in conflict_items):
            raise DataValidationError("conflicts must contain Conflict objects")
        conflict_pairs = {(item.document_id, item.expert_id) for item in conflict_items}
        unknown_conflicts = sorted(
            pair
            for pair in conflict_pairs
            if pair[0] not in self.documents or pair[1] not in self.experts
        )
        if unknown_conflicts:
            raise DataValidationError(
                f"conflicts reference unknown document/expert pairs: {unknown_conflicts}"
            )
        scores: list[MatchScore] = []
        seen: set[tuple[str, str]] = set()
        for affinity in affinities:
            if not isinstance(affinity, Affinity):
                raise DataValidationError("affinities must contain Affinity objects")
            pair = (affinity.document_id, affinity.expert_id)
            if pair in seen:
                raise DataValidationError(f"duplicate affinity pair: {pair}")
            seen.add(pair)
            if pair[0] not in self.documents or pair[1] not in self.experts:
                raise DataValidationError(f"affinity references unknown pair: {pair}")
            conflicted = pair in conflict_pairs
            scores.append(
                MatchScore(
                    document_id=pair[0],
                    expert_id=pair[1],
                    total=affinity.score,
                    content=0.0,
                    topics=0.0,
                    bid=0.0,
                    recency=0.0,
                    seniority=0.0,
                    affinity=affinity.score,
                    eligible=not conflicted,
                    reasons=(
                        "declared hard conflict"
                        if conflicted
                        else "externally supplied affinity score",
                    ),
                )
            )
        self._scores = tuple(sorted(scores, key=lambda item: (item.document_id, item.expert_id)))

    def matrix(self) -> tuple[MatchScore, ...]:
        return self._scores


def load_affinities_csv(path: str | Path) -> tuple[Affinity, ...]:
    """Load sparse score rows with an optional canonical header.

    Headerless ``paper ID, profile ID, score`` files produced by OpenReview's
    expertise workflow are accepted directly. Missing pairs remain absent.
    """

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise DataValidationError("affinity CSV must contain a header or score row")
    data_rows = rows[1:] if tuple(rows[0]) == ("document_id", "expert_id", "score") else rows
    affinities: list[Affinity] = []
    first_line = 2 if data_rows is not rows else 1
    for line_number, row in enumerate(data_rows, start=first_line):
        if not row:
            continue
        if len(row) != 3:
            raise DataValidationError(
                f"affinity CSV line {line_number} must contain exactly 3 fields"
            )
        try:
            score = float(row[2])
        except (OverflowError, ValueError) as error:
            raise DataValidationError(
                f"affinity CSV line {line_number} has an invalid score"
            ) from error
        affinities.append(Affinity(row[0], row[1], score))
    # AffinityScorer performs reference validation; duplicates are a file-level error.
    pairs = [(item.document_id, item.expert_id) for item in affinities]
    if len(pairs) != len(set(pairs)):
        raise DataValidationError("affinity CSV contains duplicate document/expert pairs")
    return tuple(affinities)
