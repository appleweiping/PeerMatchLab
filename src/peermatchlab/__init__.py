"""Transparent expert matching with explicit evidence and constraints."""

from peermatchlab.affinity import Affinity, AffinityScorer, load_affinities_csv
from peermatchlab.assignment import AssignmentEngine, AssignmentStrategy
from peermatchlab.models import (
    Assignment,
    AssignmentDiagnostics,
    DemandDiagnostic,
    Document,
    Expert,
    FeasibilityStatus,
    MatchPlan,
    Publication,
    UnmetReason,
)
from peermatchlab.openreview import (
    load_openreview_submissions,
    load_reviewer_ids,
    openreview_submissions_from_records,
    reviewer_ids_to_experts,
)
from peermatchlab.openreview_api import (
    OpenReviewClient,
    OpenReviewClientConfig,
    OpenReviewError,
    OpenReviewHttpError,
    OpenReviewProtocolError,
    OpenReviewSnapshot,
    OpenReviewTransportError,
    RetryPolicy,
    fetch_openreview_snapshot,
    write_openreview_snapshot,
)
from peermatchlab.scoring import MatchScorer, ScoreWeights

__all__ = [
    "Affinity",
    "AffinityScorer",
    "Assignment",
    "AssignmentDiagnostics",
    "AssignmentEngine",
    "AssignmentStrategy",
    "DemandDiagnostic",
    "Document",
    "Expert",
    "FeasibilityStatus",
    "MatchPlan",
    "MatchScorer",
    "OpenReviewClient",
    "OpenReviewClientConfig",
    "OpenReviewError",
    "OpenReviewHttpError",
    "OpenReviewProtocolError",
    "OpenReviewSnapshot",
    "OpenReviewTransportError",
    "Publication",
    "RetryPolicy",
    "ScoreWeights",
    "UnmetReason",
    "fetch_openreview_snapshot",
    "load_affinities_csv",
    "load_openreview_submissions",
    "load_reviewer_ids",
    "openreview_submissions_from_records",
    "reviewer_ids_to_experts",
    "write_openreview_snapshot",
]

__version__ = "0.5.0"
