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
from peermatchlab.openreview import load_openreview_submissions, load_reviewer_ids
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
    "Publication",
    "ScoreWeights",
    "UnmetReason",
    "load_affinities_csv",
    "load_openreview_submissions",
    "load_reviewer_ids",
]

__version__ = "0.4.0"
