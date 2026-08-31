"""Transparent expert matching with explicit evidence and constraints."""

from peermatchlab.assignment import AssignmentEngine, AssignmentStrategy
from peermatchlab.models import Assignment, Document, Expert, MatchPlan, Publication
from peermatchlab.scoring import MatchScorer, ScoreWeights

__all__ = [
    "Assignment",
    "AssignmentEngine",
    "AssignmentStrategy",
    "Document",
    "Expert",
    "MatchPlan",
    "MatchScorer",
    "Publication",
    "ScoreWeights",
]

__version__ = "0.1.0"
