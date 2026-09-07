"""Configuration parsing and validation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from peermatchlab.io import load_json_text
from peermatchlab.models import DataValidationError


@dataclass(frozen=True, slots=True)
class MatchConfig:
    """Runtime controls for scoring and constrained assignment."""

    reviewers_per_document: int = 2
    minimum_score: float = 0.0
    strategy: str = "optimal"
    current_year: int = 2026
    publication_half_life: float = 6.0
    require_distinct_institutions: bool = False
    load_balance_penalty: float = 0.0
    minimum_senior_reviewers: int = 0
    senior_threshold: float = 0.75
    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "content": 0.50,
            "topics": 0.20,
            "bid": 0.15,
            "recency": 0.10,
            "seniority": 0.05,
        }
    )

    def __post_init__(self) -> None:
        if isinstance(self.reviewers_per_document, bool) or not isinstance(
            self.reviewers_per_document, int
        ):
            raise DataValidationError("reviewers_per_document must be an integer")
        if self.reviewers_per_document < 1:
            raise DataValidationError("reviewers_per_document must be positive")
        if self.strategy not in {"optimal", "greedy", "minmax"}:
            raise DataValidationError("strategy must be 'optimal', 'greedy', or 'minmax'")
        if isinstance(self.current_year, bool) or not isinstance(self.current_year, int):
            raise DataValidationError("current_year must be an integer")
        if not 1800 <= self.current_year <= 2200:
            raise DataValidationError("current_year must be between 1800 and 2200")
        if not _finite_number(self.minimum_score):
            raise DataValidationError("minimum_score must be a finite number")
        if not _finite_number(self.publication_half_life):
            raise DataValidationError("publication_half_life must be a finite number")
        if self.publication_half_life <= 0:
            raise DataValidationError("publication_half_life must be positive")
        if not isinstance(self.require_distinct_institutions, bool):
            raise DataValidationError("require_distinct_institutions must be a boolean")
        if not _finite_number(self.load_balance_penalty):
            raise DataValidationError("load_balance_penalty must be a finite number")
        if not 0 <= self.load_balance_penalty <= 1:
            raise DataValidationError("load_balance_penalty must be between 0 and 1")
        if isinstance(self.minimum_senior_reviewers, bool) or not isinstance(
            self.minimum_senior_reviewers, int
        ):
            raise DataValidationError("minimum_senior_reviewers must be an integer")
        if self.minimum_senior_reviewers < 0:
            raise DataValidationError("minimum_senior_reviewers must not be negative")
        if not _finite_number(self.senior_threshold):
            raise DataValidationError("senior_threshold must be a finite number")
        if not 0 <= self.senior_threshold <= 1:
            raise DataValidationError("senior_threshold must be between 0 and 1")
        if self.minimum_senior_reviewers and self.require_distinct_institutions:
            raise DataValidationError(
                "minimum_senior_reviewers cannot be combined with require_distinct_institutions"
            )
        if not isinstance(self.weights, Mapping):
            raise DataValidationError("weights must be an object")
        if any(not isinstance(key, str) for key in self.weights):
            raise DataValidationError("score weight names must be strings")
        unknown = set(self.weights) - {"content", "topics", "bid", "recency", "seniority"}
        if unknown:
            raise DataValidationError(f"unknown score weights: {sorted(unknown)}")
        if any(not _finite_number(value) for value in self.weights.values()):
            raise DataValidationError("score weights must be finite numbers")
        normalized_values = {key: float(value) for key, value in self.weights.items()}
        if any(value < 0 for value in normalized_values.values()):
            raise DataValidationError("score weights must not be negative")
        if sum(normalized_values.values()) <= 0:
            raise DataValidationError("at least one score weight must be positive")
        object.__setattr__(self, "minimum_score", float(self.minimum_score))
        object.__setattr__(self, "publication_half_life", float(self.publication_half_life))
        object.__setattr__(self, "load_balance_penalty", float(self.load_balance_penalty))
        object.__setattr__(self, "senior_threshold", float(self.senior_threshold))
        object.__setattr__(self, "weights", MappingProxyType(normalized_values))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MatchConfig:
        """Construct a configuration while rejecting unknown keys."""

        if not isinstance(value, Mapping):
            raise DataValidationError("configuration must be an object")
        if any(not isinstance(key, str) for key in value):
            raise DataValidationError("configuration keys must be strings")
        allowed = {
            "reviewers_per_document",
            "minimum_score",
            "strategy",
            "current_year",
            "publication_half_life",
            "require_distinct_institutions",
            "load_balance_penalty",
            "minimum_senior_reviewers",
            "senior_threshold",
            "weights",
        }
        unknown = set(value) - allowed
        if unknown:
            raise DataValidationError(f"unknown configuration keys: {sorted(unknown)}")
        return cls(**dict(value))

    @classmethod
    def from_json(cls, path: str | Path) -> MatchConfig:
        """Load a configuration from UTF-8 JSON."""

        data = load_json_text(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise DataValidationError("configuration must be a JSON object")
        return cls.from_mapping(data)


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False
