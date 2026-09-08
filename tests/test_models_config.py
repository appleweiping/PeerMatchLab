from __future__ import annotations

import json
import math

import pytest

from peermatchlab.config import MatchConfig
from peermatchlab.models import (
    Assignment,
    Conflict,
    DataValidationError,
    Document,
    Expert,
    MatchPlan,
    MatchScore,
    Publication,
)
from peermatchlab.scoring import ScoreWeights


def test_publication_requires_title() -> None:
    with pytest.raises(DataValidationError):
        Publication("  ")


@pytest.mark.parametrize("identifier", [" leading", "trailing ", "bad\nvalue", "bad\x00value"])
def test_public_identifiers_reject_whitespace_and_controls(identifier: str) -> None:
    with pytest.raises(DataValidationError, match="id"):
        Publication("Title", id=identifier)
    with pytest.raises(DataValidationError, match="id"):
        Document(identifier, "Title")
    with pytest.raises(DataValidationError, match="id"):
        Expert(identifier, "Name")


@pytest.mark.parametrize("year", [1799, 2201])
def test_publication_year_bounds(year: int) -> None:
    with pytest.raises(DataValidationError):
        Publication("Title", year=year)


def test_document_cleans_and_deduplicates_terms() -> None:
    document = Document("d", "Title", topics=(" IR ", "IR", ""))
    assert document.topics == ("IR",)


def test_document_metadata_is_read_only() -> None:
    document = Document("d", "Title", metadata={"track": "search"})
    with pytest.raises(TypeError):
        document.metadata["track"] = "other"  # type: ignore[index]


def test_document_requires_positive_demand() -> None:
    with pytest.raises(DataValidationError):
        Document("d", "Title", required_experts=0)


def test_expert_rejects_negative_capacity() -> None:
    with pytest.raises(DataValidationError):
        Expert("e", "Name", capacity=-1)


def test_expert_rejects_invalid_bid() -> None:
    with pytest.raises(DataValidationError):
        Expert("e", "Name", bids={"d": 1.1})


def test_expert_text_contains_publications() -> None:
    expert = Expert("e", "Name", publications=(Publication("Vector search", "indexes"),))
    assert "Vector search" in expert.text
    assert "indexes" in expert.text


def test_assignment_rejects_zero_rank() -> None:
    with pytest.raises(DataValidationError):
        Assignment("d", "e", 0.4, 0)


def test_plan_filters_and_orders_document_assignments() -> None:
    plan = MatchPlan(
        assignments=(Assignment("d", "e2", 0.3, 2), Assignment("d", "e1", 0.4, 1)),
        unmet={},
        strategy="test",
        total_score=0.7,
    )
    assert [item.expert_id for item in plan.for_document("d")] == ["e1", "e2"]


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(DataValidationError, match="unknown configuration"):
        MatchConfig.from_mapping({"mystery": True})


def test_config_reads_json(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"reviewers_per_document": 1}), encoding="utf-8")
    assert MatchConfig.from_json(path).reviewers_per_document == 1


@pytest.mark.parametrize("strategy", ["optimal", "greedy"])
def test_config_accepts_supported_strategies(strategy: str) -> None:
    assert MatchConfig(strategy=strategy).strategy == strategy


def test_config_rejects_negative_weight() -> None:
    with pytest.raises(DataValidationError):
        MatchConfig(weights={"content": -1.0})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reviewers_per_document": 0}, "positive"),
        ({"strategy": "random"}, "strategy"),
        ({"current_year": 1700}, "current_year"),
        ({"current_year": 2201}, "current_year"),
        ({"publication_half_life": 0}, "half_life"),
        ({"weights": {"content": 0.0}}, "at least one"),
        ({"weights": {"content": 1.0, "mystery": 1.0}}, "unknown score"),
    ],
)
def test_config_validation_branches(kwargs, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        MatchConfig(**kwargs)


def test_config_json_must_be_an_object(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(DataValidationError, match="JSON object"):
        MatchConfig.from_json(path)


def test_score_weights_normalize() -> None:
    normalized = ScoreWeights(content=2, topics=1, bid=1, recency=0, seniority=0).normalized()
    assert normalized == {
        "content": 0.5,
        "topics": 0.25,
        "bid": 0.25,
        "recency": 0.0,
        "seniority": 0.0,
    }


def test_score_weights_mapping_treats_omitted_components_as_zero() -> None:
    assert ScoreWeights.from_mapping({"content": 1.0}).normalized() == {
        "content": 1.0,
        "topics": 0.0,
        "bid": 0.0,
        "recency": 0.0,
        "seniority": 0.0,
    }


def test_score_weights_reject_unknown_component() -> None:
    with pytest.raises(ValueError, match="unknown score"):
        ScoreWeights.from_mapping({"content": 1.0, "luck": 1.0})


def test_score_weights_reject_all_zero() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ScoreWeights(0, 0, 0, 0, 0)


def test_expert_rejects_invalid_seniority() -> None:
    with pytest.raises(DataValidationError, match="seniority"):
        Expert("e", "Name", seniority=1.2)


def test_plan_rejects_negative_unmet_count() -> None:
    with pytest.raises(DataValidationError, match="unmet"):
        MatchPlan((), {"d": -1}, "manual", 0.0)


def test_plan_rejects_inconsistent_total_score() -> None:
    with pytest.raises(DataValidationError, match="score sum"):
        MatchPlan((Assignment("d", "e", 0.5, 1),), {}, "manual", 99.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_score": math.nan},
        {"minimum_score": 10**1_000},
        {"publication_half_life": math.inf},
        {"reviewers_per_document": True},
        {"current_year": 2026.0},
        {"require_distinct_institutions": 1},
        {"load_balance_penalty": float("nan")},
        {"load_balance_penalty": -0.1},
        {"load_balance_penalty": 1.1},
        {"weights": {"content": math.nan}},
    ],
)
def test_config_rejects_non_finite_and_wrong_scalar_types(kwargs) -> None:
    with pytest.raises(DataValidationError):
        MatchConfig.from_mapping(kwargs)


def test_config_weights_are_immutable() -> None:
    config = MatchConfig(weights={"content": 1})
    with pytest.raises(TypeError):
        config.weights["content"] = 0.5  # type: ignore[index]


def test_domain_models_reject_non_finite_values_and_boolean_integers() -> None:
    with pytest.raises(DataValidationError):
        Expert("e", "Expert", seniority=math.nan)
    with pytest.raises(DataValidationError):
        Expert("e", "Expert", capacity=True)
    with pytest.raises(DataValidationError):
        Document("d", "Document", required_experts=True)
    with pytest.raises(DataValidationError):
        Assignment("d", "e", math.inf, 1)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Publication(1),
        lambda: Publication("Title", year=2025.0),
        lambda: Document(1, "Title"),
        lambda: Document(" ", "Title"),
        lambda: Document("d", " "),
        lambda: Document("d", "Title", topics=(1,)),
        lambda: Document("d", "Title", metadata=[]),
        lambda: Expert(1, "Name"),
        lambda: Expert(" ", "Name"),
        lambda: Expert("e", " "),
        lambda: Expert("e", "Name", institution=3),
        lambda: Expert("e", "Name", publications=("not-publication",)),
        lambda: Expert("e", "Name", bids=[]),
        lambda: Expert("e", "Name", bids={"": 0.0}),
        lambda: Expert("e", "Name", bids={"d": math.nan}),
        lambda: Expert("e", "Name", seniority=10**1_000),
        lambda: Conflict(1, "e"),
        lambda: Conflict(" ", "e"),
        lambda: MatchScore(1, "e", 0, 0, 0, 0, 0, 0),
        lambda: MatchScore("d", "e", math.nan, 0, 0, 0, 0, 0),
        lambda: MatchScore("d", "e", 1.1, 0, 0, 0, 0, 0),
        lambda: MatchScore("d", "e", 0, 0, 0, 0, 0, 0, affinity=math.nan),
        lambda: MatchScore("d", "e", 0, 0, 0, 0, 0, 0, affinity=1.1),
        lambda: MatchScore("d", "e", 0, 0, 0, 0, 0, 0, eligible=1),
        lambda: MatchScore("d", "e", 0, 0, 0, 0, 0, 0, reasons=(1,)),
        lambda: Assignment(1, "e", 0.5, 1),
        lambda: Assignment(" ", "e", 0.5, 1),
        lambda: Assignment("d", "e", 0.5, True),
        lambda: Assignment("d", "e", -0.1, 1),
        lambda: Assignment("d", "e", 0.5, 1, components=[]),
        lambda: Assignment("d", "e", 0.5, 1, components={1: 0.5}),
        lambda: Assignment("d", "e", 0.5, 1, components={"x": math.nan}),
        lambda: Assignment("d", "e", 0.5, 1, components={"x": 1.1}),
        lambda: MatchPlan(("not-assignment",), {}, "manual", 0),
        lambda: MatchPlan((), [], "manual", 0),
        lambda: MatchPlan((), {"": 1}, "manual", 0),
        lambda: MatchPlan((), {"d": True}, "manual", 0),
        lambda: MatchPlan((), {}, " ", 0),
        lambda: MatchPlan((), {}, "manual", math.nan),
    ],
)
def test_domain_model_validation_paths(factory) -> None:
    with pytest.raises(DataValidationError):
        factory()


def test_match_score_preserves_positional_eligibility_and_reason_arguments() -> None:
    score = MatchScore("d", "e", 0.5, 0.5, 0, 0, 0, 0, False, ("conflict",))

    assert not score.eligible
    assert score.reasons == ("conflict",)
    assert score.affinity is None


@pytest.mark.parametrize(
    "value",
    [
        [],
        {1: 2},
        {"weights": []},
        {"weights": {1: 1.0}},
    ],
)
def test_config_rejects_non_object_shapes(value) -> None:
    with pytest.raises(DataValidationError):
        MatchConfig.from_mapping(value)


@pytest.mark.parametrize("value", ["NaN", "1e999"])
def test_config_json_rejects_non_finite_numbers(tmp_path, value: str) -> None:
    path = tmp_path / "config.json"
    path.write_text(f'{{"minimum_score":{value}}}', encoding="utf-8")

    with pytest.raises(DataValidationError, match="non-finite JSON number"):
        MatchConfig.from_json(path)
