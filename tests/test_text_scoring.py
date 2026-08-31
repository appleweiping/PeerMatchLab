from __future__ import annotations

import pytest

from peermatchlab.models import Conflict, Document, Expert, Publication
from peermatchlab.scoring import MatchScorer, ScoreWeights
from peermatchlab.text import TfIdfSpace, cosine, set_overlap, tokenize


def test_tokenize_is_unicode_aware_and_casefolds() -> None:
    assert tokenize("RÉSUMÉ Search 搜索") == ("résumé", "search", "搜索")


def test_tokenize_removes_stopwords_and_single_characters() -> None:
    assert tokenize("A model for the X graph") == ("model", "graph")


def test_cosine_identical_vectors() -> None:
    assert cosine({"a": 2.0}, {"a": 4.0}) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors() -> None:
    assert cosine({"a": 1.0}, {"b": 1.0}) == 0.0


def test_cosine_empty_vector() -> None:
    assert cosine({}, {"b": 1.0}) == 0.0


def test_set_overlap_handles_empty_sets() -> None:
    assert set_overlap([], []) == 0.0


def test_set_overlap_normalizes_case() -> None:
    assert set_overlap(["Search", "ML"], ["search", "IR"]) == pytest.approx(1 / 3)


def test_tfidf_ignores_unknown_terms() -> None:
    space = TfIdfSpace.fit(["known tokens"])
    assert space.transform("unknown") == {}


def test_tfidf_uses_sublinear_term_frequency() -> None:
    space = TfIdfSpace.fit(["rank rank rank index", "index"])
    vector = space.transform("rank rank rank index")
    assert vector["rank"] > vector["index"]


def test_matching_prefers_relevant_expert(documents, experts) -> None:
    scorer = MatchScorer(documents, experts)
    assert scorer.score("paper-1", "expert-a").total > scorer.score("paper-1", "expert-b").total


def test_conflict_makes_pair_ineligible(documents, experts) -> None:
    scorer = MatchScorer(
        documents,
        experts,
        conflicts=[Conflict("paper-1", "expert-a", "same lab")],
    )
    score = scorer.score("paper-1", "expert-a")
    assert not score.eligible
    assert score.reasons == ("excluded: same lab",)


def test_zero_capacity_makes_pair_ineligible() -> None:
    scorer = MatchScorer([Document("d", "Search")], [Expert("e", "E", capacity=0)])
    assert not scorer.score("d", "e").eligible


def test_positive_bid_increases_score() -> None:
    document = Document("d", "Topic")
    positive = Expert("positive", "P", bids={"d": 1.0})
    negative = Expert("negative", "N", bids={"d": -1.0})
    scorer = MatchScorer([document], [positive, negative], weights=ScoreWeights(0, 0, 1, 0, 0))
    assert scorer.score("d", "positive").total == 1.0
    assert scorer.score("d", "negative").total == 0.0


def test_recent_publication_scores_above_old_publication() -> None:
    document = Document("d", "Topic")
    recent = Expert("recent", "R", publications=(Publication("Recent", year=2026),))
    old = Expert("old", "O", publications=(Publication("Old", year=2000),))
    scorer = MatchScorer(
        [document], [recent, old], weights=ScoreWeights(0, 0, 0, 1, 0), current_year=2026
    )
    assert scorer.score("d", "recent").total > scorer.score("d", "old").total


def test_matrix_order_is_deterministic(documents, experts) -> None:
    matrix = MatchScorer(reversed(documents), reversed(experts)).matrix()
    assert [(item.document_id, item.expert_id) for item in matrix] == sorted(
        (item.document_id, item.expert_id) for item in matrix
    )


def test_unknown_id_raises_key_error(documents, experts) -> None:
    with pytest.raises(KeyError):
        MatchScorer(documents, experts).score("missing", "expert-a")


def test_duplicate_document_ids_are_rejected(experts) -> None:
    with pytest.raises(ValueError, match="document identifiers"):
        MatchScorer([Document("d", "One"), Document("d", "Two")], experts)


def test_duplicate_expert_ids_are_rejected(documents) -> None:
    with pytest.raises(ValueError, match="expert identifiers"):
        MatchScorer(documents, [Expert("e", "One"), Expert("e", "Two")])


def test_scorer_requires_documents_and_experts(documents, experts) -> None:
    with pytest.raises(ValueError, match="document"):
        MatchScorer([], experts)
    with pytest.raises(ValueError, match="expert"):
        MatchScorer(documents, [])


@pytest.mark.parametrize("value", [True, float("nan"), 10**1_000])
def test_score_weights_reject_invalid_numeric_types(value) -> None:
    with pytest.raises(ValueError, match="finite"):
        ScoreWeights(content=value)


@pytest.mark.parametrize("value", [[], {1: 1.0}])
def test_score_weights_mapping_rejects_invalid_shapes(value) -> None:
    with pytest.raises(ValueError):
        ScoreWeights.from_mapping(value)


def test_scorer_rejects_wrong_weight_object(documents, experts) -> None:
    with pytest.raises(ValueError, match="ScoreWeights"):
        MatchScorer(documents, experts, weights=True)


def test_unknown_conflict_references_are_rejected(documents, experts) -> None:
    with pytest.raises(ValueError, match="unknown identifiers"):
        MatchScorer(documents, experts, conflicts=[Conflict("missing", "expert-a")])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_year": True},
        {"current_year": 1700},
        {"current_year": 2201},
        {"publication_half_life": float("nan")},
        {"publication_half_life": True},
    ],
)
def test_scorer_rejects_invalid_runtime_controls(documents, experts, kwargs) -> None:
    with pytest.raises(ValueError):
        MatchScorer(documents, experts, **kwargs)
