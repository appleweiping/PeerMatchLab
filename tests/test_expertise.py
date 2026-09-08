from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import pytest

import peermatchlab.expertise as expertise_module
from peermatchlab.expertise import (
    EvidenceDocument,
    ExpertiseConfig,
    ExpertiseCorpus,
    ExpertiseModel,
    ExpertiseScore,
    TermContribution,
    build_expertise_corpus,
    generate_expertise,
)
from peermatchlab.models import DataValidationError, Document, Expert, Publication


def _documents() -> tuple[Document, ...]:
    return (Document(id="paper", title="graph neural"),)


def _experts() -> tuple[Expert, ...]:
    return (
        Expert(
            id="reviewer-a",
            name="A",
            publications=(
                Publication("graph graph systems", year=2024),
                Publication("neural methods", year=2023),
            ),
        ),
        Expert(
            id="reviewer-b",
            name="B",
            publications=(Publication("biology cells", year=2022),),
        ),
    )


def _score(run: object, reviewer_id: str) -> ExpertiseScore:
    scores = run.scores  # type: ignore[attr-defined]
    return next(item for item in scores if item.expert_id == reviewer_id)


def test_tfidf_aggregate_matches_independent_hand_calculation() -> None:
    config = ExpertiseConfig(model="tfidf", aggregation="aggregate", include_profile=False)
    run = generate_expertise(_documents(), _experts(), config=config)
    actual = _score(run, "reviewer-a")

    graph_tf = 1.0 + math.log(2.0)
    expected = (graph_tf + 1.0) / (math.sqrt(2.0) * math.sqrt(graph_tf**2 + 3.0))
    assert actual.score == pytest.approx(expected, abs=1e-15)
    assert actual.raw_score == actual.score
    assert actual.selected_evidence_id == "aggregate:reviewer-a"
    assert sum(item.value for item in actual.contributions) == pytest.approx(expected)
    assert [item.term for item in actual.contributions] == ["graph", "neural"]
    assert _score(run, "reviewer-b").score == 0.0


@pytest.mark.parametrize("aggregation", ["max", "average"])
def test_tfidf_atomic_aggregations_match_independent_oracle(aggregation: str) -> None:
    config = ExpertiseConfig(
        model="tfidf",
        aggregation=aggregation,
        include_profile=False,  # type: ignore[arg-type]
    )
    run = generate_expertise(_documents(), _experts(), config=config)
    actual = _score(run, "reviewer-a")

    graph_tf = 1.0 + math.log(2.0)
    graph_document = graph_tf / (math.sqrt(2.0) * math.sqrt(graph_tf**2 + 1.0))
    neural_document = 0.5
    expected = (
        max(graph_document, neural_document)
        if aggregation == "max"
        else (graph_document + neural_document) / 2.0
    )
    assert actual.score == pytest.approx(expected, abs=1e-15)
    assert sum(item.value for item in actual.contributions) == pytest.approx(expected)
    assert actual.evidence_count == 2
    if aggregation == "max":
        assert actual.selected_evidence_id == "publication:reviewer-a:position:1"
    else:
        assert actual.selected_evidence_id is None


@pytest.mark.parametrize("aggregation", ["aggregate", "max", "average"])
def test_bm25_variants_match_independent_sparse_math(aggregation: str) -> None:
    config = ExpertiseConfig(
        model="bm25",
        aggregation=aggregation,
        include_profile=False,  # type: ignore[arg-type]
    )
    run = generate_expertise(_documents(), _experts(), config=config)
    actual = _score(run, "reviewer-a")
    k1 = 1.2
    b = 0.75
    if aggregation == "aggregate":
        # Two aggregate reviewer documents of lengths 5 and 2.
        idf = math.log(1.0 + (2.0 - 1.0 + 0.5) / (1.0 + 0.5))
        avgdl = 3.5
        norm = k1 * (1.0 - b + b * 5.0 / avgdl)
        graph = idf * (2.0 * (k1 + 1.0)) / (2.0 + norm)
        neural = idf * (k1 + 1.0) / (1.0 + norm)
        raw = graph + neural
    else:
        # Three atomic publications of lengths 3, 2, and 2.
        idf = math.log(1.0 + (3.0 - 1.0 + 0.5) / (1.0 + 0.5))
        avgdl = 7.0 / 3.0
        graph_norm = k1 * (1.0 - b + b * 3.0 / avgdl)
        neural_norm = k1 * (1.0 - b + b * 2.0 / avgdl)
        graph = idf * (2.0 * (k1 + 1.0)) / (2.0 + graph_norm)
        neural = idf * (k1 + 1.0) / (1.0 + neural_norm)
        raw = max(graph, neural) if aggregation == "max" else (graph + neural) / 2.0
    assert actual.raw_score == pytest.approx(raw, abs=1e-15)
    assert actual.score == pytest.approx(raw / (1.0 + raw), abs=1e-15)
    assert sum(item.value for item in actual.contributions) == pytest.approx(raw)


def test_tokenizer_fields_dates_content_and_duplicate_filters_are_explicit() -> None:
    expert = Expert(
        id="r",
        name="R",
        summary="THE profile_only",
        topics=("Graph_Nets",),
        publications=(
            Publication("Graph Étude", "useful abstract", 2022),
            Publication("graph étude", "useful abstract", 2022),
            Publication("Old evidence", "useful", 2019),
            Publication("No abstract", "", 2023),
            Publication("Undated evidence", "useful", None),
        ),
    )
    config = ExpertiseConfig(
        submission_fields=("keywords",),
        profile_fields=("topics",),
        publication_fields=("title", "abstract"),
        minimum_publication_year=2020,
        maximum_publication_year=2024,
        undated_publications="exclude",
        require_publication_abstract=True,
        stopwords=("the",),
        minimum_token_length=2,
    )
    corpus = build_expertise_corpus(
        (Document("p", "ignored", keywords=("GRAPH", "x", "THE", "étude")),),
        (expert,),
        config=config,
    )
    assert corpus.submissions[0].fields == {"keywords": "GRAPH x THE étude"}
    assert corpus.submissions[0].tokens == ("graph", "étude")
    evidence = corpus.reviewer_evidence["r"]
    assert evidence[0].tokens == ("graph", "nets")
    assert evidence[1].tokens == ("graph", "étude", "useful", "abstract")
    assert corpus.filters == {
        "duplicate_publications": 1,
        "date_or_content_filtered_publications": 3,
    }


def test_empty_evidence_and_stopword_only_query_produce_finite_zero() -> None:
    config = ExpertiseConfig(include_profile=False, undated_publications="exclude")
    run = generate_expertise(
        (Document("p", "the and"),),
        (Expert("r", "R", publications=(Publication("unused", year=None),)),),
        config=config,
    )
    assert run.model.document_count == 0
    assert run.model.average_document_length == 0.0
    assert run.model.inverse_document_frequency == {}
    assert run.scores[0].score == 0.0
    assert run.scores[0].contributions == ()


def test_model_round_trip_replays_bit_identical_scores(tmp_path: Path) -> None:
    run = generate_expertise(
        _documents(),
        _experts(),
        config=ExpertiseConfig(model="bm25", aggregation="average", include_profile=False),
    )
    model_path = tmp_path / "model.json"
    run.model.save(model_path)
    loaded = ExpertiseModel.load(model_path)
    assert loaded.to_dict() == run.model.to_dict()
    assert loaded.score(run.corpus.submissions) == run.scores
    assert loaded.score_documents(_documents()) == run.scores


def _model_payload(model: ExpertiseModel) -> bytes:
    return (
        json.dumps(model.to_dict(), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _model_at_its_exact_file_limit() -> ExpertiseModel:
    limit = 10_000
    for _attempt in range(5):
        model = generate_expertise(
            _documents(),
            _experts(),
            config=ExpertiseConfig(
                model="bm25",
                aggregation="average",
                include_profile=False,
                max_model_file_bytes=limit,
            ),
        ).model
        measured = len(_model_payload(model))
        if measured == limit:
            return model
        limit = measured
    raise AssertionError("model-file limit did not converge to its deterministic payload size")


def test_model_save_accepts_exact_byte_boundary_and_rejects_one_below(tmp_path: Path) -> None:
    exact = _model_at_its_exact_file_limit()
    path = tmp_path / "model.json"
    exact.save(path)
    assert path.read_bytes() == _model_payload(exact)
    assert path.stat().st_size == exact.config.max_model_file_bytes
    assert ExpertiseModel.load(path).to_dict() == exact.to_dict()

    too_small = generate_expertise(
        _documents(),
        _experts(),
        config=ExpertiseConfig(
            model="bm25",
            aggregation="average",
            include_profile=False,
            max_model_file_bytes=exact.config.max_model_file_bytes - 1,
        ),
    ).model
    assert len(_model_payload(too_small)) == exact.config.max_model_file_bytes
    rejected = tmp_path / "rejected.json"
    with pytest.raises(DataValidationError, match="max_model_file_bytes"):
        too_small.save(rejected)
    assert not rejected.exists()


def test_model_save_limit_failure_never_touches_missing_existing_or_hardlink_targets(
    tmp_path: Path,
) -> None:
    model = generate_expertise(
        (Document("p", "Graph"),),
        (Expert("r", "R", summary="Graph"),),
        config=ExpertiseConfig(max_model_file_bytes=1),
    ).model
    missing = tmp_path / "missing.json"
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"existing-sentinel")
    hardlink_source = tmp_path / "hardlink-source.json"
    hardlink_source.write_bytes(b"hardlink-sentinel")
    hardlink_target = tmp_path / "hardlink-target.json"
    os.link(hardlink_source, hardlink_target)

    for target in (missing, existing, hardlink_target):
        before = target.read_bytes() if target.exists() else None
        with pytest.raises(DataValidationError, match="max_model_file_bytes"):
            model.save(target)
        assert (target.read_bytes() if target.exists() else None) == before
    assert os.path.samefile(hardlink_source, hardlink_target)
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_model_save_replaces_only_the_requested_hardlink_name(tmp_path: Path) -> None:
    model = generate_expertise(_documents(), _experts()).model
    untouched = tmp_path / "source.json"
    untouched.write_bytes(b"hardlink-sentinel")
    target = tmp_path / "model.json"
    os.link(untouched, target)

    model.save(target)

    assert untouched.read_bytes() == b"hardlink-sentinel"
    assert not os.path.samefile(untouched, target)
    assert ExpertiseModel.load(target).to_dict() == model.to_dict()


@pytest.mark.parametrize("target_exists", [False, True])
@pytest.mark.parametrize("interruption", [OSError, KeyboardInterrupt, SystemExit])
def test_model_save_rolls_back_replace_failures_and_base_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_exists: bool,
    interruption: type[BaseException],
) -> None:
    model = generate_expertise(_documents(), _experts()).model
    target = tmp_path / "model.json"
    if target_exists:
        target.write_bytes(b"existing-sentinel")

    def interrupt(_source: object, _destination: object) -> None:
        raise interruption("simulated replace failure")

    monkeypatch.setattr(expertise_module.os, "replace", interrupt)
    with pytest.raises(interruption, match="simulated replace failure"):
        model.save(target)
    if target_exists:
        assert target.read_bytes() == b"existing-sentinel"
    else:
        assert not target.exists()
    assert not list(tmp_path.glob(".model.json-*"))


def test_legacy_model_without_model_file_limit_uses_compatible_default(tmp_path: Path) -> None:
    model = generate_expertise(
        (Document("p", "Graph"),),
        (Expert("r", "R", summary="Graph"),),
        config=ExpertiseConfig(max_input_file_bytes=1),
    ).model
    state = model.to_dict()
    del state["config"]["max_model_file_bytes"]  # type: ignore[index]
    path = tmp_path / "legacy-model.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    loaded = ExpertiseModel.load(path)
    assert loaded.config.max_input_file_bytes == 1
    assert loaded.config.max_model_file_bytes == 64 * 1024 * 1024


def test_fit_rejects_mismatched_preprocessing_configuration() -> None:
    corpus = build_expertise_corpus(
        _documents(), _experts(), config=ExpertiseConfig(include_profile=False)
    )
    with pytest.raises(DataValidationError, match="configurations must match"):
        ExpertiseModel.fit(
            corpus,
            config=ExpertiseConfig(include_profile=False, minimum_token_length=3),
        )


def test_duplicate_publication_ids_are_rejected_even_when_text_differs() -> None:
    expert = Expert(
        "r",
        "R",
        publications=(
            Publication("Graph", id="same"),
            Publication("Retrieval", id="same"),
        ),
    )
    with pytest.raises(DataValidationError, match="duplicate publication id"):
        build_expertise_corpus((Document("p", "Graph"),), (expert,))


def test_explicit_and_fallback_publication_ids_have_disjoint_namespaces() -> None:
    expert = Expert(
        "r",
        "R",
        publications=(
            Publication("Graph systems", id="2"),
            Publication("Neural retrieval"),
        ),
    )
    corpus = build_expertise_corpus(
        (Document("p", "Graph"),),
        (expert,),
        config=ExpertiseConfig(include_profile=False),
    )
    assert [item.id for item in corpus.reviewer_evidence["r"]] == [
        "publication:r:id:2",
        "publication:r:position:2",
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema_version=2), "schema_version"),
        (lambda value: value["statistics"].update(document_count=99), "document_count"),
        (
            lambda value: value["statistics"]["inverse_document_frequency"].update(graph=99),
            "IDF statistics",
        ),
        (
            lambda value: value["reviewers"][0]["evidence"][0].update(owner_id="wrong"),
            "owner or kind",
        ),
        (
            lambda value: value["reviewers"][0]["evidence"][0]["tokens"].append("forged"),
            "tokens disagree",
        ),
        (
            lambda value: value["config"].update(max_total_publications=1),
            "max_total_publications",
        ),
        (
            lambda value: value["config"].update(require_publication_abstract=True),
            "required abstract",
        ),
    ],
)
def test_model_loader_rejects_tampered_derived_state(
    tmp_path: Path, mutate: object, message: str
) -> None:
    model = generate_expertise(
        _documents(), _experts(), config=ExpertiseConfig(include_profile=False)
    ).model
    value = copy.deepcopy(model.to_dict())
    mutate(value)  # type: ignore[operator]
    path = tmp_path / "model.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DataValidationError, match=message):
        ExpertiseModel.load(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "cosine"},
        {"aggregation": "median"},
        {"submission_fields": ()},
        {"profile_fields": ("unknown",)},
        {"publication_fields": ("title", "title")},
        {"include_profile": 1},
        {"require_submission_abstract": 1},
        {"require_submission_abstract": True, "submission_fields": ("title",)},
        {"require_publication_abstract": "yes"},
        {"require_publication_abstract": True, "publication_fields": ("title",)},
        {"minimum_publication_year": True},
        {"minimum_publication_year": 2025, "maximum_publication_year": 2024},
        {"undated_publications": "guess"},
        {"stopwords": ("",)},
        {"minimum_token_length": 0},
        {"bm25_k1": float("nan")},
        {"bm25_b": 1.1},
        {"minimum_output_score": float("inf")},
    ],
)
def test_configuration_rejects_ambiguous_or_non_finite_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(DataValidationError):
        ExpertiseConfig(**kwargs)  # type: ignore[arg-type]


def test_configuration_json_round_trip_and_unknown_key(tmp_path: Path) -> None:
    config = ExpertiseConfig(
        stopwords=("Beta", "alpha", "beta"),
        max_document_characters=1234,
        max_document_bytes=2345,
        max_scanned_matches=345,
        max_token_characters=123,
    )
    assert config.stopwords == ("alpha", "beta")
    assert ExpertiseConfig.from_mapping(config.as_dict()) == config
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    assert ExpertiseConfig.from_json(path) == config
    with pytest.raises(DataValidationError, match="unknown expertise"):
        ExpertiseConfig.from_mapping({"surprise": 1})


def test_tokenizer_fails_early_for_character_byte_match_and_token_limits() -> None:
    expert = (Expert("r", "R", summary="Graph"),)
    with pytest.raises(DataValidationError, match="max_document_characters"):
        generate_expertise(
            (Document("p", "the " * (10 * 1024 * 1024 // 4)),),
            expert,
            config=ExpertiseConfig(max_document_characters=1_000),
        )
    with pytest.raises(DataValidationError, match="max_document_bytes"):
        generate_expertise(
            (Document("p", "é" * 600),),
            expert,
            config=ExpertiseConfig(
                max_document_characters=1_000,
                max_document_bytes=1_000,
            ),
        )
    with pytest.raises(DataValidationError, match="max_scanned_matches"):
        generate_expertise(
            (Document("p", "a a a a a a"),),
            expert,
            config=ExpertiseConfig(max_scanned_matches=5),
        )
    with pytest.raises(DataValidationError, match="max_token_characters"):
        generate_expertise(
            (Document("p", "x" * 2_000_000),),
            expert,
            config=ExpertiseConfig(
                max_document_characters=3_000_000,
                max_document_bytes=3_000_000,
                max_token_characters=100_000,
            ),
        )


@pytest.mark.parametrize(
    ("config", "documents", "experts", "message"),
    [
        (
            ExpertiseConfig(max_submissions=1),
            (Document("a", "A"), Document("b", "B")),
            (Expert("r", "R"),),
            "max_submissions",
        ),
        (
            ExpertiseConfig(max_reviewers=1),
            (Document("a", "A"),),
            (Expert("r1", "R1"), Expert("r2", "R2")),
            "max_reviewers",
        ),
        (
            ExpertiseConfig(max_total_publications=1),
            (Document("a", "A"),),
            (
                Expert(
                    "r",
                    "R",
                    publications=(Publication("One"), Publication("Two")),
                ),
            ),
            "max_total_publications",
        ),
        (
            ExpertiseConfig(max_tokens_per_document=1),
            (Document("a", "two tokens"),),
            (Expert("r", "R"),),
            "max_tokens_per_document",
        ),
    ],
)
def test_corpus_resource_limits_fail_closed(
    config: ExpertiseConfig,
    documents: tuple[Document, ...],
    experts: tuple[Expert, ...],
    message: str,
) -> None:
    with pytest.raises(DataValidationError, match=message):
        build_expertise_corpus(documents, experts, config=config)


def test_pair_and_vocabulary_limits_fail_before_large_outputs() -> None:
    pair_config = ExpertiseConfig(max_pairs=1, include_profile=False)
    corpus = build_expertise_corpus(
        (Document("p", "Graph"),),
        (Expert("a", "A"), Expert("b", "B")),
        config=pair_config,
    )
    with pytest.raises(DataValidationError, match="max_pairs"):
        ExpertiseModel.fit(corpus, config=pair_config).score(corpus.submissions)

    vocabulary_config = ExpertiseConfig(max_vocabulary_terms=1, include_profile=False)
    vocabulary_corpus = build_expertise_corpus(
        (Document("p", "Graph"),),
        (Expert("r", "R", publications=(Publication("graph systems"),)),),
        config=vocabulary_config,
    )
    with pytest.raises(DataValidationError, match="max_vocabulary_terms"):
        ExpertiseModel.fit(vocabulary_corpus, config=vocabulary_config)

    contribution_config = ExpertiseConfig(max_total_contributions=1)
    contribution_corpus = build_expertise_corpus(
        (Document("p", "Graph"),),
        (Expert("r1", "R1", summary="Graph"), Expert("r2", "R2", summary="Graph")),
        config=contribution_config,
    )
    with pytest.raises(DataValidationError, match="max_total_contributions"):
        ExpertiseModel.fit(contribution_corpus, config=contribution_config).score(
            contribution_corpus.submissions
        )


def test_required_submission_abstract_applies_to_fit_and_replayed_queries() -> None:
    config = ExpertiseConfig(require_submission_abstract=True)
    with pytest.raises(DataValidationError, match="must contain an abstract"):
        generate_expertise(
            (Document("p", "Graph"),), (Expert("r", "R", summary="Graph"),), config=config
        )

    run = generate_expertise(
        (Document("p", "Graph", abstract="Ranking"),),
        (Expert("r", "R", summary="Graph"),),
        config=config,
    )
    with pytest.raises(DataValidationError, match="must contain an abstract"):
        run.model.score_documents((Document("new", "Graph"),))

    missing = EvidenceDocument("new", "new", "submission", {"title": "Graph"}, ("graph",))
    with pytest.raises(DataValidationError, match="required abstract"):
        run.model.score((missing,))
    forged = EvidenceDocument(
        "new",
        "new",
        "submission",
        {"title": "Graph", "abstract": "Ranking"},
        ("unrelated",),
    )
    with pytest.raises(DataValidationError, match="tokens disagree"):
        run.model.score((forged,))


def test_public_models_validate_non_finite_and_wrong_evidence() -> None:
    with pytest.raises(DataValidationError, match="add to raw_score"):
        ExpertiseScore(
            "p",
            "r",
            0.5,
            0.5,
            "tfidf",
            "aggregate",
            1,
            None,
            (TermContribution("graph", 0.25),),
        )
    model = generate_expertise(
        _documents(), _experts(), config=ExpertiseConfig(include_profile=False)
    ).model
    wrong = EvidenceDocument("x", "x", "profile", {}, ())
    with pytest.raises(DataValidationError, match="submission evidence"):
        model.score((wrong,))

    with pytest.raises(DataValidationError, match="identifiers"):
        EvidenceDocument(" ", "r", "profile", {}, ())
    with pytest.raises(DataValidationError, match="term"):
        TermContribution(" ", 0.0)
    with pytest.raises(DataValidationError, match="identifiers"):
        ExpertiseScore(" ", "r", 0.0, 0.0, "tfidf", "aggregate", 0, None, ())
    with pytest.raises(DataValidationError, match="selected_evidence_id"):
        ExpertiseScore("p", "r", 0.0, 0.0, "tfidf", "max", 0, " ", ())


def test_expertise_score_is_deeply_frozen_and_enforces_model_semantics() -> None:
    mutable = [TermContribution("graph", 0.5)]
    score = ExpertiseScore(
        "p",
        "r",
        0.5,
        0.5,
        "tfidf",
        "max",
        1,
        "profile:r",
        mutable,  # type: ignore[arg-type]
    )
    mutable.clear()
    assert score.contributions == (TermContribution("graph", 0.5),)
    with pytest.raises(DataValidationError, match="model normalization"):
        ExpertiseScore("p", "r", 0.4, 0.5, "tfidf", "max", 1, "profile:r", score.contributions)
    with pytest.raises(DataValidationError, match="model normalization"):
        ExpertiseScore(
            "p",
            "r",
            0.4,
            1.0,
            "bm25",
            "max",
            1,
            "profile:r",
            (TermContribution("graph", 1.0),),
        )
    with pytest.raises(DataValidationError, match="without evidence"):
        ExpertiseScore("p", "r", 0.1, 0.1, "tfidf", "max", 0, None, ())
    with pytest.raises(DataValidationError, match="unique terms"):
        ExpertiseScore(
            "p",
            "r",
            0.5,
            0.5,
            "tfidf",
            "max",
            1,
            "profile:r",
            (TermContribution("graph", 0.25), TermContribution("graph", 0.25)),
        )
    with pytest.raises(DataValidationError, match="zero-valued"):
        ExpertiseScore(
            "p",
            "r",
            0.0,
            0.0,
            "tfidf",
            "max",
            1,
            "profile:r",
            (TermContribution("graph", 0.0),),
        )


def test_evidence_document_snapshots_mutable_inputs_and_rejects_string_subclasses() -> None:
    fields = {"title": "Graph"}
    tokens = ["graph"]
    evidence = EvidenceDocument(
        "profile:r",
        "r",
        "profile",
        fields,
        tokens,  # type: ignore[arg-type]
    )
    fields["title"] = "Mutated"
    tokens[0] = "mutated"
    assert dict(evidence.fields) == {"title": "Graph"}
    assert evidence.tokens == ("graph",)
    with pytest.raises(TypeError):
        evidence.fields["title"] = "Mutated"  # type: ignore[index]

    class DerivedString(str):
        pass

    with pytest.raises(DataValidationError, match="fields must map"):
        EvidenceDocument("profile:r", "r", "profile", {"title": DerivedString("Graph")}, ("graph",))
    with pytest.raises(DataValidationError, match="tokens must be"):
        EvidenceDocument("profile:r", "r", "profile", {"title": "Graph"}, (DerivedString("graph"),))


@pytest.mark.parametrize(
    "override",
    [
        {"max_submissions": 10**100},
        {"max_input_file_bytes": 10**100},
        {"max_model_file_bytes": 10**100},
        {"max_token_characters": 10**100},
    ],
)
def test_expertise_config_rejects_extreme_public_limits_as_domain_errors(
    override: dict[str, int],
) -> None:
    with pytest.raises(DataValidationError, match="integer between"):
        ExpertiseConfig(**override)  # type: ignore[arg-type]


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_model_loader_rejects_non_integer_schema_versions(
    tmp_path: Path, schema_version: object
) -> None:
    state = generate_expertise(
        _documents(), _experts(), config=ExpertiseConfig(include_profile=False)
    ).model.to_dict()
    state["schema_version"] = schema_version
    path = tmp_path / "model.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(DataValidationError, match="schema_version"):
        ExpertiseModel.load(path)


def test_model_loader_rejects_whitespace_only_reviewer_id(tmp_path: Path) -> None:
    state = generate_expertise(
        (Document("p", "Graph"),),
        (Expert("r", "R"),),
        config=ExpertiseConfig(include_profile=False),
    ).model.to_dict()
    state["reviewers"][0]["id"] = " "  # type: ignore[index]
    path = tmp_path / "model.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(DataValidationError, match="reviewer ids"):
        ExpertiseModel.load(path)


@pytest.mark.parametrize(
    ("target", "replacement", "message"),
    [
        ("profile_id", "profile:other", "owner or kind"),
        ("profile_year", 2024, "cannot have a year"),
        ("publication_owner", "publication:other:id:pub", "owner or kind"),
        ("publication_position", "publication:r:position:01", "owner or kind"),
        ("publication_control", "publication:r:id:bad\u0001id", "identifiers"),
    ],
)
def test_model_loader_rejects_forged_evidence_namespaces(
    tmp_path: Path, target: str, replacement: object, message: str
) -> None:
    state = generate_expertise(
        (Document("p", "Graph"),),
        (
            Expert(
                "r",
                "R",
                summary="Graph",
                publications=(Publication("Graph systems", id="pub"),),
            ),
        ),
    ).model.to_dict()
    evidence = state["reviewers"][0]["evidence"]  # type: ignore[index]
    if target == "profile_id":
        evidence[0]["id"] = replacement
    elif target == "profile_year":
        evidence[0]["year"] = replacement
    else:
        evidence[1]["id"] = replacement
    path = tmp_path / "model.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(DataValidationError, match=message):
        ExpertiseModel.load(path)


def test_score_rejects_forged_selected_evidence_namespaces() -> None:
    base = ("p", "r", 0.0, 0.0, "tfidf")
    with pytest.raises(DataValidationError, match="reviewer aggregate"):
        ExpertiseScore(*base, "aggregate", 1, "aggregate:other", ())
    with pytest.raises(DataValidationError, match="owned by their reviewer"):
        ExpertiseScore(*base, "max", 1, "publication:other:id:x", ())
    with pytest.raises(DataValidationError, match="average"):
        ExpertiseScore(*base, "average", 1, "publication:r:id:x", ())


def test_corpus_rejects_owner_key_mismatch_and_successful_fit_replays() -> None:
    config = ExpertiseConfig()
    corpus = build_expertise_corpus(
        (Document("p", "Graph"),), (Expert("r", "R", summary="Graph"),), config=config
    )
    with pytest.raises(DataValidationError, match="owner or kind"):
        ExpertiseCorpus(
            corpus.submissions,
            {"other": corpus.reviewer_evidence["r"]},
            corpus.filters,
            config,
        )
    forged_submission = EvidenceDocument("p", "other", "submission", {"title": "Graph"}, ("graph",))
    with pytest.raises(DataValidationError, match="owner id"):
        ExpertiseCorpus((forged_submission,), corpus.reviewer_evidence, corpus.filters, config)
    model = ExpertiseModel.fit(corpus, config=config)
    assert ExpertiseModel.from_mapping(model.to_dict()).to_dict() == model.to_dict()


def test_public_iterables_are_consumed_only_to_limit_plus_one() -> None:
    def documents() -> object:
        yield Document("p1", "Graph")
        yield Document("p2", "Graph")
        raise AssertionError("submission iterator was over-consumed")

    with pytest.raises(DataValidationError, match="max_submissions"):
        build_expertise_corpus(
            documents(),  # type: ignore[arg-type]
            (Expert("r", "R"),),
            config=ExpertiseConfig(max_submissions=1),
        )

    def reviewers() -> object:
        yield Expert("r1", "R1")
        yield Expert("r2", "R2")
        raise AssertionError("reviewer iterator was over-consumed")

    with pytest.raises(DataValidationError, match="max_reviewers"):
        build_expertise_corpus(
            (Document("p", "Graph"),),
            reviewers(),  # type: ignore[arg-type]
            config=ExpertiseConfig(max_reviewers=1),
        )

    expert = Expert("r", "R")

    def publications() -> object:
        yield Publication("One")
        yield Publication("Two")
        raise AssertionError("publication iterator was over-consumed")

    object.__setattr__(expert, "publications", publications())
    with pytest.raises(DataValidationError, match="configured limit"):
        build_expertise_corpus(
            (Document("p", "Graph"),),
            (expert,),
            config=ExpertiseConfig(max_publications_per_reviewer=1),
        )

    model = generate_expertise(
        (Document("p", "Graph"),), (Expert("r", "R", summary="Graph"),)
    ).model

    def queries() -> object:
        yield EvidenceDocument("q1", "q1", "submission", {"title": "Graph"}, ("graph",))
        yield EvidenceDocument("q2", "q2", "submission", {"title": "Graph"}, ("graph",))
        raise AssertionError("query iterator was over-consumed")

    limited = generate_expertise(
        (Document("p", "Graph"),),
        (Expert("r", "R", summary="Graph"),),
        config=ExpertiseConfig(max_submissions=1),
    ).model
    with pytest.raises(DataValidationError, match="configured limit"):
        limited.score(queries())  # type: ignore[arg-type]
    assert model.score_documents((Document("new", "Graph"),))[0].score > 0
