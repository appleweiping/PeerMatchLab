from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from peermatchlab import keyphrases as keyphrase_module
from peermatchlab.cli import main
from peermatchlab.keyphrases import (
    KeyphraseConfig,
    extract_keyphrases,
    rank_keyphrases,
    read_keyphrase_source,
    write_keyphrase_run,
)
from peermatchlab.models import DataValidationError, Document, Expert, Publication


def test_three_term_chain_has_independent_pagerank_oracle() -> None:
    # For undirected alpha--beta--gamma at damping=1/2:
    # x = 1/6 + y/4; y = 1/6 + x, hence x=5/18 and y=4/9.
    ranked = rank_keyphrases(
        "alpha beta gamma", config=KeyphraseConfig(damping=0.5, iterations=100)
    )
    assert [item.term for item in ranked] == ["beta", "alpha", "gamma"]
    assert [item.score for item in ranked] == pytest.approx([4 / 9, 5 / 18, 5 / 18])
    assert sum(item.score for item in ranked) == pytest.approx(1.0)


def test_disconnected_pair_and_isolate_have_dangling_mass_oracle() -> None:
    # For alpha--beta plus isolated gamma and d=1/2, gamma=1/5 and
    # each connected term=2/5. The sentence boundary prevents beta--gamma.
    ranked = rank_keyphrases(
        "alpha beta. gamma", config=KeyphraseConfig(damping=0.5, iterations=100)
    )
    assert [item.term for item in ranked] == ["alpha", "beta", "gamma"]
    assert [item.score for item in ranked] == pytest.approx([2 / 5, 2 / 5, 1 / 5])
    assert sum(item.score for item in ranked) == pytest.approx(1.0)


def test_lexical_preprocessing_is_deterministic_and_handles_dangling_nodes() -> None:
    assert rank_keyphrases("THE Résumé résumé") == rank_keyphrases("the RÉSUMÉ RÉSUMÉ")
    assert rank_keyphrases("alpha")[0].score == 1.0
    assert rank_keyphrases("the and with") == ()
    top_one = rank_keyphrases("alpha beta gamma", config=KeyphraseConfig(top_k=1))
    assert [item.term for item in top_one] == ["beta"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 1},
        {"window_size": True},
        {"top_k": 0},
        {"iterations": 101},
        {"max_records": 10_001},
        {"max_work": False},
        {"damping": 0.0},
        {"damping": 1.0},
        {"damping": float("nan")},
        {"damping": float("inf")},
        {"damping": True},
        {"damping": "0.5"},
        {"damping": 10**1_000},
    ],
)
def test_invalid_controls_fail_closed(kwargs) -> None:
    with pytest.raises(DataValidationError):
        KeyphraseConfig(**kwargs)


def test_character_token_term_edge_and_work_limits() -> None:
    with pytest.raises(DataValidationError, match="character"):
        rank_keyphrases("alpha", config=KeyphraseConfig(max_text_characters=4))
    with pytest.raises(DataValidationError, match="max_tokens"):
        rank_keyphrases("alpha beta", config=KeyphraseConfig(max_tokens=1))
    with pytest.raises(DataValidationError, match="max_terms"):
        rank_keyphrases("alpha beta", config=KeyphraseConfig(max_terms=1))
    with pytest.raises(DataValidationError, match="max_edges"):
        rank_keyphrases("alpha beta gamma", config=KeyphraseConfig(max_edges=1))
    with pytest.raises(DataValidationError, match="max_work"):
        rank_keyphrases("alpha beta gamma", config=KeyphraseConfig(max_work=1))
    with pytest.raises(DataValidationError, match="not a string"):
        rank_keyphrases(None)  # type: ignore[arg-type]


def test_domain_extraction_keeps_units_separate_and_stable() -> None:
    document = Document("s", "alpha beta gamma")
    expert = Expert(
        "r",
        "Reviewer",
        summary="alpha beta",
        publications=(Publication("beta gamma", id="0"), Publication("gamma alpha")),
    )
    records = extract_keyphrases([document], [expert])
    assert [(item.kind, item.owner_id, item.evidence_id) for item in records] == [
        ("profile", "r", "r"),
        ("publication", "r", "id:0"),
        ("publication", "r", "position:1"),
        ("submission", "s", "s"),
    ]
    assert records[-1].token_count == 3
    assert extract_keyphrases([document], [expert]) == records


def test_duplicate_and_aggregate_work_bounds() -> None:
    document = Document("s", "alpha beta")
    with pytest.raises(DataValidationError, match="unique"):
        extract_keyphrases([document, document], [])
    expert = Expert("r", "Reviewer")
    with pytest.raises(DataValidationError, match="unique"):
        extract_keyphrases([], [expert, expert])
    with pytest.raises(DataValidationError, match="max_records"):
        extract_keyphrases([document], [expert], config=KeyphraseConfig(max_records=1))
    with pytest.raises(DataValidationError, match="max_work"):
        extract_keyphrases([document], [], config=KeyphraseConfig(max_work=1))
    with pytest.raises(DataValidationError, match="max_text_characters"):
        extract_keyphrases([document], [], config=KeyphraseConfig(max_text_characters=4))
    with pytest.raises(DataValidationError, match="max_tokens"):
        extract_keyphrases([document], [], config=KeyphraseConfig(max_tokens=1))
    duplicated = Expert(
        "r",
        "Reviewer",
        publications=(Publication("alpha", id="same"), Publication("beta", id="same")),
    )
    with pytest.raises(DataValidationError, match="duplicate publication"):
        extract_keyphrases([], [duplicated])


def test_artifact_records_exact_source_hashes_and_refuses_overwrite(tmp_path) -> None:
    documents = b'[{"id":"s","title":"alpha beta gamma"}]\n'
    experts = b'[{"id":"r","name":"Reviewer","summary":"beta gamma"}]\n'
    destination = tmp_path / "keyphrases"
    manifest = write_keyphrase_run(
        destination,
        documents_source=documents,
        experts_source=experts,
        config=KeyphraseConfig(damping=0.5, iterations=100),
    )
    output = (destination / "keyphrases.jsonl").read_bytes()
    assert manifest["schema_version"] == 1
    assert manifest["config"]["tokenizer"] == "unicode-casefold-alnum-v1"
    assert "the" in manifest["config"]["stopwords"]
    assert manifest["documents_sha256"] == hashlib.sha256(documents).hexdigest()
    assert manifest["experts_sha256"] == hashlib.sha256(experts).hexdigest()
    assert manifest["keyphrases_sha256"] == hashlib.sha256(output).hexdigest()
    assert manifest["keyphrases_bytes"] == len(output)
    assert len([json.loads(line) for line in output.splitlines()]) == 2
    assert json.loads((destination / "manifest.json").read_text()) == manifest
    with pytest.raises(DataValidationError, match="already exists"):
        write_keyphrase_run(destination, documents_source=documents, experts_source=experts)
    assert (destination / "keyphrases.jsonl").read_bytes() == output


def test_concurrent_install_has_one_complete_winner(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "race"
    experts = b"[]"
    sources = (
        b'[{"id":"s","title":"alpha beta"}]',
        b'[{"id":"s","title":"gamma delta"}]',
    )
    barrier = threading.Barrier(2)
    original_install = keyphrase_module._install_directory_no_replace

    def synchronized_install(staging, target) -> None:
        barrier.wait(timeout=5)
        original_install(staging, target)

    monkeypatch.setattr(keyphrase_module, "_install_directory_no_replace", synchronized_install)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                write_keyphrase_run,
                destination,
                documents_source=source,
                experts_source=experts,
            )
            for source in sources
        ]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except FileExistsError as error:
                outcomes.append(error)
    assert sum(isinstance(outcome, dict) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, FileExistsError) for outcome in outcomes) == 1
    manifest = json.loads((destination / "manifest.json").read_text())
    artifact = (destination / "keyphrases.jsonl").read_bytes()
    assert manifest["documents_sha256"] in {
        hashlib.sha256(source).hexdigest() for source in sources
    }
    assert manifest["keyphrases_sha256"] == hashlib.sha256(artifact).hexdigest()


def test_artifact_rejects_malformed_sources_before_publishing(tmp_path) -> None:
    target = tmp_path / "bad"
    for source in (b"\xff", b'{"id":"s","title":"one","title":"two"}'):
        with pytest.raises(DataValidationError):
            write_keyphrase_run(target, documents_source=source, experts_source=b"[]")
        assert not target.exists()
    with pytest.raises(DataValidationError, match="byte limit"):
        write_keyphrase_run(
            target,
            documents_source=b"x" * (16 * 1024 * 1024 + 1),
            experts_source=b"[]",
        )
    with pytest.raises(DataValidationError, match="immutable byte snapshots"):
        write_keyphrase_run(
            target,
            documents_source="[]",
            experts_source=b"[]",  # type: ignore[arg-type]
        )
    assert not target.exists()


def test_json_renderer_rejects_unpaired_surrogate() -> None:
    with pytest.raises(DataValidationError, match="encoded as JSON"):
        keyphrase_module._json_bytes({"invalid": "\ud800"})


def test_output_bound_rejects_before_creating_destination(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(keyphrase_module, "_MAX_OUTPUT_BYTES", 1)
    destination = tmp_path / "small"
    with pytest.raises(DataValidationError, match="output exceeds"):
        write_keyphrase_run(
            destination,
            documents_source=b'[{"id":"s","title":"alpha beta"}]',
            experts_source=b"[]",
        )
    assert not destination.exists()


def test_source_reader_rejects_oversize_before_parse(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(keyphrase_module, "_MAX_SOURCE_BYTES", 3)
    source = tmp_path / "source.json"
    source.write_bytes(b"abcd")
    with pytest.raises(DataValidationError, match="byte limit"):
        read_keyphrase_source(source)


def test_bounded_source_reader_and_cli_smoke(tmp_path, capsys) -> None:
    documents = tmp_path / "documents.json"
    experts = tmp_path / "experts.json"
    documents.write_text('[{"id":"s","title":"alpha beta gamma"}]\n', encoding="utf-8")
    experts.write_text('[{"id":"r","name":"Reviewer","summary":"alpha beta"}]\n', encoding="utf-8")
    assert read_keyphrase_source(documents) == documents.read_bytes()
    target = tmp_path / "run"
    assert (
        main(
            [
                "extract-keyphrases",
                "--documents",
                str(documents),
                "--experts",
                str(experts),
                "--directory",
                str(target),
                "--damping",
                "0.5",
                "--iterations",
                "100",
            ]
        )
        == 0
    )
    assert "extracted 2 evidence records" in capsys.readouterr().out
    assert (target / "keyphrases.jsonl").exists()
    assert (
        main(
            [
                "extract-keyphrases",
                "--documents",
                str(documents),
                "--experts",
                str(experts),
                "--directory",
                str(tmp_path / "invalid"),
                "--top-k",
                "0",
            ]
        )
        == 2
    )
    assert not (tmp_path / "invalid").exists()
