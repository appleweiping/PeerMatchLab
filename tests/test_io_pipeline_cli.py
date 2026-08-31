from __future__ import annotations

import json

import pytest

from peermatchlab.cli import main
from peermatchlab.config import MatchConfig
from peermatchlab.io import (
    load_conflicts,
    load_documents,
    load_experts,
    plan_from_dict,
    plan_to_dict,
)
from peermatchlab.models import DataValidationError
from peermatchlab.pipeline import run_matching


def _write(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_load_documents_accepts_jsonl(tmp_path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text('{"id":"d1","title":"One"}\n{"id":"d2","title":"Two"}\n', encoding="utf-8")
    assert [item.id for item in load_documents(path)] == ["d1", "d2"]


def test_load_documents_accepts_one_json_object(tmp_path) -> None:
    path = tmp_path / "document.json"
    _write(path, {"id": "d", "title": "One"})
    assert load_documents(path)[0].id == "d"


def test_load_documents_rejects_non_object_records(tmp_path) -> None:
    path = tmp_path / "documents.json"
    _write(path, ["not-an-object"])
    with pytest.raises(DataValidationError, match="must contain"):
        load_documents(path)


def test_load_documents_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "documents.json"
    _write(path, [{"id": "d", "title": "One"}, {"id": "d", "title": "Two"}])
    with pytest.raises(DataValidationError, match="duplicate"):
        load_documents(path)


def test_load_documents_rejects_unknown_fields(tmp_path) -> None:
    path = tmp_path / "documents.json"
    _write(path, [{"id": "d", "title": "One", "surprise": True}])
    with pytest.raises(DataValidationError, match="unknown document"):
        load_documents(path)


def test_json_inputs_reject_duplicate_object_fields(tmp_path) -> None:
    documents = tmp_path / "documents.json"
    config = tmp_path / "config.json"
    documents.write_text('[{"id":"d","id":"other","title":"One"}]', encoding="utf-8")
    config.write_text('{"strategy":"optimal","strategy":"greedy"}', encoding="utf-8")

    with pytest.raises(DataValidationError, match="duplicate JSON field"):
        load_documents(documents)
    with pytest.raises(DataValidationError, match="duplicate JSON field"):
        MatchConfig.from_json(config)


def test_load_documents_requires_title(tmp_path) -> None:
    path = tmp_path / "documents.json"
    _write(path, [{"id": "d"}])
    with pytest.raises(DataValidationError, match="missing document"):
        load_documents(path)


def test_load_documents_rejects_non_string_terms(tmp_path) -> None:
    path = tmp_path / "documents.json"
    _write(path, [{"id": "d", "title": "One", "topics": [3]}])
    with pytest.raises(DataValidationError, match="array of strings"):
        load_documents(path)


def test_load_documents_rejects_non_object_metadata(tmp_path) -> None:
    path = tmp_path / "documents.json"
    _write(path, [{"id": "d", "title": "One", "metadata": []}])
    with pytest.raises(DataValidationError, match="metadata"):
        load_documents(path)


def test_load_experts_reads_nested_publications(tmp_path) -> None:
    path = tmp_path / "experts.json"
    _write(path, [{"id": "e", "name": "E", "publications": [{"title": "Work", "year": 2025}]}])
    assert load_experts(path)[0].publications[0].title == "Work"


def test_load_experts_rejects_non_array_publications(tmp_path) -> None:
    path = tmp_path / "experts.json"
    _write(path, [{"id": "e", "name": "E", "publications": {}}])
    with pytest.raises(DataValidationError, match="publications"):
        load_experts(path)


def test_load_experts_rejects_unknown_publication_fields(tmp_path) -> None:
    path = tmp_path / "experts.json"
    _write(path, [{"id": "e", "name": "E", "publications": [{"title": "T", "doi": "x"}]}])
    with pytest.raises(DataValidationError, match="unknown publication"):
        load_experts(path)


def test_load_experts_requires_name(tmp_path) -> None:
    path = tmp_path / "experts.json"
    _write(path, [{"id": "e"}])
    with pytest.raises(DataValidationError, match="missing expert"):
        load_experts(path)


def test_load_conflicts_none_is_empty() -> None:
    assert load_conflicts(None) == ()


def test_load_conflicts_requires_identifiers(tmp_path) -> None:
    path = tmp_path / "conflicts.json"
    _write(path, [{"expert_id": "e"}])
    with pytest.raises(DataValidationError, match="missing conflict"):
        load_conflicts(path)


def test_load_conflicts_rejects_unknown_fields(tmp_path) -> None:
    path = tmp_path / "conflicts.json"
    _write(path, [{"document_id": "d", "expert_id": "e", "extra": True}])
    with pytest.raises(DataValidationError, match="unknown conflict"):
        load_conflicts(path)


def test_plan_round_trip(documents, experts) -> None:
    plan = run_matching(documents, experts, config=MatchConfig(reviewers_per_document=1)).plan
    assert plan_from_dict(plan_to_dict(plan)) == plan


def test_plan_loader_rejects_unknown_fields() -> None:
    with pytest.raises(DataValidationError, match="unknown plan"):
        plan_from_dict({"assignments": [], "surprise": True})


@pytest.mark.parametrize(
    "value",
    [
        [],
        {1: []},
        {"assignments": [], "unmet": None},
        {"assignments": ["not-an-object"]},
        {"assignments": [{1: "bad-field-name"}]},
        {
            "assignments": [
                {
                    "document_id": "d",
                    "expert_id": "e",
                    "score": 0.5,
                    "rank": 1,
                    "components": None,
                }
            ]
        },
        {"assignments": [{"document_id": "d", "expert_id": "e", "score": 0.5}]},
        {"assignments": [{"document_id": "d", "expert_id": "e", "score": True, "rank": 1}]},
        {"assignments": [{"document_id": "d", "expert_id": "e", "score": 10**1_000, "rank": 1}]},
    ],
)
def test_plan_loader_rejects_malformed_structures(value) -> None:
    with pytest.raises(DataValidationError):
        plan_from_dict(value)


def test_loaders_do_not_coerce_scalar_types(tmp_path) -> None:
    documents = tmp_path / "documents.json"
    experts = tmp_path / "experts.json"
    _write(documents, [{"id": 7, "title": "Document"}])
    _write(experts, [{"id": "e", "name": "Expert", "capacity": True}])

    with pytest.raises(DataValidationError, match="document id must be a string"):
        load_documents(documents)
    with pytest.raises(DataValidationError, match="capacity must be an integer"):
        load_experts(experts)


@pytest.mark.parametrize("value", ["NaN", "1e999"])
def test_loaders_reject_non_finite_json_numbers(tmp_path, value: str) -> None:
    experts = tmp_path / "experts.json"
    experts.write_text(f'{{"id":"e","name":"Expert","seniority":{value}}}', encoding="utf-8")

    with pytest.raises(DataValidationError, match="non-finite JSON number"):
        load_experts(experts)


def test_pipeline_returns_safe_audit(documents, experts) -> None:
    run = run_matching(documents, experts, config=MatchConfig(reviewers_per_document=1))
    assert run.audit.safe
    assert run.audit.demand_coverage == 1.0


def test_pipeline_supports_greedy_strategy(documents, experts) -> None:
    run = run_matching(
        documents,
        experts,
        config=MatchConfig(strategy="greedy", reviewers_per_document=1),
    )
    assert run.plan.strategy == "greedy"


def test_cli_validate_examples(capsys) -> None:
    code = main(
        [
            "validate",
            "--documents",
            "examples/documents.json",
            "--experts",
            "examples/experts.json",
            "--conflicts",
            "examples/conflicts.json",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_cli_validate_rejects_unknown_conflict_references(tmp_path, capsys) -> None:
    documents = tmp_path / "documents.json"
    experts = tmp_path / "experts.json"
    conflicts = tmp_path / "conflicts.json"
    _write(documents, [{"id": "d", "title": "Document"}])
    _write(experts, [{"id": "e", "name": "Expert"}])
    _write(conflicts, [{"document_id": "missing", "expert_id": "e"}])

    code = main(
        [
            "validate",
            "--documents",
            str(documents),
            "--experts",
            str(experts),
            "--conflicts",
            str(conflicts),
        ]
    )

    assert code == 2
    assert "unknown identifiers" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("documents_value", "experts_value", "message"),
    [
        ([], [{"id": "e", "name": "Expert"}], "document"),
        ([{"id": "d", "title": "Document"}], [], "expert"),
    ],
)
def test_cli_validate_rejects_empty_required_inputs(
    tmp_path, capsys, documents_value, experts_value, message
) -> None:
    documents = tmp_path / "documents.json"
    experts = tmp_path / "experts.json"
    _write(documents, documents_value)
    _write(experts, experts_value)

    code = main(["validate", "--documents", str(documents), "--experts", str(experts)])

    assert code == 2
    assert message in capsys.readouterr().err


def test_cli_match_writes_plan(tmp_path) -> None:
    output = tmp_path / "plan.json"
    code = main(
        [
            "match",
            "--documents",
            "examples/documents.json",
            "--experts",
            "examples/experts.json",
            "--conflicts",
            "examples/conflicts.json",
            "--config",
            "examples/config.json",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["assignments"]
    assert result["audit"]["safe"] is True


def test_cli_match_can_write_score_matrix(tmp_path) -> None:
    output = tmp_path / "plan.json"
    scores = tmp_path / "scores.json"
    code = main(
        [
            "match",
            "--documents",
            "examples/documents.json",
            "--experts",
            "examples/experts.json",
            "--output",
            str(output),
            "--scores",
            str(scores),
        ]
    )
    assert code == 0
    assert len(json.loads(scores.read_text(encoding="utf-8"))) == 15


def test_cli_match_can_write_html_report(tmp_path) -> None:
    output = tmp_path / "plan.json"
    report = tmp_path / "report.html"
    code = main(
        [
            "match",
            "--documents",
            "examples/documents.json",
            "--experts",
            "examples/experts.json",
            "--output",
            str(output),
            "--html",
            str(report),
        ]
    )
    assert code == 0
    assert "<!doctype html>" in report.read_text(encoding="utf-8")


def test_cli_score_known_pair(capsys) -> None:
    code = main(
        [
            "score",
            "--documents",
            "examples/documents.json",
            "--experts",
            "examples/experts.json",
            "--document-id",
            "paper-search",
            "--expert-id",
            "expert-a",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["eligible"] is True


def test_cli_audit_existing_plan(tmp_path, capsys) -> None:
    output = tmp_path / "plan.json"
    assert (
        main(
            [
                "match",
                "--documents",
                "examples/documents.json",
                "--experts",
                "examples/experts.json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()
    code = main(
        [
            "audit",
            "--documents",
            "examples/documents.json",
            "--experts",
            "examples/experts.json",
            "--plan",
            str(output),
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["safe"] is True


def test_cli_audit_marks_unknown_plan_references_unsafe(tmp_path, capsys) -> None:
    plan = tmp_path / "external-plan.json"
    _write(
        plan,
        {
            "assignments": [
                {
                    "document_id": "missing-document",
                    "expert_id": "expert-a",
                    "score": 0.5,
                    "rank": 1,
                }
            ],
            "strategy": "external",
            "total_score": 0.5,
        },
    )

    code = main(
        [
            "audit",
            "--documents",
            "examples/documents.json",
            "--experts",
            "examples/experts.json",
            "--plan",
            str(plan),
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert code == 2
    assert result["unknown_documents"] == ["missing-document"]
    assert result["safe"] is False


def test_cli_score_unknown_pair_returns_error(capsys) -> None:
    code = main(
        [
            "score",
            "--documents",
            "examples/documents.json",
            "--experts",
            "examples/experts.json",
            "--document-id",
            "missing",
            "--expert-id",
            "expert-a",
        ]
    )
    assert code == 2
    assert "unknown document/expert pair" in capsys.readouterr().err


def test_cli_invalid_json_returns_error(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not-json", encoding="utf-8")
    code = main(["validate", "--documents", str(bad), "--experts", "examples/experts.json"])
    assert code == 2
    assert "invalid JSON" in capsys.readouterr().err


def test_cli_missing_file_returns_error(capsys) -> None:
    code = main(
        [
            "validate",
            "--documents",
            "missing-documents.json",
            "--experts",
            "examples/experts.json",
        ]
    )

    assert code == 2
    assert "missing-documents.json" in capsys.readouterr().err


def test_cli_audit_rejects_non_positive_default_demand(tmp_path, capsys) -> None:
    plan = tmp_path / "plan.json"
    _write(plan, {"assignments": []})

    code = main(
        [
            "audit",
            "--documents",
            "examples/documents.json",
            "--experts",
            "examples/experts.json",
            "--plan",
            str(plan),
            "--default-demand",
            "0",
        ]
    )

    assert code == 2
    assert "default_demand must be positive" in capsys.readouterr().err
