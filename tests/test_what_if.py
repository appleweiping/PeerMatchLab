"""Independent capacity/conflict what-if oracles and CLI boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from peermatchlab.cli import main
from peermatchlab.config import MatchConfig
from peermatchlab.models import Conflict, DataValidationError, Document, Expert
from peermatchlab.what_if import WhatIfPlan, WhatIfScenario, compare_what_if


def _inputs() -> tuple[tuple[Document, ...], tuple[Expert, ...]]:
    documents = (Document("d", "Graph neural networks", required_experts=1),)
    experts = (
        Expert("a", "Graph specialist", topics=("graph",), capacity=1),
        Expert("b", "General reviewer", topics=("review",), capacity=1),
    )
    return documents, experts


def _plan() -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "scenarios": [
                {
                    "id": "disable-a",
                    "capacity_overrides": {"a": 0},
                    "add_conflicts": [],
                    "remove_conflicts": [],
                },
                {
                    "id": "conflict-a",
                    "capacity_overrides": {},
                    "add_conflicts": [["d", "a"]],
                    "remove_conflicts": [],
                },
            ],
        },
        sort_keys=True,
    ).encode()


def test_capacity_and_conflict_scenarios_are_independent_and_audited() -> None:
    documents, experts = _inputs()
    before = (experts[0].capacity, experts[1].capacity)
    report = compare_what_if(
        documents,
        experts,
        WhatIfPlan.from_bytes(_plan()),
        config=MatchConfig(reviewers_per_document=1),
    )
    assert report == compare_what_if(
        documents,
        experts,
        WhatIfPlan.from_bytes(_plan()),
        config=MatchConfig(reviewers_per_document=1),
    )
    assert report["baseline"]["assigned"] == 1
    assert report["baseline"]["unmet"] == 0
    assert report["baseline"]["safe"] is True
    baseline = report["baseline"]["assignments"]
    assert len(baseline) == 1
    assert baseline[0] == ["d", "a"]
    assert [row["id"] for row in report["scenarios"]] == ["conflict-a", "disable-a"]
    for scenario in report["scenarios"]:
        assert scenario["result"]["assignments"] == [["d", "b"]]
        assert scenario["result"]["safe"] is True
        assert scenario["delta"]["added_assignments"] == [["d", "b"]]
        assert scenario["delta"]["removed_assignments"] == [["d", "a"]]
    assert (experts[0].capacity, experts[1].capacity) == before


def test_removing_declared_conflict_is_explicit_not_inferred() -> None:
    documents, experts = _inputs()
    raw = json.loads(_plan())
    raw["scenarios"] = [
        {
            "id": "remove",
            "capacity_overrides": {},
            "add_conflicts": [],
            "remove_conflicts": [["d", "a"]],
        }
    ]
    plan = WhatIfPlan.from_bytes(json.dumps(raw).encode())
    with pytest.raises(DataValidationError, match="does not exist"):
        compare_what_if(documents, experts, plan)
    report = compare_what_if(
        documents,
        experts,
        plan,
        conflicts=(Conflict("d", "a"),),
        config=MatchConfig(reviewers_per_document=1),
    )
    assert report["baseline"]["assignments"] == [["d", "b"]]
    assert report["scenarios"][0]["result"]["assignments"] == [["d", "a"]]


def test_complete_capacity_removal_creates_one_unmet_slot_and_negative_delta() -> None:
    documents, experts = _inputs()
    raw = json.loads(_plan())
    raw["scenarios"] = [
        {
            "id": "none-available",
            "capacity_overrides": {"a": 0, "b": 0},
            "add_conflicts": [],
            "remove_conflicts": [],
        }
    ]
    report = compare_what_if(
        documents,
        experts,
        WhatIfPlan.from_bytes(json.dumps(raw).encode()),
        config=MatchConfig(reviewers_per_document=1),
    )
    result = report["scenarios"][0]
    assert result["result"]["assigned"] == 0
    assert result["result"]["unmet"] == 1
    assert result["result"]["status"] == "infeasible"
    assert result["delta"]["assigned"] == -1
    assert result["delta"]["unmet"] == 1
    assert result["delta"]["removed_assignments"] == [["d", "a"]]
    assert result["delta"]["workload"] == {"a": -1, "b": 0}


def test_zero_explicit_document_demand_is_invalid_domain_input() -> None:
    with pytest.raises(DataValidationError, match="required_experts must be positive"):
        Document("d", "Graph neural networks", required_experts=0)


def test_plan_duplicate_json_fields_and_resource_limits_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="valid JSON"):
        WhatIfPlan.from_bytes(b"{")
    with pytest.raises(DataValidationError, match="UTF-8"):
        WhatIfPlan.from_bytes(b"\xff")
    with pytest.raises(DataValidationError, match="duplicate JSON field"):
        WhatIfPlan.from_bytes(b'{"schema_version":1,"schema_version":1,"scenarios":[]}')
    with pytest.raises(DataValidationError, match="bounded immutable"):
        WhatIfPlan.from_bytes(b" " * (64 * 1024 + 1))
    raw = json.loads(_plan())
    raw["scenarios"] = raw["scenarios"] * 5
    with pytest.raises(DataValidationError, match="1 to 8"):
        WhatIfPlan.from_bytes(json.dumps(raw).encode())
    _, experts = _inputs()
    with pytest.raises(DataValidationError, match="requested"):
        compare_what_if(
            (Document("d", "Graph neural networks", required_experts=129),),
            experts,
            WhatIfPlan.from_bytes(_plan()),
        )
    with pytest.raises(DataValidationError, match="1 to 8 unique"):
        WhatIfPlan(())
    with pytest.raises(DataValidationError, match="capacity overrides"):
        WhatIfScenario("bad", (("a", True),), (), ())
    with pytest.raises(DataValidationError, match="pairs are invalid"):
        WhatIfScenario("bad", (), (("d", "a"), ("d", "a")), ())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema_version=2),
        lambda p: p["scenarios"][1].update(id="disable-a"),
        lambda p: p["scenarios"][0]["capacity_overrides"].update(a=True),
        lambda p: p["scenarios"][0]["capacity_overrides"].update(unknown=1),
        lambda p: p["scenarios"][0]["add_conflicts"].append(["d", "unknown"]),
        lambda p: p["scenarios"][0].update(other=1),
    ],
)
def test_bad_plan_or_reference_is_rejected(mutate: object) -> None:
    documents, experts = _inputs()
    raw = json.loads(_plan())
    mutate(raw)
    with pytest.raises(DataValidationError):
        compare_what_if(documents, experts, WhatIfPlan.from_bytes(json.dumps(raw).encode()))


def test_cli_smoke_and_input_output_alias_guard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    documents = tmp_path / "documents.json"
    experts = tmp_path / "experts.json"
    plan = tmp_path / "scenarios.json"
    output = tmp_path / "report.json"
    documents.write_text(
        json.dumps([{"id": "d", "title": "Graph neural networks", "required_experts": 1}])
    )
    experts.write_text(
        json.dumps(
            [
                {"id": "a", "name": "Graph specialist", "topics": ["graph"], "capacity": 1},
                {"id": "b", "name": "General reviewer", "topics": ["review"], "capacity": 1},
            ]
        )
    )
    plan.write_bytes(_plan())
    args = [
        "what-if",
        "--documents",
        str(documents),
        "--experts",
        str(experts),
        "--scenarios",
        str(plan),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    saved = json.loads(output.read_text())
    assert saved["baseline"]["safe"] is True
    assert len(saved["scenarios"]) == 2
    assert main([*args[:-1], str(plan)]) == 2
    assert "must not refer" in capsys.readouterr().err


def test_cli_rejects_oversize_source_before_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    documents = tmp_path / "documents.json"
    experts = tmp_path / "experts.json"
    plan = tmp_path / "scenarios.json"
    output = tmp_path / "report.json"
    documents.write_bytes(b" " * (1024 * 1024 + 1))
    experts.write_text('[{"id":"a","name":"A"}]', encoding="utf-8")
    plan.write_bytes(_plan())
    assert (
        main(
            [
                "what-if",
                "--documents",
                str(documents),
                "--experts",
                str(experts),
                "--scenarios",
                str(plan),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "exceeds" in capsys.readouterr().err
    assert not output.exists()
