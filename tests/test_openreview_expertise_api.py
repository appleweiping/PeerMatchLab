"""Frozen OpenReview API-v2 protocol fixtures and independent filter oracles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

import peermatchlab.cli as cli_module
import peermatchlab.openreview_api as api_module
from peermatchlab.cli import main
from peermatchlab.models import DataValidationError
from peermatchlab.openreview_api import (
    ExpertiseFetchPolicy,
    HttpResponse,
    OpenReviewClient,
    OpenReviewClientConfig,
    OpenReviewExpertiseEvidence,
    OpenReviewProtocolError,
    OpenReviewSnapshot,
    fetch_reviewer_expertise,
    write_openreview_snapshot,
)

FIXTURE = Path(__file__).parent / "fixtures" / "openreview_expertise_api_v2.json"


class FixtureTransport:
    def __init__(self, payloads: list[object]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> HttpResponse:
        assert url.startswith("https://openreview.test/")
        assert timeout_seconds > 0
        assert max_response_bytes > 0
        assert headers["Accept"] == "application/json"
        self.calls.append(url)
        return HttpResponse(200, {}, json.dumps(self.payloads.pop(0)).encode())


def _fixture() -> list[object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["responses"]


def _client(transport: FixtureTransport, **kwargs: Any) -> OpenReviewClient:
    return OpenReviewClient(
        config=OpenReviewClientConfig(
            base_url="https://openreview.test", page_size=2, requests_per_second=1000, **kwargs
        ),
        transport=transport,
        sleeper=lambda _: None,
        monotonic=lambda: 1_000_000.0,
    )


def _snapshot(client: OpenReviewClient) -> OpenReviewSnapshot:
    from peermatchlab.openreview_api import fetch_openreview_snapshot

    return fetch_openreview_snapshot(
        client, invitation="Venue/-/Submission", reviewer_group="Venue/Reviewers"
    )


def _policy(**kwargs: Any) -> ExpertiseFetchPolicy:
    return ExpertiseFetchPolicy(
        invitations=("Public/-/Paper",),
        minimum_date_ms=1_640_995_200_000,
        require_abstract=True,
        **kwargs,
    )


def test_official_envelopes_apply_independent_set_oracle_and_replay_offline(tmp_path: Path) -> None:
    transport = FixtureTransport(_fixture())
    client = _client(transport)
    snapshot = _snapshot(client)
    expertise = fetch_reviewer_expertise(client, snapshot, _policy())

    # Independent expected set: invitation ∩ date ∩ abstract ∩ explicit author ID.
    assert [(row["reviewer_id"], row["note"]["id"]) for row in expertise.reviewer_publications] == [
        ("~Reviewer_One1", "pub-1"),
        ("~Reviewer_Two1", "pub-5"),
    ]
    assert dict(expertise.filter_counts) == {
        "scanned": 5,
        "invitation": 1,
        "date": 1,
        "content": 1,
        "retained": 2,
    }
    assert [row["content"]["expertise"] for row in expertise.profiles] == [
        ("graph", "retrieval"),
        ("ranking",),
    ]
    assert all("emails" not in row["content"] for row in expertise.profiles)
    assert [row["note"]["content"]["year"]["value"] for row in expertise.reviewer_publications] == [
        2024,
        2024,
    ]
    assert [urlsplit(url).path for url in transport.calls] == [
        "/notes",
        "/groups",
        "/profiles",
        "/notes",
        "/notes",
        "/profiles",
        "/notes",
    ]
    assert parse_qs(urlsplit(transport.calls[3]).query) == {
        "content.authorids": ["~Reviewer_One1"],
        "count": ["True"],
        "limit": ["2"],
        "sort": ["id"],
    }
    assert parse_qs(urlsplit(transport.calls[4]).query)["after"] == ["pub-2"]
    assert parse_qs(urlsplit(transport.calls[5]).query) == {
        "id": ["~Reviewer_Two1"],
        "limit": ["2"],
    }
    destination = tmp_path / "snapshot"
    manifest = write_openreview_snapshot(
        snapshot, destination, reviewer_capacity=1, expertise=expertise
    )
    assert manifest["schema_version"] == 2
    assert manifest["records"]["reviewer_publications"] == 2
    assert "private-one@example.org" not in (destination / "profiles.jsonl").read_text()
    for name, info in manifest["files"].items():
        raw = (destination / name).read_bytes()
        assert info == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    assert not transport.payloads

    expertise_run = tmp_path / "expertise"
    assert (
        main(["expertise", "--snapshot", str(destination), "--directory", str(expertise_run)]) == 0
    )
    plan = tmp_path / "plan.json"
    assert (
        main(
            [
                "match-affinity",
                "--documents",
                str(expertise_run / "documents.json"),
                "--experts",
                str(expertise_run / "experts.json"),
                "--affinities",
                str(expertise_run / "affinities.csv"),
                "--output",
                str(plan),
            ]
        )
        == 0
    )
    assert json.loads(plan.read_text())["assignments"]


def test_opt_in_fetch_cli_flows_through_expertise_and_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FixtureTransport(_fixture())
    client = _client(transport)
    monkeypatch.setattr(cli_module, "OpenReviewClient", lambda *, config, token: client)
    snapshot = tmp_path / "fetched"
    assert (
        main(
            [
                "fetch-openreview",
                "--invitation",
                "Venue/-/Submission",
                "--reviewer-group",
                "Venue/Reviewers",
                "--reviewer-capacity",
                "1",
                "--publication-invitation",
                "Public/-/Paper",
                "--minimum-publication-date-ms",
                "1640995200000",
                "--require-publication-abstract",
                "--fetch-expertise",
                "--directory",
                str(snapshot),
            ]
        )
        == 0
    )
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["records"]["reviewer_publications"] == 2
    expertise_run = tmp_path / "expertise"
    assert main(["expertise", "--snapshot", str(snapshot), "--directory", str(expertise_run)]) == 0
    plan = tmp_path / "plan.json"
    assert (
        main(
            [
                "match-affinity",
                "--documents",
                str(expertise_run / "documents.json"),
                "--experts",
                str(expertise_run / "experts.json"),
                "--affinities",
                str(expertise_run / "affinities.csv"),
                "--output",
                str(plan),
            ]
        )
        == 0
    )
    assert json.loads(plan.read_text(encoding="utf-8"))["assignments"]
    assert len(transport.calls) == 7


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (
            lambda rows: rows[3]["notes"][0]["content"]["authorids"].update(value=["~Other1"]),
            "without the reviewer",
        ),
        (lambda rows: rows[2]["profiles"][0].update(id="~Wrong1"), "id does not match"),
        (lambda rows: (rows[3].update(count=5), rows[4]["notes"].pop()), "pagination ended"),
        (lambda rows: rows[3]["notes"][0]["content"].update(authorids={"bad": 1}), "no value"),
        (lambda rows: rows[3]["notes"][0].update(invitations="Public/-/Paper"), "string list"),
        (lambda rows: rows[3]["notes"][0].update(pdate=True), "pdate is invalid"),
    ],
)
def test_malformed_or_misattributed_live_data_fails_closed(
    tmp_path: Path, change: Any, error: str
) -> None:
    responses = _fixture()
    change(responses)
    client = _client(FixtureTransport(responses))
    snapshot = _snapshot(client)
    with pytest.raises(OpenReviewProtocolError, match=error):
        fetch_reviewer_expertise(client, snapshot, _policy())
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"invitations": ()}, "invitations"),
        ({"invitations": ("A", "A")}, "unique"),
        ({"minimum_date_ms": True}, "minimum_date_ms"),
        ({"maximum_date_ms": -1}, "maximum_date_ms"),
        ({"require_abstract": 1}, "require_abstract"),
        ({"max_scanned_notes": 0}, "max_scanned_notes"),
        ({"max_publications": True}, "max_publications"),
    ],
)
def test_policy_rejects_invalid_or_unbounded_selection(
    override: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {"invitations": ("Public/-/Paper",)}
    values.update(override)
    with pytest.raises(ValueError, match=message):
        ExpertiseFetchPolicy(**values)  # type: ignore[arg-type]


def test_cli_opt_in_uses_transport_fixture_and_preserves_legacy_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FixtureTransport(_fixture())
    monkeypatch.setattr(
        cli_module,
        "OpenReviewClient",
        lambda *, config, token: OpenReviewClient(
            config=config,
            token=token,
            transport=transport,
            sleeper=lambda _: None,
            monotonic=lambda: 1_000_000.0,
        ),
    )
    destination = tmp_path / "fetched"
    assert (
        main(
            [
                "fetch-openreview",
                "--invitation",
                "Venue/-/Submission",
                "--reviewer-group",
                "Venue/Reviewers",
                "--reviewer-capacity",
                "1",
                "--directory",
                str(destination),
                "--base-url",
                "https://openreview.test",
                "--page-size",
                "2",
                "--fetch-expertise",
                "--publication-invitation",
                "Public/-/Paper",
                "--minimum-publication-date-ms",
                "1640995200000",
                "--require-publication-abstract",
            ]
        )
        == 0
    )
    assert json.loads((destination / "manifest.json").read_text())["adapter"] == (
        "openreview-api-v2-expertise-snapshot"
    )


def test_total_request_and_record_bounds_fail_before_unbounded_work() -> None:
    responses = _fixture()
    client = _client(FixtureTransport(responses), max_total_requests=3)
    snapshot = _snapshot(client)
    with pytest.raises(OpenReviewProtocolError, match="max_total_requests"):
        fetch_reviewer_expertise(client, snapshot, _policy())

    responses = _fixture()
    client = _client(FixtureTransport(responses))
    snapshot = _snapshot(client)
    with pytest.raises(OpenReviewProtocolError, match="max_scanned_notes"):
        fetch_reviewer_expertise(client, snapshot, _policy(max_scanned_notes=1))


def test_cli_requires_explicit_publication_scope(tmp_path: Path) -> None:
    common = [
        "fetch-openreview",
        "--invitation",
        "Venue/-/Submission",
        "--reviewer-group",
        "Venue/Reviewers",
        "--reviewer-capacity",
        "1",
        "--directory",
        str(tmp_path / "no-output"),
    ]
    assert main([*common, "--fetch-expertise"]) == 2
    assert main([*common, "--publication-invitation", "Public/-/Paper"]) == 2
    assert not (tmp_path / "no-output").exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ({"names": []}, "no names"),
        ({"names": [{"fullname": ""}]}, "fullname"),
        ({"names": [{"fullname": "R", "preferred": 1}]}, "preferred"),
        ({"names": [{"fullname": "R"}], "bio": 1}, "bio"),
        ({"names": [{"fullname": "R"}], "keywords": "wrong"}, "keywords"),
        ({"names": [{"fullname": "R"}], "research_interests": [1]}, "research_interests"),
        ({"names": [{"fullname": "R"}], "expertise": "wrong"}, "expertise"),
        ({"names": [{"fullname": "R"}], "expertise": [{"keywords": [1]}]}, "keywords"),
    ],
)
def test_profile_projection_rejects_malformed_public_content(
    content: dict[str, object], message: str
) -> None:
    with pytest.raises(OpenReviewProtocolError, match=message):
        api_module._profile_projection({"id": "~R1", "content": content})


def test_profile_projection_flattens_official_keywords_and_minimizes_private_fields() -> None:
    result = api_module._profile_projection(
        {
            "id": "~R1",
            "content": {
                "names": [{"fullname": "R", "preferred": False}],
                "bio": "search researcher",
                "keywords": ["retrieval"],
                "research_interests": ["ranking"],
                "expertise": [
                    "indexing",
                    {"keywords": ["search", "indexing"]},
                    {"keywords": None, "start": 2020},
                ],
                "emails": ["private@example.org"],
            },
        }
    )
    assert result["content"] == {
        "names": [{"fullname": "R", "preferred": False}],
        "bio": "search researcher",
        "keywords": ["retrieval"],
        "research_interests": ["ranking"],
        "expertise": ["indexing", "search"],
    }


def test_official_first_last_name_without_fullname_projects_safely() -> None:
    responses = _fixture()
    responses[2]["profiles"][0]["content"]["names"] = [
        {
            "first": "Ada",
            "middle": "M.",
            "last": "Lovelace",
            "preferred": True,
            "email": "private@example.org",
        }
    ]
    client = _client(FixtureTransport(responses))
    evidence = fetch_reviewer_expertise(client, _snapshot(client), _policy())
    assert evidence.profiles[0]["content"]["names"] == (
        {"fullname": "Ada M. Lovelace", "preferred": True},
    )
    assert "private@example.org" not in str(evidence.profiles)


def test_policy_snapshots_mutable_invitations() -> None:
    invitations = ["Public/-/Paper"]
    policy = ExpertiseFetchPolicy(invitations)  # type: ignore[arg-type]
    invitations.append("Private/-/Paper")
    assert policy.invitations == ("Public/-/Paper",)
    assert policy.record()["invitations"] == ["Public/-/Paper"]


def test_cli_preflights_expertise_policy_before_any_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_client(*, config: object, token: object) -> None:
        raise AssertionError("network client must not be created")

    monkeypatch.setattr(cli_module, "OpenReviewClient", forbidden_client)
    destination = tmp_path / "must-not-exist"
    assert (
        main(
            [
                "fetch-openreview",
                "--invitation",
                "Venue/-/Submission",
                "--reviewer-group",
                "Venue/Reviewers",
                "--reviewer-capacity",
                "1",
                "--directory",
                str(destination),
                "--fetch-expertise",
                "--publication-invitation",
                "Public/-/Paper",
                "--minimum-publication-date-ms",
                "2000",
                "--maximum-publication-date-ms",
                "1000",
            ]
        )
        == 2
    )
    assert not destination.exists()


def test_projected_expertise_bytes_are_bounded_before_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {"reviewer_id": "~R1", "note": {"id": "p", "content": {"title": "x"}}}
    one_row_bytes = (
        len(
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        )
        + 1
    )
    monkeypatch.setattr(api_module, "_MAX_EXPERTISE_JSONL_FILE_BYTES", one_row_bytes)
    assert (
        api_module._account_projected_jsonl_row(row, 0, "reviewer-publications.jsonl")
        == one_row_bytes
    )
    with pytest.raises(OpenReviewProtocolError, match="max_input_file_bytes"):
        api_module._account_projected_jsonl_row(row, one_row_bytes, "reviewer-publications.jsonl")


def test_acquisition_stops_when_projected_rows_exceed_cumulative_file_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both first-page publications fit individually, but not together. A later
    # page must not be fetched after the budget fails on this page.
    monkeypatch.setattr(api_module, "_MAX_EXPERTISE_JSONL_FILE_BYTES", 300)
    responses = _fixture()
    responses[3]["notes"][1]["invitations"] = ["Public/-/Paper"]
    transport = FixtureTransport(responses)
    client = _client(transport)
    snapshot = _snapshot(client)
    with pytest.raises(
        OpenReviewProtocolError, match=r"reviewer-publications\.jsonl.*max_input_file_bytes"
    ):
        fetch_reviewer_expertise(client, snapshot, _policy())
    assert [urlsplit(url).path for url in transport.calls] == [
        "/notes",
        "/groups",
        "/profiles",
        "/notes",
    ]


def _publication_note(**changes: object) -> dict[str, object]:
    note: dict[str, object] = {
        "id": "p",
        "invitations": ["Public/-/Paper"],
        "cdate": 1_704_067_200_000,
        "content": {
            "title": {"value": "Graph retrieval"},
            "abstract": {"value": "Ranking"},
            "authorids": {"value": ["~R1"]},
        },
    }
    note.update(changes)
    return note


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"content": []}, "content"),
        ({"content": {"authorids": "~R1"}}, "authorids"),
        ({"invitations": None}, "invitations"),
        ({"cdate": -1}, "cdate"),
        ({"odate": "2024"}, "odate"),
        ({"pdate": True}, "pdate"),
        ({"content": {"authorids": ["~R1"], "title": 2}}, "title"),
        ({"content": {"authorids": ["~R1"], "title": "T", "abstract": 2}}, "abstract"),
        ({"content": {"authorids": ["~R1"], "title": "T", "abstract": "A", "year": True}}, "year"),
    ],
)
def test_publication_protocol_fields_fail_closed(change: dict[str, object], message: str) -> None:
    with pytest.raises(OpenReviewProtocolError, match=message):
        api_module._publication_projection(_publication_note(**change), "~R1", _policy())


def test_publication_filter_edges_are_independent_and_inclusive() -> None:
    note = _publication_note(invitation="Public/-/Paper", invitations=None)
    assert api_module._publication_projection(note, "~R1", _policy())[1] == "retained"
    undated = _publication_note()
    undated.pop("cdate")
    assert api_module._publication_projection(undated, "~R1", _policy())[1] == "date"
    lower = _publication_note(cdate=1_640_995_200_000)
    assert api_module._publication_projection(lower, "~R1", _policy())[1] == "retained"
    upper = _publication_note(cdate=1_704_067_200_000)
    policy = ExpertiseFetchPolicy(("Public/-/Paper",), maximum_date_ms=1_704_067_200_000)
    assert api_module._publication_projection(upper, "~R1", policy)[1] == "retained"
    assert (
        api_module._publication_projection(
            _publication_note(cdate=1_704_067_200_001), "~R1", policy
        )[1]
        == "date"
    )
    assert (
        api_module._publication_projection(
            _publication_note(content={"authorids": ["~R1"], "title": "   "}),
            "~R1",
            ExpertiseFetchPolicy(("Public/-/Paper",)),
        )[1]
        == "content"
    )
    retained, reason = api_module._publication_projection(
        _publication_note(content={"authorids": ["~R1"], "title": "T", "year": 2023}),
        "~R1",
        ExpertiseFetchPolicy(("Public/-/Paper",)),
    )
    assert reason == "retained"
    assert retained["content"]["year"]["value"] == 2023


@pytest.mark.parametrize(
    ("profiles", "message"),
    [
        ([], "0 profiles"),
        ([{"id": "~Other", "content": {}}], "id does not match"),
        ([{"id": "~R1", "content": {}}, {"id": "~R1", "content": {}}], "2 profiles"),
    ],
)
def test_exact_profile_lookup_rejects_missing_wrong_and_multiple_profiles(
    profiles: list[dict[str, object]], message: str
) -> None:
    client = _client(FixtureTransport([{"profiles": profiles}]))
    with pytest.raises(OpenReviewProtocolError, match=message):
        client.get_profile("~R1")


def _evidence(**changes: object) -> OpenReviewExpertiseEvidence:
    values: dict[str, Any] = {
        "profiles": ({"id": "~R1", "content": {"names": [{"fullname": "R"}]}},),
        "reviewer_publications": (
            {
                "reviewer_id": "~R1",
                "note": {
                    "id": "p",
                    "invitations": ["Public/-/Paper"],
                    "content": {
                        "title": {"value": "T"},
                        "abstract": {"value": "A"},
                        "authorids": {"value": ["~R1"]},
                        "year": {"value": None},
                    },
                },
            },
        ),
        "policy": ExpertiseFetchPolicy(("Public/-/Paper",)),
        "filter_counts": {"scanned": 1, "invitation": 0, "date": 0, "content": 0, "retained": 1},
    }
    values.update(changes)
    return OpenReviewExpertiseEvidence(**values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"policy": None}, "policy"),
        ({"profiles": ({"id": "~R1"},)}, "profile evidence content"),
        (
            {"profiles": ({"id": "~R1", "content": {}}, {"id": "~R1", "content": {}})},
            "duplicate IDs",
        ),
        ({"reviewer_publications": ({"reviewer_id": "~R1"},)}, "exactly"),
        (
            {"reviewer_publications": ({"reviewer_id": "~Other", "note": {"id": "p"}},)},
            "unknown profile",
        ),
        ({"reviewer_publications": ({"reviewer_id": "~R1", "note": {"content": {}}},)}, "note id"),
        (
            {
                "reviewer_publications": (
                    {
                        "reviewer_id": "~R1",
                        "note": {"id": "p", "content": {"authorids": ["~Other"]}},
                    },
                )
            },
            "author identity",
        ),
        ({"filter_counts": {"scanned": 1}}, "filter counts are invalid"),
        ({"filter_counts": []}, "must be a mapping"),
        (
            {
                "filter_counts": {
                    "scanned": 1,
                    "invitation": 0,
                    "date": 0,
                    "content": 0,
                    "retained": -1,
                }
            },
            "filter counts are invalid",
        ),
        (
            {
                "filter_counts": {
                    "scanned": 2,
                    "invitation": 0,
                    "date": 0,
                    "content": 0,
                    "retained": 1,
                }
            },
            "do not match",
        ),
    ],
)
def test_public_evidence_boundary_rejects_forged_join_or_counts(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises((OpenReviewProtocolError, ValueError), match=message):
        _evidence(**changes)


def test_public_evidence_boundary_rejects_duplicate_association() -> None:
    row = _evidence().reviewer_publications[0]
    with pytest.raises(OpenReviewProtocolError, match="duplicate reviewer-publication"):
        _evidence(
            reviewer_publications=(row, row),
            filter_counts={"scanned": 2, "invitation": 0, "date": 0, "content": 0, "retained": 2},
        )


def test_public_evidence_boundary_rejects_unrelated_private_fields() -> None:
    with pytest.raises(OpenReviewProtocolError, match="unrelated fields"):
        _evidence(
            profiles=(
                {
                    "id": "~R1",
                    "content": {"names": [{"fullname": "R"}], "emails": ["private@example.org"]},
                },
            )
        )
    with pytest.raises(OpenReviewProtocolError, match="author identity"):
        _evidence(
            reviewer_publications=(
                {
                    "reviewer_id": "~R1",
                    "note": {
                        "id": "p",
                        "content": {
                            "title": {"value": "T"},
                            "authorids": {"value": ["~R1", "coauthor@example.org"]},
                        },
                    },
                },
            )
        )


@pytest.mark.parametrize("reviewer_id", ["reviewer@example.org", "~", "~bad@example.org"])
def test_public_expertise_evidence_rejects_non_profile_reviewer_ids(
    reviewer_id: str,
) -> None:
    with pytest.raises(OpenReviewProtocolError, match="tilde reviewer"):
        _evidence(
            profiles=({"id": reviewer_id, "content": {"names": [{"fullname": "R"}]}},),
            reviewer_publications=(),
            filter_counts={"scanned": 0, "invitation": 0, "date": 0, "content": 0, "retained": 0},
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda profile, publication: profile["content"]["names"][0].update(
            email="private@example.org"
        ),
        lambda profile, publication: profile["content"].update(names="not-an-array"),
        lambda profile, publication: profile["content"]["names"][0].update(fullname=7),
        lambda profile, publication: profile["content"]["names"][0].update(preferred="yes"),
        lambda profile, publication: profile["content"].update(
            bio={"value": "Bio", "email": "private@example.org"}
        ),
        lambda profile, publication: profile["content"].update(
            research_interests={"value": ["IR"], "email": "private@example.org"}
        ),
        lambda profile, publication: profile["content"].update(
            keywords={"value": ["IR"], "email": "private@example.org"}
        ),
        lambda profile, publication: profile["content"].update(
            expertise=[{"keywords": ["IR"], "email": "private@example.org"}]
        ),
        lambda profile, publication: publication["note"]["content"]["title"].update(
            email="coauthor@example.org"
        ),
        lambda profile, publication: publication["note"].update(readers=["private@example.org"]),
        lambda profile, publication: publication["note"]["content"].update(
            emails=["private@example.org"]
        ),
        lambda profile, publication: publication["note"].update(invitations=[42]),
        lambda profile, publication: publication["note"]["content"]["title"].update(value=42),
        lambda profile, publication: publication["note"]["content"].update(
            abstract={"value": "A", "email": "coauthor@example.org"}
        ),
        lambda profile, publication: publication["note"]["content"]["authorids"].update(
            email="coauthor@example.org"
        ),
        lambda profile, publication: publication["note"]["content"].update(
            year={"value": 2024, "email": "coauthor@example.org"}
        ),
        lambda profile, publication: publication["note"]["content"]["year"].update(value=2201),
        lambda profile, publication: publication["note"].update(pdate={"value": 2024}),
    ],
)
def test_public_evidence_rejects_nested_private_or_noncanonical_values(change: Any) -> None:
    evidence = _evidence()
    profile = api_module._thaw_json(evidence.profiles[0])
    publication = api_module._thaw_json(evidence.reviewer_publications[0])
    change(profile, publication)
    with pytest.raises(OpenReviewProtocolError):
        _evidence(profiles=(profile,), reviewer_publications=(publication,))


def test_public_evidence_retained_rows_must_satisfy_declared_policy() -> None:
    evidence = _evidence()
    publication = api_module._thaw_json(evidence.reviewer_publications[0])
    publication["note"]["invitations"] = ["Other/-/Paper"]
    with pytest.raises(OpenReviewProtocolError, match="invitation policy"):
        _evidence(reviewer_publications=(publication,))

    with pytest.raises(OpenReviewProtocolError, match="date policy"):
        _evidence(policy=ExpertiseFetchPolicy(("Public/-/Paper",), minimum_date_ms=1))

    publication = api_module._thaw_json(evidence.reviewer_publications[0])
    publication["note"]["content"]["abstract"]["value"] = ""
    with pytest.raises(OpenReviewProtocolError, match="content policy"):
        _evidence(
            reviewer_publications=(publication,),
            policy=ExpertiseFetchPolicy(("Public/-/Paper",), require_abstract=True),
        )

    publication = api_module._thaw_json(evidence.reviewer_publications[0])
    del publication["note"]["content"]["year"]
    with pytest.raises(OpenReviewProtocolError, match="canonical projection"):
        _evidence(reviewer_publications=(publication,))


def test_public_evidence_respects_declared_scan_and_per_reviewer_limits() -> None:
    evidence = _evidence()
    first = api_module._thaw_json(evidence.reviewer_publications[0])
    second = api_module._thaw_json(evidence.reviewer_publications[0])
    second["note"]["id"] = "p2"
    counts = {"scanned": 2, "invitation": 0, "date": 0, "content": 0, "retained": 2}
    with pytest.raises(OpenReviewProtocolError, match="max_publications_per_reviewer"):
        _evidence(
            reviewer_publications=(first, second),
            filter_counts=counts,
            policy=ExpertiseFetchPolicy(("Public/-/Paper",), max_publications_per_reviewer=1),
        )
    with pytest.raises(ValueError, match="filter counts"):
        _evidence(
            reviewer_publications=(first, second),
            filter_counts=counts,
            policy=ExpertiseFetchPolicy(("Public/-/Paper",), max_scanned_notes=1),
        )


def _single_reviewer_snapshot() -> OpenReviewSnapshot:
    return OpenReviewSnapshot(
        notes=({"id": "s", "content": {"title": {"value": "Submission"}}},),
        reviewer_ids=("~R1",),
        base_url="https://openreview.test",
        paper_filter={"invitation": "Venue/-/Submission"},
        reviewer_group="Venue/Reviewers",
    )


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("policy", None, "valid selection policy"),
        ("filter_counts", [], "filter counts"),
        (
            "filter_counts",
            {"scanned": 1, "invitation": 0, "date": 0, "content": 0, "retained": 0},
            "filter counts",
        ),
    ],
)
def test_direct_writer_rejects_tampered_evidence_contract(
    tmp_path: Path, attribute: str, value: object, message: str
) -> None:
    evidence = _evidence()
    object.__setattr__(evidence, attribute, value)
    destination = tmp_path / "tampered"
    with pytest.raises(OpenReviewProtocolError, match=message):
        write_openreview_snapshot(
            _single_reviewer_snapshot(), destination, reviewer_capacity=1, expertise=evidence
        )
    assert not destination.exists()


def test_direct_writer_rechecks_publication_and_per_reviewer_limits(tmp_path: Path) -> None:
    first = api_module._thaw_json(_evidence().reviewer_publications[0])
    second = api_module._thaw_json(_evidence().reviewer_publications[0])
    second["note"]["id"] = "p2"
    evidence = _evidence(
        reviewer_publications=(first, second),
        filter_counts={"scanned": 2, "invitation": 0, "date": 0, "content": 0, "retained": 2},
        policy=ExpertiseFetchPolicy(
            ("Public/-/Paper",), max_publications=2, max_publications_per_reviewer=2
        ),
    )
    object.__setattr__(
        evidence,
        "policy",
        ExpertiseFetchPolicy(
            ("Public/-/Paper",), max_publications=1, max_publications_per_reviewer=2
        ),
    )
    with pytest.raises(OpenReviewProtocolError, match="max_publications"):
        write_openreview_snapshot(
            _single_reviewer_snapshot(),
            tmp_path / "over-total",
            reviewer_capacity=1,
            expertise=evidence,
        )
    assert not (tmp_path / "over-total").exists()

    object.__setattr__(
        evidence,
        "policy",
        ExpertiseFetchPolicy(
            ("Public/-/Paper",), max_publications=2, max_publications_per_reviewer=1
        ),
    )
    with pytest.raises(OpenReviewProtocolError, match="max_publications_per_reviewer"):
        write_openreview_snapshot(
            _single_reviewer_snapshot(),
            tmp_path / "over-reviewer",
            reviewer_capacity=1,
            expertise=evidence,
        )
    assert not (tmp_path / "over-reviewer").exists()


def test_direct_writer_rejects_extra_join_field_before_install(tmp_path: Path) -> None:
    evidence = _evidence()
    row = dict(evidence.reviewer_publications[0])
    row["email"] = "private@example.org"
    object.__setattr__(evidence, "reviewer_publications", (row,))
    destination = tmp_path / "extra-join"
    with pytest.raises(OpenReviewProtocolError, match="unrelated fields"):
        write_openreview_snapshot(
            _single_reviewer_snapshot(), destination, reviewer_capacity=1, expertise=evidence
        )
    assert not destination.exists()


def test_direct_writer_rechecks_evidence_privacy_before_creating_destination(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    object.__setattr__(
        evidence,
        "reviewer_publications",
        (
            {
                "reviewer_id": "~R1",
                "note": {
                    "id": "p",
                    "content": {
                        "title": {"value": "T"},
                        "authorids": {"value": ["~R1", "coauthor@example.org"]},
                    },
                },
            },
        ),
    )
    snapshot = OpenReviewSnapshot(
        notes=({"id": "s", "content": {"title": {"value": "Submission"}}},),
        reviewer_ids=("~R1",),
        base_url="https://openreview.test",
        paper_filter={"invitation": "Venue/-/Submission"},
        reviewer_group="Venue/Reviewers",
    )
    destination = tmp_path / "private-leak"
    with pytest.raises(OpenReviewProtocolError, match="author identity"):
        write_openreview_snapshot(snapshot, destination, reviewer_capacity=1, expertise=evidence)
    assert not destination.exists()


def test_direct_writer_rechecks_declared_policy_before_install(tmp_path: Path) -> None:
    evidence = _evidence()
    object.__setattr__(
        evidence, "policy", ExpertiseFetchPolicy(("Public/-/Paper",), require_abstract=True)
    )
    publication = api_module._thaw_json(evidence.reviewer_publications[0])
    publication["note"]["invitations"] = tuple(publication["note"]["invitations"])
    publication["note"]["content"]["authorids"]["value"] = tuple(
        publication["note"]["content"]["authorids"]["value"]
    )
    publication["note"]["content"]["abstract"]["value"] = ""
    object.__setattr__(evidence, "reviewer_publications", (publication,))
    snapshot = OpenReviewSnapshot(
        notes=({"id": "s", "content": {"title": {"value": "Submission"}}},),
        reviewer_ids=("~R1",),
        base_url="https://openreview.test",
        paper_filter={"invitation": "Venue/-/Submission"},
        reviewer_group="Venue/Reviewers",
    )
    destination = tmp_path / "policy-bypass"
    with pytest.raises(OpenReviewProtocolError, match="content policy"):
        write_openreview_snapshot(snapshot, destination, reviewer_capacity=1, expertise=evidence)
    assert not destination.exists()


def test_direct_writer_rejects_nested_private_field_after_evidence_mutation(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    publication = api_module._thaw_json(evidence.reviewer_publications[0])
    publication["note"]["invitations"] = tuple(publication["note"]["invitations"])
    publication["note"]["content"]["authorids"]["value"] = tuple(
        publication["note"]["content"]["authorids"]["value"]
    )
    publication["note"]["content"]["title"]["email"] = "coauthor@example.org"
    object.__setattr__(evidence, "reviewer_publications", (publication,))
    snapshot = OpenReviewSnapshot(
        notes=({"id": "s", "content": {"title": {"value": "Submission"}}},),
        reviewer_ids=("~R1",),
        base_url="https://openreview.test",
        paper_filter={"invitation": "Venue/-/Submission"},
        reviewer_group="Venue/Reviewers",
    )
    destination = tmp_path / "nested-private-leak"
    with pytest.raises(OpenReviewProtocolError, match="unrelated fields"):
        write_openreview_snapshot(snapshot, destination, reviewer_capacity=1, expertise=evidence)
    assert not destination.exists()


def test_direct_writer_rejects_email_profile_id_after_evidence_mutation(tmp_path: Path) -> None:
    evidence = _evidence()
    profile = api_module._thaw_json(evidence.profiles[0])
    profile["id"] = "reviewer@example.org"
    object.__setattr__(evidence, "profiles", (profile,))
    snapshot = OpenReviewSnapshot(
        notes=({"id": "s", "content": {"title": {"value": "Submission"}}},),
        reviewer_ids=("reviewer@example.org",),
        base_url="https://openreview.test",
        paper_filter={"invitation": "Venue/-/Submission"},
        reviewer_group="Venue/Reviewers",
    )
    destination = tmp_path / "email-profile-leak"
    with pytest.raises(OpenReviewProtocolError, match="tilde reviewer"):
        write_openreview_snapshot(snapshot, destination, reviewer_capacity=1, expertise=evidence)
    assert not destination.exists()


def test_author_acquisition_caps_global_and_per_reviewer_retained_publications() -> None:
    responses = _fixture()
    client = _client(FixtureTransport(responses))
    snapshot = _snapshot(client)
    with pytest.raises(OpenReviewProtocolError, match="max_publications"):
        fetch_reviewer_expertise(client, snapshot, _policy(max_publications=1))

    responses = _fixture()
    responses[3]["notes"][1]["invitations"] = ["Public/-/Paper"]
    client = _client(FixtureTransport(responses))
    snapshot = _snapshot(client)
    with pytest.raises(OpenReviewProtocolError, match="max_publications_per_reviewer"):
        fetch_reviewer_expertise(client, snapshot, _policy(max_publications_per_reviewer=1))


def test_scan_budget_stops_author_pagination_before_fetching_excess_pages() -> None:
    transport = FixtureTransport(_fixture())
    client = _client(transport)
    snapshot = _snapshot(client)
    with pytest.raises(OpenReviewProtocolError, match="max_scanned_notes"):
        fetch_reviewer_expertise(client, snapshot, _policy(max_scanned_notes=1))
    assert [urlsplit(url).path for url in transport.calls] == [
        "/notes",
        "/groups",
        "/profiles",
        "/notes",
    ]
    assert parse_qs(urlsplit(transport.calls[-1]).query)["limit"] == ["1"]


def test_author_pagination_reduces_every_page_to_remaining_scan_budget() -> None:
    responses = [
        {"count": 3, "notes": [{"id": "a"}, {"id": "b"}]},
        {"notes": [{"id": "c"}]},
    ]
    transport = FixtureTransport(responses)
    notes = _client(transport).get_all_author_notes("~R1", max_scanned_notes=3)
    assert [note["id"] for note in notes] == ["a", "b", "c"]
    assert [parse_qs(urlsplit(url).query)["limit"] for url in transport.calls] == [["2"], ["1"]]

    responses = [
        {"count": 3, "notes": [{"id": "a"}, {"id": "b"}]},
        {"notes": [{"id": "c"}, {"id": "d"}]},
    ]
    with pytest.raises(OpenReviewProtocolError, match="requested limit"):
        _client(FixtureTransport(responses)).get_all_author_notes("~R1", max_scanned_notes=3)


def test_author_scan_budget_rejects_invalid_controls_before_network() -> None:
    transport = FixtureTransport([])
    client = _client(transport)
    for invalid in (-1, True, "two"):
        with pytest.raises(ValueError, match="max_records"):
            client.get_all_author_notes("~R1", max_scanned_notes=invalid)
    assert transport.calls == []


def test_retry_after_without_timezone_and_snapshot_budgets_fail_safely() -> None:
    client = _client(FixtureTransport([]))
    assert client._retry_after_seconds({"retry-after": "Sun, 06 Nov 1994 08:49:37"}) is None
    with pytest.raises(ValueError, match="record hard limit"):
        api_module._bounded_snapshot_values(range(3), 2, "fixture")
    budget = api_module._SnapshotJsonBudget(max_items=10, max_utf8_bytes=2)
    with pytest.raises(ValueError, match="UTF-8 byte budget"):
        budget.consume_text("abc")
    with pytest.raises(ValueError, match="UTF-8 byte budget"):
        budget.consume_cached(0, 3)
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="circular references"):
        api_module._snapshot_json(cyclic)


def test_scan_budget_carries_across_reviewers_without_fetching_excess_pages() -> None:
    transport = FixtureTransport(_fixture())
    client = _client(transport)
    snapshot = _snapshot(client)
    with pytest.raises(OpenReviewProtocolError, match="max_scanned_notes"):
        fetch_reviewer_expertise(client, snapshot, _policy(max_scanned_notes=4))
    assert [urlsplit(url).path for url in transport.calls] == [
        "/notes",
        "/groups",
        "/profiles",
        "/notes",
        "/notes",
        "/profiles",
        "/notes",
    ]
    assert parse_qs(urlsplit(transport.calls[-1]).query)["limit"] == ["1"]


def test_publication_evidence_does_not_persist_unrelated_author_emails(tmp_path: Path) -> None:
    responses = _fixture()
    responses[3]["notes"][0]["content"]["authorids"]["value"].append("coauthor@example.org")
    client = _client(FixtureTransport(responses))
    snapshot = _snapshot(client)
    expertise = fetch_reviewer_expertise(client, snapshot, _policy())
    assert expertise.reviewer_publications[0]["note"]["content"]["authorids"]["value"] == (
        "~Reviewer_One1",
    )
    destination = tmp_path / "minimized"
    write_openreview_snapshot(snapshot, destination, reviewer_capacity=1, expertise=expertise)
    assert "coauthor@example.org" not in (destination / "reviewer-publications.jsonl").read_text(
        encoding="utf-8"
    )


def test_snapshot_writer_validates_more_than_default_offline_submission_limit(
    tmp_path: Path,
) -> None:
    snapshot = OpenReviewSnapshot(
        notes=tuple(
            {"id": f"submission-{index}", "content": {"title": {"value": "Title"}}}
            for index in range(10_001)
        ),
        reviewer_ids=("~R1",),
        base_url="https://openreview.test",
        paper_filter={"invitation": "Venue/-/Submission"},
        reviewer_group="Venue/Reviewers",
    )
    destination = tmp_path / "large-valid"
    manifest = write_openreview_snapshot(
        snapshot, destination, reviewer_capacity=1, expertise=_evidence()
    )
    assert manifest["records"]["documents"] == 10_001
    assert (destination / "manifest.json").is_file()


def test_expertise_acquisition_rejects_unresolved_email_members() -> None:
    client = _client(FixtureTransport([]))
    snapshot = OpenReviewSnapshot(
        notes=({"id": "p", "content": {"title": {"value": "T"}}},),
        reviewer_ids=("reviewer@example.org",),
        base_url="https://openreview.test",
        paper_filter={"invitation": "Venue/-/Submission"},
        reviewer_group="Venue/Reviewers",
    )
    with pytest.raises(OpenReviewProtocolError, match="tilde reviewer"):
        fetch_reviewer_expertise(client, snapshot, _policy())


def test_filter_policy_rejects_inverted_dates_and_accepts_undated_without_date_filter() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        ExpertiseFetchPolicy(("Public/-/Paper",), minimum_date_ms=2_000, maximum_date_ms=1_000)
    projected, reason = api_module._publication_projection(
        _publication_note(cdate=None), "~R1", ExpertiseFetchPolicy(("Public/-/Paper",))
    )
    assert reason == "retained"
    assert "pdate" not in projected
    assert projected["content"]["year"]["value"] is None
    assert api_module._profile_projection({"id": "~R1", "content": {"names": [{"fullname": "R"}]}})[
        "content"
    ] == {"names": [{"fullname": "R", "preferred": False}]}


def test_snapshot_writer_rejects_profile_order_mismatch_atomically(tmp_path: Path) -> None:
    transport = FixtureTransport(_fixture())
    client = _client(transport)
    snapshot = _snapshot(client)
    expertise = fetch_reviewer_expertise(client, snapshot, _policy())
    swapped = OpenReviewExpertiseEvidence(
        tuple(reversed(expertise.profiles)),
        expertise.reviewer_publications,
        expertise.policy,
        expertise.filter_counts,
    )
    destination = tmp_path / "not-created"
    with pytest.raises(DataValidationError, match="group order"):
        write_openreview_snapshot(snapshot, destination, reviewer_capacity=1, expertise=swapped)
    assert not destination.exists()


def test_group_and_profile_envelopes_reject_oversize_or_invalid_rows() -> None:
    group_client = _client(
        FixtureTransport([{"groups": [{"id": "Reviewers", "members": ["~R1", "~R2"]}]}]),
        max_records=1,
    )
    with pytest.raises(OpenReviewProtocolError, match="max_records"):
        group_client.get_group_members("Reviewers")
    for payload, message in (
        ({"profiles": [3]}, "profile must be"),
        ({"profiles": [{"id": "~R1"}]}, "profile content"),
        ({"profiles": "wrong"}, "profiles must be"),
    ):
        client = _client(FixtureTransport([payload]))
        with pytest.raises(OpenReviewProtocolError, match=message):
            client.get_profile("~R1")


def test_publication_dates_prefer_publication_over_original_over_creation() -> None:
    note = _publication_note(
        pdate=1_704_067_200_000, odate=1_577_836_800_000, cdate=1_451_606_400_000
    )
    projected, reason = api_module._publication_projection(note, "~R1", _policy())
    assert reason == "retained"
    assert projected["content"]["year"]["value"] == 2024
    assert projected["pdate"] == 1_704_067_200_000
