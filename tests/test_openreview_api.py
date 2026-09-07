from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

import pytest

import peermatchlab.cli as cli_module
import peermatchlab.openreview_api as api_module
from peermatchlab.cli import main
from peermatchlab.models import DataValidationError
from peermatchlab.openreview_api import (
    HttpResponse,
    OpenReviewClient,
    OpenReviewClientConfig,
    OpenReviewHttpError,
    OpenReviewProtocolError,
    OpenReviewSnapshot,
    OpenReviewTransportError,
    RetryPolicy,
    UrllibTransport,
    fetch_openreview_snapshot,
    write_openreview_snapshot,
)


def _response(
    payload: object,
    *,
    status: int = 200,
    headers: Mapping[str, str] | None = None,
) -> HttpResponse:
    return HttpResponse(
        status,
        headers or {},
        json.dumps(payload, separators=(",", ":")).encode(),
    )


class QueueTransport:
    def __init__(self, *responses: HttpResponse | OpenReviewTransportError) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, str], float, int]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> HttpResponse:
        self.calls.append((url, headers, timeout_seconds, max_response_bytes))
        outcome = self.responses.pop(0)
        if isinstance(outcome, OpenReviewTransportError):
            raise outcome
        return outcome


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def read(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _client(
    transport: QueueTransport,
    *,
    clock: FakeClock | None = None,
    token: str | None = None,
    **config: Any,
) -> OpenReviewClient:
    timer = clock or FakeClock()
    config_values: dict[str, Any] = {
        "base_url": "https://openreview.test",
        "requests_per_second": 4,
    }
    config_values.update(config)
    return OpenReviewClient(
        config=OpenReviewClientConfig(**config_values),
        token=token,
        transport=transport,
        sleeper=timer.sleep,
        monotonic=timer.read,
        wall_clock=timer.read,
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://api2.openreview.net",
        "https://user:secret@api2.openreview.net",
        "https://api2.openreview.net/path",
        "https://api2.openreview.net?token=secret",
        "https://api2.openreview.net/#fragment",
        "https://api2.openreview.net:99999",
        "not-a-url",
        "",
    ],
)
def test_client_config_rejects_unsafe_base_urls(url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        OpenReviewClientConfig(base_url=url)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("page_size", 0),
        ("page_size", 1001),
        ("max_pages", True),
        ("max_records", -1),
        ("max_response_bytes", 0),
        ("timeout_seconds", float("inf")),
        ("requests_per_second", 0.0),
        ("retry", "invalid"),
    ],
)
def test_client_config_rejects_invalid_resource_bounds(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        OpenReviewClientConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"max_attempts": 21},
        {"max_attempts": True},
        {"initial_delay_seconds": -1},
        {"max_backoff_seconds": float("nan")},
        {"max_retry_after_seconds": "1"},
        {"retryable_status_codes": frozenset()},
        {"retryable_status_codes": frozenset({True})},
        {"retryable_status_codes": frozenset({700})},
    ],
)
def test_retry_policy_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)  # type: ignore[arg-type]


def test_http_response_normalizes_headers_and_validates_values() -> None:
    response = HttpResponse(429, {"Retry-After": "3"}, b"{}")
    assert response.headers["retry-after"] == "3"
    with pytest.raises(ValueError, match="between"):
        HttpResponse(99, {}, b"")
    with pytest.raises(TypeError, match="integer"):
        HttpResponse(True, {}, b"")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bytes"):
        HttpResponse(200, {}, "")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="headers"):
        HttpResponse(200, {"x": 1}, b"")  # type: ignore[dict-item]


def test_cursor_pagination_group_lookup_auth_and_rate_limit_are_explicit() -> None:
    transport = QueueTransport(
        _response({"count": 3, "notes": [{"id": "a"}, {"id": "b"}]}),
        _response({"notes": [{"id": "c"}]}),
        _response({"groups": [{"id": "Venue/Reviewers", "members": ["~R1", "~R2"]}]}),
    )
    clock = FakeClock()
    client = _client(transport, clock=clock, token="secret-token", page_size=2)

    snapshot = fetch_openreview_snapshot(
        client,
        invitation="Venue/-/Submission",
        reviewer_group="Venue/Reviewers",
    )

    assert [note["id"] for note in snapshot.notes] == ["a", "b", "c"]
    assert snapshot.reviewer_ids == ("~R1", "~R2")
    first = urlsplit(transport.calls[0][0])
    second = urlsplit(transport.calls[1][0])
    group = urlsplit(transport.calls[2][0])
    assert first.path == "/notes"
    assert parse_qs(first.query) == {
        "count": ["True"],
        "invitation": ["Venue/-/Submission"],
        "limit": ["2"],
        "sort": ["id"],
    }
    assert parse_qs(second.query)["after"] == ["b"]
    assert parse_qs(group.query) == {"id": ["Venue/Reviewers"], "limit": ["2"]}
    assert all(call[1]["Authorization"] == "Bearer secret-token" for call in transport.calls)
    assert all("secret-token" not in call[0] for call in transport.calls)
    assert clock.sleeps == [0.25, 0.25]


def test_venue_id_uses_official_content_filter_and_empty_result_is_valid() -> None:
    transport = QueueTransport(_response({"count": 0, "notes": []}))
    client = _client(transport)
    assert client.get_all_notes(venue_id="Venue.cc/2026/Conference") == ()
    assert parse_qs(urlsplit(transport.calls[0][0]).query)["content.venueid"] == [
        "Venue.cc/2026/Conference"
    ]


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"invitation": "a", "venue_id": "b"}, {"invitation": ""}, {"venue_id": "\n"}],
)
def test_note_filter_requires_exactly_one_non_empty_value(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        _client(QueueTransport()).get_all_notes(**kwargs)


def test_retry_honors_retry_after_then_exponential_backoff() -> None:
    transport = QueueTransport(
        _response({}, status=429, headers={"Retry-After": "7"}),
        _response({}, status=503),
        _response({"groups": [{"id": "Reviewers", "members": ["~R1"]}]}),
    )
    clock = FakeClock()
    client = _client(
        transport,
        clock=clock,
        requests_per_second=1_000_000,
        retry=RetryPolicy(max_attempts=3, initial_delay_seconds=1),
    )

    assert client.get_group_members("Reviewers") == ("~R1",)
    assert clock.sleeps == [7.0, 2.0]


def test_retry_after_http_date_is_parsed_and_capped() -> None:
    transport = QueueTransport(
        _response({}, status=429, headers={"Retry-After": "Thu, 01 Jan 1970 00:21:40 GMT"}),
        _response({"groups": [{"id": "Reviewers", "members": ["~R1"]}]}),
    )
    clock = FakeClock(now=1_000)
    client = _client(
        transport,
        clock=clock,
        requests_per_second=1_000_000,
        retry=RetryPolicy(max_attempts=2, max_retry_after_seconds=12),
    )
    client.get_group_members("Reviewers")
    assert clock.sleeps == [12]


@pytest.mark.parametrize("header", ["nonsense", "-2", "NaN"])
def test_invalid_retry_after_falls_back_to_exponential_delay(header: str) -> None:
    transport = QueueTransport(
        _response({}, status=429, headers={"Retry-After": header}),
        _response({"groups": [{"id": "Reviewers", "members": ["~R1"]}]}),
    )
    clock = FakeClock()
    client = _client(
        transport,
        clock=clock,
        requests_per_second=1_000_000,
        retry=RetryPolicy(max_attempts=2, initial_delay_seconds=1.5),
    )
    client.get_group_members("Reviewers")
    assert clock.sleeps == [1.5]


def test_transport_failures_are_retried_but_finite() -> None:
    transport = QueueTransport(
        OpenReviewTransportError("offline"),
        _response({"groups": [{"id": "Reviewers", "members": ["~R1"]}]}),
    )
    clock = FakeClock()
    client = _client(
        transport,
        clock=clock,
        requests_per_second=1_000_000,
        retry=RetryPolicy(max_attempts=2, initial_delay_seconds=0),
    )
    assert client.get_group_members("Reviewers") == ("~R1",)
    assert len(transport.calls) == 2

    failing = QueueTransport(
        OpenReviewTransportError("offline"),
        OpenReviewTransportError("still offline with super-secret"),
    )
    with pytest.raises(OpenReviewTransportError, match="after 2 attempt") as captured:
        _client(
            failing,
            requests_per_second=1_000_000,
            retry=RetryPolicy(max_attempts=2, initial_delay_seconds=0),
        ).get_group_members("Reviewers")
    assert "super-secret" not in str(captured.value)
    assert captured.value.__suppress_context__
    assert captured.value.__context__ is None


def test_non_retryable_and_exhausted_http_errors_report_bounded_context() -> None:
    unauthorized = QueueTransport(_response({"secret": "do not echo"}, status=401))
    with pytest.raises(OpenReviewHttpError) as captured:
        _client(unauthorized).get_group_members("Reviewers")
    assert captured.value.status_code == 401
    assert captured.value.attempts == 1
    assert "do not echo" not in str(captured.value)

    unavailable = QueueTransport(_response({}, status=503), _response({}, status=503))
    with pytest.raises(OpenReviewHttpError) as exhausted:
        _client(
            unavailable,
            requests_per_second=1_000_000,
            retry=RetryPolicy(max_attempts=2, initial_delay_seconds=0),
        ).get_group_members("Reviewers")
    assert exhausted.value.attempts == 2

    redirect = QueueTransport(_response({}, status=302, headers={"Location": "https://other.test"}))
    with pytest.raises(OpenReviewHttpError) as redirected:
        _client(redirect, token="secret").get_group_members("Reviewers")
    assert redirected.value.status_code == 302
    assert len(redirect.calls) == 1


@pytest.mark.parametrize(
    "body",
    [b"not json", b'{"groups":[],"groups":[]}', b"\xff"],
)
def test_response_body_must_be_utf8_strict_json(body: bytes) -> None:
    transport = QueueTransport(HttpResponse(200, {}, body))
    with pytest.raises(OpenReviewProtocolError, match="strict JSON"):
        _client(transport).get_group_members("Reviewers")


def test_invalid_json_error_does_not_retain_or_echo_response_body() -> None:
    transport = QueueTransport(HttpResponse(200, {}, b'{"super-secret":'))
    with pytest.raises(OpenReviewProtocolError) as captured:
        _client(transport).get_group_members("Reviewers")
    assert "super-secret" not in str(captured.value)
    assert captured.value.__suppress_context__
    assert captured.value.__context__ is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"notes": []}, "count"),
        ({"count": True, "notes": []}, "count"),
        ({"count": 0, "notes": {}}, "array"),
        ({"count": 1, "notes": [1]}, "JSON object"),
        ({"count": 1, "notes": [{}]}, "non-empty id"),
        ({"count": 1, "notes": [{"id": ""}]}, "non-empty id"),
        ({"count": 1, "notes": [{"id": "a"}, {"id": "b"}]}, "more notes"),
        ({"count": 2, "notes": [{"id": "a"}]}, "ended"),
    ],
)
def test_note_page_protocol_fails_closed(payload: object, message: str) -> None:
    with pytest.raises(OpenReviewProtocolError, match=message):
        _client(QueueTransport(_response(payload)), page_size=2).get_all_notes(invitation="S")


def test_note_count_and_page_resource_limits_fail_before_unbounded_work() -> None:
    with pytest.raises(OpenReviewProtocolError, match="max_records"):
        _client(QueueTransport(_response({"count": 3, "notes": []})), max_records=2).get_all_notes(
            invitation="S"
        )
    with pytest.raises(OpenReviewProtocolError, match="requested limit"):
        _client(
            QueueTransport(_response({"count": 3, "notes": [{"id": "a"}, {"id": "b"}]})),
            page_size=1,
        ).get_all_notes(invitation="S")
    with pytest.raises(OpenReviewProtocolError, match="max_pages"):
        _client(
            QueueTransport(_response({"count": 2, "notes": [{"id": "a"}]})),
            page_size=1,
            max_pages=1,
        ).get_all_notes(invitation="S")


def test_repeated_cursor_page_is_detected_as_duplicate() -> None:
    transport = QueueTransport(
        _response({"count": 2, "notes": [{"id": "a"}]}),
        _response({"notes": [{"id": "a"}]}),
    )
    with pytest.raises(OpenReviewProtocolError, match="duplicate note id"):
        _client(transport, page_size=1).get_all_notes(invitation="S")


def test_non_increasing_cursor_page_is_rejected() -> None:
    transport = QueueTransport(
        _response({"count": 2, "notes": [{"id": "b"}]}),
        _response({"notes": [{"id": "a"}]}),
    )
    with pytest.raises(OpenReviewProtocolError, match="strictly increasing"):
        _client(transport, page_size=1).get_all_notes(invitation="S")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "array"),
        ({"groups": []}, "returned 0"),
        ({"groups": [{"id": "Other", "members": []}]}, "does not match"),
        ({"groups": [{"id": "R", "members": "~R1"}]}, "array"),
        ({"groups": [{"id": "R", "members": [1]}]}, "non-empty string"),
        ({"groups": [{"id": "R", "members": ["~R1", "~R1"]}]}, "duplicate"),
        ({"groups": [{"id": "R", "members": []}]}, "contains no members"),
    ],
)
def test_group_protocol_is_strict(payload: object, message: str) -> None:
    with pytest.raises(OpenReviewProtocolError, match=message):
        _client(QueueTransport(_response(payload))).get_group_members("R")


def test_group_id_and_token_reject_header_injection() -> None:
    with pytest.raises(ValueError, match="newlines"):
        _client(QueueTransport()).get_group_members("R\nInjected")
    with pytest.raises(ValueError, match="token"):
        _client(QueueTransport(), token="secret\rheader")


def _snapshot() -> OpenReviewSnapshot:
    return OpenReviewSnapshot(
        notes=(
            {
                "id": "paper-1",
                "forum": "paper-1",
                "content": {
                    "title": {"value": "A paper"},
                    "abstract": {"value": "Summary"},
                    "keywords": {"value": ["IR"]},
                },
            },
        ),
        reviewer_ids=("~Reviewer1", "~Reviewer2"),
        base_url="https://api2.openreview.net",
        paper_filter={"invitation": "Venue/-/Submission"},
        reviewer_group="Venue/Reviewers",
    )


def test_snapshot_writer_is_canonical_converted_hashed_and_non_overwriting(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "snapshot"
    manifest = write_openreview_snapshot(_snapshot(), destination, reviewer_capacity=3)

    assert sorted(path.name for path in destination.iterdir()) == [
        "documents.json",
        "experts.json",
        "manifest.json",
        "reviewer-ids.txt",
        "submissions.jsonl",
    ]
    documents = json.loads((destination / "documents.json").read_text(encoding="utf-8"))
    experts = json.loads((destination / "experts.json").read_text(encoding="utf-8"))
    assert documents[0]["id"] == "paper-1"
    assert experts[0]["capacity"] == 3
    assert manifest["records"] == {"documents": 1, "experts": 2}
    assert manifest["conversion"] == {"reviewer_capacity": 3}
    assert "super-secret" not in json.dumps(dict(manifest))
    for name, evidence in manifest["files"].items():
        assert evidence["bytes"] == (destination / name).stat().st_size
        assert evidence["sha256"] == hashlib.sha256((destination / name).read_bytes()).hexdigest()
    with pytest.raises(DataValidationError, match="already exists"):
        write_openreview_snapshot(_snapshot(), destination, reviewer_capacity=3)


def test_snapshot_writer_removes_staging_directory_after_conversion_failure(
    tmp_path: Path,
) -> None:
    invalid = OpenReviewSnapshot(
        notes=({"id": "paper-1", "content": {}},),
        reviewer_ids=("~R1",),
        base_url="https://api2.openreview.net",
        paper_filter={"invitation": "S"},
        reviewer_group="R",
    )
    destination = tmp_path / "snapshot"
    with pytest.raises(DataValidationError, match="title"):
        write_openreview_snapshot(invalid, destination, reviewer_capacity=1)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_snapshot_writer_publishes_nothing_when_manifest_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write_json = api_module.write_json

    def fail_manifest(path: str | Path, value: object) -> None:
        if Path(path).name == "manifest.json":
            raise OSError("simulated disk failure")
        original_write_json(path, value)

    monkeypatch.setattr(api_module, "write_json", fail_manifest)
    destination = tmp_path / "snapshot"
    with pytest.raises(OSError, match="simulated disk failure"):
        write_openreview_snapshot(_snapshot(), destination, reviewer_capacity=2)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_snapshot_requires_non_empty_source_sets() -> None:
    with pytest.raises(ValueError, match="require notes"):
        OpenReviewSnapshot((), ("~R1",), "https://api2.openreview.net", {}, "R")
    with pytest.raises(ValueError, match="reviewers"):
        OpenReviewSnapshot(({"id": "a"},), (), "https://api2.openreview.net", {}, "R")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_url": "http://api2.openreview.net"}, "base_url"),
        ({"paper_filter": {}}, "paper_filter"),
        ({"paper_filter": {"forum": "paper-1"}}, "paper_filter"),
        ({"paper_filter": {"invitation": ""}}, "paper_filter value"),
        ({"paper_filter": {"invitation": " S "}}, "surrounding whitespace"),
        ({"reviewer_group": " R "}, "surrounding whitespace"),
        ({"notes": ({"content": {}},)}, "non-empty string ids"),
        ({"notes": ({"id": "a"}, {"id": "a"})}, "unique"),
        ({"notes": (1,)}, "JSON objects"),
        ({"reviewer_ids": ("",)}, "snapshot reviewer id"),
        ({"reviewer_ids": (" ~R1 ",)}, "surrounding whitespace"),
        ({"reviewer_ids": ("~R1", "~R1")}, "unique"),
    ],
)
def test_snapshot_validates_its_public_boundary(overrides: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "notes": ({"id": "paper-1", "content": {"title": {"value": "T"}}},),
        "reviewer_ids": ("~R1",),
        "base_url": "https://api2.openreview.net",
        "paper_filter": {"invitation": "S"},
        "reviewer_group": "R",
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        OpenReviewSnapshot(**values)  # type: ignore[arg-type]


def test_fetch_openreview_cli_uses_env_token_and_writes_no_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    class StubClient:
        base_url = "https://api2.openreview.net"

        def __init__(self, *, config: OpenReviewClientConfig, token: str | None) -> None:
            captured["config"] = config
            captured["token"] = token

        def get_all_notes(
            self, *, invitation: str | None = None, venue_id: str | None = None
        ) -> tuple[Mapping[str, Any], ...]:
            captured["filter"] = (invitation, venue_id)
            return _snapshot().notes

        def get_group_members(self, group_id: str) -> tuple[str, ...]:
            captured["group"] = group_id
            return _snapshot().reviewer_ids

    monkeypatch.setenv("TEST_OPENREVIEW_TOKEN", "super-secret")
    monkeypatch.setattr(cli_module, "OpenReviewClient", StubClient)
    destination = tmp_path / "fetched"
    code = main(
        [
            "fetch-openreview",
            "--invitation",
            "Venue/-/Submission",
            "--reviewer-group",
            "Venue/Reviewers",
            "--reviewer-capacity",
            "4",
            "--directory",
            str(destination),
            "--token-env",
            "TEST_OPENREVIEW_TOKEN",
            "--page-size",
            "100",
            "--max-records",
            "500",
            "--max-attempts",
            "3",
            "--requests-per-second",
            "2",
        ]
    )

    assert code == 0
    assert captured["token"] == "super-secret"
    assert captured["filter"] == ("Venue/-/Submission", None)
    assert captured["group"] == "Venue/Reviewers"
    assert isinstance(captured["config"], OpenReviewClientConfig)
    assert captured["config"].page_size == 100
    assert "super-secret" not in (destination / "manifest.json").read_text(encoding="utf-8")
    assert "fetched 1 submissions and 2 reviewers" in capsys.readouterr().out


def test_fetch_openreview_cli_reports_missing_token_environment_variable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "fetch-openreview",
            "--venue-id",
            "Venue",
            "--reviewer-group",
            "Reviewers",
            "--reviewer-capacity",
            "1",
            "--directory",
            str(tmp_path / "snapshot"),
            "--token-env",
            "CERTAINLY_MISSING_OPENREVIEW_TOKEN",
        ]
    )
    assert code == 2
    assert "is not defined" in capsys.readouterr().err


class _ContextResponse:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self._body = io.BytesIO(body)
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"

    def read(self, amount: int) -> bytes:
        return self._body.read(amount)

    def __enter__(self) -> _ContextResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_urllib_transport_success_http_error_network_error_and_size_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = UrllibTransport()
    monkeypatch.setattr(
        api_module, "_open_https_request", lambda request, timeout: _ContextResponse(b"{}")
    )
    response = transport.get(
        "https://openreview.test/groups",
        headers={},
        timeout_seconds=1,
        max_response_bytes=10,
    )
    assert response.status_code == 200

    def http_error(request: object, timeout: float) -> None:
        headers = Message()
        headers["Retry-After"] = "1"
        raise HTTPError("https://openreview.test", 429, "rate", headers, io.BytesIO(b"{}"))

    monkeypatch.setattr(api_module, "_open_https_request", http_error)
    response = transport.get(
        "https://openreview.test/groups",
        headers={},
        timeout_seconds=1,
        max_response_bytes=10,
    )
    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"

    monkeypatch.setattr(
        api_module,
        "_open_https_request",
        lambda request, timeout: (_ for _ in ()).throw(URLError("offline")),
    )
    with pytest.raises(OpenReviewTransportError, match="offline"):
        transport.get(
            "https://openreview.test/groups",
            headers={},
            timeout_seconds=1,
            max_response_bytes=10,
        )

    monkeypatch.setattr(
        api_module,
        "_open_https_request",
        lambda request, timeout: _ContextResponse(b"too large"),
    )
    with pytest.raises(OpenReviewProtocolError, match="exceeds"):
        transport.get(
            "https://openreview.test/groups",
            headers={},
            timeout_seconds=1,
            max_response_bytes=2,
        )


def test_urllib_transport_refuses_non_bytes_body() -> None:
    class BadResponse:
        def read(self, amount: int) -> str:
            return "not bytes"

    with pytest.raises(OpenReviewProtocolError, match="non-bytes"):
        UrllibTransport._read_body(BadResponse(), 10)


def test_transport_redirect_handler_refuses_redirects() -> None:
    assert (
        api_module._NoRedirectHandler().redirect_request(
            Request("https://openreview.test"),
            None,
            302,
            "redirect",
            {},
            "https://other.test",
        )
        is None
    )
