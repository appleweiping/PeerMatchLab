"""Bounded, read-only OpenReview API v2 synchronization.

The transport and clocks are injectable so pagination, retry, and rate-limit
behaviour can be verified without contacting OpenReview in the test suite.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from peermatchlab.io import load_json_text, write_json
from peermatchlab.models import DataValidationError
from peermatchlab.openreview import openreview_submissions_from_records, reviewer_ids_to_experts

DEFAULT_OPENREVIEW_API_V2_URL = "https://api2.openreview.net"
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_SNAPSHOT_RECORDS = 10_000_000
_MAX_SNAPSHOT_CONTAINER_ITEMS = 100_000
_MAX_SNAPSHOT_DEPTH = 100
_MAX_SNAPSHOT_EXPANDED_ITEMS = 10_000_000
_MAX_SNAPSHOT_UTF8_BYTES = 1024 * 1024 * 1024
_MAX_HTTP_RESPONSE_BYTES = 1024 * 1024 * 1024
_MAX_EXPERTISE_JSONL_FILE_BYTES = 64 * 1024 * 1024


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


class OpenReviewError(RuntimeError):
    """Base class for bounded OpenReview synchronization failures."""


class OpenReviewTransportError(OpenReviewError):
    """Raised when no HTTP response can be obtained."""


class OpenReviewProtocolError(OpenReviewError):
    """Raised when an OpenReview response violates the expected protocol."""


class OpenReviewHttpError(OpenReviewError):
    """Raised for a terminal non-success HTTP response."""

    def __init__(self, status_code: int, path: str, attempts: int) -> None:
        self.status_code = status_code
        self.path = path
        self.attempts = attempts
        super().__init__(
            f"OpenReview GET {path} failed with HTTP {status_code} after {attempts} attempt(s)"
        )


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Minimal response value understood by :class:`OpenReviewClient`."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes

    def __post_init__(self) -> None:
        if isinstance(self.status_code, bool) or not isinstance(self.status_code, int):
            raise TypeError("HTTP status_code must be an integer")
        if not 100 <= self.status_code <= 599:
            raise ValueError("HTTP status_code must be between 100 and 599")
        if not isinstance(self.body, bytes):
            raise TypeError("HTTP body must be bytes")
        if not isinstance(self.headers, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.headers.items()
        ):
            raise TypeError("HTTP headers must map strings to strings")
        object.__setattr__(
            self,
            "headers",
            MappingProxyType({key.lower(): value for key, value in self.headers.items()}),
        )


class HttpTransport(Protocol):
    """Injectable HTTP GET boundary used by the OpenReview client."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> HttpResponse:
        """Fetch one bounded response or raise :class:`OpenReviewTransportError`."""


class _NoRedirectHandler(HTTPRedirectHandler):
    """Keep bearer credentials on the explicitly configured API origin."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _open_https_request(request: Request, timeout_seconds: float) -> Any:
    opener = build_opener(_NoRedirectHandler())
    return opener.open(request, timeout=timeout_seconds)  # nosec B310


class UrllibTransport:
    """Standard-library HTTPS transport with bounded response reads."""

    @staticmethod
    def _read_body(response: Any, max_response_bytes: int) -> bytes:
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or not 1 <= max_response_bytes <= _MAX_HTTP_RESPONSE_BYTES
        ):
            raise OpenReviewProtocolError(
                f"max_response_bytes must be an integer between 1 and {_MAX_HTTP_RESPONSE_BYTES}"
            )
        body: object = response.read(max_response_bytes + 1)
        if not isinstance(body, bytes):
            raise OpenReviewProtocolError("HTTP transport returned a non-bytes response body")
        if len(body) > max_response_bytes:
            raise OpenReviewProtocolError(f"OpenReview response exceeds {max_response_bytes} bytes")
        return body

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> HttpResponse:
        request = Request(url, headers=dict(headers), method="GET")
        try:
            # The client validates the base URL as HTTPS before this boundary.
            with _open_https_request(request, timeout_seconds) as response:
                return HttpResponse(
                    status_code=int(response.status),
                    headers=dict(response.headers.items()),
                    body=self._read_body(response, max_response_bytes),
                )
        except HTTPError as error:
            return HttpResponse(
                status_code=error.code,
                headers=dict(error.headers.items()) if error.headers is not None else {},
                body=self._read_body(error, max_response_bytes),
            )
        except OpenReviewProtocolError:
            raise
        except (URLError, TimeoutError, OSError) as error:
            raise OpenReviewTransportError(f"OpenReview transport failed: {error}") from error


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Finite retry policy for transient OpenReview responses."""

    max_attempts: int = 5
    initial_delay_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    max_retry_after_seconds: float = 120.0
    retryable_status_codes: frozenset[int] = field(default_factory=lambda: _RETRYABLE_STATUS_CODES)

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or not 1 <= self.max_attempts <= 20
        ):
            raise ValueError("max_attempts must be an integer between 1 and 20")
        for name, value in (
            ("initial_delay_seconds", self.initial_delay_seconds),
            ("max_backoff_seconds", self.max_backoff_seconds),
            ("max_retry_after_seconds", self.max_retry_after_seconds),
        ):
            if not _finite_number(value) or float(value) < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if not self.retryable_status_codes or any(
            isinstance(value, bool) or not isinstance(value, int) or not 100 <= value <= 599
            for value in self.retryable_status_codes
        ):
            raise ValueError("retryable_status_codes must contain valid HTTP status integers")


@dataclass(frozen=True, slots=True)
class OpenReviewClientConfig:
    """Network and resource bounds for one OpenReview client."""

    base_url: str = DEFAULT_OPENREVIEW_API_V2_URL
    page_size: int = 1000
    max_pages: int = 1000
    max_records: int = 100_000
    max_total_requests: int = 100_000
    timeout_seconds: float = 30.0
    max_response_bytes: int = 16 * 1024 * 1024
    requests_per_second: float = 4.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    def __post_init__(self) -> None:
        _validate_base_url(self.base_url)
        for name, integer_value, maximum in (
            ("page_size", self.page_size, 1000),
            ("max_pages", self.max_pages, 100_000),
            ("max_records", self.max_records, 100_000),
            ("max_total_requests", self.max_total_requests, 1_000_000),
            ("max_response_bytes", self.max_response_bytes, _MAX_HTTP_RESPONSE_BYTES),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or not 1 <= integer_value <= maximum
            ):
                raise ValueError(f"{name} must be an integer between 1 and {maximum}")
        for name, numeric_value in (
            ("timeout_seconds", self.timeout_seconds),
            ("requests_per_second", self.requests_per_second),
        ):
            if not _finite_number(numeric_value) or float(numeric_value) <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if not isinstance(self.retry, RetryPolicy):
            raise TypeError("retry must be a RetryPolicy")


def _validate_base_url(base_url: str) -> None:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty HTTPS URL")
    parsed = urlsplit(base_url)
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("base_url must contain a valid port") from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("base_url must be an origin-only HTTPS URL without credentials")


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if "\r" in value or "\n" in value:
        raise ValueError(f"{field_name} must not contain newlines")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value


def _required_profile_id(value: object) -> str:
    reviewer_id = _required_text(value, "reviewer_id")
    if (
        not reviewer_id.startswith("~")
        or len(reviewer_id) < 2
        or "@" in reviewer_id
        or any(character.isspace() for character in reviewer_id)
    ):
        raise OpenReviewProtocolError("expertise requires exact tilde reviewer profile IDs")
    return reviewer_id


def _strict_object(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise OpenReviewProtocolError(f"OpenReview {context} must be a JSON object")
    return value


def _strict_array(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise OpenReviewProtocolError(f"OpenReview {context} must be a JSON array")
    return value


class OpenReviewClient:
    """Read-only API v2 client with cursor pagination and finite retries."""

    def __init__(
        self,
        *,
        config: OpenReviewClientConfig | None = None,
        token: str | None = None,
        transport: HttpTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config or OpenReviewClientConfig()
        if token is not None:
            _required_text(token, "token")
        self._token = token
        self._transport = transport or UrllibTransport()
        self._sleeper = sleeper
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._last_request_at: float | None = None
        self._request_count = 0

    @property
    def base_url(self) -> str:
        """Return the sanitized API origin used by this client."""

        return self.config.base_url.rstrip("/")

    def _headers(self) -> Mapping[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "PeerMatchLab/0.4 (read-only OpenReview sync)",
        }
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        return MappingProxyType(headers)

    def _throttle(self) -> None:
        now = self._monotonic()
        if self._last_request_at is not None:
            minimum_interval = 1.0 / self.config.requests_per_second
            remaining = minimum_interval - (now - self._last_request_at)
            if remaining > 0:
                self._sleeper(remaining)
                now = self._monotonic()
        self._last_request_at = now

    def _retry_after_seconds(self, headers: Mapping[str, str]) -> float | None:
        raw = headers.get("retry-after")
        if raw is None:
            return None
        try:
            delay = float(raw)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(raw)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                delay = parsed.timestamp() - self._wall_clock()
            except (TypeError, ValueError, OverflowError, OSError):
                return None
        if not math.isfinite(delay) or delay < 0:
            return None
        return min(delay, self.config.retry.max_retry_after_seconds)

    def _retry_delay(self, response: HttpResponse | None, failure_index: int) -> float:
        if response is not None:
            directed = self._retry_after_seconds(response.headers)
            if directed is not None:
                return directed
        return float(
            min(
                self.config.retry.initial_delay_seconds * (2**failure_index),
                self.config.retry.max_backoff_seconds,
            )
        )

    def _get_json(self, path: str, params: Mapping[str, str | int | bool]) -> object:
        query = urlencode(params)
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        last_transport_error: OpenReviewTransportError | None = None
        for attempt in range(self.config.retry.max_attempts):
            if self._request_count >= self.config.max_total_requests:
                raise OpenReviewProtocolError(
                    f"OpenReview request count exceeds max_total_requests "
                    f"{self.config.max_total_requests}"
                )
            self._request_count += 1
            self._throttle()
            response: HttpResponse | None = None
            try:
                response = self._transport.get(
                    url,
                    headers=self._headers(),
                    timeout_seconds=float(self.config.timeout_seconds),
                    max_response_bytes=self.config.max_response_bytes,
                )
            except OpenReviewTransportError as error:
                last_transport_error = error
            if response is not None and 200 <= response.status_code < 300:
                parsed_value: object = None
                parse_error: str | None = None
                try:
                    text = response.body.decode("utf-8")
                    parsed_value = load_json_text(text)
                except (UnicodeDecodeError, json.JSONDecodeError, DataValidationError) as error:
                    parse_error = str(error)
                    text = ""
                if parse_error is None:
                    return parsed_value
                response = None
                raise OpenReviewProtocolError(
                    f"OpenReview GET {path} returned invalid strict JSON: {parse_error}"
                ) from None
            attempts = attempt + 1
            retryable = response is None or (
                response.status_code in self.config.retry.retryable_status_codes
            )
            if not retryable or attempts == self.config.retry.max_attempts:
                if response is not None:
                    raise OpenReviewHttpError(response.status_code, path, attempts)
                if last_transport_error is None:
                    raise OpenReviewTransportError(
                        f"OpenReview GET {path} failed without a response"
                    )
                raise OpenReviewTransportError(
                    f"OpenReview GET {path} failed after {attempts} attempt(s)"
                ) from None
            self._sleeper(self._retry_delay(response, attempt))
        raise OpenReviewTransportError("finite OpenReview retry loop terminated unexpectedly")

    def get_all_notes(
        self,
        *,
        invitation: str | None = None,
        venue_id: str | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Fetch every note matching exactly one stable paper filter.

        Results are cursor-paginated by ``id``. The first page count is treated
        as a completeness contract and duplicate IDs or truncated pages fail
        closed.
        """

        if (invitation is None) == (venue_id is None):
            raise ValueError("exactly one of invitation or venue_id is required")
        params: dict[str, str | int | bool] = {
            "limit": self.config.page_size,
            "sort": "id",
        }
        if invitation is not None:
            params["invitation"] = _required_text(invitation, "invitation")
        else:
            if venue_id is None:
                raise ValueError("venue_id is required when invitation is absent")
            params["content.venueid"] = _required_text(venue_id, "venue_id")

        return self._all_notes(params)

    def get_all_author_notes(
        self, reviewer_id: str, *, max_scanned_notes: int | None = None
    ) -> tuple[Mapping[str, Any], ...]:
        """Fetch all Notes explicitly indexed under one reviewer author ID."""

        return tuple(self.iter_author_notes(reviewer_id, max_scanned_notes=max_scanned_notes))

    def iter_author_notes(
        self, reviewer_id: str, *, max_scanned_notes: int | None = None
    ) -> Iterable[Mapping[str, Any]]:
        """Yield validated author Notes by page without retaining all raw Notes."""

        return self._iter_notes(
            {
                "limit": self.config.page_size,
                "sort": "id",
                "content.authorids": _required_text(reviewer_id, "reviewer_id"),
            },
            max_records=max_scanned_notes,
        )

    def _all_notes(
        self, params: Mapping[str, str | int | bool], *, max_records: int | None = None
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._iter_notes(params, max_records=max_records))

    def _iter_notes(
        self, params: Mapping[str, str | int | bool], *, max_records: int | None = None
    ) -> Iterable[Mapping[str, Any]]:
        if max_records is not None and (
            isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 0
        ):
            raise ValueError("max_records must be a non-negative integer")
        record_limit = (
            min(self.config.max_records, max_records)
            if max_records is not None
            else self.config.max_records
        )
        limit_name = (
            "max_scanned_notes"
            if max_records is not None and max_records < self.config.max_records
            else "max_records"
        )
        page_size = min(self.config.page_size, max(1, record_limit))
        received = 0
        seen: set[str] = set()
        previous_id: str | None = None
        expected_count: int | None = None
        after: str | None = None
        for page_number in range(1, self.config.max_pages + 1):
            page_params = dict(params)
            request_limit = min(page_size, max(1, record_limit - received))
            page_params["limit"] = request_limit
            if after is None:
                page_params["count"] = True
            else:
                page_params["after"] = after
            payload = _strict_object(self._get_json("/notes", page_params), "note response")
            rows = _strict_array(payload.get("notes"), "notes")
            if expected_count is None:
                count = payload.get("count")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise OpenReviewProtocolError(
                        "OpenReview first note page must contain a non-negative integer count"
                    )
                if count > record_limit:
                    raise OpenReviewProtocolError(
                        f"OpenReview note count {count} exceeds {limit_name} {record_limit}"
                    )
                expected_count = count
            if len(rows) > request_limit:
                raise OpenReviewProtocolError("OpenReview notes page exceeds the requested limit")

            for index, row in enumerate(rows):
                note = _strict_object(row, f"note {index} on page {page_number}")
                note_id = note.get("id")
                if (
                    not isinstance(note_id, str)
                    or not note_id.strip()
                    or note_id != note_id.strip()
                    or "\r" in note_id
                    or "\n" in note_id
                ):
                    raise OpenReviewProtocolError("every OpenReview note must have a non-empty id")
                if note_id in seen:
                    raise OpenReviewProtocolError(
                        f"OpenReview pagination returned duplicate note id: {note_id}"
                    )
                if previous_id is not None and note_id <= previous_id:
                    raise OpenReviewProtocolError(
                        "OpenReview notes are not strictly increasing under sort=id"
                    )
                seen.add(note_id)
                received += 1
                previous_id = note_id
                yield note
            if received > record_limit:
                raise OpenReviewProtocolError(
                    f"OpenReview result exceeds {limit_name} {record_limit}"
                )
            if expected_count is None:
                raise OpenReviewProtocolError("OpenReview note count was not initialized")
            if received > expected_count:
                raise OpenReviewProtocolError("OpenReview returned more notes than its count")
            if received == expected_count:
                return
            if not rows or len(rows) < request_limit:
                raise OpenReviewProtocolError(
                    f"OpenReview note pagination ended at {received} of {expected_count} records"
                )
            if previous_id is None:
                raise OpenReviewProtocolError("OpenReview note cursor must be a string")
            after = previous_id
        raise OpenReviewProtocolError(
            f"OpenReview note pagination exceeds max_pages {self.config.max_pages}"
        )

    def get_group_members(self, group_id: str) -> tuple[str, ...]:
        """Fetch one exact OpenReview group and validate its direct members."""

        normalized_id = _required_text(group_id, "group_id")
        payload = _strict_object(
            self._get_json("/groups", {"id": normalized_id, "limit": 2}),
            "group response",
        )
        groups = _strict_array(payload.get("groups"), "groups")
        if len(groups) != 1:
            raise OpenReviewProtocolError(
                f"OpenReview group lookup for {normalized_id!r} returned {len(groups)} groups"
            )
        group = _strict_object(groups[0], "group")
        if group.get("id") != normalized_id:
            raise OpenReviewProtocolError("OpenReview group response id does not match the request")
        members = _strict_array(group.get("members"), "group members")
        if any(
            not isinstance(member, str)
            or not member.strip()
            or member != member.strip()
            or "\r" in member
            or "\n" in member
            for member in members
        ):
            raise OpenReviewProtocolError(
                "OpenReview group members must be non-empty string identifiers"
            )
        reviewer_ids = tuple(member for member in members if isinstance(member, str))
        if len(reviewer_ids) > self.config.max_records:
            raise OpenReviewProtocolError("OpenReview reviewer group exceeds max_records")
        if len(reviewer_ids) != len(set(reviewer_ids)):
            raise OpenReviewProtocolError("OpenReview group contains duplicate members")
        if not reviewer_ids:
            raise OpenReviewProtocolError("OpenReview reviewer group contains no members")
        return reviewer_ids

    def get_profile(self, reviewer_id: str) -> Mapping[str, Any]:
        """Fetch exactly one profile, without a broad name/email search."""

        requested = _required_text(reviewer_id, "reviewer_id")
        payload = _strict_object(
            self._get_json("/profiles", {"id": requested, "limit": 2}),
            "profile response",
        )
        rows = _strict_array(payload.get("profiles"), "profiles")
        if len(rows) != 1:
            raise OpenReviewProtocolError(
                f"OpenReview profile lookup for {requested!r} returned {len(rows)} profiles"
            )
        profile = _strict_object(rows[0], "profile")
        if profile.get("id") != requested:
            raise OpenReviewProtocolError("OpenReview profile id does not match reviewer id")
        _strict_object(profile.get("content"), "profile content")
        return profile


@dataclass(frozen=True, slots=True)
class ExpertiseFetchPolicy:
    """Explicit, bounded selection of authored publication Notes."""

    invitations: tuple[str, ...]
    minimum_date_ms: int | None = None
    maximum_date_ms: int | None = None
    require_abstract: bool = False
    max_scanned_notes: int = 100_000
    max_publications: int = 100_000
    max_publications_per_reviewer: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.invitations, (list, tuple)) or not 1 <= len(self.invitations) <= 100:
            raise ValueError("publication invitations must contain 1 to 100 exact IDs")
        invitations = tuple(self.invitations)
        for invitation in invitations:
            _required_text(invitation, "publication invitation")
        if len(set(invitations)) != len(invitations):
            raise ValueError("publication invitations must be unique")
        object.__setattr__(self, "invitations", invitations)
        for name in ("minimum_date_ms", "maximum_date_ms"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 9_999_999_999_999
            ):
                raise ValueError(f"{name} must be a non-negative Unix millisecond date")
        if (
            self.minimum_date_ms is not None
            and self.maximum_date_ms is not None
            and self.minimum_date_ms > self.maximum_date_ms
        ):
            raise ValueError("minimum_date_ms must not exceed maximum_date_ms")
        if not isinstance(self.require_abstract, bool):
            raise ValueError("require_abstract must be a boolean")
        for name in ("max_scanned_notes", "max_publications", "max_publications_per_reviewer"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 10_000_000
            ):
                raise ValueError(f"{name} must be an integer between 1 and 10000000")

    def record(self) -> Mapping[str, object]:
        return {
            "invitations": list(self.invitations),
            "minimum_date_ms": self.minimum_date_ms,
            "maximum_date_ms": self.maximum_date_ms,
            "require_abstract": self.require_abstract,
            "max_scanned_notes": self.max_scanned_notes,
            "max_publications": self.max_publications,
            "max_publications_per_reviewer": self.max_publications_per_reviewer,
        }


def _content_value(content: Mapping[str, Any], field_name: str) -> object:
    value = content.get(field_name)
    if isinstance(value, Mapping):
        if "value" not in value:
            raise OpenReviewProtocolError(f"OpenReview {field_name} wrapper has no value")
        return value["value"]
    return value


def _profile_projection(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep only expertise fields; do not persist email or affiliation data."""

    content = _strict_object(profile.get("content"), "profile content")
    names = _strict_array(_content_value(content, "names"), "profile names")
    projected_names: list[dict[str, object]] = []
    for row in names:
        name = _strict_object(row, "profile name")
        fullname = name.get("fullname")
        if fullname is None or (isinstance(fullname, str) and not fullname.strip()):
            parts = [name.get(field_name) for field_name in ("first", "middle", "last")]
            if any(part is not None and not isinstance(part, str) for part in parts):
                raise OpenReviewProtocolError("OpenReview profile name parts must be strings")
            fullname = " ".join(
                part.strip() for part in parts if isinstance(part, str) and part.strip()
            )
        if not isinstance(fullname, str) or not fullname.strip():
            raise OpenReviewProtocolError("OpenReview profile name requires fullname or name parts")
        preferred = name.get("preferred", False)
        if not isinstance(preferred, bool):
            raise OpenReviewProtocolError("OpenReview profile preferred flag must be boolean")
        projected_names.append({"fullname": fullname, "preferred": preferred})
    if not projected_names:
        raise OpenReviewProtocolError("OpenReview profile has no names")
    projected: dict[str, object] = {"names": projected_names}
    for field_name in ("bio", "research_interests", "keywords"):
        value = _content_value(content, field_name)
        if value is None:
            continue
        if field_name == "bio":
            if not isinstance(value, str):
                raise OpenReviewProtocolError("OpenReview profile bio must be a string")
            projected[field_name] = value
        else:
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise OpenReviewProtocolError(
                    f"OpenReview profile {field_name} must be string list"
                )
            projected[field_name] = value
    expertise = _content_value(content, "expertise")
    if expertise is not None:
        entries = _strict_array(expertise, "profile expertise")
        keywords: list[str] = []
        for entry in entries:
            if isinstance(entry, str):
                keywords.append(entry)
            else:
                item = _strict_object(entry, "profile expertise entry")
                raw_keywords = item.get("keywords")
                if raw_keywords is None:
                    continue
                if not isinstance(raw_keywords, list) or any(
                    not isinstance(keyword, str) for keyword in raw_keywords
                ):
                    raise OpenReviewProtocolError("OpenReview expertise keywords must be strings")
                keywords.extend(raw_keywords)
        projected["expertise"] = list(dict.fromkeys(keywords))
    return {"id": profile["id"], "content": projected}


def _publication_date(note: Mapping[str, Any]) -> int | None:
    for field_name in ("pdate", "odate", "cdate"):
        date = note.get(field_name)
        if date is None:
            continue
        if (
            isinstance(date, bool)
            or not isinstance(date, int)
            or not 0 <= date <= 9_999_999_999_999
        ):
            raise OpenReviewProtocolError(f"OpenReview publication {field_name} is invalid")
        return date
    return None


def _publication_projection(
    note: Mapping[str, Any], reviewer_id: str, policy: ExpertiseFetchPolicy
) -> tuple[Mapping[str, Any] | None, str]:
    content = _strict_object(note.get("content"), "publication content")
    authorids = _content_value(content, "authorids")
    if not isinstance(authorids, list) or any(not isinstance(item, str) for item in authorids):
        raise OpenReviewProtocolError("OpenReview publication authorids must be a string list")
    if reviewer_id not in authorids:
        raise OpenReviewProtocolError(
            "OpenReview author query returned a note without the reviewer"
        )
    invitations = note.get("invitations")
    if invitations is None and "invitation" in note:
        invitations = [note["invitation"]]
    if not isinstance(invitations, list) or any(not isinstance(item, str) for item in invitations):
        raise OpenReviewProtocolError("OpenReview publication invitations must be a string list")
    if not any(invitation in policy.invitations for invitation in invitations):
        return None, "invitation"
    date = _publication_date(note)
    if (policy.minimum_date_ms is not None and (date is None or date < policy.minimum_date_ms)) or (
        policy.maximum_date_ms is not None and (date is None or date > policy.maximum_date_ms)
    ):
        return None, "date"
    title = _content_value(content, "title")
    abstract = _content_value(content, "abstract")
    if title is not None and not isinstance(title, str):
        raise OpenReviewProtocolError("OpenReview publication title must be a string")
    if abstract is not None and not isinstance(abstract, str):
        raise OpenReviewProtocolError("OpenReview publication abstract must be a string")
    if (
        not title
        or not title.strip()
        or (policy.require_abstract and (not abstract or not abstract.strip()))
    ):
        return None, "content"
    year = _content_value(content, "year")
    if year is not None and (
        isinstance(year, bool) or not isinstance(year, int) or not 1800 <= year <= 2200
    ):
        raise OpenReviewProtocolError("OpenReview publication year is invalid")
    if year is None and date is not None:
        try:
            year = datetime.fromtimestamp(date / 1000, tz=UTC).year
        except (OverflowError, OSError, ValueError) as error:
            raise OpenReviewProtocolError(
                "OpenReview publication date is outside supported years"
            ) from error
    projected_note = {
        "id": note["id"],
        "invitations": invitations,
        "content": {
            "title": {"value": title},
            "abstract": {"value": abstract or ""},
            "authorids": {"value": [reviewer_id]},
            "year": {"value": year},
        },
    }
    if date is not None:
        projected_note["pdate"] = date
    return projected_note, "retained"


def _assert_minimized_profile(profile: Mapping[str, Any]) -> None:
    content = _strict_object(profile.get("content"), "profile evidence content")
    if set(profile) != {"id", "content"} or set(content) - {
        "names",
        "bio",
        "research_interests",
        "keywords",
        "expertise",
    }:
        raise OpenReviewProtocolError("profile evidence contains unrelated fields")
    _required_profile_id(profile.get("id"))
    names = content.get("names")
    if names is not None:
        if not isinstance(names, tuple):
            raise OpenReviewProtocolError("profile evidence names must be a projected array")
        for name in names:
            row = _strict_object(name, "profile evidence name")
            if set(row) - {"fullname", "preferred"}:
                raise OpenReviewProtocolError("profile evidence name contains unrelated fields")
            if not isinstance(row.get("fullname"), str) or not isinstance(
                row.get("preferred", False), bool
            ):
                raise OpenReviewProtocolError("profile evidence name is invalid")
    bio = content.get("bio")
    if bio is not None and not isinstance(bio, str):
        raise OpenReviewProtocolError("profile evidence bio must be a string")
    for field_name in ("research_interests", "keywords", "expertise"):
        value = content.get(field_name)
        if value is not None and (
            not isinstance(value, tuple) or any(not isinstance(item, str) for item in value)
        ):
            raise OpenReviewProtocolError(f"profile evidence {field_name} must be strings")


def _minimized_publication_value(content: Mapping[str, Any], field_name: str) -> object:
    value = content.get(field_name)
    if isinstance(value, Mapping):
        if set(value) != {"value"}:
            raise OpenReviewProtocolError(
                f"reviewer-publication {field_name} contains unrelated fields"
            )
        return value["value"]
    return value


def _assert_minimized_publication(joined: Mapping[str, Any]) -> None:
    if set(joined) != {"reviewer_id", "note"}:
        raise OpenReviewProtocolError("reviewer-publication contains unrelated fields")
    reviewer_id = _required_profile_id(joined.get("reviewer_id"))
    note = _strict_object(joined.get("note"), "reviewer-publication note")
    _required_text(note.get("id"), "reviewer-publication note id")
    content = _strict_object(note.get("content"), "reviewer-publication content")
    if set(note) - {"id", "invitations", "content", "pdate"} or set(content) - {
        "title",
        "abstract",
        "authorids",
        "year",
    }:
        raise OpenReviewProtocolError("reviewer-publication contains unrelated fields")
    invitations = note.get("invitations")
    if invitations is not None and (
        not isinstance(invitations, tuple)
        or any(not isinstance(invitation, str) for invitation in invitations)
    ):
        raise OpenReviewProtocolError("reviewer-publication invitations must be strings")
    date = note.get("pdate")
    if date is not None and (
        isinstance(date, bool) or not isinstance(date, int) or not 0 <= date <= 9_999_999_999_999
    ):
        raise OpenReviewProtocolError("reviewer-publication pdate must be an integer")
    for field_name in ("title", "abstract"):
        value = _minimized_publication_value(content, field_name)
        if value is not None and not isinstance(value, str):
            raise OpenReviewProtocolError(f"reviewer-publication {field_name} must be a string")
    year = _minimized_publication_value(content, "year")
    if year is not None and (
        isinstance(year, bool) or not isinstance(year, int) or not 1800 <= year <= 2200
    ):
        raise OpenReviewProtocolError("reviewer-publication year must be an integer")
    authorids = _minimized_publication_value(content, "authorids")
    if authorids != (reviewer_id,):
        raise OpenReviewProtocolError("reviewer-publication does not prove author identity")


def _assert_retained_publication(joined: Mapping[str, Any], policy: ExpertiseFetchPolicy) -> None:
    _assert_minimized_publication(joined)
    note = _strict_object(joined["note"], "reviewer-publication note")
    raw_note = _thaw_json(note)
    if not isinstance(raw_note, Mapping):
        raise OpenReviewProtocolError("reviewer-publication note is not an object")
    projected, reason = _publication_projection(raw_note, joined["reviewer_id"], policy)
    if projected is None:
        raise OpenReviewProtocolError(f"retained publication violates {reason} policy")
    if projected != raw_note:
        raise OpenReviewProtocolError("retained publication is not a canonical projection")


@dataclass(frozen=True, slots=True)
class OpenReviewExpertiseEvidence:
    """Privacy-minimized and explicitly joined offline expertise evidence."""

    profiles: tuple[Mapping[str, Any], ...]
    reviewer_publications: tuple[Mapping[str, Any], ...]
    policy: ExpertiseFetchPolicy
    filter_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.policy, ExpertiseFetchPolicy):
            raise ValueError("expertise policy must be ExpertiseFetchPolicy")
        budget = _SnapshotJsonBudget(
            max_items=_MAX_SNAPSHOT_EXPANDED_ITEMS, max_utf8_bytes=_MAX_SNAPSHOT_UTF8_BYTES
        )
        profiles = _bounded_snapshot_values(self.profiles, _MAX_SNAPSHOT_RECORDS, "profiles")
        publications = _bounded_snapshot_values(
            self.reviewer_publications, self.policy.max_publications, "reviewer publications"
        )
        frozen_profiles = tuple(_snapshot_json(row, budget=budget) for row in profiles)
        profile_ids: set[str] = set()
        for row in frozen_profiles:
            profile = _strict_object(row, "profile evidence")
            profile_id = _required_profile_id(profile.get("id"))
            _assert_minimized_profile(profile)
            if profile_id in profile_ids:
                raise OpenReviewProtocolError("expertise profiles contain duplicate IDs")
            profile_ids.add(profile_id)
        object.__setattr__(self, "profiles", frozen_profiles)
        frozen_publications = tuple(_snapshot_json(row, budget=budget) for row in publications)
        pairs: set[tuple[str, str]] = set()
        publications_per_reviewer: dict[str, int] = {}
        for row in frozen_publications:
            joined = _strict_object(row, "reviewer-publication evidence")
            if set(joined) != {"reviewer_id", "note"}:
                raise OpenReviewProtocolError(
                    "reviewer-publication evidence must contain exactly reviewer_id and note"
                )
            reviewer_id = _required_profile_id(joined["reviewer_id"])
            if reviewer_id not in profile_ids:
                raise OpenReviewProtocolError("reviewer-publication references an unknown profile")
            note = _strict_object(joined["note"], "reviewer-publication note")
            note_id = _required_text(note.get("id"), "reviewer-publication note id")
            _assert_retained_publication(joined, self.policy)
            pair = (reviewer_id, note_id)
            if pair in pairs:
                raise OpenReviewProtocolError("duplicate reviewer-publication association")
            pairs.add(pair)
            publications_per_reviewer[reviewer_id] = (
                publications_per_reviewer.get(reviewer_id, 0) + 1
            )
            if publications_per_reviewer[reviewer_id] > self.policy.max_publications_per_reviewer:
                raise OpenReviewProtocolError("reviewer exceeds max_publications_per_reviewer")
        object.__setattr__(
            self,
            "reviewer_publications",
            frozen_publications,
        )
        if not isinstance(self.filter_counts, Mapping):
            raise ValueError("expertise filter counts must be a mapping")
        counts = dict(self.filter_counts)
        if set(counts) != {"scanned", "invitation", "date", "content", "retained"} or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts.values()
        ):
            raise ValueError("expertise filter counts are invalid")
        if (
            counts["scanned"]
            != sum(counts[name] for name in ("invitation", "date", "content", "retained"))
            or counts["retained"] != len(frozen_publications)
            or counts["scanned"] > self.policy.max_scanned_notes
        ):
            raise ValueError("expertise filter counts do not match publications")
        object.__setattr__(self, "filter_counts", MappingProxyType(counts))

    def assert_minimized(self) -> None:
        """Recheck the public writer boundary before publishing evidence."""

        if not isinstance(self.policy, ExpertiseFetchPolicy):
            raise OpenReviewProtocolError("expertise evidence has no valid selection policy")
        if len(self.reviewer_publications) > self.policy.max_publications:
            raise OpenReviewProtocolError("expertise evidence exceeds max_publications")
        if not isinstance(self.filter_counts, Mapping):
            raise OpenReviewProtocolError("expertise filter counts are invalid")
        counts = dict(self.filter_counts)
        if (
            set(counts) != {"scanned", "invitation", "date", "content", "retained"}
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts.values()
            )
            or counts["scanned"]
            != sum(counts[name] for name in ("invitation", "date", "content", "retained"))
            or counts["retained"] != len(self.reviewer_publications)
            or counts["scanned"] > self.policy.max_scanned_notes
        ):
            raise OpenReviewProtocolError("expertise filter counts do not match publications")
        publications_per_reviewer: dict[str, int] = {}
        for row in self.profiles:
            profile = _strict_object(row, "profile evidence")
            _assert_minimized_profile(profile)
        for row in self.reviewer_publications:
            joined = _strict_object(row, "reviewer-publication evidence")
            _assert_retained_publication(joined, self.policy)
            reviewer_id = joined["reviewer_id"]
            publications_per_reviewer[reviewer_id] = (
                publications_per_reviewer.get(reviewer_id, 0) + 1
            )
            if publications_per_reviewer[reviewer_id] > self.policy.max_publications_per_reviewer:
                raise OpenReviewProtocolError("reviewer exceeds max_publications_per_reviewer")


def _account_projected_jsonl_row(row: Mapping[str, Any], used_bytes: int, label: str) -> int:
    """Apply the downstream loader's UTF-8 file limit before retaining a row."""

    try:
        row_bytes = (
            len(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            )
            + 1
        )
    except (TypeError, ValueError, UnicodeError) as error:
        raise OpenReviewProtocolError(f"{label} cannot be serialized as JSONL") from error
    if row_bytes > _MAX_EXPERTISE_JSONL_FILE_BYTES - used_bytes:
        raise OpenReviewProtocolError(f"{label} exceeds max_input_file_bytes")
    return used_bytes + row_bytes


def fetch_reviewer_expertise(
    client: OpenReviewClient,
    snapshot: OpenReviewSnapshot,
    policy: ExpertiseFetchPolicy,
) -> OpenReviewExpertiseEvidence:
    """Fetch bounded profiles and Notes with a verified author-ID join."""

    profiles: list[Mapping[str, Any]] = []
    rows: list[Mapping[str, Any]] = []
    profile_bytes = 0
    publication_bytes = 0
    counts = {"scanned": 0, "invitation": 0, "date": 0, "content": 0, "retained": 0}
    for reviewer_id in snapshot.reviewer_ids:
        _required_profile_id(reviewer_id)
        profile = _profile_projection(client.get_profile(reviewer_id))
        profile_bytes = _account_projected_jsonl_row(profile, profile_bytes, "profiles.jsonl")
        profiles.append(profile)
        retained_for_reviewer = 0
        for note in client.iter_author_notes(
            reviewer_id, max_scanned_notes=policy.max_scanned_notes - counts["scanned"]
        ):
            counts["scanned"] += 1
            if counts["scanned"] > policy.max_scanned_notes:
                raise OpenReviewProtocolError("publication scan exceeds max_scanned_notes")
            projected, reason = _publication_projection(note, reviewer_id, policy)
            counts[reason] += 1
            if projected is not None:
                retained_for_reviewer += 1
                if retained_for_reviewer > policy.max_publications_per_reviewer:
                    raise OpenReviewProtocolError("reviewer exceeds max_publications_per_reviewer")
                if len(rows) >= policy.max_publications:
                    raise OpenReviewProtocolError("publication count exceeds max_publications")
                joined = {"reviewer_id": reviewer_id, "note": projected}
                publication_bytes = _account_projected_jsonl_row(
                    joined, publication_bytes, "reviewer-publications.jsonl"
                )
                rows.append(joined)
    return OpenReviewExpertiseEvidence(tuple(profiles), tuple(rows), policy, counts)


@dataclass(frozen=True, slots=True)
class OpenReviewSnapshot:
    """Validated source records for one matching-data snapshot."""

    notes: tuple[Mapping[str, Any], ...]
    reviewer_ids: tuple[str, ...]
    base_url: str
    paper_filter: Mapping[str, str]
    reviewer_group: str

    def __post_init__(self) -> None:
        budget = _SnapshotJsonBudget(
            max_items=_MAX_SNAPSHOT_EXPANDED_ITEMS,
            max_utf8_bytes=_MAX_SNAPSHOT_UTF8_BYTES,
        )
        notes = _bounded_snapshot_values(self.notes, _MAX_SNAPSHOT_RECORDS, "snapshot notes")
        reviewer_ids = _bounded_snapshot_values(
            self.reviewer_ids, _MAX_SNAPSHOT_RECORDS, "snapshot reviewer ids"
        )
        if not notes or not reviewer_ids:
            raise ValueError("OpenReview snapshots require notes and reviewers")
        _validate_base_url(self.base_url)
        budget.consume_text(self.base_url)
        _required_text(self.reviewer_group, "reviewer_group")
        budget.consume_text(self.reviewer_group)
        paper_filter = _snapshot_json(self.paper_filter, budget=budget)
        if (
            not isinstance(paper_filter, Mapping)
            or len(paper_filter) != 1
            or next(iter(paper_filter), None) not in {"invitation", "content.venueid"}
        ):
            raise ValueError("paper_filter must contain exactly one invitation or content.venueid")
        for key, value in paper_filter.items():
            if not isinstance(key, str):
                raise ValueError("paper_filter keys must be strings")
            _required_text(value, "paper_filter value")
        note_ids: list[str] = []
        frozen_notes: list[Mapping[str, Any]] = []
        for raw_note in notes:
            note = _snapshot_json(raw_note, budget=budget)
            if not isinstance(note, Mapping) or any(not isinstance(key, str) for key in note):
                raise ValueError("snapshot notes must be JSON objects with string keys")
            note_id = note.get("id")
            if not isinstance(note_id, str):
                raise ValueError("snapshot notes must have non-empty string ids")
            note_ids.append(_required_text(note_id, "snapshot note id"))
            frozen_notes.append(note)
        if len(note_ids) != len(set(note_ids)):
            raise ValueError("snapshot note ids must be unique")
        for reviewer_id in reviewer_ids:
            _required_text(reviewer_id, "snapshot reviewer id")
            budget.consume_text(reviewer_id)
        if len(reviewer_ids) != len(set(reviewer_ids)):
            raise ValueError("snapshot reviewer ids must be unique")
        object.__setattr__(self, "notes", tuple(frozen_notes))
        object.__setattr__(self, "reviewer_ids", reviewer_ids)
        object.__setattr__(self, "paper_filter", paper_filter)


def _bounded_snapshot_values(values: Iterable[Any], limit: int, label: str) -> tuple[Any, ...]:
    try:
        iterator = iter(values)
    except TypeError as error:
        raise ValueError(f"{label} must be iterable") from error
    result = tuple(islice(iterator, limit + 1))
    if len(result) > limit:
        raise ValueError(f"{label} exceeds the {limit}-record hard limit")
    return result


@dataclass(slots=True)
class _SnapshotJsonBudget:
    max_items: int
    max_utf8_bytes: int
    used_items: int = 0
    used_utf8_bytes: int = 0
    cache: dict[int, tuple[object, Any, int, int]] = field(default_factory=dict)
    active: set[int] = field(default_factory=set)

    def consume_items(self, amount: int) -> None:
        if amount > self.max_items - self.used_items:
            raise ValueError("snapshot JSON exceeds the aggregate expanded-item budget")
        self.used_items += amount

    def consume_text(self, value: str) -> None:
        remaining = self.max_utf8_bytes - self.used_utf8_bytes
        if len(value) > remaining:
            raise ValueError("snapshot JSON exceeds the aggregate UTF-8 byte budget")
        size = 0
        try:
            for position in range(0, len(value), 64 * 1024):
                size += len(value[position : position + 64 * 1024].encode("utf-8"))
                if size > remaining:
                    raise ValueError("snapshot JSON exceeds the aggregate UTF-8 byte budget")
        except UnicodeEncodeError as error:
            raise ValueError("snapshot JSON strings must be valid Unicode") from error
        self.used_utf8_bytes += size

    def consume_cached(self, items: int, utf8_bytes: int) -> None:
        self.consume_items(items)
        if utf8_bytes > self.max_utf8_bytes - self.used_utf8_bytes:
            raise ValueError("snapshot JSON exceeds the aggregate UTF-8 byte budget")
        self.used_utf8_bytes += utf8_bytes


def _snapshot_json(
    value: object,
    *,
    budget: _SnapshotJsonBudget | None = None,
    depth: int = 0,
) -> Any:
    """Copy one JSON-like value into immutable built-in containers exactly once."""

    active_budget = budget or _SnapshotJsonBudget(
        max_items=_MAX_SNAPSHOT_EXPANDED_ITEMS,
        max_utf8_bytes=_MAX_SNAPSHOT_UTF8_BYTES,
    )
    if depth > _MAX_SNAPSHOT_DEPTH:
        raise ValueError("snapshot JSON nesting exceeds the supported depth")
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_budget.active:
            raise ValueError("snapshot JSON must not contain circular references")
        cached = active_budget.cache.get(identity)
        if cached is not None and cached[0] is value:
            active_budget.consume_cached(cached[2], cached[3])
            return cached[1]
        rows = _bounded_snapshot_values(
            value.items(), _MAX_SNAPSHOT_CONTAINER_ITEMS, "snapshot JSON object"
        )
        if any(type(key) is not str for key, _item in rows):
            raise ValueError("snapshot JSON object keys must be strings")
        if len({key for key, _item in rows}) != len(rows):
            raise ValueError("snapshot JSON object keys must be unique")
        start_items = active_budget.used_items
        start_bytes = active_budget.used_utf8_bytes
        active_budget.consume_items(len(rows))
        for key, _item in rows:
            active_budget.consume_text(key)
        active_budget.active.add(identity)
        try:
            frozen_mapping = MappingProxyType(
                {
                    key: _snapshot_json(item, budget=active_budget, depth=depth + 1)
                    for key, item in rows
                }
            )
        finally:
            active_budget.active.remove(identity)
        active_budget.cache[identity] = (
            value,
            frozen_mapping,
            active_budget.used_items - start_items,
            active_budget.used_utf8_bytes - start_bytes,
        )
        return frozen_mapping
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active_budget.active:
            raise ValueError("snapshot JSON must not contain circular references")
        cached = active_budget.cache.get(identity)
        if cached is not None and cached[0] is value:
            active_budget.consume_cached(cached[2], cached[3])
            return cached[1]
        items = _bounded_snapshot_values(
            value, _MAX_SNAPSHOT_CONTAINER_ITEMS, "snapshot JSON array"
        )
        start_items = active_budget.used_items
        start_bytes = active_budget.used_utf8_bytes
        active_budget.consume_items(len(items))
        active_budget.active.add(identity)
        try:
            frozen_array = tuple(
                _snapshot_json(item, budget=active_budget, depth=depth + 1) for item in items
            )
        finally:
            active_budget.active.remove(identity)
        active_budget.cache[identity] = (
            value,
            frozen_array,
            active_budget.used_items - start_items,
            active_budget.used_utf8_bytes - start_bytes,
        )
        return frozen_array
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        try:
            integer_text = str(value)
        except ValueError as error:
            raise ValueError("snapshot JSON integer exceeds the supported size") from error
        active_budget.consume_text(integer_text)
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("snapshot JSON numbers must be finite")
        active_budget.consume_text(repr(value))
        return value
    if type(value) is str:
        active_budget.consume_text(value)
        return value
    raise ValueError("snapshot notes must contain JSON-compatible values")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def fetch_openreview_snapshot(
    client: OpenReviewClient,
    *,
    reviewer_group: str,
    invitation: str | None = None,
    venue_id: str | None = None,
) -> OpenReviewSnapshot:
    """Fetch the paper notes and reviewer membership for one venue snapshot."""

    notes = client.get_all_notes(invitation=invitation, venue_id=venue_id)
    members = client.get_group_members(reviewer_group)
    paper_filter = (
        {"invitation": invitation} if invitation is not None else {"content.venueid": venue_id}
    )
    return OpenReviewSnapshot(
        notes=notes,
        reviewer_ids=members,
        base_url=client.base_url,
        paper_filter={key: value for key, value in paper_filter.items() if value is not None},
        reviewer_group=reviewer_group,
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_openreview_snapshot(
    snapshot: OpenReviewSnapshot,
    directory: str | Path,
    *,
    reviewer_capacity: int,
    expertise: OpenReviewExpertiseEvidence | None = None,
) -> Mapping[str, Any]:
    """Atomically create a raw and converted matching-data snapshot directory.

    Existing destinations are refused so a previous evidence snapshot cannot be
    silently mixed with a new API response.
    """

    if expertise is not None:
        if not isinstance(expertise, OpenReviewExpertiseEvidence):
            raise TypeError("expertise must be OpenReviewExpertiseEvidence")
        expertise.assert_minimized()
    destination = Path(directory)
    if destination.exists():
        raise DataValidationError(f"snapshot destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name or 'openreview'}-", dir=destination.parent)
    )
    installed = False
    try:
        submissions_path = staging / "submissions.jsonl"
        with submissions_path.open("w", encoding="utf-8", newline="\n") as stream:
            for note in snapshot.notes:
                json.dump(
                    _thaw_json(note),
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                stream.write("\n")
        reviewer_path = staging / "reviewer-ids.txt"
        with reviewer_path.open("w", encoding="utf-8", newline="\n") as stream:
            for reviewer_id in snapshot.reviewer_ids:
                stream.write(f"{reviewer_id}\n")
        if expertise is not None:
            profile_ids = [profile.get("id") for profile in expertise.profiles]
            if profile_ids != list(snapshot.reviewer_ids):
                raise DataValidationError(
                    "expertise profiles must match reviewer group order exactly"
                )
            for name, rows in (
                ("profiles.jsonl", expertise.profiles),
                ("reviewer-publications.jsonl", expertise.reviewer_publications),
            ):
                with (staging / name).open("w", encoding="utf-8", newline="\n") as stream:
                    for row in rows:
                        json.dump(
                            _thaw_json(row),
                            stream,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        stream.write("\n")
            from peermatchlab.expertise import ExpertiseConfig
            from peermatchlab.expertise_io import load_openreview_expertise_snapshot

            publication_counts: dict[str, int] = {}
            for row in expertise.reviewer_publications:
                reviewer_id = row["reviewer_id"]
                publication_counts[reviewer_id] = publication_counts.get(reviewer_id, 0) + 1
            validation_config = ExpertiseConfig(
                max_submissions=max(1, len(snapshot.notes)),
                max_reviewers=max(1, len(snapshot.reviewer_ids)),
                max_publications_per_reviewer=max(publication_counts.values(), default=1),
                max_total_publications=max(1, len(expertise.reviewer_publications)),
            )

            loaded = load_openreview_expertise_snapshot(
                staging, config=validation_config, reviewer_capacity=reviewer_capacity
            )
            if {item.id for item in loaded.experts} != set(snapshot.reviewer_ids):
                raise DataValidationError("expertise snapshot lost reviewer identities")
        documents = openreview_submissions_from_records(
            snapshot.notes, max_records=max(1, len(snapshot.notes))
        )
        experts = reviewer_ids_to_experts(
            snapshot.reviewer_ids,
            capacity=reviewer_capacity,
            max_reviewers=max(1, len(snapshot.reviewer_ids)),
        )
        write_json(
            staging / "documents.json",
            [
                {
                    "id": item.id,
                    "title": item.title,
                    "abstract": item.abstract,
                    "topics": list(item.topics),
                    "keywords": list(item.keywords),
                    "metadata": dict(item.metadata),
                }
                for item in documents
            ],
        )
        write_json(
            staging / "experts.json",
            [
                {
                    "id": item.id,
                    "name": item.name,
                    "capacity": item.capacity,
                    "metadata": dict(item.metadata),
                }
                for item in experts
            ],
        )
        file_names = [
            "submissions.jsonl",
            "reviewer-ids.txt",
            "documents.json",
            "experts.json",
        ]
        if expertise is not None:
            file_names.extend(("profiles.jsonl", "reviewer-publications.jsonl"))
        files = {
            name: {
                "bytes": (staging / name).stat().st_size,
                "sha256": _sha256(staging / name),
            }
            for name in file_names
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "adapter": "openreview-api-v2-snapshot",
            "source": {
                "base_url": snapshot.base_url,
                "paper_filter": dict(snapshot.paper_filter),
                "reviewer_group": snapshot.reviewer_group,
            },
            "conversion": {"reviewer_capacity": reviewer_capacity},
            "records": {"documents": len(documents), "experts": len(experts)},
            "files": files,
            "limitations": [
                "the snapshot contains direct group members only",
                "reviewer shells do not contain publications or inferred expertise",
                "authentication tokens are never written to disk",
                "venue authorization and handling of private data remain operator responsibilities",
            ],
        }
        if expertise is not None:
            manifest["schema_version"] = 2
            manifest["adapter"] = "openreview-api-v2-expertise-snapshot"
            manifest["acquisition"] = {
                "policy": dict(expertise.policy.record()),
                "filters": dict(expertise.filter_counts),
                "privacy": "profiles retain only names and expertise fields",
            }
            manifest["records"].update(
                profiles=len(expertise.profiles),
                reviewer_publications=len(expertise.reviewer_publications),
            )
            manifest["limitations"] = [
                "direct group members only; no nested group expansion",
                "only explicitly attributed, allowed-invitation publication Notes are retained",
                "venue authorization and private-data handling remain operator responsibilities",
                "authentication tokens are never written to disk",
            ]
        write_json(staging / "manifest.json", manifest)
        from peermatchlab.expertise_io import _install_directory_no_replace

        try:
            _install_directory_no_replace(staging, destination)
        except FileExistsError as error:
            raise DataValidationError(
                f"snapshot destination already exists: {destination}"
            ) from error
        installed = True
        return MappingProxyType(manifest)
    finally:
        if not installed:
            shutil.rmtree(staging, ignore_errors=True)
