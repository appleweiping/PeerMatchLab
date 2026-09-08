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
from datetime import UTC
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
    timeout_seconds: float = 30.0
    max_response_bytes: int = 16 * 1024 * 1024
    requests_per_second: float = 4.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    def __post_init__(self) -> None:
        _validate_base_url(self.base_url)
        for name, integer_value, maximum in (
            ("page_size", self.page_size, 1000),
            ("max_pages", self.max_pages, 100_000),
            ("max_records", self.max_records, 10_000_000),
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

        result: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        previous_id: str | None = None
        expected_count: int | None = None
        after: str | None = None
        for page_number in range(1, self.config.max_pages + 1):
            page_params = dict(params)
            if after is None:
                page_params["count"] = True
            else:
                page_params["after"] = after
            payload = _strict_object(self._get_json("/notes", page_params), "note response")
            rows = _strict_array(payload.get("notes"), "notes")
            if len(rows) > self.config.page_size:
                raise OpenReviewProtocolError("OpenReview notes page exceeds the requested limit")
            if expected_count is None:
                count = payload.get("count")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise OpenReviewProtocolError(
                        "OpenReview first note page must contain a non-negative integer count"
                    )
                if count > self.config.max_records:
                    raise OpenReviewProtocolError(
                        f"OpenReview note count {count} exceeds max_records "
                        f"{self.config.max_records}"
                    )
                expected_count = count

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
                result.append(note)
                previous_id = note_id
            if len(result) > self.config.max_records:
                raise OpenReviewProtocolError(
                    f"OpenReview result exceeds max_records {self.config.max_records}"
                )
            if expected_count is None:
                raise OpenReviewProtocolError("OpenReview note count was not initialized")
            if len(result) > expected_count:
                raise OpenReviewProtocolError("OpenReview returned more notes than its count")
            if len(result) == expected_count:
                return tuple(result)
            if not rows or len(rows) < self.config.page_size:
                raise OpenReviewProtocolError(
                    f"OpenReview note pagination ended at {len(result)} of {expected_count} records"
                )
            last_id = result[-1].get("id")
            if not isinstance(last_id, str):
                raise OpenReviewProtocolError("OpenReview note cursor must be a string")
            after = last_id
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
        if len(reviewer_ids) != len(set(reviewer_ids)):
            raise OpenReviewProtocolError("OpenReview group contains duplicate members")
        if not reviewer_ids:
            raise OpenReviewProtocolError("OpenReview reviewer group contains no members")
        return reviewer_ids


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
) -> Mapping[str, Any]:
    """Atomically create a raw and converted matching-data snapshot directory.

    Existing destinations are refused so a previous evidence snapshot cannot be
    silently mixed with a new API response.
    """

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
        files = {
            name: {
                "bytes": (staging / name).stat().st_size,
                "sha256": _sha256(staging / name),
            }
            for name in (
                "submissions.jsonl",
                "reviewer-ids.txt",
                "documents.json",
                "experts.json",
            )
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
