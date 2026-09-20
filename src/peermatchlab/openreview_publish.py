"""Offline-first assignment edge plans and an injectable, fail-closed write boundary.

This module intentionally does not contain an OpenReview HTTP POST implementation.
An operator must supply a transport appropriate to an authorized venue. Tests use
only an in-memory transport; importing or preparing a plan never contacts a service.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Protocol, TypeVar

from peermatchlab.audit import audit_plan
from peermatchlab.models import Conflict, DataValidationError, Document, Expert, MatchPlan

_MAX_ASSIGNMENTS = 1_000
_MAX_BATCH_SIZE = 100
_MAX_PAPERS = 1_000
_MAX_REVIEWERS = 10_000
_MAX_CONFLICTS = 100_000
_Record = TypeVar("_Record")


def _exact_id(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or not value.isprintable()
        or any(character.isspace() for character in value)
        or len(value.encode("utf-8")) > 512
    ):
        raise DataValidationError(
            f"{name} must be an exact, printable identifier of at most 512 bytes"
        )
    return value


def _bounded_unique(values: Iterable[str], name: str, *, maximum: int) -> tuple[str, ...]:
    items = tuple(_exact_id(value, name) for value in islice(values, maximum + 1))
    if not items or len(items) > maximum or len(items) != len(set(items)):
        raise DataValidationError(f"{name} must contain 1 to {maximum} unique exact identifiers")
    return tuple(sorted(items))


def _bounded_records(values: Iterable[_Record], name: str, *, maximum: int) -> tuple[_Record, ...]:
    items = tuple(islice(values, maximum + 1))
    if len(items) > maximum:
        raise DataValidationError(f"{name} exceeds maximum of {maximum} records")
    return items


@dataclass(frozen=True, slots=True)
class PublishScope:
    """Venue identifiers that are bound into one immutable write plan."""

    assignment_invitation: str
    score_invitation: str
    reviewer_group: str
    label: str
    paper_ids: tuple[str, ...]
    reviewer_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("assignment_invitation", "score_invitation", "reviewer_group", "label"):
            _exact_id(getattr(self, name), name)
        if self.assignment_invitation == self.score_invitation:
            raise DataValidationError("assignment and score invitations must differ")
        object.__setattr__(
            self, "paper_ids", _bounded_unique(self.paper_ids, "paper_ids", maximum=_MAX_PAPERS)
        )
        object.__setattr__(
            self,
            "reviewer_ids",
            _bounded_unique(self.reviewer_ids, "reviewer_ids", maximum=_MAX_REVIEWERS),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "assignment_invitation": self.assignment_invitation,
            "score_invitation": self.score_invitation,
            "reviewer_group": self.reviewer_group,
            "label": self.label,
            "paper_ids": list(self.paper_ids),
            "reviewer_ids": list(self.reviewer_ids),
        }


@dataclass(frozen=True, slots=True)
class AssignmentEdge:
    """The exact content of an assignment or score edge in the scoped protocol."""

    invitation: str
    head: str
    tail: str
    label: str
    weight: float

    def __post_init__(self) -> None:
        for name in ("invitation", "head", "tail", "label"):
            _exact_id(getattr(self, name), name)
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise DataValidationError("edge weight must be a finite score in [0, 1]")
        if not 0 <= self.weight <= 1:
            raise DataValidationError("edge weight must be a finite score in [0, 1]")

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.invitation, self.head, self.tail, self.label)

    def as_dict(self) -> dict[str, object]:
        return {
            "invitation": self.invitation,
            "head": self.head,
            "tail": self.tail,
            "label": self.label,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class PublishPlan:
    """Canonical two-invitation edge set and SHA-256 of its exact scoped content."""

    scope: PublishScope
    edges: tuple[AssignmentEdge, ...]
    digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "peermatchlab-openreview-edge-plan-v1",
            "scope": self.scope.as_dict(),
            "edges": [edge.as_dict() for edge in self.edges],
            "sha256": self.digest,
        }


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def prepare_publish_plan(
    plan: MatchPlan,
    scope: PublishScope,
    *,
    documents: Iterable[Document],
    experts: Iterable[Expert],
    conflicts: Iterable[Conflict] = (),
    default_demand: int = 2,
    require_distinct_institutions: bool = False,
) -> PublishPlan:
    """Reject unsafe/incomplete plans before any external transport is consulted."""

    document_items = _bounded_records(documents, "documents", maximum=_MAX_PAPERS)
    expert_items = _bounded_records(experts, "experts", maximum=_MAX_REVIEWERS)
    conflict_items = _bounded_records(conflicts, "conflicts", maximum=_MAX_CONFLICTS)
    if set(scope.paper_ids) != {item.id for item in document_items}:
        raise DataValidationError("paper_ids must exactly match audited documents")
    if set(scope.reviewer_ids) != {item.id for item in expert_items}:
        raise DataValidationError("reviewer_ids must exactly match audited experts")
    if not plan.assignments or len(plan.assignments) > _MAX_ASSIGNMENTS:
        raise DataValidationError(f"publish plan requires 1 to {_MAX_ASSIGNMENTS} assignments")
    audit = audit_plan(
        plan,
        document_items,
        expert_items,
        conflict_items,
        default_demand=default_demand,
        require_distinct_institutions=require_distinct_institutions,
    )
    if not audit.safe or audit.demand_coverage != 1 or any(plan.unmet.values()):
        raise DataValidationError(
            "only independently audited, complete assignments can be published"
        )
    edges = tuple(
        sorted(
            (
                AssignmentEdge(
                    invitation, item.document_id, item.expert_id, scope.label, item.score
                )
                for item in plan.assignments
                for invitation in (scope.assignment_invitation, scope.score_invitation)
            ),
            key=lambda item: item.key,
        )
    )
    if len({edge.key for edge in edges}) != len(edges):
        raise DataValidationError("publish plan contains duplicate edges")
    body: dict[str, object] = {
        "schema": "peermatchlab-openreview-edge-plan-v1",
        "scope": scope.as_dict(),
        "edges": [edge.as_dict() for edge in edges],
    }
    return PublishPlan(scope, edges, hashlib.sha256(_canonical_bytes(body)).hexdigest())


@dataclass(frozen=True, slots=True)
class RemoteContext:
    """Exact invitation/group/paper evidence fetched by an injected transport."""

    assignment_invitation: str
    score_invitation: str
    reviewer_group: str
    paper_ids: tuple[str, ...]
    reviewer_ids: tuple[str, ...]


class AssignmentWriteTransport(Protocol):
    """Operator-provided boundary requiring atomic create-only edge writes.

    An adapter based on unconditional bulk upsert does not implement this
    contract: it could replace an edge inserted between read and write.
    """

    def inspect_context(self, scope: PublishScope) -> RemoteContext:
        """Read exact invitation IDs, group members, and paper IDs."""

    def read_edges(self, scope: PublishScope) -> tuple[AssignmentEdge, ...]:
        """Read *all* edges for both invitations, label, and paper IDs in scope."""

    def post_edges(self, edges: tuple[AssignmentEdge, ...]) -> None:
        """Atomically create each edge or conflict on an existing identity.

        Never overwrite an existing edge, even when its content is identical.
        A batch may partially succeed before a conflict/transport error.
        """


@dataclass(frozen=True, slots=True)
class PublishReceipt:
    """Auditable result; incomplete results are safe to reconcile and retry."""

    plan_sha256: str
    status: str
    expected_edges: int
    confirmed_edges: int
    attempted_batches: int
    failure: str | None = None

    def __post_init__(self) -> None:
        if len(self.plan_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.plan_sha256
        ):
            raise DataValidationError("receipt plan_sha256 must be a lowercase SHA-256 digest")
        if self.status not in {"dry_run", "incomplete", "complete"}:
            raise DataValidationError("receipt status is unknown")
        for name in ("expected_edges", "confirmed_edges", "attempted_batches"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DataValidationError(f"receipt {name} must be a non-negative integer")
        if self.expected_edges > 2 * _MAX_ASSIGNMENTS or self.confirmed_edges > self.expected_edges:
            raise DataValidationError("receipt edge counts exceed the bounded plan")
        if self.status == "complete" and self.confirmed_edges != self.expected_edges:
            raise DataValidationError("complete receipt must confirm every edge")
        if self.status == "dry_run" and (self.confirmed_edges or self.attempted_batches):
            raise DataValidationError("dry-run receipt cannot claim remote writes or confirmation")
        if (self.status == "incomplete") != (self.failure is not None):
            raise DataValidationError("only incomplete receipts carry a failure code")
        if self.failure is not None and self.failure not in {
            "unconfirmed_batch",
            "readback_unavailable",
        }:
            raise DataValidationError("receipt failure code is unknown")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "peermatchlab-openreview-publish-receipt-v1",
            "plan_sha256": self.plan_sha256,
            "status": self.status,
            "expected_edges": self.expected_edges,
            "confirmed_edges": self.confirmed_edges,
            "attempted_batches": self.attempted_batches,
            "failure": self.failure,
        }


def _verify_plan(plan: PublishPlan) -> None:
    body = plan.as_dict()
    body.pop("sha256")
    if hashlib.sha256(_canonical_bytes(body)).hexdigest() != plan.digest:
        raise DataValidationError("publish plan digest does not match its content")
    if not plan.edges or len(plan.edges) > 2 * _MAX_ASSIGNMENTS or len(plan.edges) % 2:
        raise DataValidationError("publish plan has an invalid edge count")
    invitations_by_pair: dict[tuple[str, str], set[str]] = {}
    weights: dict[tuple[str, str], float] = {}
    for edge in plan.edges:
        if (
            edge.invitation not in {plan.scope.assignment_invitation, plan.scope.score_invitation}
            or edge.head not in plan.scope.paper_ids
            or edge.tail not in plan.scope.reviewer_ids
            or edge.label != plan.scope.label
        ):
            raise DataValidationError("publish plan contains an edge outside its exact scope")
        pair = (edge.head, edge.tail)
        invitations_by_pair.setdefault(pair, set()).add(edge.invitation)
        if pair in weights and weights[pair] != edge.weight:
            raise DataValidationError("assignment and score edge weights must agree")
        weights[pair] = edge.weight
    if any(
        invitations != {plan.scope.assignment_invitation, plan.scope.score_invitation}
        for invitations in invitations_by_pair.values()
    ) or len({edge.key for edge in plan.edges}) != len(plan.edges):
        raise DataValidationError(
            "publish plan must contain exactly one edge per invitation and pair"
        )


def _remote_edges(plan: PublishPlan, transport: AssignmentWriteTransport) -> set[AssignmentEdge]:
    rows = transport.read_edges(plan.scope)
    if not isinstance(rows, tuple) or len(rows) > 2 * _MAX_ASSIGNMENTS:
        raise DataValidationError("remote edge response exceeds the bounded exact scope")
    if any(not isinstance(row, AssignmentEdge) for row in rows):
        raise DataValidationError("remote edge response contains a malformed edge")
    keys = [row.key for row in rows]
    if len(keys) != len(set(keys)):
        raise DataValidationError("remote edge response contains duplicate identities")
    expected = set(plan.edges)
    if any(row not in expected for row in rows):
        raise DataValidationError("remote edge differs from or exceeds the immutable plan")
    return set(rows)


def _verify_context(scope: PublishScope, context: RemoteContext) -> None:
    if not isinstance(context, RemoteContext):
        raise DataValidationError("remote context must contain exact venue evidence")
    papers = _bounded_unique(context.paper_ids, "remote paper_ids", maximum=1_000)
    reviewers = _bounded_unique(context.reviewer_ids, "remote reviewer_ids", maximum=10_000)
    if (
        context.assignment_invitation != scope.assignment_invitation
        or context.score_invitation != scope.score_invitation
        or context.reviewer_group != scope.reviewer_group
        or papers != scope.paper_ids
        or reviewers != scope.reviewer_ids
    ):
        raise DataValidationError("remote invitations, group membership, or papers changed")


def publish_assignment_plan(
    plan: PublishPlan,
    *,
    transport: AssignmentWriteTransport | None = None,
    publish: bool = False,
    confirm_sha256: str | None = None,
    batch_size: int = 50,
) -> PublishReceipt:
    """Dry-run by default; explicit writes reconcile every batch, never overwrite.

    An ambiguous POST returns ``incomplete`` after one read-back. A subsequent
    invocation reads the remote set afresh and sends only truly missing edges.
    Divergent edges and changed invitation/group/paper context always fail closed.
    This guarantee depends on the injected transport's atomic create-only
    operation; ordinary last-writer-wins bulk upsert is not a valid transport.
    """

    _verify_plan(plan)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= _MAX_BATCH_SIZE
    ):
        raise DataValidationError(f"batch_size must be an integer between 1 and {_MAX_BATCH_SIZE}")
    if not isinstance(publish, bool):
        raise DataValidationError("publish must be a boolean")
    if not publish:
        return PublishReceipt(plan.digest, "dry_run", len(plan.edges), 0, 0)
    if transport is None or confirm_sha256 != plan.digest:
        raise DataValidationError(
            "writing requires an injected transport and exact plan SHA-256 confirmation"
        )
    _verify_context(plan.scope, transport.inspect_context(plan.scope))
    current = _remote_edges(plan, transport)
    batches = 0
    while len(current) < len(plan.edges):
        previous = current
        _verify_context(plan.scope, transport.inspect_context(plan.scope))
        current = _remote_edges(plan, transport)
        if not previous.issubset(current):
            raise DataValidationError("remote edges disappeared during reconciliation")
        pending = tuple(edge for edge in plan.edges if edge not in current)
        if not pending:
            return PublishReceipt(plan.digest, "complete", len(plan.edges), len(current), batches)
        batch = pending[:batch_size]
        batches += 1
        failed = False
        try:
            transport.post_edges(batch)
        except Exception:  # An ambiguous partial write must be reconciled, not blindly retried.
            failed = True
        try:
            _verify_context(plan.scope, transport.inspect_context(plan.scope))
            current = _remote_edges(plan, transport)
        except Exception:
            return PublishReceipt(
                plan.digest,
                "incomplete",
                len(plan.edges),
                len(current),
                batches,
                "readback_unavailable",
            )
        if not previous.issubset(current):
            return PublishReceipt(
                plan.digest,
                "incomplete",
                len(plan.edges),
                len(current),
                batches,
                "unconfirmed_batch",
            )
        if failed or not set(batch).issubset(current):
            return PublishReceipt(
                plan.digest,
                "incomplete",
                len(plan.edges),
                len(current),
                batches,
                "unconfirmed_batch",
            )
    return PublishReceipt(plan.digest, "complete", len(plan.edges), len(current), batches)


def write_publish_artifact(plan: PublishPlan, receipt: PublishReceipt, path: str | Path) -> None:
    """Atomically link one evidence file to a destination that must not exist."""

    _verify_plan(plan)
    if receipt.plan_sha256 != plan.digest:
        raise DataValidationError("receipt does not belong to this plan")
    if receipt.expected_edges != len(plan.edges):
        raise DataValidationError("receipt expected edge count does not match the plan")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, object] = {
        "schema": "peermatchlab-openreview-publish-artifact-v1",
        "plan": plan.as_dict(),
        "receipt": receipt.as_dict(),
    }
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
    )
    staging = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(_canonical_bytes(body))
            output.flush()
            os.fsync(output.fileno())
        os.link(staging, destination)
    finally:
        if staging.exists():
            staging.unlink()
