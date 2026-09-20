"""No test in this module contacts or writes to OpenReview."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from itertools import count
from pathlib import Path

import pytest

import peermatchlab.openreview_publish as publisher
from peermatchlab.models import (
    Assignment,
    Conflict,
    DataValidationError,
    Document,
    Expert,
    MatchPlan,
)
from peermatchlab.openreview_publish import (
    AssignmentEdge,
    PublishPlan,
    PublishScope,
    RemoteContext,
    prepare_publish_plan,
    publish_assignment_plan,
    write_publish_artifact,
)


def fixture() -> tuple[MatchPlan, PublishScope, tuple[Document, ...], tuple[Expert, ...]]:
    documents = (
        Document("paper-A", "A", required_experts=1),
        Document("paper-B", "B", required_experts=1),
    )
    experts = (Expert("~Alice1", "Alice", capacity=1), Expert("~Bob1", "Bob", capacity=1))
    plan = MatchPlan(
        assignments=(
            Assignment("paper-A", "~Alice1", 0.8, 1),
            Assignment("paper-B", "~Bob1", 0.6, 1),
        ),
        unmet={"paper-A": 0, "paper-B": 0},
        strategy="optimal",
        total_score=1.4,
    )
    scope = PublishScope(
        "Venue/-/Assignment",
        "Venue/-/Aggregate_Score",
        "Venue/Reviewers",
        "peer-matchlab-v1",
        ("paper-B", "paper-A"),
        ("~Bob1", "~Alice1"),
    )
    return plan, scope, documents, experts


def prepared() -> PublishPlan:
    plan, scope, documents, experts = fixture()
    return prepare_publish_plan(plan, scope, documents=documents, experts=experts, default_demand=1)


class MockTransport:
    def __init__(self, plan: PublishPlan) -> None:
        self.scope = plan.scope
        self.remote: dict[tuple[str, str, str, str], AssignmentEdge] = {}
        self.writes: list[tuple[AssignmentEdge, ...]] = []
        self.inspections = 0
        self.reads = 0
        self.fail_after: int | None = None
        self.stale = False

    def inspect_context(self, scope: PublishScope) -> RemoteContext:
        self.inspections += 1
        return RemoteContext(
            scope.assignment_invitation,
            scope.score_invitation,
            scope.reviewer_group,
            scope.paper_ids,
            scope.reviewer_ids,
        )

    def read_edges(self, scope: PublishScope) -> tuple[AssignmentEdge, ...]:
        self.reads += 1
        assert scope == self.scope
        return tuple(self.remote.values())

    def post_edges(self, edges: tuple[AssignmentEdge, ...]) -> None:
        self.writes.append(edges)
        if self.stale:
            return
        for index, edge in enumerate(edges):
            if self.fail_after is not None and index == self.fail_after:
                raise RuntimeError("simulated ambiguous transport failure")
            if edge.key in self.remote:
                raise RuntimeError("simulated conditional-insert conflict")
            self.remote[edge.key] = edge


def test_dry_run_never_calls_even_an_injected_transport() -> None:
    plan = prepared()
    transport = MockTransport(plan)
    result = publish_assignment_plan(plan, transport=transport)
    assert result.status == "dry_run"
    assert result.expected_edges == 4
    assert (transport.inspections, transport.reads, transport.writes) == (0, 0, [])
    assert result.as_dict()["plan_sha256"] == plan.digest
    assert publish_assignment_plan(plan).status == "dry_run"


def test_explicit_publish_reconciles_complete_remote_state_without_new_writes() -> None:
    plan = prepared()
    transport = MockTransport(plan)
    with pytest.raises(DataValidationError, match="exact plan SHA-256"):
        publish_assignment_plan(plan, transport=transport, publish=True)
    assert not transport.writes and transport.inspections == 0
    result = publish_assignment_plan(
        plan, transport=transport, publish=True, confirm_sha256=plan.digest, batch_size=2
    )
    assert (result.status, result.confirmed_edges, result.attempted_batches) == ("complete", 4, 2)
    assert all(len(batch) <= 2 for batch in transport.writes)
    assert set(transport.remote.values()) == set(plan.edges)
    writes = len(transport.writes)
    again = publish_assignment_plan(
        plan, transport=transport, publish=True, confirm_sha256=plan.digest
    )
    assert again.status == "complete" and again.attempted_batches == 0
    assert len(transport.writes) == writes


def rehashed(plan: PublishPlan) -> PublishPlan:
    body = plan.as_dict()
    body.pop("sha256")
    return replace(plan, digest=hashlib.sha256(publisher._canonical_bytes(body)).hexdigest())


def test_partial_transport_failure_returns_incomplete_and_retries_only_missing_edges() -> None:
    plan = prepared()
    transport = MockTransport(plan)
    transport.fail_after = 1
    first = publish_assignment_plan(
        plan, transport=transport, publish=True, confirm_sha256=plan.digest
    )
    assert (first.status, first.confirmed_edges, first.attempted_batches, first.failure) == (
        "incomplete",
        1,
        1,
        "unconfirmed_batch",
    )
    first_key = next(iter(transport.remote))
    transport.fail_after = None
    second = publish_assignment_plan(
        plan, transport=transport, publish=True, confirm_sha256=plan.digest, batch_size=2
    )
    assert second.status == "complete" and second.confirmed_edges == 4
    assert all(edge.key != first_key for batch in transport.writes[1:] for edge in batch)


def test_concurrent_divergent_insert_between_read_and_post_is_never_overwritten() -> None:
    plan = prepared()
    transport = MockTransport(plan)
    original_post = transport.post_edges
    first = plan.edges[0]
    divergent = replace(first, weight=0.11)

    def concurrent_insert(edges: tuple[AssignmentEdge, ...]) -> None:
        transport.remote[first.key] = divergent
        original_post(edges)

    transport.post_edges = concurrent_insert  # type: ignore[method-assign]
    result = publish_assignment_plan(
        plan, transport=transport, publish=True, confirm_sha256=plan.digest
    )
    assert result.status == "incomplete" and result.attempted_batches == 1
    assert transport.remote[first.key] == divergent
    assert len(transport.remote) == 1
    with pytest.raises(DataValidationError, match="remote edge differs"):
        publish_assignment_plan(plan, transport=transport, publish=True, confirm_sha256=plan.digest)
    assert transport.remote[first.key] == divergent


def test_unconfirmed_write_stops_after_one_batch() -> None:
    plan = prepared()
    transport = MockTransport(plan)
    transport.stale = True
    result = publish_assignment_plan(
        plan, transport=transport, publish=True, confirm_sha256=plan.digest, batch_size=1
    )
    assert result.status == "incomplete" and result.confirmed_edges == 0
    assert len(transport.writes) == 1


def test_readback_failure_reports_uncertainty_and_retry_reconciles() -> None:
    plan = prepared()
    transport = MockTransport(plan)
    original_read = transport.read_edges

    def intermittent_read(scope: PublishScope) -> tuple[AssignmentEdge, ...]:
        if transport.writes:
            raise RuntimeError("simulated unavailable read-back")
        return original_read(scope)

    transport.read_edges = intermittent_read  # type: ignore[method-assign]
    first = publish_assignment_plan(
        plan, transport=transport, publish=True, confirm_sha256=plan.digest, batch_size=2
    )
    assert (first.status, first.confirmed_edges, first.failure) == (
        "incomplete",
        0,
        "readback_unavailable",
    )
    assert len(transport.writes) == 1
    transport.read_edges = original_read  # type: ignore[method-assign]
    second = publish_assignment_plan(
        plan, transport=transport, publish=True, confirm_sha256=plan.digest
    )
    assert (second.status, second.confirmed_edges, second.attempted_batches) == ("complete", 4, 1)
    assert set(transport.writes[1]) == set(plan.edges) - set(transport.writes[0])


def test_context_drift_after_write_reports_incomplete() -> None:
    plan = prepared()
    transport = MockTransport(plan)
    original_inspect = transport.inspect_context

    def drifting_context(scope: PublishScope) -> RemoteContext:
        value = original_inspect(scope)
        return replace(value, reviewer_group="changed") if transport.writes else value

    transport.inspect_context = drifting_context  # type: ignore[method-assign]
    result = publish_assignment_plan(
        plan, transport=transport, publish=True, confirm_sha256=plan.digest, batch_size=1
    )
    assert (result.status, result.confirmed_edges, result.failure) == (
        "incomplete",
        0,
        "readback_unavailable",
    )
    assert len(transport.writes) == 1


def test_remote_deletion_between_batches_never_causes_unbounded_reposting() -> None:
    plan = prepared()
    transport = MockTransport(plan)
    original_post = transport.post_edges

    def replacing_post(edges: tuple[AssignmentEdge, ...]) -> None:
        transport.remote.clear()
        original_post(edges)

    transport.post_edges = replacing_post  # type: ignore[method-assign]
    receipt = publish_assignment_plan(
        plan, transport=transport, publish=True, confirm_sha256=plan.digest, batch_size=1
    )
    assert receipt.status == "incomplete" and receipt.attempted_batches == 2
    assert len(transport.writes) == 2


@pytest.mark.parametrize("case", ["wrong_weight", "unknown_edge", "duplicate"])
def test_existing_divergent_or_duplicate_edges_fail_closed(case: str) -> None:
    plan = prepared()
    transport = MockTransport(plan)
    edge = plan.edges[0]
    if case == "wrong_weight":
        transport.remote[edge.key] = replace(edge, weight=0.11)
    elif case == "unknown_edge":
        alien = replace(edge, tail="~Unknown")
        transport.remote[alien.key] = alien
    else:
        transport.remote[edge.key] = edge
        transport.read_edges = lambda scope: (edge, edge)  # type: ignore[method-assign]
    with pytest.raises(DataValidationError, match="remote edge"):
        publish_assignment_plan(plan, transport=transport, publish=True, confirm_sha256=plan.digest)
    assert not transport.writes


@pytest.mark.parametrize("case", ["wrong_type", "oversized", "malformed_row"])
def test_untrusted_remote_edge_response_is_bounded_and_typed(case: str) -> None:
    plan = prepared()
    transport = MockTransport(plan)
    if case == "wrong_type":
        transport.read_edges = lambda scope: []  # type: ignore[method-assign]
    elif case == "oversized":
        transport.read_edges = lambda scope: (plan.edges[0],) * 2_001  # type: ignore[method-assign]
    else:
        transport.read_edges = lambda scope: ("not-an-edge",)  # type: ignore[method-assign]
    with pytest.raises(DataValidationError, match="remote edge response"):
        publish_assignment_plan(plan, transport=transport, publish=True, confirm_sha256=plan.digest)
    assert not transport.writes


def test_remote_context_must_be_typed_and_bounded() -> None:
    plan = prepared()
    transport = MockTransport(plan)
    transport.inspect_context = lambda scope: {}  # type: ignore[method-assign]
    with pytest.raises(DataValidationError, match="remote context"):
        publish_assignment_plan(plan, transport=transport, publish=True, confirm_sha256=plan.digest)
    assert not transport.writes


def test_context_drift_fails_before_any_write() -> None:
    plan = prepared()
    transport = MockTransport(plan)
    transport.inspect_context = lambda scope: RemoteContext(  # type: ignore[method-assign]
        scope.assignment_invitation,
        "other-score-invitation",
        scope.reviewer_group,
        scope.paper_ids,
        scope.reviewer_ids,
    )
    with pytest.raises(DataValidationError, match="remote invitations"):
        publish_assignment_plan(plan, transport=transport, publish=True, confirm_sha256=plan.digest)
    assert not transport.writes


def test_prepare_rejects_unsafe_incomplete_or_wrong_scope() -> None:
    plan, scope, documents, experts = fixture()
    with pytest.raises(DataValidationError, match="paper_ids"):
        prepare_publish_plan(
            plan,
            replace(scope, paper_ids=("paper-A",)),
            documents=documents,
            experts=experts,
            default_demand=1,
        )
    with pytest.raises(DataValidationError, match="only independently audited"):
        prepare_publish_plan(
            plan,
            scope,
            documents=documents,
            experts=experts,
            conflicts=(Conflict("paper-A", "~Alice1"),),
            default_demand=1,
        )
    incomplete = MatchPlan((plan.assignments[0],), {"paper-A": 0, "paper-B": 1}, "optimal", 0.8)
    with pytest.raises(DataValidationError, match="only independently audited"):
        prepare_publish_plan(
            incomplete, scope, documents=documents, experts=experts, default_demand=1
        )


def test_plan_digest_is_order_independent_but_binds_all_scoped_content() -> None:
    plan, scope, documents, experts = fixture()
    first = prepared()
    reverse = MatchPlan(
        tuple(reversed(plan.assignments)), plan.unmet, plan.strategy, plan.total_score
    )
    assert (
        first.digest
        == prepare_publish_plan(
            reverse, scope, documents=documents, experts=experts, default_demand=1
        ).digest
    )
    changed = replace(scope, label="a-different-run")
    assert (
        first.digest
        != prepare_publish_plan(
            plan, changed, documents=documents, experts=experts, default_demand=1
        ).digest
    )
    tampered = replace(first, edges=(replace(first.edges[0], weight=0.01), *first.edges[1:]))
    with pytest.raises(DataValidationError, match="digest"):
        publish_assignment_plan(tampered)


def test_rehashed_but_malformed_edge_sets_still_fail_closed() -> None:
    plan = prepared()
    malformed = (
        (replace(plan, edges=()), "invalid edge count"),
        (
            replace(plan, edges=(replace(plan.edges[0], tail="~Alien"), *plan.edges[1:])),
            "outside its exact scope",
        ),
        (
            replace(plan, edges=(replace(plan.edges[0], weight=0.11), *plan.edges[1:])),
            "weights must agree",
        ),
        (replace(plan, edges=(plan.edges[0], plan.edges[0], *plan.edges[2:])), "exactly one edge"),
    )
    for tampered, message in malformed:
        with pytest.raises(DataValidationError, match=message):
            publish_assignment_plan(rehashed(tampered))


def test_duplicate_pair_is_rejected_before_edge_plan_is_emitted() -> None:
    plan, scope, documents, experts = fixture()
    duplicated = MatchPlan(
        (plan.assignments[0], plan.assignments[0], plan.assignments[1]),
        {"paper-A": 0, "paper-B": 0},
        "synthetic",
        2.2,
    )
    with pytest.raises(DataValidationError, match="only independently audited"):
        prepare_publish_plan(
            duplicated, scope, documents=documents, experts=experts, default_demand=1
        )


def test_peer_completes_plan_between_preflight_reads_without_empty_post() -> None:
    plan = prepared()
    transport = MockTransport(plan)
    original_read = transport.read_edges

    def peer_completion(scope: PublishScope) -> tuple[AssignmentEdge, ...]:
        if transport.reads == 1:
            transport.remote = {edge.key: edge for edge in plan.edges}
        return original_read(scope)

    transport.read_edges = peer_completion  # type: ignore[method-assign]
    receipt = publish_assignment_plan(
        plan, transport=transport, publish=True, confirm_sha256=plan.digest
    )
    assert receipt.status == "complete" and receipt.attempted_batches == 0
    assert not transport.writes


def test_bounds_and_ids_are_validated() -> None:
    plan = prepared()
    with pytest.raises(DataValidationError, match="batch_size"):
        publish_assignment_plan(plan, batch_size=101)
    with pytest.raises(DataValidationError, match="exact, printable"):
        replace(plan.scope, reviewer_group="bad group")
    with pytest.raises(DataValidationError, match="unique"):
        replace(plan.scope, reviewer_ids=("~Alice1", "~Alice1"))
    with pytest.raises(DataValidationError, match="finite score"):
        replace(plan.edges[0], weight=float("nan"))
    with pytest.raises(DataValidationError, match="finite score"):
        replace(plan.edges[0], weight=True)
    with pytest.raises(DataValidationError, match="publish must be"):
        publish_assignment_plan(plan, publish=1)  # type: ignore[arg-type]
    with pytest.raises(DataValidationError, match="invitations must differ"):
        replace(plan.scope, score_invitation=plan.scope.assignment_invitation)


def test_receipt_cannot_claim_unverified_or_invalid_state() -> None:
    plan = prepared()
    receipt = publish_assignment_plan(plan)
    invalid = (
        ({"plan_sha256": "wrong"}, "lowercase SHA-256"),
        ({"status": "bogus"}, "status is unknown"),
        ({"attempted_batches": -1}, "non-negative integer"),
        ({"expected_edges": 2_001}, "edge counts exceed"),
        ({"status": "incomplete"}, "only incomplete"),
        ({"failure": "unknown"}, "only incomplete"),
        ({"confirmed_edges": 1}, "dry-run receipt"),
    )
    for changes, message in invalid:
        with pytest.raises(DataValidationError, match=message):
            replace(receipt, **changes)


def test_unbounded_generators_are_stopped_at_the_declared_ceiling() -> None:
    plan, scope, documents, experts = fixture()
    seen = 0

    def endless_ids():
        nonlocal seen
        for index in count():
            seen += 1
            yield f"paper-{index}"

    with pytest.raises(DataValidationError, match="1 to 1000"):
        replace(scope, paper_ids=endless_ids())
    assert seen == 1_001

    seen = 0

    def endless_documents():
        nonlocal seen
        for _ in count():
            seen += 1
            yield documents[0]

    with pytest.raises(DataValidationError, match="documents exceeds maximum"):
        prepare_publish_plan(
            plan, scope, documents=endless_documents(), experts=experts, default_demand=1
        )
    assert seen == 1_001

    seen = 0

    def endless_experts():
        nonlocal seen
        for _ in count():
            seen += 1
            yield experts[0]

    with pytest.raises(DataValidationError, match="experts exceeds maximum"):
        prepare_publish_plan(
            plan, scope, documents=documents, experts=endless_experts(), default_demand=1
        )
    assert seen == 10_001

    seen = 0

    def endless_conflicts():
        nonlocal seen
        for _ in count():
            seen += 1
            yield Conflict("paper-A", "~Bob1")

    with pytest.raises(DataValidationError, match="conflicts exceeds maximum"):
        prepare_publish_plan(
            plan,
            scope,
            documents=documents,
            experts=experts,
            conflicts=endless_conflicts(),
            default_demand=1,
        )
    assert seen == 100_001


def test_no_overwrite_local_artifact_contains_same_digest(tmp_path: Path) -> None:
    plan = prepared()
    receipt = publish_assignment_plan(plan)
    output = tmp_path / "publish-evidence.json"
    write_publish_artifact(plan, receipt, output)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["plan"]["sha256"] == plan.digest
    assert artifact["receipt"]["status"] == "dry_run"
    with pytest.raises(FileExistsError):
        write_publish_artifact(plan, receipt, output)
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(DataValidationError, match="expected edge count"):
        write_publish_artifact(plan, replace(receipt, expected_edges=2), tmp_path / "other")
    with pytest.raises(DataValidationError, match="complete receipt"):
        replace(receipt, status="complete")


def test_concurrent_destination_creation_is_never_replaced(tmp_path: Path, monkeypatch) -> None:
    plan = prepared()
    receipt = publish_assignment_plan(plan)
    output = tmp_path / "race.json"
    original_link = os.link

    def create_target_before_link(source: Path, destination: Path) -> None:
        output.write_text("created concurrently", encoding="utf-8")
        original_link(source, destination)

    monkeypatch.setattr(publisher.os, "link", create_target_before_link)
    with pytest.raises(FileExistsError):
        write_publish_artifact(plan, receipt, output)
    assert output.read_text(encoding="utf-8") == "created concurrently"
    assert list(tmp_path.iterdir()) == [output]
