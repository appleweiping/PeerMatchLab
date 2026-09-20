"""Synthetic assignment-publish replay; no HTTP client or venue credentials."""

from peermatchlab import (
    Assignment,
    AssignmentEdge,
    Document,
    Expert,
    MatchPlan,
    PublishScope,
    RemoteContext,
    prepare_publish_plan,
    publish_assignment_plan,
)


class MemoryTransport:
    def __init__(self) -> None:
        self.edges: dict[tuple[str, str, str, str], AssignmentEdge] = {}

    def inspect_context(self, scope: PublishScope) -> RemoteContext:
        return RemoteContext(
            scope.assignment_invitation,
            scope.score_invitation,
            scope.reviewer_group,
            scope.paper_ids,
            scope.reviewer_ids,
        )

    def read_edges(self, scope: PublishScope) -> tuple[AssignmentEdge, ...]:
        return tuple(self.edges.values())

    def post_edges(self, edges: tuple[AssignmentEdge, ...]) -> None:
        for edge in edges:
            if edge.key in self.edges:
                raise RuntimeError("synthetic conditional-insert conflict")
            self.edges[edge.key] = edge


documents = (Document("synthetic-paper", "Synthetic retrieval paper", required_experts=1),)
experts = (Expert("~SyntheticReviewer1", "Synthetic Reviewer", capacity=1),)
plan = MatchPlan(
    (Assignment("synthetic-paper", "~SyntheticReviewer1", 0.75, 1),),
    {"synthetic-paper": 0},
    "synthetic",
    0.75,
)
scope = PublishScope(
    "Synthetic/-/Assignment",
    "Synthetic/-/Aggregate_Score",
    "Synthetic/Reviewers",
    "peer-matchlab-demo",
    ("synthetic-paper",),
    ("~SyntheticReviewer1",),
)
prepared = prepare_publish_plan(plan, scope, documents=documents, experts=experts, default_demand=1)
transport = MemoryTransport()
print(prepared.digest)
print(publish_assignment_plan(prepared, transport=transport).status)
print(
    publish_assignment_plan(
        prepared, transport=transport, publish=True, confirm_sha256=prepared.digest
    ).status
)
print(
    publish_assignment_plan(
        prepared, transport=transport, publish=True, confirm_sha256=prepared.digest
    ).attempted_batches
)
