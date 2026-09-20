# Assignment edge preparation and mock write-back

This is an offline-first protocol slice, **not a production OpenReview publisher**.
The frozen Matcher reference creates an assignment edge and a matching aggregate-score
edge for every paper/reviewer pair. PeerMatchLab prepares those two edge sets and
audits an independently produced `MatchPlan`; it does not ship an HTTP POST client,
authentication, invitation discovery, or a hosted matching service. No test or
example contacts a real venue.

`prepare_publish_plan` accepts the exact assignment invitation, score invitation,
reviewer group, label, paper IDs, and reviewer IDs in a `PublishScope`. The IDs must
exactly match the audited domain inputs. The plan must have complete demand, no
conflicts, capacity excess, duplicate pair, or other hard-audit violation. Input
scores are copied verbatim to both invitation-specific edges; the canonical JSON
of all scoped IDs and edge content determines the SHA-256 plan digest. The maximum
is 1,000 assignments and 2,000 edges. This preparation makes no network calls.

`publish_assignment_plan` is dry-run by default, even if a transport was supplied.
An actual write requires **all** of `publish=True`, a caller-supplied
`AssignmentWriteTransport`, and `confirm_sha256` equal to the immutable plan digest.
The transport must fetch exact invitation IDs, group membership, and paper IDs,
and must return the complete set of edges for the specified two invitations,
label, and paper heads. Both returned context and existing edge identity/content
are checked before a write. Any unexpected or divergent edge aborts; this code
never deletes or overwrites an edge **provided the injected adapter honors its
required atomic create-only contract**. Each edge insertion must fail on an
existing identity; a plain OpenReview bulk upsert cannot safely implement this
transport because another actor may insert an edge after the preflight read.
Batches contain at most 100 edges.

After every batch, the context and complete edge set are read back. An ambiguous
partial transport failure, unavailable read-back, or not-yet-visible edge returns
an `incomplete` receipt, not a blind automatic retry. Repeating the call re-reads
the remote set and sends only missing edges; existing exact edges produce a
zero-write `complete` receipt. A concurrent actor that changes a label-scoped
edge or venue context causes fail-closed behavior when the conditional-insert
contract is upheld. Operators must inspect
`confirmed_edges`, `failure`, and the authoritative venue state before retrying.
`write_publish_artifact` stores both objects in one canonical JSON file. It
writes a temporary file beside the requested path and hard-links it into place,
so an existing or concurrently created destination is never replaced. The
temporary name is removed even when installation fails. Paper IDs are limited
to 1,000, reviewer IDs to 10,000, and conflicts to 100,000; generator inputs
are read only up to one record past these ceilings before rejection.

Try the purely in-memory synthetic replay:

```bash
uv run python examples/openreview_publish_mock.py
```

It prints one digest, then `dry_run`, `complete`, and `0` attempted batches
on an idempotent re-run. `tests/test_openreview_publish.py` also simulates a
partial POST, stale read-back, changed context, duplicate/foreign edges, and
read-back failure. There are no credentials, real invitation IDs, or network
operations in either sample.

This boundary intentionally leaves OpenReview API v1/v2 edge payload shape,
invitation authorization, pagination of existing edges, eventual-consistency
timing, atomic conditional-insert support, and production conflict policy to a
future separately reviewed adapter. If a real endpoint cannot provide atomic
create-only semantics, the write path must remain disabled; preflight alone
is insufficient.
The caller must not treat a mock-transport `complete` receipt as proof that a
real venue was updated.
