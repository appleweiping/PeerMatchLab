# Changelog

All notable changes are documented here. The project follows semantic versioning once the public API reaches 1.0.

## Unreleased

_No changes yet._

## 0.5.0 - 2026-09-07

- Added a bounded, read-only OpenReview API v2 synchronization client and `fetch-openreview` CLI.
  It implements ID-cursor pagination with a count completeness contract, strict payload validation,
  finite retry with bounded `Retry-After`, proactive request pacing, injectable transport/clocks,
  HTTPS and response-size safeguards, environment-only bearer-token loading, and atomic snapshot
  output containing raw evidence, converted interchange files, SHA-256 digests, and provenance.
- Added offline protocol/oracle tests for pagination, truncation, duplicate cursors, response schema,
  group membership, authentication placement, resource bounds, rate limiting, retry timing, transport
  failures, terminal HTTP behavior, atomic cleanup, manifest integrity, and the public CLI.

## 0.4.0 - 2026-09-07

- Added a real `minmax` assignment strategy. It preserves the maximum
  achievable assignment cardinality, binary-searches the smallest per-expert
  load cap that can attain it, and runs the integral evidence-maximizing flow
  under that cap. The strategy is distinct from `optimal` and `greedy`, works
  with the existing score/diversity/seniority constraints, and is covered by
  deterministic fairness and cardinality regression tests.

## 0.3.0 - 2026-09-07

- Added a tag-gated release pipeline with locked builds, clean wheel and sdist installation
  checks, CycloneDX SBOM, SHA-256 manifest, and GitHub provenance.
- Added typed, machine-readable assignment diagnostics to `MatchPlan`, stable JSON output, CLI
  summaries, and HTML reports. They distinguish observed conflict, zero-capacity, score-threshold,
  sparse-matrix, seniority, institution, and shared-capacity evidence without claiming a unique cause.
- Optimal shortfalls are marked globally `infeasible` only after maximum flow is exhausted. Greedy
  shortfalls are explicitly `not_certified`; a greedy ordering is never presented as proof of
  infeasibility. Independent randomized exhaustive tests verify the maximum assigned count and status.
- Kept audited legacy plans compatible: diagnostics are optional when loading externally produced
  plans, while any supplied diagnostic object is strictly validated against assignment and unmet totals.
- Added `minimum_senior_reviewers` and `senior_threshold`, reserving a number of each document's
  slots for experts at or above a seniority threshold. The optimal solver enforces the floor by
  splitting each document's demand at the source, so reserved units can only reach senior pair
  nodes; both sides meet at one capacity-one node per document-expert pair, which keeps an expert
  from filling a reserved and a free slot for the same document. The greedy baseline honours the
  same floor rather than merely preferring senior experts.
- The reservation is hard: a reserved slot no senior expert can fill is reported as unmet instead of
  being handed to a junior, and the rest of the document is still filled. Strategy names gain
  `-senior`.
- The objective is unchanged under the constraint, checked by exhaustive search over small instances
  rather than assumed. Across 500 randomized instances the flow plan was never beaten by enumeration
  and never placed a junior in a reserved slot.
- Restructuring the network changed nothing when the reservation is not requested: 3,600 plans over
  150 randomized instances, across both strategies, three demand levels, institution diversity and
  load balancing, are identical to the plans produced before the change.
- `minimum_senior_reviewers` cannot be combined with `require_distinct_institutions`. The two are not
  jointly expressible in this flow network -- a unit passing through the capacity-one institution
  gate no longer carries which side of the source split it came from -- so the combination is refused
  rather than approximated into a plan that satisfies one constraint and quietly relaxes the other.

## 0.2.0 - 2026-08-31

- Added strict sparse external-affinity ingestion and matching through the same constrained optimizer.
- Added loss-aware local OpenReview submission and reviewer-ID import without network access.
- Made institution diversity an exact global min-cost-flow constraint for the optimal strategy.
- Added an optional convex workload-balancing penalty to both optimal and greedy strategies.
- Added strict duplicate-key rejection for JSON, JSONL, configuration, and audited-plan inputs.
- Added randomized exhaustive-oracle tests for small institution-diverse assignment instances.

## 0.1.0 - 2026-08-31

- Added strict document, expert, publication, bid, and conflict models.
- Added deterministic TF-IDF evidence scoring with five decomposed components.
- Added optimal capacity-constrained and diversity-aware greedy assignment.
- Added independent safety, coverage, workload, and institution audits.
- Added JSON/JSONL adapters, CLI workflows, examples, tests, and packaging.
- Hardened external-plan audits against unknown references, duplicate pairs, and excess demand.
- Made JSON numeric validation strict and kept score maximization ahead of large-pool tie breaking.
