# Changelog

All notable changes are documented here. The project follows semantic versioning once the public API reaches 1.0.

## Unreleased

_No changes yet._

## 0.7.0 - 2026-09-19

- Added opt-in, read-only OpenReview API v2 profile and publication acquisition
  with exact author-ID joins, invitation/date/content filters, privacy-minimized
  snapshots, finite aggregate request limits, and atomically published version-2
  provenance manifests. The original reviewer-shell snapshot remains unchanged.
- Added frozen API-v2 transport fixtures, an acquisition-to-expertise-to-match
  smoke test, and an independent ≥90% branch-coverage gate in CI and release.
- Bounded the aggregate publication scan before and during author pagination,
  minimized persisted author IDs, and rechecked privacy, identities, declared
  filters, and canonical evidence at the public writer boundary.

## 0.6.0 - 2026-09-08

- Added a dependency-free, typed expertise-generation layer with distinct TF-IDF cosine and BM25
  implementations, sparse corpus statistics, additive term explanations, and aggregate, maximum,
  or average reviewer-evidence semantics.
- Added explicit tokenizer, stopword, field, publication date/content, duplicate, score-threshold,
  and resource-limit configuration, including document character/byte, scanned-match, and token-length
  ceilings. Empty and filtered evidence stays auditable and all numeric controls reject booleans and
  non-finite values.
- Added a strict offline OpenReview-shaped snapshot contract for submissions, profiles, and explicit
  reviewer-publication joins; it never infers authorship or contacts a service.
- Added atomic expertise artifact directories containing normalized assignment inputs, sparse affinity
  CSV, exact token documents, explanations, a schema-versioned replayable model, and byte/SHA-256
  provenance for all original files and every derived artifact except the manifest itself.
- Kept explicit publication IDs and positional fallbacks in disjoint namespaces, reload-verified every
  model before artifact installation, and made persisted total-publication limits and required abstracts
  on retained evidence fail closed. Configuration provenance now hashes the same bounded bytes used for
  parsing.
- Closed local provenance around immutable causal source bytes and adapter parameters, deep-froze score
  explanations, enforced score-model invariants, re-parsed configuration bytes at publication time,
  and made staging cleanup cover process interruptions. Local JSON record limits now stop parsing at
  the first excess record, and public resource controls reject platform-sized integers predictably.
- Deep-snapshotted live OpenReview note mappings before raw and converted output, and refused matching
  output paths that alias any input through a path, symbolic link, or hard link. Snapshot freezing now
  has aggregate expanded-item and UTF-8 work budgets that account for aliased deep/wide structures.
- Separated corpus-input and persisted-model byte ceilings. Direct model saves render and validate the
  exact deterministic payload before atomically replacing any destination, and legacy models receive
  the bounded model-file default when loaded.
- Added independent hand-calculated TF-IDF/BM25 oracles, model-tampering, empty/non-finite/duplicate,
  date/content-filter, resource-bound, atomic-cleanup, manifest-integrity, CLI, and CI smoke tests.

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
