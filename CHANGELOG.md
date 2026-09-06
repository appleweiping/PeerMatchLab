# Changelog

All notable changes are documented here. The project follows semantic versioning once the public API reaches 1.0.

## Unreleased

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
