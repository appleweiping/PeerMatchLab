# Changelog

All notable changes are documented here. The project follows semantic versioning once the public API reaches 1.0.

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
