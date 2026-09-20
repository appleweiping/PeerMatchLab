# Exact maximin assignment for small panels

Set `"strategy": "maximin"` in the same JSON config accepted by `peermatch match`
and `peermatch match-affinity`, or pass `MatchConfig(strategy="maximin")` to the
Python pipeline. No new file format is required: exported plans retain the
ordinary assignments, unmet demand, independent audit, and diagnostics.

The objective is lexicographic:

1. Fill as many requested slots as the hard constraints allow.
2. Maximize the weakest document's **sum** of selected pair scores. An unfilled
   document with no selected expert has a total of zero.
3. Maximize the sum over all selected pair scores.
4. Break exact ties by the lexicographically first sorted document/expert pairs.

Unlike `minmax`, which bounds the *largest reviewer workload*, this objective
protects the lowest-scoring document. It is neither a guarantee of reviewer
qualification nor a group-fairness metric: input affinities and their biases
remain the caller's responsibility. As with other strategies, zero scores are
eligible unless `minimum_score` excludes them. The plan's feasibility status
is certified for the declared constraints because this bounded search explores
every admissible assignment.

The solver enumerates subsets per document and checks global expert capacities.
It accepts at most **6 documents, 8 experts, and 16 eligible pairs** after
eligibility/minimum-score filtering. This is exponential-time by design; it
rejects larger instances with a `ValueError` so callers do not accidentally
use it on full-conference workloads. `load_balance_penalty` is unsupported
because it would introduce an additional competing objective. Use the
scalable flow-based `optimal` or `minmax` strategy for larger cases.

Institution diversity treats unknown affiliations as distinct, matching the
existing flow solver. Reserved senior slots cannot be filled by junior
reviewers; if senior capacity is short, free slots can still be filled and the
remaining demand is reported. The pre-existing rule forbidding simultaneous
seniority floors and institution diversity applies here too.

This original bounded solver does not implement reviewer minimum loads,
random tie breaking, a scalable approximate makespan method, or a hosted
matching service.
