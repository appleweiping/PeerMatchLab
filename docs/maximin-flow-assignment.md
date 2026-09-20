# Bounded edge-bottleneck flow assignment

Set `"strategy": "maximin-flow"` in the normal `peermatch match` or
`match-affinity` configuration, or call
`AssignmentEngine.assign(strategy="maximin-flow")`. The output remains a
normal independently auditable assignment plan with unmet-demand diagnostics.
The [example configuration](../examples/maximin-flow-config.json) works with
the repository's synthetic affinity CSV.

The optimization order is:

1. Fill as many requested reviewer slots as the declared hard constraints
   allow (maximum cardinality).
2. Among plans with that cardinality, maximize the **lowest score on any
   selected document–reviewer pair**. With no selected pairs, the floor is
   defined as zero.
3. At that floor, maximize the sum of the existing flow solver's six-decimal
   fixed-point scores. Its stable, deterministic edge order resolves remaining
   ties; no random seed or hidden input is used.

The solver obtains a maximum-cardinality integral flow, sorts the distinct
eligible input scores, then binary-searches the highest threshold whose
pruned network can still carry that many assignments. Monotonicity of this
feasibility predicate proves the selected-edge floor is optimal within the
declared constraints. Each flow also obeys hard conflicts, score threshold,
per-expert capacity, per-document demand, optional distinct institution gates,
and optional reserved senior slots. Senior reservations and institution gates
remain mutually incompatible, as for the existing flow solver. A nonzero
`load_balance_penalty` is rejected instead of silently changing the objective.

This is **not** the existing `maximin` objective, which maximizes the weakest
document's **sum** of selected scores after cardinality. For example, on a
two-paper, two-reviewer-per-paper fixture, `maximin-flow` can improve the
lowest individual edge from 0.1 to 0.2 while reducing the weakest paper sum
from 0.9 to 0.4. It is not a document-fairness guarantee. If full coverage is
infeasible, the cardinality stays optimal, but the identity of an unfilled
document can change during the floor search. Use the independent audit and
inspect all `unmet` entries before any operational decision.

This is also **not** the frozen OpenReview Matcher `FairFlow` solver. That
approximation targets paper-score makespan, considers reviewer minimum loads,
and uses a different reassignment flow. `maximin-flow` has no reviewer minimum
load, no paper-score makespan certificate, and no official OpenReview dataset,
runtime, or assignment parity claim. All input scores and their potential bias
remain the caller's responsibility.

For predictable local resource use, reject more than **24 documents, 48
experts, 256 eligible pairs after minimum-score filtering, 64 requested
slots, or 256 total expert capacity**. The graph and the number of flow solves
are bounded (one baseline plus at most nine threshold trials). This is wider
than the exact exhaustive `maximin` limit of 6/8/16, but is not a conference
scale solver or a latency guarantee. Tests include an independent exhaustive
small-graph feasibility/objective oracle; run the synthetic, audited comparison
locally with `uv run --locked python scripts/benchmark_maximin_flow.py`.
