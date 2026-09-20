# Bounded document-side fair-local assignment

Set `"strategy": "fair-local"` in a `peermatch match` or `match-affinity`
JSON config, or use `MatchConfig(strategy="fair-local")` in the Python API.
The normal plan, independent audit, unmet-demand diagnostics, and CLI output
formats are unchanged. [This example](../examples/fair-local-config.json)
uses the default resource budget.

The objective has two stages:

1. Compute an integral maximum-cardinality assignment under the same
   eligibility, hard-conflict, score-threshold, expert-capacity, document-demand,
   institution-diversity, and reserved-seniority constraints as `optimal`.
2. Accept only moves that lexicographically improve the vector of **all**
   document score sums sorted from weakest to strongest. Among exactly equal
   vectors, select the lexicographically first sorted pair list. A document
   with no selected expert has a score sum of zero.

The local neighborhood comprises a selected edge replaced by another
admissible expert for the same document, an edge transferred to another
document with an open slot, or two selected edges on different documents
whose experts can be swapped. Every candidate is rechecked against the hard
constraints before use. Replacements may change expert loads; swaps and
transfers do not change their total. All moves preserve the number of filled
slots, and strictly ordered accepted states cannot cycle. Raw unrounded scores
are compared; no fairness score is inferred from demographic attributes.

This is **not** the [OpenReview Matcher `FairFlow` algorithm](https://github.com/openreview/openreview-matcher/blob/e6a2dad82880b45560d5b09b5b2236bec13a6cec/matcher/solvers/fairflow.py). That algorithm
searches a paper-score makespan with a lower-bound flow construction and a
different approximation procedure. `fair-local` can stop at a local optimum
that is worse than the global `maximin` result; for example, an improving
three-document cycle may have no improving one- or two-edge move. The tests
include that counterexample. A budget cutoff may stop even before a local
fixed point. The plan's `diagnostics.certified` certifies maximum assignment
cardinality / whole-run demand feasibility, **not** globally optimal fairness.

The default budget is 32 accepted steps and 200,000 considered moves. Set
`fair_local_max_steps` from 1 to 128 and `fair_local_max_checks` from 1 to
2,000,000 if needed; non-default controls with another strategy are rejected.
To bound the initial flow and the local search, the strategy accepts at most
128 documents, 256 experts, 4,096 eligible pairs after minimum-score
filtering, 512 total requested slots, and 4,096 total expert capacity.
Instances outside these limits fail explicitly. This is larger than the
bounded exact `maximin` strategy, but is **not** a full-conference scale claim.
The current scalar `load_balance_penalty` is incompatible with this fairness
objective. As elsewhere in this package, simultaneous seniority floors and
institution diversity are rejected because the flow representation cannot
express both without relaxing one constraint.

`fair-local` does not implement reviewer minimum-load quotas, FairIR's
threshold constraints, group-fairness guarantees, live OpenReview matching,
or an exact global maximin solver at larger scale. Affinity quality and bias
remain the caller's responsibility.
