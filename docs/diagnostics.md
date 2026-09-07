# Assignment diagnostics API

`AssignmentEngine.assign()` attaches an `AssignmentDiagnostics` object to every newly generated
`MatchPlan`. It is serialized under the top-level `diagnostics` key by `plan_to_dict()` and therefore
appears in `match` and `match-affinity` JSON output.

```json
{
  "status": "infeasible",
  "certified": true,
  "requested": 2,
  "assigned": 1,
  "unmet": 1,
  "documents": [
    {
      "document_id": "paper-1",
      "requested": 2,
      "assigned": 1,
      "unmet": 1,
      "reason_codes": ["expert_capacity", "global_capacity_coupling"],
      "evidence": {
        "admissible_pairs": 2,
        "saturated_admissible_experts": 1
      },
      "saturated_experts": ["reviewer-7"]
    }
  ]
}
```

The engine emits a complete fixed set of evidence counters; the shortened object above is only an
illustration. JSON ordering is deterministic.

## Run status and certification

| `status` | `certified` | Meaning |
|---|---:|---|
| `satisfied` | `true` | The returned assignments themselves prove that every requested slot was filled. |
| `infeasible` | `true` | The optimal flow could not send the complete requested flow. No full assignment exists in the active network. |
| `not_certified` | `false` | Greedy left demand unmet. A different ordering or the optimal strategy may still fill it. |

`infeasible` is a **run-level** result. Maximum flow proves that all document demands cannot be met
simultaneously; which document is short may differ between equally large assignments. Per-document
reason codes are therefore constraint evidence, not individual infeasibility certificates.

## Reason codes

| Code | Evidence boundary |
|---|---|
| `hard_conflict` | A pair was present in the exact conflict set carried by the built-in text or affinity scorer. |
| `other_ineligible` | A custom scorer marked a pair ineligible without identifying it as a hard conflict. |
| `zero_capacity` | A scored expert has configured capacity zero. This may overlap another exclusion. |
| `minimum_score` | An otherwise eligible, positive-capacity pair fell below the configured threshold. |
| `sparse_score_matrix` | At least one expert has no score row for the document. Missing sparse affinities are not synthesized. |
| `candidate_scarcity` | Fewer locally admissible scored experts exist than the document requests. |
| `expert_capacity` | At least one unselected admissible expert is at capacity in the returned global plan. |
| `seniority_floor` | The returned document assignment did not fill its reserved senior slots. |
| `institution_diversity` | Available institution groups locally bound demand, or an otherwise available pair shares a used group. Unknown institutions remain expert-specific groups. |
| `global_capacity_coupling` | Local candidate, institution, and senior counts are individually large enough, but optimal global flow still leaves the document short. Shared capacities or graph coupling are involved. |
| `greedy_not_certified` | The shortage came from the deterministic greedy baseline and is not an infeasibility proof. |

Codes can appear together. They answer “which filters or bottlenecks are evidenced here?”, not “which
single input caused the shortage?”. Counterfactual claims would require a separately defined rerun and
are intentionally outside this schema.

## Evidence counters

Every document includes `total_experts`, `scored_pairs`, `unscored_experts`, `hard_conflict_pairs`,
`other_ineligible_pairs`, `zero_capacity_pairs`, `eligible_pairs`, `below_minimum_score_pairs`,
`admissible_pairs`, `senior_admissible_pairs`, `senior_assigned`, `institution_groups`,
`institution_blocked_pairs`, and `saturated_admissible_experts`. Counts are non-negative integers.
`saturated_experts` lists the corresponding expert identifiers in deterministic order.

The public immutable models are:

- `FeasibilityStatus`
- `UnmetReason`
- `DemandDiagnostic`
- `AssignmentDiagnostics`
- `MatchPlan.diagnostics`

`plan_from_dict()` accepts older external plans with no `diagnostics` field. If the field is supplied,
it is strict: unknown fields or reason codes, invalid counts, a mismatched `certified` flag, and totals
that disagree with the plan are rejected. Diagnostics describe feasibility; `audit_plan()` remains the
independent check for conflicts, capacity, duplicate assignments, references, and demand safety.
