# Bounded capacity and conflict what-if comparisons

`peermatch what-if` compares a baseline match with 1–8 independent local
scenarios. A scenario can change expert capacities or explicitly add/remove
declared document–expert conflicts. Every scenario starts from the same
unchanged baseline, runs the existing matcher, and undergoes the independent
assignment audit. The JSON report includes assigned and unmet demand, status,
scores, assignments, per-expert workload, audit and unmet reason codes, plus
signed deltas and added/removed assignment pairs.

```sh
peermatch what-if \
  --documents examples/documents.json \
  --experts examples/experts.json \
  --conflicts examples/conflicts.json \
  --config examples/config.json \
  --scenarios examples/what-if-scenarios.json \
  --output what-if-report.json
```

The schema-1 plan has a `scenarios` array. Each scenario requires a unique,
short lowercase `id` and all three change fields: `capacity_overrides` maps
existing expert IDs to integer capacities from 0 to 128; `add_conflicts` and
`remove_conflicts` are arrays of `[document_id, expert_id]` pairs. Adding an
existing conflict or removing a nonexistent one is an error. A scenario must
have at least one change. Scenario IDs are evaluated and reported in sorted
order, regardless of plan order. Conflict changes affect *only* the named
scenario. The source files and Python domain objects are never modified.

The local comparison is limited to 32 documents, 64 experts, 128 requested
slots, 2,048 conflicts, 8 scenarios, 1 MiB per document/expert/conflict CLI
input, and 64 KiB per plan/config CLI input. The output path cannot alias an
input path, including through symlinks and hard links. It is an ordinary
replaceable local JSON output, not a signed or create-only registry record.

This is a deterministic sensitivity *observation*, not a causal attribution
or proof that a particular constraint is the unique cause of unmet demand.
Changing capacity or conflict declarations can alter assignments globally.
The tool neither infers real-world conflicts nor recommends removing them;
human operators must verify the declarations and approve any actual change.
It does not implement an interactive web UI, OpenReview write-back, or the
frozen Matcher solver families still listed as gaps in the capability map.
