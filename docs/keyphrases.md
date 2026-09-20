# Local graph-ranked keyphrase preprocessing

`extract-keyphrases` is an opt-in lexical preprocessing stage for local
PeerMatchLab `documents.json` and `experts.json` (or JSONL). It produces
inspectable ranked words for each submission, reviewer profile, and
publication separately. It is useful for evidence review and for building
future trainable keyphrase models; it does not change assignments or replace
the existing TF-IDF/BM25 affinity generator.

```bash
peermatch extract-keyphrases \
  --documents examples/documents.json \
  --experts examples/experts.json \
  --directory scratch/keyphrases
```

The output directory is new and never overwritten. `keyphrases.jsonl` has
`kind`, `owner_id`, `evidence_id`, `token_count`, and a descending list of
`{term, score}`. A publication with a source ID uses `id:<value>`; otherwise
it uses `position:<zero-based-index>`. `manifest.json` records the algorithm
name, complete extraction config, record count, exact SHA-256 of both input
byte streams and the output bytes, and the output byte count. The manifest
does not hash itself. The writer renders the complete output before creating
the destination and atomically installs a staged directory without replacing
an existing one. Inputs are captured once as immutable bounded bytes; the
captured bytes, not a later read of the paths, are the causal inputs.

The lexical tokenizer uses Unicode case-folded alphanumeric runs, drops
single-character words and a fixed English stopword list, and excludes
underscores. For each text unit, an undirected edge connects two distinct
words occurring within a configurable window **inside one sentence or input
field**. Periods, exclamation points, question marks, and newlines close a
window; a title and abstract therefore do not create a spurious bridge.
Repeated co-occurrences do not increase edge weight. With `n` terms, damping
`d`, degree `deg(v)`, and rank
`r_t`, every fixed iteration computes:

```text
r_(t+1)(v) = (1-d)/n + d × dangling_mass/n
               + d × Σ_(u adjacent to v) r_t(u)/deg(u)
```

Ranks start at `1/n`. The default is a two-word window, damping 0.85, 30
iterations, and top 20 terms. Equal scores break by the term's Unicode
lexical order; graph traversal is sorted so results are hash-seed independent.
For the three-word chain `alpha beta gamma` with damping 0.5, the stationary
ranks are exactly `5/18, 4/9, 5/18`; this independent hand calculation is
covered by a test. For `alpha beta. gamma` at the same damping, the disconnected
pair ranks `2/5` each and isolated `gamma` ranks `1/5`; this is also tested.
Empty/stopword-only text has a defined empty result.

Per-source input is limited to 16 MiB. A run is capped at 10,000 evidence
records and 16 MiB of rendered output. Each evidence item is limited to one
million characters, 100,000 retained tokens, 20,000 distinct terms, and
200,000 graph edges. The estimated cumulative graph-iteration work is capped
at ten million units across the run. CLI controls are `--top-k`,
`--window-size`, `--iterations`, and `--damping`; the typed Python API exposes
the remaining resource limits. Invalid inputs and exhausted limits fail
before output publication.

This is a dependency-free TextRank-*style* word graph, not an implementation
of the frozen OpenReview Expertise spaCy pipeline. It has no part-of-speech
tags, lemmatization, phrase chunks, or model training. Scores are graph
centrality, not reviewer affinity, model confidence, or evidence of reviewer
competence. No network access occurs and the caller remains responsible for
private submission and reviewer data.
