# Frozen SPECTER-family embedding interchange

PeerMatchLab can score precomputed publication and submission embeddings through
an `EmbeddingProvider` protocol. It does **not** bundle SPECTER, SPECTER2,
SciNCL, a tokenizer, neural weights, or an inference implementation. The
included vectors are small synthetic math oracles, not model output. A real
encoder must be run separately under its own license, with an operator-supplied
model identity and weights digest; those declarations are provenance, not a
verification that the declared model actually produced the vectors.

The frozen OpenReview Expertise SPECTER2/SciNCL predictor at `3e2803a`
interchanges JSONL records shaped `{"paper_id": ..., "embedding": [...]}`.
Its scoring normalizes vectors to unit length, takes submission-publication
cosines, min-max normalizes over the **whole** paper-pair matrix (clamping a
constant matrix to `[0, 1]`), drops empty publication embeddings from each
reviewer's evidence, computes max or average, and rounds to four decimals.
This slice implements those two reviewer aggregations. It does not implement
percentile selection, venue-specific logit weights, ensemble blending, GPU
inference, or the comparator's other model families. Profiles are not encoded;
only reviewer publications contribute.

Run the deterministic local fixture, then feed its affinities to the existing
assignment pipeline:

```sh
peermatch expertise-embedding \
  --snapshot examples/expertise/snapshot \
  --embeddings examples/expertise/embeddings \
  --directory embedding-run
peermatch match-affinity \
  --documents embedding-run/documents.json \
  --experts embedding-run/experts.json \
  --affinities embedding-run/affinities.csv \
  --config examples/expertise/match-config.json \
  --output embedding-plan.json
```

`--documents` plus `--experts` can replace `--snapshot`. `--aggregation` is
`max` (default) or `average`; `--max-paper-pairs` and `--max-candidate-pairs`
default to one million each. Paper comparisons are hard-capped at ten million
and scanned in two passes, retaining only one submission row at a time.
Candidate pairs are hard-capped at one million because the public API returns
an in-memory score tuple; larger result sets are not claimed to stream. The
local source adapter also enforces its existing byte/record/publication limits.

The embedding directory has a strict `manifest.json` (schema 1),
`submissions.jsonl`, and `publications.jsonl`. The latter two use exactly the
comparator's `paper_id`/`embedding` row shape, with 768 finite coordinates or
an empty vector. Empty publication vectors participate as zero in global
normalization, then are excluded from reviewer aggregation. Each JSONL file
is byte-counted and SHA-256 checked. The manifest also holds canonical SHA-256
hashes of the requested `(paper_id, title, abstract)` sequences, binding
vectors to the exact source text and association order. Duplicate/missing IDs,
unknown keys, wrong dimension, non-finite numbers, changed bytes, and source
text drift are errors. Synthetic fixtures must declare a null weights digest.

Outputs are installed in a new directory only after replay against captured
source bytes. The run contains normalized assignment inputs, `scores.jsonl`
for **all** candidate pairs, positive-only `affinities.csv`, exact copies of
source/embedding bytes, and a schema-1 `manifest.json` recording generator,
scoring settings, counts, and SHA-256 digests. The manifest cannot hash itself.
The 64 MiB per-file ceiling applies to **input** files only. Generated files
are UTF-8 streamed and SHA-256 hashed as written, with a 256 MiB per-file
ceiling and 512 MiB aggregate ceiling across generated files (including the
manifest). Both limits are enforced before each write and recorded in the run
manifest. Source-byte copies retain their separate input limits. Exceeding an
output ceiling removes the staging directory and publishes nothing.
No network request is made by this command.

The typed API accepts any provider implementing `EmbeddingProvider.embed` and
`provenance`. Providers are checked for complete IDs, 768-dimensional finite
vectors, and matrix limits before scoring. This is an executable adapter
boundary for future licensed encoders, not a claim of live model parity.

Embedding similarity is only evidence. It does not account for conflicts,
author identity errors, missing/biased publication histories, language or
field coverage, or reviewer willingness. Apply the existing assignment
constraints and human review before any use in a real venue.
