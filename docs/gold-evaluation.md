# Gold-standard expertise evaluation

`peermatch evaluate-gold` compares a local sparse affinity CSV with explicitly
judged paper/reviewer pairs. It does not download a gold dataset, call
OpenReview, infer neural embeddings, or disclose paper text. Supply a dataset
you are permitted to use; the report records input hashes, not raw text.

For a quick local demonstration, `examples/gold-judgments.csv` is an entirely
synthetic set of grades compatible with `examples/affinities.csv`; it is not
an official or licensed gold benchmark.

```powershell
peermatch evaluate-gold --gold judgments.csv --gold-format triples `
  --affinities affinities.csv --k 1 --k 5 --k 10 `
  --relevance-threshold 1 --output evaluation.json
```

From the repository root, substitute `examples/gold-judgments.csv` and
`examples/affinities.csv` to run the synthetic example without private data.

The canonical `judgments.csv` has the exact header
`document_id,expert_id,relevance`, with one integer grade per pair. The
alternative `--gold-format openreview` accepts the public gold-standard
`evaluations.csv` shape: `ParticipantID,Paper1,Expertise1,...,Paper10,Expertise10`.
CSV or tab-separated rows are accepted. Each participant becomes an expert;
each nonempty paper cell becomes one judged document/expert pair. No reviewer,
paper ID, or grade is discarded. The adapter does **not** reproduce the
reference repository's pairwise-loss, bootstrap, or cross-validation pipeline.

The affinity file is the same sparse comma-separated CSV used by
`match-affinity`: optional exact header `document_id,expert_id,score`, or
headerless rows. Scores are finite in `[0, 1]`; they are similarities, so
larger is better. This report evaluates **only judged pairs**. A scored pair
with no gold judgment is counted as ignored, never invented as a negative.
Each judged document must have at least one judged pair with a score. Missing
judged scores rank after every scored judged pair, with `expert_id` breaking
ties. `--strict-coverage` rejects any missing judged score instead. Exact
score ties also break by ascending `expert_id`; CSV row order does not matter.

Grades at or above `--relevance-threshold` (default 1) are positive. A
document with no judged positive is skipped with reason `no_judged_positive`
and excluded from macro denominators. If all documents are skipped, macro
metrics are zero and `evaluated_documents` is zero; the report does not imply
that the system performed well. For each `k`:

- `precision`: positives among the first `k` divided by `k`, even when fewer
  than `k` judged experts exist.
- `recall`: positives among the first `k` divided by **all judged positives**.
- `hits`: 1 if any positive appears among the first `k`, otherwise 0.
- `average_precision`: precision at every positive rank up to `k`, summed and
  divided by **all judged positives**. This is truncated AP, not AP divided
  by the smaller number of positives encountered before `k`.

`macro[k]` is the unweighted mean of each metric over evaluated documents;
`macro[k].average_precision` is MAP@k. The JSON includes per-document counts,
per-document metrics, skipped documents, source byte counts and SHA-256, and
`result_fingerprint_sha256`. The fingerprint hashes the canonical normalized
result **without** raw source hashes, so permuting rows preserves it and the
metrics. Raw source SHA-256 values intentionally change if row bytes change;
therefore the entire report cannot be byte-identical across such permutations.
The same exact source bytes and options do produce byte-identical reports.

Inputs are bounded to 16 MiB per file, 64 KiB per physical row, 100,000 rows,
10,000 judged documents, grades and cutoffs at most 1,000, and at most 20
distinct cutoffs. Duplicate pairs, bad UTF-8/CSV, non-finite scores, malformed
grades, missing scored documents, and output/input aliases fail before output.
The output is written through a temporary file and atomically replaced after
validation; a failed installation leaves an existing report intact. Input IDs
appear in the per-document report, so keep the report private if identifiers
themselves are sensitive.

This slice does not implement hyperparameter search, cross-validation,
real-time inference, or parity with any external benchmark score.
