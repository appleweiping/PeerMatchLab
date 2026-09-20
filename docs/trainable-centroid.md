# Bounded trainable keyphrase-centroid expertise

`peermatch expertise-centroid` is an original, dependency-free offline
training/inference workflow. It consumes a **previously generated**
[`extract-keyphrases` run](keyphrases.md), the exact document/expert source
bytes named in that run, and caller-authored train/validation triplet JSONL.
It does not download data, call a provider, run PyTorch or the frozen OpenReview
Expertise model, or reproduce its official gold-data scores.

## Synthetic end-to-end run

These files are fictional. From the repository root:

```bash
peermatch extract-keyphrases \
  --documents examples/centroid/documents.json \
  --experts examples/centroid/experts.json \
  --directory centroid-keyphrases

peermatch expertise-centroid \
  --keyphrases centroid-keyphrases \
  --documents examples/centroid/documents.json \
  --experts examples/centroid/experts.json \
  --train examples/centroid/train.jsonl \
  --validation examples/centroid/validation.jsonl \
  --dimensions 2 --epochs 3 --seed 17 \
  --directory centroid-run

peermatch match-affinity \
  --documents centroid-run/documents.json \
  --experts centroid-run/experts.json \
  --conflicts centroid-run/conflicts.json \
  --affinities centroid-run/affinities.csv \
  --config examples/expertise/match-config.json \
  --output centroid-assignment.json
```

`--conflicts` can be supplied during centroid preparation. A positive
training/validation pair that is a declared hard conflict is rejected; the
holdout conflict subset is copied to the run for assignment. Inference emits
scores for all holdout document/reviewer pairs; `match-affinity` still applies
the hard constraints. Neither a high score nor a training label overrides a
declared conflict, capacity, seniority, or institution rule.

Both output directories are create-only. Choose new names for reruns; neither
command overwrites existing files. The run contains a versioned `model.json`,
`affinities.csv`, normalized **holdout-only** `documents.json`, `experts.json`,
`conflicts.json`, and an aggregate hash/byte/count `manifest.json`. The
checkpoint stores a learned vocabulary and weights, so a real venue must
handle it as private derived data; the manifest does not include raw text or
labels. Preserve the original inputs to replay or audit the declared hashes.

## Data separation and scoring

Each JSONL row is exactly:

```json
{"document_id":"d-train","positive_expert_id":"e-math","negative_expert_id":"e-graph"}
```

Train and validation must contain **disjoint submission IDs**. Repeated
triplets, contradictory labels for one submission/reviewer pair, missing
submission or reviewer keyphrases, and unknown IDs fail closed. Every source
byte stream is capped and SHA-256-bound to the keyphrase extraction manifest
or checkpoint. Vocabulary is formed only from training submissions and
reviewers occurring in training triplets. Reviewer evidence combines its
profile and publication keyphrases, deduplicating terms. Reviewer text may be
shared as *unlabeled features* across partitions; validation labels never
enter weight updates. A validation or holdout entity with no known term is
rejected rather than scored with untrained random coordinates. All
submission IDs absent from both label partitions are holdout scoring inputs.
The tool does **not** infer that this split is unbiased or temporally safe;
the caller must build partitions before examining holdout outcomes and avoid
source/author leakage across their own data.

For a document `q` and reviewer `r`, take the arithmetic mean of their
learned term vectors and score their dot product. A training triplet has
margin `q·positive - q·negative` and loss `log(1 + exp(-margin))`.
Synchronous, deterministic SGD differentiates through each centroid and
applies optional L2 weight decay to touched term vectors. The seed controls
local uniform initialization; training order is sorted by IDs. At the end of
each epoch, the untouched validation labels evaluate per-submission average
precision over *judged* positive/negative reviewers, with reviewer-ID tie
breaking. The first epoch attaining the highest validation MAP is selected;
validation never changes weights. Raw holdout dot products pass through a
logistic transform and are rounded to six decimals to satisfy the existing
`[0,1]` affinity contract. These numbers are **not calibrated probabilities**.

## Finite bounds and audit

Input files: keyphrase/document/expert sources at most 16 MiB each;
train/validation/conflicts at most 2 MiB each; keyphrase manifest 64 KiB.
At most 10,000 keyphrase records, 2,000 train and 500 validation triplets,
4,096 learned terms, 256 terms per entity, 32 dimensions, 50 epochs, and
10,000 holdout/reviewer affinities are accepted. A configured coordinate-work
bound defaults to 20 million and covers training, validation, and inference;
generated model and run byte bounds are independently enforced. Empty/missing
evidence, nonfinite coordinates/loss, invalid dimensions/IDs, forged source
hashes, or a changed input path before publication are errors.

`load_centroid_model` verifies the checkpoint's canonical payload digest and
shape; `verify_keyphrase_centroid` refits from the exact captured inputs and
compares checkpoint bytes. The run writer performs that replay and rechecks
source paths before publishing a staged directory with no replacement. An
independent hand calculation in the tests checks one pairwise BCE gradient
step; a separate two-submission ranking oracle checks MAP. Real-world use
still requires representative labels, licensing, bias/contamination review,
held-out evaluation, declared conflicts, and human oversight.

The frozen OpenReview Expertise centroid uses a different PyTorch training,
optimizer, batcher, keyphrase and evaluation pipeline. This slice closes a
local trainable-keyphrase outcome, **not** full comparator parity, TPMS,
multifacet neural scoring, SPECTER inference, or official gold evaluation.
