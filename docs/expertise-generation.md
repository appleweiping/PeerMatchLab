# Local expertise generation

PeerMatchLab 0.6 adds a separate information-retrieval stage that turns local
submission and reviewer evidence into a sparse affinity CSV. It performs no
network requests and does not infer authorship, conflicts, or reviewer
identity. The output can be inspected, replayed, and then passed unchanged to
`peermatch match-affinity`.

## Inputs

The `expertise` command accepts either PeerMatchLab `documents.json` and
`experts.json` files or a local OpenReview-shaped snapshot directory:

```text
snapshot/
├── submissions.jsonl
├── profiles.jsonl
└── reviewer-publications.jsonl
```

`submissions.jsonl` contains OpenReview API-v1 or API-v2-shaped notes.
`profiles.jsonl` contains profile objects with `id`, `content.names`, and
optional `bio`, `expertise`, `research_interests`, and `keywords`. Scalar v2
content values may be wrapped as `{"value": ...}`.

Each line in `reviewer-publications.jsonl` is an explicit join:

```json
{
  "reviewer_id": "~Ada_Reviewer1",
  "note": {
    "id": "publication-1",
    "content": {
      "title": {"value": "Sparse retrieval"},
      "abstract": {"value": "Lexical ranking for scholarly search."},
      "year": {"value": 2024}
    }
  }
}
```

The join is intentional. PeerMatchLab never guesses a reviewer from author
names or publication text. Duplicate profile IDs, duplicate
reviewer/publication associations, unknown reviewer references, malformed v2
wrappers, duplicate JSON keys, and non-finite numbers fail closed. An optional
source `manifest.json` is deliberately not incorporated because it is not a
causal input to the join. Provenance includes only the three byte streams from
which the adapter can independently reconstruct every returned domain object.

The opt-in [read-only API v2 acquisition](openreview-sync.md#opt-in-expertise-acquisition)
now produces these three streams. It verifies each retrieved publication's
`content.authorids` against the exact reviewer profile ID before writing the
explicit join. Existing hand-authored offline snapshots retain their original
contract; no network access occurs during `expertise`.

Normalized `experts.json` publications may likewise carry an optional `id`.
It is preserved in evidence and max-score explanations; duplicate IDs within
one reviewer are rejected.

## Text and filtering contract

The complete configuration is embedded in both `model.json` and
`manifest.json`. Defaults are explicit:

- tokenizer: Unicode alphanumeric runs excluding underscores, Unicode
  case-folding, minimum length two, plus persisted ceilings for document
  characters/UTF-8 bytes, scanned matches, and token characters;
- stopwords: a fixed, persisted English function-word list;
- submission fields: `title`, `abstract`, `topics`, and `keywords`;
- profile fields: `summary`, `topics`, and `keywords`;
- publication fields: `title` and `abstract`;
- dates: undated publications included, with no minimum or maximum year;
- content: a profile/publication must retain at least one token; stopword-only
  submissions remain present and receive zero scores;
- duplicates: exact case-folded title/abstract/year duplicates within one
  reviewer are counted once;
- resource ceilings: source `max_input_file_bytes`, persisted-model
  `max_model_file_bytes`, records, publications per reviewer, document
  characters/bytes, regex matches, token length, tokens per document and corpus,
  vocabulary terms, candidate pairs, and total explanation contributions. The
  two byte controls are independent: lowering a corpus-input ceiling does not
  make an otherwise valid fitted model unloadable.

`require_submission_abstract`, `require_publication_abstract`, `minimum_publication_year`,
`maximum_publication_year`, and `undated_publications` make common content and
date policies reproducible. Filtering happens before global statistics are
fit, and filter counts are written to the manifest. When either required-abstract
control is enabled, `abstract` must also be present in the corresponding selected
field list. A persisted model can therefore recheck the retained evidence and new
queries. It cannot reconstruct source publications filtered out before fitting;
the source digests and manifest filter counts are the audit record for that step.

## Algorithms and aggregation

TF-IDF uses reviewer evidence as the fitted document collection. For term
`t`, evidence document `d`, and corpus size `N`:

```text
idf(t) = log((1 + N) / (1 + df(t))) + 1
tfidf(t, d) = (1 + log(tf(t, d))) × idf(t)
```

Paper/evidence similarity is sparse cosine similarity. Every reported term
contribution is the corresponding normalized dot-product term and therefore
sums to the reported score.

BM25 uses Robertson's positive IDF:

```text
idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(q, d) = Σ idf(t) × tf(t,d)(k1+1)
                         / (tf(t,d) + k1(1-b+b|d|/avgdl))
```

Query terms are binary: repeating a word in a submission does not multiply
its BM25 query contribution. The additive `raw_score` is mapped into the
assignment affinity interval by `raw_score / (1 + raw_score)`. Explanations
retain raw additive contributions, so no term weight is hidden by the
normalization.

Three genuinely different reviewer representations are supported:

- `aggregate`: concatenate all eligible profile/publication units for a
  reviewer, fit one corpus document per non-empty reviewer, and score once;
- `max`: fit atomic evidence documents and retain the best evidence score;
- `average`: fit atomic evidence documents and average every eligible evidence
  score, including non-matching zeroes.

Profile evidence participates as its own unit when `include_profile` is true.
Set it to false for publication-only models. Empty reviewers remain in the
candidate set with a finite zero score but do not alter IDF or average document
length.

These formulas and preprocessing rules are PeerMatchLab's documented contract;
they are not claimed to be numerically interchangeable with a particular
OpenReview Expertise deployment. The three-column affinity output is
interoperable, while tokenization, corpus selection, BM25 normalization, and
filters must be compared explicitly before scores from different generators
are mixed.

## CLI and artifacts

Run the checked-in offline example:

```bash
peermatch expertise \
  --snapshot examples/expertise/snapshot \
  --config examples/expertise/config.json \
  --reviewer-capacity 3 \
  --directory scratch/expertise

peermatch match-affinity \
  --documents scratch/expertise/documents.json \
  --experts scratch/expertise/experts.json \
  --affinities scratch/expertise/affinities.csv \
  --config examples/expertise/match-config.json \
  --output scratch/plan.json
```

The destination must not exist and is installed atomically only after every
file succeeds:

- `documents.json` and `experts.json`: normalized assignment inputs;
- `submission-documents.jsonl` and `reviewer-documents.jsonl`: selected fields
  plus exact tokens used by the model;
- `affinities.csv`: positive paper/reviewer scores at or above
  `minimum_output_score`;
- `explanations.jsonl`: model, aggregation, raw and normalized score, evidence
  count, selected evidence (for aggregate/max), and every non-zero term
  contribution;
- `model.json`: schema-versioned configuration, evidence tokens, IDF, document
  count, and average length; loading re-tokenizes selected fields, recomputes
  statistics, rechecks total-publication limits and required abstracts on retained
  evidence, and rejects disagreement;
- `manifest.json`: input and derived byte counts/SHA-256 digests, generator
  version, configuration, filter counts, corpus statistics, and record counts.

The manifest cannot hash itself without a recursive definition; it hashes
every other derived artifact and every causal source byte stream instead. The
adapter name and all parsing/join limits are stored beside those digests, and
the writer re-derives the domain objects from its immutable captured bytes
before publishing. Existing
destinations are never overwritten. A failure during any write removes the
staging directory even for interpreter-level interruption, so a previous run
cannot be mixed with partial new output.
The CLI reads a configuration once into an immutable bounded byte snapshot,
parses and hashes that same snapshot, re-parses those bytes when the artifact is
published, and confirms that the on-disk file is unchanged immediately before
installation. The re-derived configuration must equal the fitted model configuration.
The staged model is also loaded and rescored before the directory is installed.

Generated publication evidence IDs use disjoint `id:` and `position:`
namespaces. An explicit identifier such as `"2"` therefore cannot collide with
the fallback for the second publication, and the corpus builder rejects any
remaining evidence-ID collision before fitting.

The same contract is available as a typed Python API:

```python
from peermatchlab import ExpertiseConfig, ExpertiseModel, generate_expertise
from peermatchlab.io import load_documents, load_experts

run = generate_expertise(
    load_documents("documents.json"),
    load_experts("experts.json"),
    config=ExpertiseConfig(model="bm25", aggregation="max"),
)
run.model.save("model.json")

reloaded = ExpertiseModel.load("model.json")
scores = reloaded.score_documents(load_documents("new-submissions.json"))
```

`ExpertiseModel.save` renders its complete deterministic UTF-8 payload before
touching the destination and enforces `max_model_file_bytes` against those exact
bytes, including the trailing newline. The exact limit is accepted. Oversized
models leave both missing and existing destinations untouched; successful saves
replace one file atomically, and failed or interrupted writes remove their
temporary sibling. `ExpertiseModel.load` enforces the same persisted model-file
limit. Models and configuration files from before this control was added remain
loadable and receive the bounded default.

`ExpertiseModel.fit` also accepts a corpus returned by
`build_expertise_corpus`; it rejects a configuration different from the one
that created the corpus, so persisted tokenizer and filter semantics cannot
drift silently.

## Safety boundary

Lexical similarity is evidence, not truth. It may amplify publication-volume,
language, field-vocabulary, and profile-completeness differences. Operators
must inspect explanations, declare conflicts separately, choose date/content
filters suitable for the venue, protect private profile and submission data,
and retain human review and appeal paths. A high score never proves competence
or absence of conflict; a zero score can mean missing or filtered evidence.
