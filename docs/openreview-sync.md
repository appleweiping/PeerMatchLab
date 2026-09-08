# OpenReview API v2 synchronization

PeerMatchLab's OpenReview synchronization is a small read-only protocol boundary. It retrieves the
two inputs needed to start a matching run—submission notes and reviewer identifiers—without coupling
the assignment engine to a live service or pretending that identifiers are expertise profiles.

## Protocol contract

The client accepts an origin-only HTTPS base URL and performs only these requests:

| Resource | Query contract | Accepted payload |
|---|---|---|
| `GET /notes` | exactly one of `invitation` or `content.venueid`; `sort=id`; bounded `limit`; `count=true` on the first page; then `after=<last id>` | object with a non-negative integer `count` on page one and an array of unique note objects with non-empty string IDs |
| `GET /groups` | exact `id` and `limit=2` | exactly one group whose ID matches and whose direct `members` are unique non-empty strings |

OpenReview documents `limit`, `offset`, `after`, and `count` on `/notes` in its official
[API v2 OpenAPI definition](https://docs.openreview.net/reference/api-v2/openapi-definition). The
official [data-retrieval guide](https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc)
uses `get_all_notes` with invitation and content filters. The official
[`openreview-py` client](https://github.com/openreview/openreview-py) orders full retrieval by ID and
advances using the last ID as the `after` cursor; it also retries 429/500/502/503/504 responses.

PeerMatchLab independently implements that narrow public protocol with only the Python standard
library. It does not import or vendor `openreview-py` and does not claim API coverage beyond the two
GET resources above.

## Completeness and resource bounds

Cursor pagination is used instead of a moving numeric offset. The first page's count is recorded as
the expected total. Synchronization stops only when that exact number of unique notes has arrived.
It raises `OpenReviewProtocolError` for a missing or invalid count, duplicate or non-increasing IDs,
an oversized page, a short or empty page before the count is reached, records beyond the declared
count, or exhaustion of `max_pages`/`max_records`.

Each HTTP body has a byte limit before JSON parsing. JSON must be UTF-8, objects may not contain
duplicate keys, and non-finite JSON numbers are rejected by the shared strict parser. Response
bodies are not copied into HTTP exceptions, limiting accidental disclosure of private venue data.
The immutable combined snapshot also has one aggregate expanded-container-item budget and one
aggregate UTF-8 text-byte budget shared by the paper filter, every note, and reviewer identifiers,
in addition to its per-container and nesting-depth ceilings. Repeated references are frozen once but
charged at their fully expanded JSON cost, so aliased deep/wide Python inputs cannot multiply into an
unbounded copy or later serialization.

## Retry and request pacing

`RetryPolicy` has a finite attempt count. Transport failures and HTTP
429/500/502/503/504 are retried with capped exponential delays. A numeric or HTTP-date
`Retry-After` header takes precedence and is capped separately. Other HTTP statuses fail
immediately. Proactive pacing applies a minimum interval derived from `requests_per_second` to every
attempt, including retries.

Transport, sleeper, monotonic clock, and wall clock are constructor dependencies. Tests therefore
verify exact URLs, headers, pagination cursors, delays, attempt counts, and terminal failures without
sleeping or contacting OpenReview.

## Authentication and data handling

The CLI accepts `--token-env NAME`, never a token value. It reads the named environment variable and
adds `Authorization: Bearer ...`; the token is not placed in query parameters, filesystem output, or
error messages. Only HTTPS origins without URL credentials, paths, query strings, or fragments are
accepted.

The output directory is staged beside its destination and renamed into place only after every raw,
converted, hashed, and manifest file succeeds. Existing destinations are refused. The manifest
records the paper filter, reviewer group, reviewer-capacity conversion input, record counts, byte
counts, and SHA-256 file digests. It
contains no retrieval timestamp, so identical source records produce byte-identical evidence files.
Fetched note mappings are deep-copied into immutable built-in containers once before either the raw
or converted file is written. Both representations therefore derive from one stable snapshot even
when a custom API transport returns stateful mapping objects. Any exception, including an interpreter-
level interruption, removes the unpublished staging directory while leaving an existing destination intact.

Private submissions and reviewer identities remain governed by venue policy. Operators must secure
the snapshot directory, verify authorization, derive conflicts separately, and use an actual
expertise model or affinity source before assignment. Direct group members are preserved in server
order; nested group expansion and profile/publication retrieval are intentionally outside this
protocol version.

The separate [local expertise-generation contract](expertise-generation.md) accepts profiles and
explicit reviewer-publication joins that an authorized operator has already synchronized. That
offline adapter does not expand the live API client's scope or silently treat the reviewer shells
created here as textual expertise.

## CLI example

```bash
peermatch fetch-openreview \
  --venue-id 'Venue.cc/2026/Conference' \
  --reviewer-group 'Venue.cc/2026/Conference/Reviewers' \
  --reviewer-capacity 4 \
  --token-env OPENREVIEW_TOKEN \
  --page-size 500 \
  --max-records 10000 \
  --max-attempts 5 \
  --requests-per-second 4 \
  --timeout-seconds 30 \
  --directory scratch/venue-snapshot
```

The generated `documents.json` and `experts.json` can be passed directly to `match-affinity` along
with an independently generated sparse affinity CSV.

## Primary-source ledger

Sources were accessed on 2026-09-07. The research stopped after the endpoint schema, complete-note
iteration behavior, retry status set, and Expertise input boundary were each supported by a current
first-party source; further searches would not change this deliberately narrow protocol.

| Supported claim | First-party source | Revision/access note |
|---|---|---|
| API v2 `/notes` and `/groups` schemas; `limit`, `offset`, `after`, and `count` parameters | OpenReview, [API v2 OpenAPI definition](https://docs.openreview.net/reference/api-v2/openapi-definition) | Live documentation accessed 2026-09-07 |
| Invitation/content retrieval patterns for all submissions | OpenReview, [How to Get all Notes](https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc) | Live guide accessed 2026-09-07 |
| ID ordering and `after=<last id>` iteration; retry on 429/500/502/503/504 | OpenReview, [`openreview-py`](https://github.com/openreview/openreview-py/tree/d53d9b3c272f204c871cfb9dc9104666b6f29789) | Commit `d53d9b3`, 2026-09-04; inspected `openreview/api/client.py` and `openreview/tools.py` |
| Expertise datasets can select papers by invitation or venue ID and reviewers by group/IDs | OpenReview, [`openreview-expertise`](https://github.com/openreview/openreview-expertise/tree/3e2803a5120aed5be3d762c98d8dd2e34a636232) | Commit `3e2803a`, 2026-08-31; inspected README configuration contract |
| Matcher provides solver and asynchronous service surfaces beyond this synchronization slice | OpenReview, [`openreview-matcher`](https://github.com/openreview/openreview-matcher/tree/e6a2dad82880b45560d5b09b5b2236bec13a6cec) | Commit `e6a2dad`, 2026-08-26; inspected README solver/service inventory |
