"""Bounded, deterministic graph-ranked keywords for local matching evidence.

This is a lexical TextRank-style preprocessing outcome, not POS tagging,
lemmatization, or a claim of numerical equivalence with OpenReview Expertise.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from peermatchlab.expertise_io import _install_directory_no_replace
from peermatchlab.io import load_documents_text, load_experts_text
from peermatchlab.models import DataValidationError, Document, Expert

_TOKENS = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCES = re.compile(r"[^.!?\n]+", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "using",
        "with",
    }
)
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_RECORDS = 10_000
_MAX_TEXT_CHARACTERS = 1_000_000
_MAX_TOKENS = 100_000
_MAX_TERMS = 20_000
_MAX_EDGES = 200_000
_MAX_WORK = 10_000_000
_MAX_OUTPUT_BYTES = 16 * 1024 * 1024


def _positive_int(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise DataValidationError(f"{name} must be an integer in [1, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class KeyphraseConfig:
    """Finite graph and artifact limits for one lexical extraction run."""

    window_size: int = 2
    top_k: int = 20
    iterations: int = 30
    damping: float = 0.85
    max_records: int = _MAX_RECORDS
    max_text_characters: int = _MAX_TEXT_CHARACTERS
    max_tokens: int = _MAX_TOKENS
    max_terms: int = _MAX_TERMS
    max_edges: int = _MAX_EDGES
    max_work: int = _MAX_WORK

    def __post_init__(self) -> None:
        for name, maximum in (
            ("window_size", 16),
            ("top_k", 1_000),
            ("iterations", 100),
            ("max_records", _MAX_RECORDS),
            ("max_text_characters", _MAX_TEXT_CHARACTERS),
            ("max_tokens", _MAX_TOKENS),
            ("max_terms", _MAX_TERMS),
            ("max_edges", _MAX_EDGES),
            ("max_work", _MAX_WORK),
        ):
            _positive_int(getattr(self, name), name, maximum)
        if self.window_size < 2:
            raise DataValidationError("window_size must be at least 2")
        try:
            damping = float(self.damping) if type(self.damping) in (int, float) else float("nan")
        except (TypeError, ValueError, OverflowError):
            damping = float("nan")
        if not math.isfinite(damping) or not 0.0 < damping < 1.0:
            raise DataValidationError("damping must be finite and strictly between zero and one")
        object.__setattr__(self, "damping", damping)

    def as_dict(self) -> dict[str, object]:
        return {
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
            "tokenizer": "unicode-casefold-alnum-v1",
            "stopwords": sorted(_STOPWORDS),
        }


@dataclass(frozen=True, slots=True)
class Keyphrase:
    term: str
    score: float


@dataclass(frozen=True, slots=True)
class KeyphraseRecord:
    kind: str
    owner_id: str
    evidence_id: str
    token_count: int
    terms: tuple[Keyphrase, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "owner_id": self.owner_id,
            "evidence_id": self.evidence_id,
            "token_count": self.token_count,
            "keyphrases": [{"term": term.term, "score": term.score} for term in self.terms],
        }


def rank_keyphrases(text: str, *, config: KeyphraseConfig | None = None) -> tuple[Keyphrase, ...]:
    """Rank a simple undirected co-occurrence graph with normalized PageRank.

    One edge is included per distinct unordered pair in each token window.
    All retained terms, including isolates, receive teleportation mass;
    dangling mass is uniformly redistributed. Equal scores break by term.
    """

    chosen = config or KeyphraseConfig()
    if type(text) is not str or len(text) > chosen.max_text_characters:
        raise DataValidationError("keyphrase text exceeds its character limit or is not a string")
    sentences: list[tuple[str, ...]] = []
    vocabulary: set[str] = set()
    token_count = 0
    for sentence in _SENTENCES.finditer(text.casefold()):
        words: list[str] = []
        for match in _TOKENS.finditer(sentence.group()):
            term = match.group()
            if len(term) > 1 and term not in _STOPWORDS:
                token_count += 1
                if token_count > chosen.max_tokens:
                    raise DataValidationError("keyphrase text exceeds max_tokens")
                words.append(term)
                vocabulary.add(term)
        if words:
            sentences.append(tuple(words))
    if len(vocabulary) > chosen.max_terms:
        raise DataValidationError("keyphrase text exceeds max_terms")
    if not vocabulary:
        return ()
    neighbors: dict[str, set[str]] = {term: set() for term in vocabulary}
    edges = 0
    for sentence_words in sentences:
        for index, left in enumerate(sentence_words):
            for right in sentence_words[index + 1 : index + chosen.window_size]:
                if left == right or right in neighbors[left]:
                    continue
                neighbors[left].add(right)
                neighbors[right].add(left)
                edges += 1
                if edges > chosen.max_edges:
                    raise DataValidationError("keyphrase graph exceeds max_edges")
    if chosen.iterations * (2 * edges + len(vocabulary)) > chosen.max_work:
        raise DataValidationError("keyphrase graph exceeds max_work")
    terms = sorted(vocabulary)
    count = len(terms)
    score = {term: 1.0 / count for term in terms}
    for _ in range(chosen.iterations):
        dangling = sum(score[term] for term in terms if not neighbors[term])
        base = (1.0 - chosen.damping + chosen.damping * dangling) / count
        next_score = {term: base for term in terms}
        for term in terms:
            links = neighbors[term]
            if links:
                portion = chosen.damping * score[term] / len(links)
                for neighbor in sorted(links):
                    next_score[neighbor] += portion
        score = next_score
    return tuple(
        Keyphrase(term, score[term])
        for term in sorted(terms, key=lambda value: (-score[value], value))[: chosen.top_k]
    )


def extract_keyphrases(
    documents: Iterable[Document],
    experts: Iterable[Expert],
    *,
    config: KeyphraseConfig | None = None,
) -> tuple[KeyphraseRecord, ...]:
    """Extract submission, reviewer profile, and publication evidence separately."""

    chosen = config or KeyphraseConfig()
    result: list[KeyphraseRecord] = []
    total_work = 0

    def add(kind: str, owner: str, evidence: str, text: str) -> None:
        nonlocal total_work
        if len(result) >= chosen.max_records:
            raise DataValidationError("keyphrase run exceeds max_records")
        if len(text) > chosen.max_text_characters:
            raise DataValidationError("keyphrase text exceeds max_text_characters")
        tokens = [
            match.group()
            for match in _TOKENS.finditer(text.casefold())
            if len(match.group()) > 1 and match.group() not in _STOPWORDS
        ]
        if len(tokens) > chosen.max_tokens:
            raise DataValidationError("keyphrase text exceeds max_tokens")
        total_work += chosen.iterations * (
            2 * len(tokens) * (chosen.window_size - 1) + len(set(tokens))
        )
        if total_work > chosen.max_work:
            raise DataValidationError("keyphrase run exceeds max_work")
        result.append(
            KeyphraseRecord(
                kind, owner, evidence, len(tokens), rank_keyphrases(text, config=chosen)
            )
        )

    document_ids: set[str] = set()
    for document in documents:
        if not isinstance(document, Document) or document.id in document_ids:
            raise DataValidationError("keyphrase documents must have unique valid identifiers")
        document_ids.add(document.id)
        add(
            "submission",
            document.id,
            document.id,
            "\n".join((document.title, document.abstract, *document.topics, *document.keywords)),
        )
    expert_ids: set[str] = set()
    for expert in experts:
        if not isinstance(expert, Expert) or expert.id in expert_ids:
            raise DataValidationError("keyphrase experts must have unique valid identifiers")
        expert_ids.add(expert.id)
        add(
            "profile",
            expert.id,
            expert.id,
            "\n".join((expert.summary, *expert.topics, *expert.keywords)),
        )
        publication_ids: set[str] = set()
        for index, publication in enumerate(expert.publications):
            evidence_id = (
                f"id:{publication.id}" if publication.id is not None else f"position:{index}"
            )
            if evidence_id in publication_ids:
                raise DataValidationError("duplicate publication evidence identifier")
            publication_ids.add(evidence_id)
            add(
                "publication",
                expert.id,
                evidence_id,
                "\n".join((publication.title, publication.abstract)),
            )
    return tuple(sorted(result, key=lambda item: (item.kind, item.owner_id, item.evidence_id)))


def read_keyphrase_source(path: str | Path) -> bytes:
    """Read at most the hard source ceiling before parsing or hashing."""

    with Path(path).open("rb") as stream:
        data = stream.read(_MAX_SOURCE_BYTES + 1)
    if len(data) > _MAX_SOURCE_BYTES:
        raise DataValidationError("keyphrase source exceeds its byte limit")
    return data


def _decode_source(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataValidationError("keyphrase source must be UTF-8") from error


def _json_bytes(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return (rendered + "\n").encode("utf-8")
    except (UnicodeError, RecursionError, ValueError) as error:
        raise DataValidationError("keyphrase artifact cannot be encoded as JSON") from error


def write_keyphrase_run(
    directory: str | Path,
    *,
    documents_source: bytes,
    experts_source: bytes,
    config: KeyphraseConfig | None = None,
) -> dict[str, object]:
    """Publish a no-overwrite, checksummed local preprocessing artifact."""

    chosen = config or KeyphraseConfig()
    if not isinstance(documents_source, bytes) or not isinstance(experts_source, bytes):
        raise DataValidationError("keyphrase sources must be immutable byte snapshots")
    if max(len(documents_source), len(experts_source)) > _MAX_SOURCE_BYTES:
        raise DataValidationError("keyphrase source exceeds its byte limit")
    documents = load_documents_text(
        _decode_source(documents_source), max_records=chosen.max_records
    )
    experts = load_experts_text(_decode_source(experts_source), max_records=chosen.max_records)
    records = extract_keyphrases(documents, experts, config=chosen)
    output_buffer = bytearray()
    for record in records:
        line = _json_bytes(record.as_dict())
        if len(output_buffer) + len(line) > _MAX_OUTPUT_BYTES:
            raise DataValidationError("keyphrase output exceeds its byte limit")
        output_buffer.extend(line)
    output = bytes(output_buffer)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "algorithm": "lexical-undirected-pagerank-v1",
        "config": chosen.as_dict(),
        "records": len(records),
        "documents_sha256": hashlib.sha256(documents_source).hexdigest(),
        "experts_sha256": hashlib.sha256(experts_source).hexdigest(),
        "keyphrases_sha256": hashlib.sha256(output).hexdigest(),
        "keyphrases_bytes": len(output),
    }
    destination = Path(directory)
    if destination.exists() or destination.is_symlink():
        raise DataValidationError("keyphrase destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as stage:
        staging = Path(stage)
        (staging / "keyphrases.jsonl").write_bytes(output)
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        _install_directory_no_replace(staging, destination)
    return manifest
