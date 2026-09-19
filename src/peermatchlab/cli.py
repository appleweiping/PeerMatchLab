"""Command-line interface for validating, scoring, matching, and auditing."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from peermatchlab import __version__
from peermatchlab.affinity import load_affinities_csv
from peermatchlab.audit import audit_plan
from peermatchlab.config import MatchConfig
from peermatchlab.expertise import ExpertiseConfig, generate_expertise
from peermatchlab.expertise_io import (
    load_expertise_config_source,
    load_local_domain_expertise_inputs,
    load_openreview_expertise_snapshot,
    write_expertise_run,
)
from peermatchlab.io import (
    load_conflicts,
    load_documents,
    load_experts,
    load_json_text,
    plan_from_dict,
    plan_to_dict,
    write_json,
)
from peermatchlab.models import Conflict, DataValidationError, Document, Expert
from peermatchlab.openreview import load_openreview_submissions, load_reviewer_ids
from peermatchlab.openreview_api import (
    DEFAULT_OPENREVIEW_API_V2_URL,
    ExpertiseFetchPolicy,
    OpenReviewClient,
    OpenReviewClientConfig,
    OpenReviewError,
    RetryPolicy,
    fetch_openreview_snapshot,
    fetch_reviewer_expertise,
    write_openreview_snapshot,
)
from peermatchlab.pipeline import MatchRun, run_affinity_matching, run_matching
from peermatchlab.report import write_html


def _inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--documents", required=True, help="JSON or JSONL documents")
    parser.add_argument("--experts", required=True, help="JSON or JSONL experts")
    parser.add_argument("--conflicts", help="optional JSON or JSONL hard conflicts")


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser, separated for documentation and tests."""

    parser = argparse.ArgumentParser(
        prog="peermatch", description="Transparent, constraint-aware expert matching"
    )
    parser.add_argument("--version", action="version", version=f"PeerMatchLab {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate input files without matching")
    _inputs(validate)

    score = commands.add_parser("score", help="show the evidence score for one pair")
    _inputs(score)
    score.add_argument("--document-id", required=True)
    score.add_argument("--expert-id", required=True)
    score.add_argument("--config")

    match = commands.add_parser("match", help="create assignments and an audit report")
    _inputs(match)
    match.add_argument("--config", help="optional JSON run configuration")
    match.add_argument("--output", required=True, help="result JSON path")
    match.add_argument("--scores", help="optional full score-matrix JSON path")
    match.add_argument("--html", help="optional self-contained HTML report path")

    affinity = commands.add_parser("match-affinity", help="assign a sparse external affinity CSV")
    _inputs(affinity)
    affinity.add_argument(
        "--affinities",
        required=True,
        help="CSV rows containing document_id,expert_id,score; canonical header optional",
    )
    affinity.add_argument("--config", help="optional JSON run configuration")
    affinity.add_argument("--output", required=True, help="result JSON path")
    affinity.add_argument("--html", help="optional self-contained HTML report path")

    audit = commands.add_parser("audit", help="independently audit an existing plan")
    _inputs(audit)
    audit.add_argument("--plan", required=True)
    audit.add_argument("--default-demand", type=int, default=2)
    audit.add_argument(
        "--require-distinct-institutions",
        action="store_true",
        help="treat repeated known institutions within a document as unsafe",
    )

    openreview = commands.add_parser(
        "import-openreview",
        help="convert local OpenReview submission JSONL and reviewer IDs",
    )
    openreview.add_argument("--submissions", required=True)
    openreview.add_argument("--reviewers", required=True)
    openreview.add_argument("--reviewer-capacity", type=int, required=True)
    openreview.add_argument("--max-records", type=int, default=100_000)
    openreview.add_argument("--max-input-file-bytes", type=int, default=64 * 1024 * 1024)
    openreview.add_argument("--max-line-bytes", type=int, default=8 * 1024 * 1024)
    openreview.add_argument("--directory", required=True)

    fetch_openreview = commands.add_parser(
        "fetch-openreview",
        help="fetch a bounded OpenReview API v2 paper/reviewer snapshot",
    )
    paper_filter = fetch_openreview.add_mutually_exclusive_group(required=True)
    paper_filter.add_argument("--invitation", help="submission invitation ID")
    paper_filter.add_argument("--venue-id", help="submission content.venueid value")
    fetch_openreview.add_argument("--reviewer-group", required=True)
    fetch_openreview.add_argument("--reviewer-capacity", type=int, required=True)
    fetch_openreview.add_argument("--directory", required=True)
    fetch_openreview.add_argument("--base-url", default=DEFAULT_OPENREVIEW_API_V2_URL)
    fetch_openreview.add_argument(
        "--token-env",
        help="read a bearer token from this environment variable; never writes it to disk",
    )
    fetch_openreview.add_argument("--page-size", type=int, default=1000)
    fetch_openreview.add_argument("--max-records", type=int, default=100_000)
    fetch_openreview.add_argument("--max-pages", type=int, default=1000)
    fetch_openreview.add_argument("--max-total-requests", type=int, default=100_000)
    fetch_openreview.add_argument("--max-attempts", type=int, default=5)
    fetch_openreview.add_argument("--requests-per-second", type=float, default=4.0)
    fetch_openreview.add_argument("--timeout-seconds", type=float, default=30.0)
    fetch_openreview.add_argument(
        "--fetch-expertise",
        action="store_true",
        help="also fetch exact reviewer profiles and explicitly attributed publications",
    )
    fetch_openreview.add_argument(
        "--publication-invitation",
        action="append",
        default=[],
        help="allowed exact publication invitation; repeatable and required with --fetch-expertise",
    )
    fetch_openreview.add_argument("--minimum-publication-date-ms", type=int)
    fetch_openreview.add_argument("--maximum-publication-date-ms", type=int)
    fetch_openreview.add_argument("--require-publication-abstract", action="store_true")
    fetch_openreview.add_argument("--max-scanned-publications", type=int, default=100_000)
    fetch_openreview.add_argument("--max-publications", type=int, default=100_000)
    fetch_openreview.add_argument("--max-publications-per-reviewer", type=int, default=10_000)

    expertise = commands.add_parser(
        "expertise",
        help="generate local, explainable TF-IDF or BM25 paper-reviewer affinities",
    )
    source = expertise.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--snapshot",
        help="directory containing submissions, profiles, and reviewer-publication JSONL",
    )
    source.add_argument("--documents", help="PeerMatchLab document JSON or JSONL fixtures")
    expertise.add_argument("--experts", help="PeerMatchLab expert JSON or JSONL fixtures")
    expertise.add_argument("--config", help="optional expertise JSON configuration")
    expertise.add_argument(
        "--reviewer-capacity",
        type=int,
        default=1,
        help="capacity assigned to snapshot profiles (default: 1)",
    )
    expertise.add_argument("--directory", required=True, help="new artifact directory")
    return parser


def _load(
    args: argparse.Namespace,
) -> tuple[tuple[Document, ...], tuple[Expert, ...], tuple[Conflict, ...]]:
    documents = load_documents(args.documents)
    experts = load_experts(args.experts)
    conflicts = load_conflicts(args.conflicts)
    if not documents:
        raise DataValidationError("at least one document is required")
    if not experts:
        raise DataValidationError("at least one expert is required")
    document_ids = {item.id for item in documents}
    expert_ids = {item.id for item in experts}
    unknown_documents = sorted(
        {item.document_id for item in conflicts if item.document_id not in document_ids}
    )
    unknown_experts = sorted(
        {item.expert_id for item in conflicts if item.expert_id not in expert_ids}
    )
    if unknown_documents or unknown_experts:
        raise DataValidationError(
            "conflicts reference unknown identifiers: "
            f"documents={unknown_documents}, experts={unknown_experts}"
        )
    return documents, experts, conflicts


def _config(path: str | None) -> MatchConfig:
    return MatchConfig.from_json(path) if path else MatchConfig()


def _import_openreview(args: argparse.Namespace) -> int:
    imported_documents = load_openreview_submissions(
        args.submissions,
        max_records=args.max_records,
        max_input_file_bytes=args.max_input_file_bytes,
        max_line_bytes=args.max_line_bytes,
    )
    imported_experts = load_reviewer_ids(
        args.reviewers,
        capacity=args.reviewer_capacity,
        max_reviewers=args.max_records,
        max_input_file_bytes=args.max_input_file_bytes,
        max_line_bytes=args.max_line_bytes,
    )
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / "documents.json",
        [
            {
                "id": item.id,
                "title": item.title,
                "abstract": item.abstract,
                "topics": list(item.topics),
                "keywords": list(item.keywords),
                "metadata": dict(item.metadata),
            }
            for item in imported_documents
        ],
    )
    write_json(
        directory / "experts.json",
        [
            {
                "id": item.id,
                "name": item.name,
                "capacity": item.capacity,
                "metadata": dict(item.metadata),
            }
            for item in imported_experts
        ],
    )
    write_json(
        directory / "metadata.json",
        {
            "adapter": "openreview-local-export-v1",
            "documents": len(imported_documents),
            "experts": len(imported_experts),
            "limitations": [
                "no network requests or OpenReview authentication are performed",
                "reviewer files provide identifiers and capacity, not expertise text",
                "use match-affinity with a separately generated sparse score CSV",
            ],
        },
    )
    print(
        f"converted {len(imported_documents)} submissions and "
        f"{len(imported_experts)} reviewers to {directory}"
    )
    return 0


def _fetch_openreview(args: argparse.Namespace) -> int:
    if not args.fetch_expertise and (
        args.publication_invitation
        or args.minimum_publication_date_ms is not None
        or args.maximum_publication_date_ms is not None
        or args.require_publication_abstract
        or args.max_scanned_publications != 100_000
        or args.max_publications != 100_000
        or args.max_publications_per_reviewer != 10_000
    ):
        raise DataValidationError("publication options require --fetch-expertise")
    if args.fetch_expertise and not args.publication_invitation:
        raise DataValidationError("--fetch-expertise requires --publication-invitation")
    policy = None
    if args.fetch_expertise:
        policy = ExpertiseFetchPolicy(
            invitations=tuple(args.publication_invitation),
            minimum_date_ms=args.minimum_publication_date_ms,
            maximum_date_ms=args.maximum_publication_date_ms,
            require_abstract=args.require_publication_abstract,
            max_scanned_notes=args.max_scanned_publications,
            max_publications=args.max_publications,
            max_publications_per_reviewer=args.max_publications_per_reviewer,
        )
    token: str | None = None
    if args.token_env is not None:
        if not isinstance(args.token_env, str) or not args.token_env.strip():
            raise DataValidationError("token environment variable name must not be empty")
        token = os.environ.get(args.token_env)
        if token is None:
            raise DataValidationError(
                f"token environment variable is not defined: {args.token_env}"
            )
    config = OpenReviewClientConfig(
        base_url=args.base_url,
        page_size=args.page_size,
        max_pages=args.max_pages,
        max_records=args.max_records,
        max_total_requests=args.max_total_requests,
        timeout_seconds=args.timeout_seconds,
        requests_per_second=args.requests_per_second,
        retry=RetryPolicy(max_attempts=args.max_attempts),
    )
    client = OpenReviewClient(config=config, token=token)
    snapshot = fetch_openreview_snapshot(
        client,
        invitation=args.invitation,
        venue_id=args.venue_id,
        reviewer_group=args.reviewer_group,
    )
    expertise = None
    if policy is not None:
        expertise = fetch_reviewer_expertise(client, snapshot, policy)
    manifest = write_openreview_snapshot(
        snapshot,
        args.directory,
        reviewer_capacity=args.reviewer_capacity,
        expertise=expertise,
    )
    records = manifest.get("records")
    if not isinstance(records, Mapping):
        raise DataValidationError("generated OpenReview manifest has invalid record counts")
    print(
        f"fetched {records['documents']} submissions and {records['experts']} reviewers "
        f"to {args.directory}"
    )
    return 0


def _expertise(args: argparse.Namespace) -> int:
    config_source = load_expertise_config_source(args.config) if args.config else None
    config = config_source.config if config_source is not None else ExpertiseConfig()
    if args.snapshot is not None:
        if args.experts is not None:
            raise DataValidationError("--experts cannot be combined with --snapshot")
        source = load_openreview_expertise_snapshot(
            args.snapshot,
            config=config,
            reviewer_capacity=args.reviewer_capacity,
        )
    else:
        if args.documents is None or args.experts is None:
            raise DataValidationError("--documents requires --experts")
        if args.reviewer_capacity != 1:
            raise DataValidationError(
                "--reviewer-capacity applies only to --snapshot; normalized experts own capacity"
            )
        source = load_local_domain_expertise_inputs(
            args.documents,
            args.experts,
            config=config,
        )
    run = generate_expertise(source.documents, source.experts, config=config)
    manifest = write_expertise_run(
        run,
        args.directory,
        source=source,
        config_source=config_source,
    )
    records = manifest["records"]
    if not isinstance(records, Mapping):
        raise DataValidationError("generated expertise manifest has invalid record counts")
    print(
        f"generated {records['emitted_pairs']} sparse affinities from "
        f"{records['candidate_pairs']} candidate pairs to {args.directory}"
    )
    return 0


def _print_match_summary(run: MatchRun, output: str) -> None:
    diagnostics = run.plan.diagnostics
    suffix = (
        f"; status={diagnostics.status.value}, unmet={diagnostics.unmet}, "
        f"certified={str(diagnostics.certified).lower()}"
        if diagnostics is not None
        else ""
    )
    print(f"wrote {len(run.plan.assignments)} assignments to {output}{suffix}")


def _same_file(left: str | Path, right: str | Path) -> bool:
    """Detect lexical, symlink, and hard-link aliases without requiring outputs to exist."""

    left_path = Path(left)
    right_path = Path(right)
    try:
        return os.path.samefile(left_path, right_path)
    except OSError:
        return left_path.resolve(strict=False) == right_path.resolve(strict=False)


def _protect_match_outputs(args: argparse.Namespace) -> None:
    if args.command not in {"match", "match-affinity"}:
        return
    inputs: list[tuple[str, str | Path]] = [
        ("documents", args.documents),
        ("experts", args.experts),
    ]
    for name in ("conflicts", "config"):
        value = getattr(args, name, None)
        if value is not None:
            inputs.append((name, value))
    if args.command == "match-affinity":
        inputs.append(("affinities", args.affinities))
    outputs = [
        (name, value)
        for name in ("output", "scores", "html")
        if (value := getattr(args, name, None)) is not None
    ]
    for output_name, output_path in outputs:
        for input_name, input_path in inputs:
            if _same_file(output_path, input_path):
                raise DataValidationError(
                    f"{output_name} path must not refer to the {input_name} input"
                )
    for index, (output_name, output_path) in enumerate(outputs):
        for other_name, other_path in outputs[index + 1 :]:
            if _same_file(output_path, other_path):
                raise DataValidationError(
                    f"{output_name} and {other_name} paths must refer to different files"
                )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-compatible exit code."""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "import-openreview":
            return _import_openreview(args)
        if args.command == "fetch-openreview":
            return _fetch_openreview(args)
        if args.command == "expertise":
            return _expertise(args)
        _protect_match_outputs(args)
        documents, experts, conflicts = _load(args)
        if args.command == "validate":
            print(
                json.dumps(
                    {
                        "documents": len(documents),
                        "experts": len(experts),
                        "conflicts": len(conflicts),
                        "valid": True,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command in {"match", "score"}:
            run = run_matching(documents, experts, conflicts=conflicts, config=_config(args.config))
            if args.command == "score":
                selected = next(
                    (
                        item
                        for item in run.scores
                        if item.document_id == args.document_id and item.expert_id == args.expert_id
                    ),
                    None,
                )
                if selected is None:
                    raise DataValidationError("unknown document/expert pair")
                print(
                    json.dumps(
                        {
                            "document_id": selected.document_id,
                            "expert_id": selected.expert_id,
                            "total": selected.total,
                            "eligible": selected.eligible,
                            "components": selected.component_map(),
                            "reasons": list(selected.reasons),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            output = plan_to_dict(run.plan)
            output["audit"] = run.audit.as_dict()
            write_json(args.output, output)
            if args.scores:
                write_json(
                    args.scores,
                    [
                        {
                            "document_id": item.document_id,
                            "expert_id": item.expert_id,
                            "total": item.total,
                            "eligible": item.eligible,
                            "components": item.component_map(),
                            "reasons": list(item.reasons),
                        }
                        for item in run.scores
                    ],
                )
            if args.html:
                write_html(args.html, run, documents, experts)
            _print_match_summary(run, args.output)
            return 0 if run.audit.safe else 2
        if args.command == "match-affinity":
            run = run_affinity_matching(
                documents,
                experts,
                load_affinities_csv(args.affinities),
                conflicts=conflicts,
                config=_config(args.config),
            )
            output = plan_to_dict(run.plan)
            output["audit"] = run.audit.as_dict()
            write_json(args.output, output)
            if args.html:
                write_html(args.html, run, documents, experts)
            _print_match_summary(run, args.output)
            return 0 if run.audit.safe else 2
        if args.command == "audit":
            raw_plan = load_json_text(Path(args.plan).read_text(encoding="utf-8"))
            if not isinstance(raw_plan, dict):
                raise DataValidationError("plan must be a JSON object")
            report = audit_plan(
                plan_from_dict(raw_plan),
                documents,
                experts,
                conflicts,
                default_demand=args.default_demand,
                require_distinct_institutions=args.require_distinct_institutions,
            )
            print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
            return 0 if report.safe else 2
    except (
        DataValidationError,
        OpenReviewError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        OSError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 1
