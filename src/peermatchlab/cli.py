"""Command-line interface for validating, scoring, matching, and auditing."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from peermatchlab import __version__
from peermatchlab.affinity import load_affinities_csv
from peermatchlab.audit import audit_plan
from peermatchlab.config import MatchConfig
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
    openreview.add_argument("--directory", required=True)
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
    imported_documents = load_openreview_submissions(args.submissions)
    imported_experts = load_reviewer_ids(args.reviewers, capacity=args.reviewer_capacity)
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


def _print_match_summary(run: MatchRun, output: str) -> None:
    diagnostics = run.plan.diagnostics
    suffix = (
        f"; status={diagnostics.status.value}, unmet={diagnostics.unmet}, "
        f"certified={str(diagnostics.certified).lower()}"
        if diagnostics is not None
        else ""
    )
    print(f"wrote {len(run.plan.assignments)} assignments to {output}{suffix}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-compatible exit code."""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "import-openreview":
            return _import_openreview(args)
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
    except (DataValidationError, ValueError, KeyError, json.JSONDecodeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 1
