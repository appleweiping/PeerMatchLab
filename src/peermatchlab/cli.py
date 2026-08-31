"""Command-line interface for validating, scoring, matching, and auditing."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from peermatchlab.audit import audit_plan
from peermatchlab.config import MatchConfig
from peermatchlab.io import (
    load_conflicts,
    load_documents,
    load_experts,
    plan_from_dict,
    plan_to_dict,
    write_json,
)
from peermatchlab.models import Conflict, DataValidationError, Document, Expert
from peermatchlab.pipeline import run_matching
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
    parser.add_argument("--version", action="version", version="PeerMatchLab 0.1.0")
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

    audit = commands.add_parser("audit", help="independently audit an existing plan")
    _inputs(audit)
    audit.add_argument("--plan", required=True)
    audit.add_argument("--default-demand", type=int, default=2)
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-compatible exit code."""

    args = build_parser().parse_args(argv)
    try:
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
            print(f"wrote {len(run.plan.assignments)} assignments to {args.output}")
            return 0 if run.audit.safe else 2
        if args.command == "audit":
            raw_plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
            report = audit_plan(
                plan_from_dict(raw_plan),
                documents,
                experts,
                conflicts,
                default_demand=args.default_demand,
            )
            print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
            return 0 if report.safe else 2
    except (DataValidationError, ValueError, KeyError, json.JSONDecodeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 1
