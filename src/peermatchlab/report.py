"""Self-contained HTML reports for human review and demonstrations."""

from __future__ import annotations

import html
from collections.abc import Iterable
from pathlib import Path

from peermatchlab.models import Document, Expert
from peermatchlab.pipeline import MatchRun

_REASON_LABELS = {
    "hard_conflict": "hard-conflict exclusions",
    "other_ineligible": "other scorer-declared ineligible pairs",
    "zero_capacity": "zero-capacity experts",
    "minimum_score": "minimum-score filtering",
    "sparse_score_matrix": "missing pairs in the sparse score matrix",
    "candidate_scarcity": "too few locally admissible candidates",
    "expert_capacity": "expert capacity consumed elsewhere",
    "seniority_floor": "reserved senior slots",
    "institution_diversity": "distinct-institution gates",
    "global_capacity_coupling": "global capacity coupling across documents",
    "greedy_not_certified": "greedy result is not an infeasibility certificate",
}

_EVIDENCE_LABELS = {
    "hard_conflict_pairs": "hard-conflict pairs",
    "other_ineligible_pairs": "other ineligible pairs",
    "zero_capacity_pairs": "zero-capacity pairs",
    "below_minimum_score_pairs": "pairs below minimum score",
    "unscored_experts": "experts without a score row",
    "admissible_pairs": "admissible pairs",
    "senior_admissible_pairs": "senior admissible pairs",
    "institution_groups": "available institution groups",
    "institution_blocked_pairs": "pairs blocked by a used institution",
    "saturated_admissible_experts": "saturated admissible experts",
}


def _percent(value: float) -> str:
    return f"{max(0.0, min(1.0, value)) * 100:.1f}%"


def render_html(
    run: MatchRun,
    documents: Iterable[Document],
    experts: Iterable[Expert],
    *,
    title: str = "PeerMatchLab assignment report",
) -> str:
    """Render one run as portable HTML without scripts or external assets."""

    document_map = {item.id: item for item in documents}
    expert_map = {item.id: item for item in experts}
    score_map = {(item.document_id, item.expert_id): item for item in run.scores}
    sections: list[str] = []
    for document_id in sorted(document_map):
        document = document_map[document_id]
        assignments = run.plan.for_document(document_id)
        rows: list[str] = []
        for assignment in assignments:
            expert = expert_map[assignment.expert_id]
            evidence = score_map[(document_id, assignment.expert_id)]
            bars = "".join(
                (
                    '<div class="component">'
                    f"<span>{html.escape(name)}</span>"
                    '<div class="track"><i style="width:'
                    f'{_percent(value)}"></i></div><strong>{value:.2f}</strong></div>'
                )
                for name, value in assignment.components.items()
            )
            reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in evidence.reasons)
            rows.append(
                '<article class="match">'
                f'<div class="rank">#{assignment.rank}</div>'
                '<div class="match-body">'
                f"<h3>{html.escape(expert.name)} "
                f"<small>{html.escape(expert.institution or 'Independent')}</small></h3>"
                f'<p class="score">Total evidence <b>{assignment.score:.3f}</b></p>'
                f'<div class="bars">{bars}</div><ul>{reasons}</ul>'
                "</div></article>"
            )
        unmet = run.plan.unmet.get(document_id, 0)
        warning = ""
        if unmet:
            diagnostic = (
                run.plan.diagnostics.for_document(document_id)
                if run.plan.diagnostics is not None
                else None
            )
            detail = ""
            if diagnostic is not None:
                reasons = ", ".join(
                    _REASON_LABELS[reason.value] for reason in diagnostic.reason_codes
                )
                constraint_evidence = "; ".join(
                    f"{label}: {diagnostic.evidence[key]}"
                    for key, label in _EVIDENCE_LABELS.items()
                    if diagnostic.evidence.get(key, 0)
                )
                saturated = (
                    "; saturated experts: " + ", ".join(diagnostic.saturated_experts)
                    if diagnostic.saturated_experts
                    else ""
                )
                detail = (
                    f'<span class="diagnostic"><b>Constraint evidence:</b> '
                    f"{html.escape(reasons or 'no local filter identified')}. "
                    f"{html.escape(constraint_evidence + saturated)}</span>"
                )
            warning = f'<p class="warning">Unmet expert slots: {unmet}{detail}</p>'
        sections.append(
            '<section class="document">'
            f"<h2>{html.escape(document.title)}</h2>"
            f'<p class="document-id">{html.escape(document.id)}</p>'
            f"{warning}{''.join(rows) if rows else '<p>No eligible assignment.</p>'}</section>"
        )
    audit = run.audit
    if not audit.safe:
        status = "Review required"
    elif run.plan.diagnostics is not None and run.plan.diagnostics.unmet:
        status = "Constraint-safe · Incomplete"
    else:
        status = "Safe plan"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#667085; --paper:#fff; --bg:#f2f5f7;
  --accent:#087e8b; --accent2:#ff5a5f; --line:#d8e0e5; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 Inter,Segoe UI,sans-serif; }}
main {{ max-width:1120px; margin:auto; padding:42px 24px 64px; }}
header {{ display:flex; justify-content:space-between; gap:24px; align-items:end; margin-bottom:24px; }}
h1 {{ font-size:36px; margin:0; letter-spacing:-.03em; }}
.eyebrow {{ color:var(--accent); font-weight:800; text-transform:uppercase; letter-spacing:.12em; }}
.status {{ border-radius:999px; padding:8px 14px; background:#daf5ed; color:#116149; font-weight:800; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:20px 0 28px; }}
.metric,.document {{ background:var(--paper); border:1px solid var(--line); border-radius:16px; box-shadow:0 8px 28px #2030400b; }}
.metric {{ padding:18px; }} .metric b {{ display:block; font-size:27px; }} .metric span {{ color:var(--muted); }}
.document {{ padding:24px; margin:16px 0; }} h2 {{ margin:0; font-size:21px; }}
.document-id {{ margin:2px 0 18px; color:var(--muted); font-family:ui-monospace,monospace; }}
.match {{ display:grid; grid-template-columns:48px 1fr; gap:12px; border-top:1px solid var(--line); padding:18px 0 4px; }}
.rank {{ width:40px; height:40px; display:grid; place-items:center; border-radius:12px; background:#e5f6f8; color:var(--accent); font-weight:900; }}
.match h3 {{ margin:0; }} .match h3 small {{ color:var(--muted); font-size:13px; font-weight:500; }}
.score {{ margin:4px 0 10px; }} .bars {{ display:grid; grid-template-columns:repeat(2,minmax(220px,1fr)); gap:7px 20px; }}
.component {{ display:grid; grid-template-columns:70px 1fr 36px; align-items:center; gap:8px; color:var(--muted); font-size:12px; }}
.track {{ height:7px; border-radius:8px; background:#e8edf0; overflow:hidden; }} .track i {{ display:block; height:100%; background:linear-gradient(90deg,var(--accent),#45b8ac); }}
.component strong {{ color:var(--ink); }} ul {{ margin:10px 0 0; padding-left:20px; color:var(--muted); }}
  .warning {{ color:#8d3b16; background:#fff1e8; padding:8px 12px; border-radius:8px; }}
  .diagnostic {{ display:block; margin-top:5px; color:#6f3518; }}
footer {{ color:var(--muted); margin-top:28px; text-align:center; }}
@media (max-width:760px) {{ .metrics {{ grid-template-columns:repeat(2,1fr); }} .bars {{ grid-template-columns:1fr; }} header {{ display:block; }} .status {{ display:inline-block; margin-top:12px; }} }}
</style>
</head>
<body><main>
<header><div><div class="eyebrow">Auditable expert allocation</div><h1>{html.escape(title)}</h1></div>
<div class="status">{status}</div></header>
<div class="metrics">
  <div class="metric"><b>{audit.demand_coverage:.0%}</b><span>Demand covered</span></div>
  <div class="metric"><b>{audit.average_score:.3f}</b><span>Average evidence</span></div>
  <div class="metric"><b>{audit.workload_gini:.3f}</b><span>Workload Gini</span></div>
  <div class="metric"><b>{len(run.plan.assignments)}</b><span>Assignments</span></div>
</div>
{"".join(sections)}
<footer>Generated locally by PeerMatchLab · strategy {html.escape(run.plan.strategy)}</footer>
</main></body></html>
"""


def write_html(
    path: str | Path,
    run: MatchRun,
    documents: Iterable[Document],
    experts: Iterable[Expert],
    *,
    title: str = "PeerMatchLab assignment report",
) -> None:
    """Write a self-contained HTML report."""

    Path(path).write_text(
        render_html(run, documents, experts, title=title),
        encoding="utf-8",
        newline="\n",
    )
