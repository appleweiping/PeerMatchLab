from __future__ import annotations

from peermatchlab.config import MatchConfig
from peermatchlab.models import Document, Expert
from peermatchlab.pipeline import run_matching
from peermatchlab.report import render_html, write_html


def test_report_contains_assignments(documents, experts) -> None:
    run = run_matching(documents, experts, config=MatchConfig(reviewers_per_document=1))
    report = render_html(run, documents, experts)
    assert "PeerMatchLab assignment report" in report
    assert "Ari" in report
    assert "Demand covered" in report


def test_report_escapes_untrusted_text() -> None:
    documents = [Document("d", "<script>alert(1)</script>")]
    experts = [Expert("e", "<b>Expert</b>", capacity=1)]
    run = run_matching(documents, experts, config=MatchConfig(reviewers_per_document=1))
    report = render_html(run, documents, experts)
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert "<b>Expert</b>" not in report


def test_report_marks_unmet_slots(documents, experts) -> None:
    run = run_matching(
        documents,
        experts,
        config=MatchConfig(reviewers_per_document=5, minimum_score=1.1),
    )
    assert "Unmet expert slots: 5" in render_html(run, documents, experts)


def test_write_report_creates_portable_document(tmp_path, documents, experts) -> None:
    run = run_matching(documents, experts, config=MatchConfig(reviewers_per_document=1))
    path = tmp_path / "report.html"
    write_html(path, run, documents, experts, title="Custom report")
    content = path.read_text(encoding="utf-8")
    assert content.startswith("<!doctype html>")
    assert "Custom report" in content
    assert "https://" not in content
    assert b"\r\n" not in path.read_bytes()
