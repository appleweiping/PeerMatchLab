from __future__ import annotations

import json
from pathlib import Path

import pytest

from peermatchlab.cli import main
from peermatchlab.models import DataValidationError
from peermatchlab.openreview import load_openreview_submissions, load_reviewer_ids


def test_openreview_adapter_accepts_v1_and_v2_content_shapes(tmp_path: Path) -> None:
    submissions = tmp_path / "submissions.jsonl"
    submissions.write_text(
        json.dumps(
            {
                "id": "note-v2",
                "forum": "forum-1",
                "invitations": ["Venue/-/Submission", "Venue/-/Camera_Ready"],
                "content": {
                    "title": {"value": "Wrapped title"},
                    "abstract": {"value": "Wrapped abstract"},
                    "keywords": {"value": ["IR", "Matching"]},
                    "subject_areas": {"value": ["Information Retrieval"]},
                    "venueid": {"value": "Venue.cc/2026/Conference"},
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "note-v1",
                "invitation": "Venue/-/Submission",
                "content": {
                    "title": "Direct title",
                    "abstract": "Direct abstract",
                    "keywords": ["Evaluation"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    documents = load_openreview_submissions(submissions)

    assert [item.id for item in documents] == ["note-v2", "note-v1"]
    assert documents[0].topics == ("Information Retrieval",)
    assert documents[0].keywords == ("IR", "Matching")
    assert documents[0].metadata["openreview_forum"] == "forum-1"
    assert documents[0].metadata["openreview_invitations"] == (
        "Venue/-/Submission",
        "Venue/-/Camera_Ready",
    )
    assert documents[0].metadata["openreview_venue"] == "Venue.cc/2026/Conference"
    assert documents[1].metadata["openreview_invitation"] == "Venue/-/Submission"


def test_openreview_adapter_accepts_common_singular_subject_area(tmp_path: Path) -> None:
    submissions = tmp_path / "submissions.jsonl"
    submissions.write_text(
        json.dumps(
            {
                "id": "note",
                "content": {
                    "title": {"value": "Title"},
                    "subject_area": {"value": "Information Retrieval"},
                    "subject_areas": {"value": ["Machine Learning", "Information Retrieval"]},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_openreview_submissions(submissions)[0].topics == (
        "Machine Learning",
        "Information Retrieval",
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('{"id":"a","id":"b","content":{"title":"T"}}\n', "duplicate JSON"),
        ('{"id":"a","content":{}}\n', "submission title"),
        ('{"id":"a","content":{"title":{"not_value":"T"}}}\n', "contain value"),
        ('{"id":"a","content":{"title":"T","keywords":"IR"}}\n', "array"),
        (
            '{"id":"a","content":{"title":"T"}}\n{"id":"a","content":{"title":"Other"}}\n',
            "duplicate OpenReview",
        ),
    ],
)
def test_openreview_adapter_rejects_invalid_notes(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "submissions.jsonl"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(DataValidationError, match=message):
        load_openreview_submissions(path)


def test_reviewer_id_adapter_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "reviewers.txt"
    path.write_text("~Reviewer1\n~Reviewer2\n", encoding="utf-8")
    experts = load_reviewer_ids(path, capacity=3)
    assert [item.id for item in experts] == ["~Reviewer1", "~Reviewer2"]
    assert all(item.capacity == 3 for item in experts)

    path.write_text("~Reviewer1\n~Reviewer1\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="duplicates"):
        load_reviewer_ids(path, capacity=3)
    with pytest.raises(DataValidationError, match="capacity"):
        load_reviewer_ids(path, capacity=True)  # type: ignore[arg-type]


def test_openreview_adapters_reject_empty_local_exports(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="no notes"):
        load_openreview_submissions(empty)
    with pytest.raises(DataValidationError, match="no identifiers"):
        load_reviewer_ids(empty, capacity=1)


def test_import_openreview_cli_writes_explicit_interchange_files(tmp_path: Path) -> None:
    submissions = tmp_path / "submissions.jsonl"
    reviewers = tmp_path / "reviewers.txt"
    output = tmp_path / "converted"
    submissions.write_text(
        '{"id":"paper-1","content":{"title":{"value":"A paper"},'
        '"abstract":{"value":"Summary"},"keywords":{"value":["IR"]}}}\n',
        encoding="utf-8",
    )
    reviewers.write_text("~Reviewer1\n~Reviewer2\n", encoding="utf-8")

    code = main(
        [
            "import-openreview",
            "--submissions",
            str(submissions),
            "--reviewers",
            str(reviewers),
            "--reviewer-capacity",
            "2",
            "--directory",
            str(output),
        ]
    )

    assert code == 0
    assert json.loads((output / "documents.json").read_text(encoding="utf-8"))[0]["id"] == "paper-1"
    assert len(json.loads((output / "experts.json").read_text(encoding="utf-8"))) == 2
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["adapter"] == "openreview-local-export-v1"
    assert len(metadata["limitations"]) == 3
