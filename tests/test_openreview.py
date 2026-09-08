from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from peermatchlab.cli import main
from peermatchlab.models import DataValidationError
from peermatchlab.openreview import (
    load_openreview_submissions,
    load_reviewer_ids,
    openreview_submissions_from_records,
    reviewer_ids_to_experts,
)


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

    for invalid in (" reviewer", "reviewer ", "reviewer\talias", "reviewer\x00alias"):
        path.write_text(invalid + "\n", encoding="utf-8")
        with pytest.raises(DataValidationError, match="without surrounding whitespace"):
            load_reviewer_ids(path, capacity=1)


def test_openreview_public_iterables_stop_at_limit_plus_one() -> None:
    def reviewer_ids() -> Iterator[str]:
        yield "r1"
        yield "r2"
        raise AssertionError("reviewer iterator was over-consumed")

    with pytest.raises(DataValidationError, match="configured limit"):
        reviewer_ids_to_experts(reviewer_ids(), capacity=1, max_reviewers=1)

    def submissions() -> Iterator[object]:
        for note_id in ("p1", "p2"):
            yield {"id": note_id, "content": {"title": "Graph"}}
        raise AssertionError("submission iterator was over-consumed")

    with pytest.raises(DataValidationError, match="configured limit"):
        openreview_submissions_from_records(submissions(), max_records=1)


def test_openreview_file_loaders_stream_with_byte_line_record_and_utf8_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submissions = tmp_path / "submissions.jsonl"
    row = b'{"id":"p","content":{"title":"Graph"}}\n'
    submissions.write_bytes(row + row.replace(b'"p"', b'"q"'))
    with pytest.raises(DataValidationError, match="configured limit"):
        load_openreview_submissions(submissions, max_records=1)
    with pytest.raises(DataValidationError, match="max_input_file_bytes"):
        load_openreview_submissions(submissions, max_input_file_bytes=len(row) - 1)
    with pytest.raises(DataValidationError, match="max_line_bytes"):
        load_openreview_submissions(submissions, max_line_bytes=10)
    submissions.write_bytes(b"\xff\n")
    with pytest.raises(DataValidationError, match="UTF-8"):
        load_openreview_submissions(submissions)
    submissions.write_text("[" * 2_000 + "0" + "]" * 2_000 + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="nesting exceeds"):
        load_openreview_submissions(submissions)

    reviewers = tmp_path / "reviewers.txt"
    reviewers.write_text("r1\nr2\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="configured limit"):
        load_reviewer_ids(reviewers, capacity=1, max_reviewers=1)
    with pytest.raises(DataValidationError, match="max_input_file_bytes"):
        load_reviewer_ids(reviewers, capacity=1, max_input_file_bytes=2)
    with pytest.raises(DataValidationError, match="max_line_bytes"):
        load_reviewer_ids(reviewers, capacity=1, max_line_bytes=1)

    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("read_text must not be used")
        ),
    )
    reviewers.write_bytes(b"r1\n")
    submissions.write_bytes(row)
    assert load_reviewer_ids(reviewers, capacity=1)[0].id == "r1"
    assert load_openreview_submissions(submissions)[0].id == "p"


@pytest.mark.parametrize(
    ("loader", "argument"),
    [
        ("submissions", "max_records"),
        ("submissions", "max_input_file_bytes"),
        ("submissions", "max_line_bytes"),
        ("reviewers", "max_reviewers"),
        ("reviewers", "max_input_file_bytes"),
        ("reviewers", "max_line_bytes"),
    ],
)
def test_openreview_file_loaders_reject_extreme_limits_as_domain_errors(
    tmp_path: Path, loader: str, argument: str
) -> None:
    source = tmp_path / "source"
    source.write_text(
        '{"id":"p","content":{"title":"Graph"}}\n' if loader == "submissions" else "r\n",
        encoding="utf-8",
    )
    kwargs = {argument: 10**100}
    with pytest.raises(DataValidationError, match="integer between"):
        if loader == "submissions":
            load_openreview_submissions(source, **kwargs)  # type: ignore[arg-type]
        else:
            load_reviewer_ids(source, capacity=1, **kwargs)  # type: ignore[arg-type]


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

    limited = tmp_path / "limited"
    assert (
        main(
            [
                "import-openreview",
                "--submissions",
                str(submissions),
                "--reviewers",
                str(reviewers),
                "--reviewer-capacity",
                "2",
                "--max-records",
                "1",
                "--max-input-file-bytes",
                "1024",
                "--max-line-bytes",
                "512",
                "--directory",
                str(limited),
            ]
        )
        == 2
    )
    assert not limited.exists()
