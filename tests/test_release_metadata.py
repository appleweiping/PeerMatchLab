from __future__ import annotations

import re
import tomllib
from pathlib import Path


def test_release_version_metadata_stays_synchronized() -> None:
    root = Path(__file__).parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        package_version = tomllib.load(stream)["project"]["version"]

    source = (root / "src" / "peermatchlab" / "__init__.py").read_text(encoding="utf-8")
    source_match = re.search(r'^__version__ = "([^"]+)"$', source, flags=re.MULTILINE)
    assert source_match is not None

    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    citation_match = re.search(r"^version: ([^\s]+)$", citation, flags=re.MULTILINE)
    assert citation_match is not None

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert source_match.group(1) == package_version
    assert citation_match.group(1) == package_version
    assert f"## {package_version} -" in changelog
