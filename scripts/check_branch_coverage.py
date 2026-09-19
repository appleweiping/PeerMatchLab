"""Fail when *branch* coverage, not blended statement coverage, is below 90%."""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> int:
    report = subprocess.run(
        [sys.executable, "-m", "coverage", "json", "-o", "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        totals = json.loads(report.stdout)["totals"]
    except (ValueError, KeyError, TypeError):
        print("branch coverage data is missing", file=sys.stderr)
        return 2
    covered = totals["covered_branches"]
    possible = totals["num_branches"]
    if not isinstance(covered, int) or not isinstance(possible, int) or possible <= 0:
        print("branch coverage data is missing", file=sys.stderr)
        return 2
    percent = 100 * covered / possible
    print(f"branch coverage: {covered}/{possible} = {percent:.2f}% (required: 90.00%)")
    return 0 if 10 * covered >= 9 * possible else 1


if __name__ == "__main__":
    raise SystemExit(main())
