# Contributing

Thank you for improving PeerMatchLab. Contributions should preserve its central properties: deterministic offline execution, explicit evidence, strict input validation, and independently auditable assignments.

## Before coding

- Search existing issues and pull requests.
- For a new scoring signal, constraint, public schema field, or optimizer, open an issue describing the semantics and failure cases first.
- Keep unrelated refactors in separate changes.
- Never add real reviewer, applicant, employee, or other sensitive personal data to tests or examples.

## Local workflow

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"

ruff check .
ruff format --check .
mypy src
pytest --cov=peermatchlab --cov-report=term-missing
python -m build
```

Tests must cover success, invalid input, deterministic ties, insufficient capacity, and relevant hard-constraint behavior. A changed scoring formula also needs a README update explaining its range and interpretation.

## Pull requests

- Explain the user-visible problem and why the chosen scope is sufficient.
- List the exact validation commands actually run.
- Include or update tests for every behavior change.
- Call out schema, compatibility, performance, or fairness implications.
- Do not claim benchmark results that cannot be reproduced from committed commands and data.

By contributing, you agree that your contribution is licensed under the repository's MIT License.
