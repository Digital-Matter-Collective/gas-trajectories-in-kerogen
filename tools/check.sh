#!/usr/bin/env bash
# Non-mutating equivalent of prepare.sh, safe to run in CI: same file list
# (tools/list_python_files.sh), no autofix, plus the test suite. This is
# what .github/workflows/ci.yml calls, and what contributors can call
# locally to see exactly what CI will see.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/list_python_files.sh"

echo "[1/5] Ruff check (no fixes)"
ruff check $PYTHON_FILES

echo "[2/5] isort --check-only"
isort --check-only --settings-path=pyproject.toml $PYTHON_FILES

echo "[3/5] black --check"
black --check --config=pyproject.toml $PYTHON_FILES

echo "[4/5] mypy"
mypy --config-file=mypy.ini .

echo "[5/5] pytest"
python -m pytest -q

echo "OK"
