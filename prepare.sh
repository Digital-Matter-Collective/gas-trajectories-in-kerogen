#!/usr/bin/env bash
set -euo pipefail

# File list is shared with tools/check.sh (the CI-equivalent, non-mutating
# script) so the two can never list different files.
source "$(dirname "${BASH_SOURCE[0]}")/tools/list_python_files.sh"

echo "[1/6] Ruff autofix (safe fixes)"
ruff check --fix $PYTHON_FILES

echo "[2/6] isort"
isort --settings-path=pyproject.toml $PYTHON_FILES

echo "[3/6] black"
black --config=pyproject.toml $PYTHON_FILES

echo "[4/6] Ruff check (no fixes)"
ruff check $PYTHON_FILES

echo "[5/6] mypy"
mypy --config-file=mypy.ini .

echo "[6/6] OK"