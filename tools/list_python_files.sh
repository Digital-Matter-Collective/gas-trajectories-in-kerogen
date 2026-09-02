#!/usr/bin/env bash
# Shared file-discovery logic for prepare.sh (local autofix) and
# tools/check.sh (non-mutating, CI-equivalent). Source this file - do not
# execute it directly - to cd into the repo root and populate $PYTHON_FILES
# with the same list in both places, so the two scripts can never drift.
#
# Uses `git ls-files` instead of `find` + a hand-maintained prune list:
# this only ever lints tracked source, so local venvs/caches/build output
# (however they're named or gitignored) can never leak in by accident.

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_FILES=$(git ls-files -- '*.py')
