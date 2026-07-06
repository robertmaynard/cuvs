#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# PR-time entry point: identify which tests cover the changes in a PR and
# write selected_tests.txt + remaining_tests.txt for two-pass CTest execution.
#
# Reproducible locally — all CI steps are driven through this script.
#
# Usage (CI):
#   ci/coverage/select_tests.sh --pr-number 1234 --mapping func2tests.json
#
# Usage (local):
#   ci/coverage/select_tests.sh --base-ref main --mapping func2tests.json
#
# Options:
#   --pr-number N       GitHub PR number; uses `gh pr diff` to get changed files
#   --base-ref REF      Git ref to diff against; uses `git diff` for local runs
#   --mapping PATH      func2tests.json produced by collect_and_map.sh
#                       (default: <repo>/func2tests.json)
#   --ctest-dir PATH    Directory from which to run ctest -N
#                       (default: cpp/build/coverage if it exists, otherwise
#                        $CONDA_PREFIX/bin/gtests/libcuvs)
#   --base-tests PATH   base_tests.txt to always include (default: auto-detected)
#   --selected PATH     Output file for selected tests  (default: selected_tests.txt)
#   --remaining PATH    Output file for remaining tests (default: remaining_tests.txt)
#   --work-dir PATH     Scratch directory for intermediate files (default: /tmp/cuvs-select)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PR_NUMBER=""
BASE_REF=""
MAPPING="$REPO_ROOT/func2tests.json"
CTEST_DIR=""
BASE_TESTS="$SCRIPT_DIR/base_tests.txt"
SELECTED_OUT="selected_tests.txt"
REMAINING_OUT="remaining_tests.txt"
WORK_DIR="/tmp/cuvs-select-$$"

while [[ $# -gt 0 ]]; do
  case $1 in
    --pr-number)   PR_NUMBER="$2";   shift 2 ;;
    --base-ref)    BASE_REF="$2";    shift 2 ;;
    --mapping)     MAPPING="$2";     shift 2 ;;
    --ctest-dir)   CTEST_DIR="$2";   shift 2 ;;
    --base-tests)  BASE_TESTS="$2";  shift 2 ;;
    --selected)    SELECTED_OUT="$2"; shift 2 ;;
    --remaining)   REMAINING_OUT="$2"; shift 2 ;;
    --work-dir)    WORK_DIR="$2";    shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$PR_NUMBER" && -z "$BASE_REF" ]]; then
  echo "ERROR: one of --pr-number or --base-ref is required" >&2
  exit 1
fi

# Auto-detect ctest directory
if [[ -z "$CTEST_DIR" ]]; then
  if [[ -f "$REPO_ROOT/cpp/build/coverage/CTestTestfile.cmake" ]]; then
    CTEST_DIR="$REPO_ROOT/cpp/build/coverage"
  elif [[ -d "${CONDA_PREFIX:-}/bin/gtests/libcuvs" ]]; then
    CTEST_DIR="$CONDA_PREFIX/bin/gtests/libcuvs"
  else
    echo "ERROR: cannot locate ctest directory; pass --ctest-dir explicitly" >&2
    exit 1
  fi
fi

mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT

CHANGED_FILES="$WORK_DIR/changed_files.txt"
CTAGS_JSONL="$WORK_DIR/changed_functions.jsonl"

# ── Step 1: Get changed source files ─────────────────────────────────────────
echo "==> Getting changed files ..."
if [[ -n "$PR_NUMBER" ]]; then
  gh pr diff "$PR_NUMBER" --name-only > "$WORK_DIR/all_changed.txt"
else
  # Committed changes on this branch vs base, plus any staged/unstaged edits
  # not yet committed.  Using separate commands and deduplicating so that a
  # local work-in-progress edit is included without pulling in every prior
  # commit on the branch when diffing against a common ancestor.
  { git -C "$REPO_ROOT" diff --name-only "${BASE_REF}...HEAD"
    git -C "$REPO_ROOT" diff --name-only HEAD
  } | sort -u > "$WORK_DIR/all_changed.txt"
fi

# Filter to C/C++/CUDA source files
grep -E '\.(c|cc|cpp|cu|cuh|hpp|h)$' "$WORK_DIR/all_changed.txt" > "$CHANGED_FILES" || true

N_ALL=$(wc -l < "$WORK_DIR/all_changed.txt")
N_SRC=$(wc -l < "$CHANGED_FILES")
echo "  $N_ALL file(s) changed; $N_SRC are C/C++/CUDA"
if [[ -s "$CHANGED_FILES" ]]; then
  while IFS= read -r f; do echo "    $f"; done < "$CHANGED_FILES"
fi

if [[ ! -s "$CHANGED_FILES" ]]; then
  echo "  No C/C++/CUDA files changed — selecting base tests only."
fi

# ── Step 2: Extract changed function names via ctags ─────────────────────────
if [[ -s "$CHANGED_FILES" ]]; then
  echo "==> Running ctags on changed source files ..."

  # Build list of files that actually exist (deleted files are skipped)
  EXISTING_FILES=()
  while IFS= read -r f; do
    full="$REPO_ROOT/$f"
    [[ -f "$full" ]] && EXISTING_FILES+=("$full")
  done < "$CHANGED_FILES"

  if [[ ${#EXISTING_FILES[@]} -gt 0 ]]; then
    ctags \
      --output-format=json \
      --fields=+nKs \
      --kinds-C++=f+p \
      --kinds-CUDA=f \
      --language-force=C++ \
      -f - \
      -- "${EXISTING_FILES[@]}" \
      > "$CTAGS_JSONL" 2>/dev/null || true
    echo "  $(wc -l < "$CTAGS_JSONL") tag(s) extracted"
  else
    touch "$CTAGS_JSONL"
  fi
fi

# ── Step 3: Select tests ──────────────────────────────────────────────────────
echo "==> Selecting tests ..."

BASE_TESTS_ARG=""
[[ -f "$BASE_TESTS" ]] && BASE_TESTS_ARG="--base-tests $BASE_TESTS"

CTAGS_ARG=""
[[ -f "$CTAGS_JSONL" ]] && CTAGS_ARG="--ctags-jsonl $CTAGS_JSONL"

python3 "$SCRIPT_DIR/select_tests.py" \
  --mapping       "$MAPPING"        \
  --changed-files "$CHANGED_FILES"  \
  --ctest-dir     "$CTEST_DIR"      \
  --selected-output  "$SELECTED_OUT"  \
  --remaining-output "$REMAINING_OUT" \
  $BASE_TESTS_ARG \
  $CTAGS_ARG

echo "==> Done."
echo "    Pass 1: ctest --tests-from-file $SELECTED_OUT"
echo "    Pass 2: ctest --tests-from-file $REMAINING_OUT"
