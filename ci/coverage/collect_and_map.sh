#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Nightly entry point: configure a gcov-instrumented cuVS build, run every
# GTest case individually to capture per-test coverage, and produce
# func2tests.json mapping source functions to the tests that cover them.
#
# Reproducible locally — all CI steps are driven through this script.
#
# Usage:
#   ci/coverage/collect_and_map.sh [OPTIONS]
#
# Options:
#   --skip-build          Reuse an existing coverage build; skip cmake/ninja
#   --build-dir PATH      CMake build directory      (default: <repo>/cpp/build/coverage)
#   --coverage-dir PATH   Per-test .cov.json/.jit.log output directory
#                         (default: <build-dir>/per_test_coverage)
#   --output PATH         func2tests.json output path (default: <repo>/func2tests.json)
#
# Prerequisites:
#   cmake, ninja, gcov (bundled with GCC), python3, and a GPU-capable host with
#   the cuVS build dependencies available (see cpp/CMakePresets.json for the
#   coverage preset).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BUILD_DIR="$REPO_ROOT/cpp/build/coverage"
COVERAGE_DIR=""
MAPPING_OUT="$REPO_ROOT/func2tests.json"
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-build)   SKIP_BUILD=1; shift ;;
    --build-dir)    BUILD_DIR="$2"; shift 2 ;;
    --coverage-dir) COVERAGE_DIR="$2"; shift 2 ;;
    --output)       MAPPING_OUT="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Default coverage dir lives inside the build tree so it is never committed
[[ -z "$COVERAGE_DIR" ]] && COVERAGE_DIR="$BUILD_DIR/per_test_coverage"

# ── Step 1: gcov-instrumented build ──────────────────────────────────────────
if [[ $SKIP_BUILD -eq 0 ]]; then
  echo "==> Configuring coverage build (cmake --preset coverage) ..."
  cmake --preset coverage -S "$REPO_ROOT/cpp"

  echo "==> Building ..."
  cmake --build --preset coverage
else
  echo "==> Skipping build (--skip-build)"
fi

# ── Step 2: Per-test gcov collection ─────────────────────────────────────────
echo "==> Collecting per-test coverage ..."
python3 "$SCRIPT_DIR/collect_coverage.py" \
  --build-dir    "$BUILD_DIR"    \
  --coverage-dir "$COVERAGE_DIR" \
  --repo-root    "$REPO_ROOT"    \
  --continue-on-failure

# ── Step 3: Build function→test mapping ──────────────────────────────────────
MAPPING_SCRIPT="$SCRIPT_DIR/build_mapping.py"
if [[ -f "$MAPPING_SCRIPT" ]]; then
  echo "==> Building func2tests.json ..."
  python3 "$MAPPING_SCRIPT" \
    --coverage-dir  "$COVERAGE_DIR" \
    --jit-sources   "$BUILD_DIR/jit_lto_sources.json" \
    --repo-root     "$REPO_ROOT" \
    --output        "$MAPPING_OUT"
  echo "==> Mapping written to: $MAPPING_OUT"
else
  echo "==> build_mapping.py not yet present (Phase 2 deliverable). Skipping."
  echo "    Coverage .info files are in: $COVERAGE_DIR"
fi

echo "==> Done."
