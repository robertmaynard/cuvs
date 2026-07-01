#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Run every CTest test labelled "gtest_case" individually, capturing:
#   - a per-test .cov.json file for gcov coverage
#   - a per-test .jit.log file listing JIT-LTO fragment keys activated at runtime
#
# Called by ci/coverage/collect_and_map.sh.  Not intended to be run directly,
# but can be for debugging: python3 collect_coverage.py --build-dir <path>
#
# Outputs (in --coverage-dir):
#   <safe_name>.cov.json  — gcov coverage for one GTest case
#   <safe_name>.jit.log   — one JIT-LTO fragment key per line (may be empty)

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_MAX_SAFE_LEN = 200


def safe_name(test_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", test_name)
    if len(sanitized) <= _MAX_SAFE_LEN:
        return sanitized
    digest = hashlib.sha1(test_name.encode()).hexdigest()[:8]
    return sanitized[:_MAX_SAFE_LEN - 9] + "_" + digest


def list_gtest_case_tests(build_dir: Path) -> list[tuple[int, str]]:
    """Return (global_ctest_index, test_name) for every gtest_case-labelled test."""
    result = subprocess.run(
        ["ctest", "-N", "-L", "gtest_case"],
        cwd=build_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    tests: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        m = re.match(r"\s+Test\s+#(\d+):\s+(.+)", line)
        if m:
            tests.append((int(m.group(1)), m.group(2).strip()))
    return tests


def delete_gcda_files(build_dir: Path) -> None:
    for subdir in ("CMakeFiles", "tests/CMakeFiles"):
        d = build_dir / subdir
        if d.exists():
            for gcda in d.rglob("*.gcda"):
                gcda.unlink(missing_ok=True)


def run_single_test(test_index: int, build_dir: Path, jit_log: Path) -> bool:
    """Run one CTest test by its global index (avoids regex issues with test names)."""
    env = os.environ.copy()
    env["CUVS_JIT_TRACE_LOG"] = str(jit_log)
    result = subprocess.run(
        ["ctest", "-I", f"{test_index},{test_index}", "--output-on-failure"],
        cwd=build_dir,
        env=env,
    )
    return result.returncode == 0


def _process_gcda(args: tuple[int, Path, Path]) -> list[dict]:
    """Run gcov --json-format on one gcda file; return parsed file entries."""
    idx, gcda, tmp = args
    work = tmp / str(idx)
    work.mkdir(exist_ok=True)
    subprocess.run(
        ["gcov", "--json-format", "--demangled-names", str(gcda)],
        cwd=work,
        capture_output=True,
    )
    results: list[dict] = []
    for gz_file in work.glob("*.gcov.json.gz"):
        try:
            with gzip.open(gz_file) as gz:
                results.extend(json.load(gz).get("files", []))
        except Exception:
            pass
    return results


def capture_coverage_gcov(
    build_dir: Path,
    cov_json: Path,
    test_name: str,
    repo_root: Path | None = None,
) -> None:
    """
    Run gcov --json-format on all gcda files produced by the test, filter to
    cuVS sources, and write a per-test .cov.json file.
    """
    gcda_files: list[Path] = []
    for subdir in ("CMakeFiles", "tests/CMakeFiles"):
        d = build_dir / subdir
        if d.exists():
            gcda_files.extend(d.rglob("*.gcda"))

    if not gcda_files:
        cov_json.write_text(json.dumps({"test_name": test_name, "covered": {}}) + "\n")
        return

    covered: dict[str, set[str]] = {}
    cpp_root = repo_root / "cpp" if repo_root else None

    with tempfile.TemporaryDirectory(prefix="cuvs_gcov_") as tmpdir:
        tmp = Path(tmpdir)
        work_args = [(i, gcda, tmp) for i, gcda in enumerate(gcda_files)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for file_entries in executor.map(_process_gcda, work_args):
                for file_data in file_entries:
                    src = file_data.get("file", "")
                    if cpp_root:
                        try:
                            rel = Path(src).resolve().relative_to(cpp_root)
                            src_key = str(Path("cpp") / rel)
                        except ValueError:
                            continue
                    elif "cuvs/cpp" in src:
                        src_key = src[src.index("cuvs/cpp"):]
                    else:
                        continue

                    if "_deps" in src_key or "__pycache__" in src_key:
                        continue

                    for fn in file_data.get("functions", []):
                        if fn.get("execution_count", 0) > 0:
                            name = fn.get("demangled_name") or fn.get("name", "")
                            if name:
                                covered.setdefault(src_key, set()).add(name)

    result = {
        "test_name": test_name,
        "covered": {k: sorted(v) for k, v in covered.items()},
    }
    cov_json.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect per-test gcov coverage for every gtest_case-labelled CTest test. "
            "Writes one .cov.json file and one .jit.log file per test into --coverage-dir."
        )
    )
    parser.add_argument(
        "--build-dir",
        default="cpp/build/coverage",
        help="CMake build directory produced by the coverage preset",
    )
    parser.add_argument(
        "--coverage-dir",
        default=None,
        help="Directory to write per-test .cov.json and .jit.log files "
             "(default: <build-dir>/per_test_coverage)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root; used to filter coverage to cuVS sources only (default: auto-detect)",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Capture coverage and continue even when a test fails",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tests whose .cov.json already exists in --coverage-dir",
    )
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    coverage_dir = Path(args.coverage_dir).resolve() if args.coverage_dir else build_dir / "per_test_coverage"
    repo_root = Path(args.repo_root).resolve() if args.repo_root else build_dir.parent.parent.parent

    if not build_dir.exists():
        sys.exit(f"ERROR: build directory does not exist: {build_dir}")

    coverage_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering gtest_case tests in {build_dir} ...", flush=True)
    tests = list_gtest_case_tests(build_dir)
    if not tests:
        sys.exit(
            "ERROR: no tests found with label 'gtest_case'. "
            "Ensure the coverage build includes gtest_discover_tests() in ConfigureTest()."
        )
    print(f"Found {len(tests)} test(s).", flush=True)

    failures: list[str] = []

    for i, (test_index, test) in enumerate(tests, 1):
        sname = safe_name(test)
        cov_json = coverage_dir / f"{sname}.cov.json"
        jit_log = coverage_dir / f"{sname}.jit.log"

        if args.resume and cov_json.exists():
            print(f"[{i}/{len(tests)}] skip (already collected): {test}", flush=True)
            continue

        print(f"[{i}/{len(tests)}] {test}", flush=True)

        delete_gcda_files(build_dir)

        success = run_single_test(test_index, build_dir, jit_log)
        if not success:
            print(f"  WARNING: test failed — coverage captured anyway: {test}", flush=True)
            failures.append(test)
            if not args.continue_on_failure:
                sys.exit(
                    "Aborting after first failure. "
                    "Use --continue-on-failure to proceed despite failures."
                )

        capture_coverage_gcov(build_dir, cov_json, test, repo_root=repo_root)

    print(f"\nCoverage files written to: {coverage_dir}", flush=True)

    if failures:
        print(f"\n{len(failures)} test(s) failed during collection:", flush=True)
        for f in failures:
            print(f"  {f}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
