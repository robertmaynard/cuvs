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
#
# Parallelism (-j N):
#   Each worker is assigned a non-overlapping slice of the test list and runs
#   its tests sequentially.  Workers are isolated via GCOV_PREFIX: each worker
#   gets its own directory under <coverage-dir>/.gcda_workers/worker_N/ so
#   gcda files from different workers never mix.  Use -j 1 on a single-GPU
#   machine; set -j to match the number of available GPUs.

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import multiprocessing
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


def _process_gcda(args: tuple[int, Path, Path, Path | None]) -> list[dict]:
    """Run gcov --json-format on one gcda file; return parsed file entries."""
    idx, gcda, tmp, prefix_dir = args
    work = tmp / str(idx)
    work.mkdir(exist_ok=True)

    if prefix_dir is not None:
        # Recover the build-tree directory where the matching .gcno lives by
        # stripping the GCOV_PREFIX from the gcda path.
        rel = gcda.relative_to(prefix_dir)
        gcno_dir = Path("/") / rel.parent
        gcov_cmd = ["gcov", "--json-format", "--demangled-names",
                    "-o", str(gcno_dir), str(gcda)]
    else:
        gcov_cmd = ["gcov", "--json-format", "--demangled-names", str(gcda)]

    subprocess.run(gcov_cmd, cwd=work, capture_output=True)

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
    prefix_dir: Path | None = None,
) -> None:
    """
    Run gcov --json-format on the gcda files produced by the test, filter to
    cuVS sources, and write a per-test .cov.json file.

    When prefix_dir is set (parallel mode), gcda files are searched under that
    directory (they were redirected there via GCOV_PREFIX).  gcov is pointed at
    the build tree for .gcno files via -o.
    """
    if prefix_dir is not None:
        gcda_files = list(prefix_dir.rglob("*.gcda"))
    else:
        gcda_files = []
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
        work_args = [(i, gcda, tmp, prefix_dir) for i, gcda in enumerate(gcda_files)]

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


# ---------------------------------------------------------------------------
# Worker — must be module-level to be picklable by multiprocessing
# ---------------------------------------------------------------------------

def _collect_worker(
    worker_id: int,
    test_slice: list[tuple[int, str]],
    build_dir_str: str,
    coverage_dir_str: str,
    repo_root_str: str,
    prefix_base_str: str,
    continue_on_failure: bool,
    resume: bool,
) -> list[str]:
    """
    Run a slice of tests sequentially, writing one .cov.json and one .jit.log
    per test.  gcda files are isolated to this worker's GCOV_PREFIX directory
    so concurrent workers never interfere with each other.

    Returns a list of test names that failed.
    """
    build_dir = Path(build_dir_str)
    coverage_dir = Path(coverage_dir_str)
    repo_root = Path(repo_root_str)
    prefix_dir = Path(prefix_base_str) / f"worker_{worker_id}"
    prefix_dir.mkdir(parents=True, exist_ok=True)

    total = len(test_slice)
    failures: list[str] = []

    for i, (test_index, test) in enumerate(test_slice, 1):
        sname = safe_name(test)
        cov_json = coverage_dir / f"{sname}.cov.json"
        jit_log = coverage_dir / f"{sname}.jit.log"

        if resume and cov_json.exists():
            print(f"[w{worker_id} {i}/{total}] skip: {test}", flush=True)
            continue

        print(f"[w{worker_id} {i}/{total}] {test}", flush=True)

        # Clear this worker's isolated gcda tree before each test
        for gcda in prefix_dir.rglob("*.gcda"):
            gcda.unlink(missing_ok=True)

        env = os.environ.copy()
        env["GCOV_PREFIX"] = str(prefix_dir)
        env["CUVS_JIT_TRACE_LOG"] = str(jit_log)

        result = subprocess.run(
            ["ctest", "-I", f"{test_index},{test_index}", "--output-on-failure"],
            cwd=build_dir,
            env=env,
        )
        if result.returncode != 0:
            print(f"  [w{worker_id}] WARNING: test failed — coverage captured anyway: {test}",
                  flush=True)
            failures.append(test)
            if not continue_on_failure:
                return failures

        capture_coverage_gcov(
            build_dir, cov_json, test,
            repo_root=repo_root,
            prefix_dir=prefix_dir,
        )

    return failures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=6,
        help="Number of parallel collection workers; set to match available GPUs (default: 6)",
    )
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    coverage_dir = (
        Path(args.coverage_dir).resolve() if args.coverage_dir
        else build_dir / "per_test_coverage"
    )
    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root
        else build_dir.parent.parent.parent
    )

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

    n_workers = min(args.jobs, len(tests))
    prefix_base = coverage_dir / ".gcda_workers"
    prefix_base.mkdir(parents=True, exist_ok=True)

    # Round-robin assignment keeps each worker's slice roughly equal in length
    # and spreads slow/fast tests across workers rather than front-loading them.
    chunks: list[list[tuple[int, str]]] = [tests[i::n_workers] for i in range(n_workers)]

    worker_args = [
        (
            i,
            chunk,
            str(build_dir),
            str(coverage_dir),
            str(repo_root),
            str(prefix_base),
            args.continue_on_failure,
            args.resume,
        )
        for i, chunk in enumerate(chunks)
    ]

    print(f"Running {n_workers} worker(s) ...", flush=True)

    if n_workers == 1:
        all_failures = [_collect_worker(*worker_args[0])]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_collect_worker, *wa) for wa in worker_args]
            all_failures = [f.result() for f in futures]

    failures = [name for worker_failures in all_failures for name in worker_failures]

    print(f"\nCoverage files written to: {coverage_dir}", flush=True)

    if failures:
        print(f"\n{len(failures)} test(s) failed during collection:", flush=True)
        for f in failures:
            print(f"  {f}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
