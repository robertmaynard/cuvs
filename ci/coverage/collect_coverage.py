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
# Test execution: ctest is invoked exactly ONCE, not once per test.
#   cuVS's tests are registered with gtest_discover_tests(DISCOVERY_MODE PRE_TEST),
#   which defers GTest enumeration to ctest run time. That means every ctest
#   invocation — even one selecting a single test — re-runs `--gtest_list_tests`
#   against every gtest_case test binary to rebuild the full catalog before it
#   can find the requested test. Measured on this build: ~109s of pure
#   discovery overhead per ctest invocation, vs. ~1-2s of actual test runtime.
#   At thousands of tests, invoking ctest per test makes discovery overhead the
#   dominant cost of collection.
#
#   Instead, `load_ctest_index()` calls `ctest --show-only=json-v1 -L gtest_case`
#   once at startup, paying that ~109s discovery cost a single time for the
#   whole run. This returns, per test, the exact argv ctest would have used
#   (binary path + `--gtest_filter=...` + any extra args GoogleTest's module
#   adds), plus its WORKING_DIRECTORY/ENVIRONMENT/TIMEOUT properties. Workers
#   then invoke that captured command directly via subprocess, bypassing ctest
#   entirely for the remaining N-1 tests. We take over what ctest previously
#   did for free: pass/fail via returncode, timeout enforcement, and
#   output-on-failure-only logging.
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
import time
from pathlib import Path

_MAX_SAFE_LEN = 200


def safe_name(test_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", test_name)
    if len(sanitized) <= _MAX_SAFE_LEN:
        return sanitized
    digest = hashlib.sha1(test_name.encode()).hexdigest()[:8]
    return sanitized[:_MAX_SAFE_LEN - 9] + "_" + digest


def detect_gpu_count() -> int:
    """Return the number of CUDA GPUs visible to nvidia-smi, or 1 if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, check=True,
        )
        count = sum(1 for line in result.stdout.splitlines() if line.strip())
        return max(1, count)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 1


def load_ctest_index(build_dir: Path) -> list[dict]:
    """
    Discover every gtest_case-labelled test exactly once via
    `ctest --show-only=json-v1`, returning an ordered list of
    {name, command, working_directory, environment, timeout} entries — one
    per ctest test *invocation*, in ctest's own discovery order.

    This is the only ctest invocation collection performs — each entry's
    `command` list is executed directly (bypassing ctest) for the remainder
    of the run, to avoid paying gtest_discover_tests(DISCOVERY_MODE
    PRE_TEST)'s full-binary rescan cost on every single test run.

    Returns a list, not a dict keyed by name: cuVS's parameterized GTest
    cases can stringify to identical ctest test names (observed: 486 duplicate
    names out of 7475 entries on a real build, e.g. two float rounding cases
    that print the same). ctest itself distinguishes these by internal numeric
    ID, not name, and runs both — collapsing to a name-keyed dict would
    silently drop one of every duplicate pair's invocation entirely.
    """
    result = subprocess.run(
        ["ctest", "--show-only=json-v1", "-L", "gtest_case"],
        cwd=build_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)

    entries: list[dict] = []
    for test in data.get("tests", []):
        command = test.get("command", [])
        props = {p["name"]: p.get("value") for p in test.get("properties", [])}

        env: dict[str, str] = {}
        for item in props.get("ENVIRONMENT") or []:
            if "=" in item:
                k, v = item.split("=", 1)
                env[k] = v

        timeout = None
        raw_timeout = props.get("TIMEOUT")
        if raw_timeout not in (None, ""):
            try:
                timeout = float(raw_timeout)
            except (TypeError, ValueError):
                timeout = None

        entries.append({
            "name": test["name"],
            "command": command,
            "working_directory": props.get("WORKING_DIRECTORY"),
            "environment": env,
            "timeout": timeout,
        })
    return entries


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
    test_slice: list[dict],
    build_dir_str: str,
    coverage_dir_str: str,
    repo_root_str: str,
    prefix_base_str: str,
    continue_on_failure: bool,
    resume: bool,
    n_gpus: int = 1,
) -> list[str]:
    """
    Run a slice of tests sequentially, writing one .cov.json and one .jit.log
    per test.  gcda files are isolated to this worker's GCOV_PREFIX directory
    so concurrent workers never interfere with each other.

    Each test is invoked directly via the command/cwd/env/timeout captured by
    load_ctest_index() — ctest itself is never invoked here, avoiding its
    per-invocation full-binary GTest rescan (see module docstring).

    Returns a list of test names that failed.
    """
    build_dir = Path(build_dir_str)
    coverage_dir = Path(coverage_dir_str)
    repo_root = Path(repo_root_str)
    prefix_dir = Path(prefix_base_str) / f"worker_{worker_id}"
    prefix_dir.mkdir(parents=True, exist_ok=True)

    # Per-test test-execution-vs-gcov-capture timing, appended as one JSON line
    # per test. Separate from .cov.json (whose schema build_mapping.py depends
    # on) — purely diagnostic, to measure how much of the per-test wall time is
    # test execution vs. gcov post-processing before deciding whether it's
    # worth overlapping the two.
    timing_log = coverage_dir / f"timing_worker_{worker_id}.jsonl"

    total = len(test_slice)
    failures: list[str] = []

    for i, entry in enumerate(test_slice, 1):
        test = entry["name"]
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

        # Apply the test's own ctest-declared environment first (if any), then
        # our worker pinning last so it always wins on conflict.
        env = os.environ.copy()
        env.update(entry["environment"])
        env["GCOV_PREFIX"] = str(prefix_dir)
        env["RTCX_JIT_TRACE_LOG"] = str(jit_log)
        env["CUDA_VISIBLE_DEVICES"] = str(worker_id % n_gpus)

        cwd = entry["working_directory"] or str(build_dir)

        test_start = time.monotonic()
        try:
            result = subprocess.run(
                entry["command"],
                cwd=cwd,
                env=env,
                timeout=entry["timeout"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            returncode = result.returncode
            output = result.stdout
        except subprocess.TimeoutExpired as exc:
            returncode = 1
            output = (exc.stdout or "") + f"\n[collect_coverage.py] TIMEOUT after {entry['timeout']}s"
        except OSError as exc:
            returncode = 1
            output = f"[collect_coverage.py] failed to launch test: {exc}"
        test_seconds = time.monotonic() - test_start

        if returncode != 0:
            print(f"  [w{worker_id}] WARNING: test failed — coverage captured anyway: {test}",
                  flush=True)
            print(output, flush=True)
            failures.append(test)
            if not continue_on_failure:
                return failures

        gcov_start = time.monotonic()
        capture_coverage_gcov(
            build_dir, cov_json, test,
            repo_root=repo_root,
            prefix_dir=prefix_dir,
        )
        gcov_seconds = time.monotonic() - gcov_start

        with open(timing_log, "a") as f:
            f.write(json.dumps({
                "test": test,
                "test_seconds": round(test_seconds, 3),
                "gcov_seconds": round(gcov_seconds, 3),
            }) + "\n")

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
        default=None,
        help="Number of parallel collection workers (default: number of GPUs detected by nvidia-smi)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs available for CUDA_VISIBLE_DEVICES pinning "
             "(default: auto-detected via nvidia-smi)",
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

    print(f"Discovering gtest_case tests in {build_dir} (one-time ctest invocation) ...",
          flush=True)
    tests = load_ctest_index(build_dir)
    if not tests:
        sys.exit(
            "ERROR: no tests found with label 'gtest_case'. "
            "Ensure the coverage build includes gtest_discover_tests() in ConfigureTest()."
        )
    unique_names = len({t["name"] for t in tests})
    print(f"Found {len(tests)} test invocation(s) ({unique_names} unique name(s)). "
          f"ctest will not be invoked again for the remainder of this run.", flush=True)

    (coverage_dir / "ctest_test_index.json").write_text(json.dumps(tests, indent=2) + "\n")

    n_gpus = args.gpus if args.gpus is not None else detect_gpu_count()
    n_workers = min(args.jobs if args.jobs is not None else n_gpus, len(tests))
    print(f"GPUs: {n_gpus}  workers: {n_workers}", flush=True)

    prefix_base = coverage_dir / ".gcda_workers"
    prefix_base.mkdir(parents=True, exist_ok=True)

    # Round-robin assignment keeps each worker's slice roughly equal in length
    # and spreads slow/fast tests across workers rather than front-loading them.
    chunks: list[list[dict]] = [tests[i::n_workers] for i in range(n_workers)]

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
            n_gpus,
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
        if not args.continue_on_failure:
            sys.exit(1)


if __name__ == "__main__":
    main()
