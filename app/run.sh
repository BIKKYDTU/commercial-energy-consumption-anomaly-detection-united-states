#!/bin/bash

### COMMON SETUP; DO NOT MODIFY ###
set -e

# --- CONFIGURE THIS SECTION ---
# Energy Consumption Analytics Pipeline — tests in tests/; pipeline imported from PIPELINE_REPO_ROOT.
# In Docker verification, tests live under /eval_assets and the solution is injected into /app.
run_all_tests() {
    local repo_root parse_script
    # Resolve symlinks (verification installs this script as /usr/local/bin/run_tests -> /eval_assets/run_tests).
    repo_root="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
    if [[ -z "${PIPELINE_REPO_ROOT:-}" ]]; then
        if [[ "$repo_root" == "/eval_assets" ]]; then
            export PIPELINE_REPO_ROOT="/app"
        else
            export PIPELINE_REPO_ROOT="$repo_root"
        fi
    fi
    parse_script="${repo_root}/parsing.py"
    [[ -f "$parse_script" ]] || parse_script="${repo_root}/parse_results"
    echo "Running pipeline tests (pytest tests/) in ${repo_root} (PIPELINE_REPO_ROOT=${PIPELINE_REPO_ROOT})..."
    cd "$repo_root"
    pip install -r "${repo_root}/requirements.txt" >/dev/null 2>&1 || true
    # Merge pytest stdout+stderr into one log for parse_results.
    # Use tee only on a real TTY (local interactive runs). If stdout is redirected
    # — e.g. verification.sh: run_tests > stdout.txt — tee would duplicate writes to the
    # same file and corrupt output; use a single redirect instead.
    if [[ -t 1 ]]; then
        pytest tests/ --tb=short --no-header -v --color=yes \
            2>&1 | tee "${repo_root}/stdout.txt" || true
    else
        pytest tests/ --tb=short --no-header -v --color=no \
            > "${repo_root}/stdout.txt" 2>&1 || true
    fi
    python3 "$parse_script" \
        "${repo_root}/stdout.txt" \
        /dev/null \
        "${repo_root}/results.json"

    # Print totals from results.json (parsed pytest output)
    python3 -c '
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    sys.exit(0)
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    sys.exit(0)
tests = data.get("tests", [])
if not isinstance(tests, list):
    tests = []
total = len(tests)
passed = sum(1 for t in tests if isinstance(t, dict) and t.get("status") == "PASSED")
failed = sum(1 for t in tests if isinstance(t, dict) and t.get("status") == "FAILED")
skipped = sum(1 for t in tests if isinstance(t, dict) and t.get("status") == "SKIPPED")
print()
print("--- Test summary ---")
print(f"Total test cases: {total}")
print(f"Passed: {passed}")
if failed or skipped:
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
print("--------------------")
' "${repo_root}/results.json" || true
}
# --- END CONFIGURATION SECTION ---

### COMMON EXECUTION; DO NOT MODIFY ###
run_all_tests
