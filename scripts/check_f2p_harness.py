#!/usr/bin/env python3
"""Non-pytest checks for F2P harness (run from repo root before packing tests.zip)."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    trp = root / "tests" / "test_run_pipeline.py"
    if not trp.is_file():
        print("ERROR: tests/test_run_pipeline.py missing", file=sys.stderr)
        return 1
    text = trp.read_text(encoding="utf-8")
    if "pipeline_output" in text:
        print(
            "ERROR: pipeline_output must not appear in test_run_pipeline.py "
            "(causes pytest ERROR on empty baseline).",
            file=sys.stderr,
        )
        return 1
    if "def multi_year_run" not in text or "return _run" not in text:
        print(
            "ERROR: multi_year_run factory pattern missing in test_run_pipeline.py.",
            file=sys.stderr,
        )
        return 1
    print("check_f2p_harness: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
