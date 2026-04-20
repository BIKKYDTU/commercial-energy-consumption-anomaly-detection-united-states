# Rubric: `load_data` / Data Ingestion (`prompt.md` §1)

This document maps automated tests in `test_load_data.py` to `prompt.md` Key Requirements §1 and the **Expected Interface → `load_data`**, records gaps and overfits, and lists items that are **manual or reviewer-only** when they cannot be asserted safely in code without overfitting.

**Authoritative source:** `prompt.md` only.

---

## Requirement checklist (what §1 + `load_data` obligate)

| ID | Requirement (verbatim intent) | Automated in `test_load_data.py` |
|----|-------------------------------|-----------------------------------|
| R1 | Pipeline accepts a **CSV file path** as input | Partially (`str(path)` used; no negative tests) |
| R2 | CSV has **exactly two columns**: `timestamp`, `consumption_kwh` | Not directly tested (output shape ≠ input column audit) |
| R3 | `timestamp` format **YYYY-MM-DD HH:MM:SS** (ISO-style per prompt) | Implicitly (fixtures use that format) |
| R4 | `consumption_kwh` values are **positive floats** in valid inputs | Partially — `test_load_data_two_columns_float64_timezone_naive_no_nan` now asserts all loaded values `> 0` for that fixture; other fixtures still rely on positive inputs only |
| R5 | Parsed index: **timezone-naive** `DatetimeIndex`, **hourly** | Partially (tz checked; hourly not explicitly asserted) |
| R6 | Missing hours in span: **forward-fill** from preceding valid | Yes (`test_load_data_forward_fills_internal_hourly_gaps`) |
| R7 | Leading NaNs after that: **back-fill** from nearest subsequent valid | Yes (`test_load_data_backfills_leading_na_after_reindex`) — see caveat below |
| R8 | `load_data` output: **monotonically increasing** hourly index, **no NaN** in `consumption_kwh`, **float64** | Partially (no NaN, float64, length; monotonic/freq not asserted) |

---

## Per-test evaluation (strict vs `prompt.md`)

### `test_load_data_two_columns_float64_timezone_naive_no_nan`

| Field | Content |
|-------|---------|
| **Maps to** | R4–R6 (partial), R8 (partial), `load_data` Output |
| **Status** | ⚠️ Overly Broad (Under-testing) |
| **Issue** | Does not assert **monotonically increasing** index or explicit **hourly frequency** (`prompt.md` `load_data` Description). Does not validate **exactly two columns** in the CSV or **positive float** as a constraint on invalid files. |
| **Fix suggestion** | Add `assert df.index.is_monotonic_increasing` and `assert pd.Timedelta(hours=1) == (df.index[1] - df.index[0])` (or `infer_freq` / `asfreq` checks) for a continuous series. Optional: negative test for `>2` columns if spec requires hard failure. |
| **Fix feasibility** | ✅ Fixable |
| **UI limitation** | ❌ Testable via Code |

---

### `test_load_data_forward_fills_internal_hourly_gaps`

| Field | Content |
|-------|---------|
| **Maps to** | R6, §1 “immediately preceding valid value” |
| **Status** | ✅ Correct |
| **Issue** | None material; spot-checks match forward-fill semantics for an internal gap. |
| **Fix feasibility** | NA |
| **UI limitation** | NA |

---

### `test_load_data_backfills_leading_na_after_reindex`

| Field | Content |
|-------|---------|
| **Maps to** | R7, `load_data` “back-fills any remaining leading NaN” |
| **Status** | ⚠️ Overly Specific **or** ⚠️ Prompt tension (see Issue) |
| **Issue** | Fixture puts **`NaN` in `consumption_kwh` cells** in the CSV. §1 states the CSV column **`consumption_kwh` (positive float)** — a literal reading excludes NaN in-file. The **same** back-fill rule can be tested by **omitting leading timestamps** (missing hourly rows) so every written cell remains a positive float, with gaps only from reindexing. |
| **Fix suggestion** | Replace explicit NaN cells with a CSV that starts at hour 6 (or only lists rows from the first valid hour) so leading hours are missing as **rows**, not invalid floats; then assert hours 0–5 match hour 6 after back-fill. |
| **Fix feasibility** | ✅ Fixable |
| **UI limitation** | ❌ Testable via Code |

---

## Gaps in the suite (suite-level Under-testing)

1. **No test** that invalid CSVs (wrong column names, extra columns, non-numeric timestamps) are rejected or handled as required — only if `prompt.md` implies strict failure (it says “MUST contain exactly two columns”; it does not specify error handling).
2. **No test** that **`filepath: str`** is the sole accepted input form (minor; Python typing is not runtime-enforced).
3. **Hourly frequency** and **monotonicity** are implied by lengths and continuity in happy paths but not asserted as general properties.

---

## Manual / reviewer rubric (when code tests are insufficient or ambiguous)

Use these for human grading or supplemental review when automated tests do not fully prove compliance.

| # | Criterion | Pass if |
|---|-----------|--------|
| M1 | CSV schema | File has exactly `timestamp` and `consumption_kwh`; no extra columns relied upon. |
| M2 | Timestamp strings | Values match `YYYY-MM-DD HH:MM:SS` as stated in §1. |
| M3 | Positivity | All loaded `consumption_kwh` values are `> 0` for valid inputs (per §1). |
| M4 | Index semantics | `DatetimeIndex` is timezone-naive and represents a complete hourly grid from min to max timestamp after load. |
| M5 | Fill logic order | Implementation applies **forward-fill for gaps** then **back-fill for leading NaNs** (order matters if both apply). |
| M6 | Documentation | `load_data` docstring or comments match `prompt.md` reindex + fill behavior (optional documentation rubric). |

---

## Quick reference: status summary

| Test | Status |
|------|--------|
| `test_load_data_two_columns_float64_timezone_naive_no_nan` | ⚠️ Overly Broad |
| `test_load_data_forward_fills_internal_hourly_gaps` | ✅ Correct |
| `test_load_data_backfills_leading_na_after_reindex` | ⚠️ Fixable prompt alignment (CSV NaN vs positive float) |

---

## Related files

- Tests: `tests/test_load_data.py` (§1 ingestion; do not duplicate §1 checks in `test_run_pipeline.py`)
- Spec: `prompt.md` §1 and Expected Interface `load_data`
- Top-level reviewer rubric: `rubrics.md` items **26–31** (Data ingestion / §1 manual criteria)
