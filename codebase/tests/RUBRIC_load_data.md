# Rubric: `load_data` / data ingestion

**Scope:** Key Requirements §1 and the Expected Interface for `load_data` in `prompt.md`.

**Authoritative source:** `prompt.md`.

---

## Grading criteria

Each criterion is **self-contained**, **positive** (describes what success looks like), and **general** (not tied to specific test names or implementation details). Use the **score** and **classification** fields when entering criteria in the rubric UI.

---

### 1. CSV schema and inputs

**Criterion:** The implementation accepts CSV input using exactly the two columns specified for timestamps and consumption, and reads them as the primary data source for ingestion.

**Score:** 5 — Mandatory  

**Classification:** Instruction Following

---

### 2. Timestamp parsing

**Criterion:** Timestamp values are parsed according to the required string format and represented as a timezone-naive datetime index suitable for time-series use.

**Score:** 5 — Mandatory  

**Classification:** Code Correctness

---

### 3. Consumption values

**Criterion:** Consumption values are loaded as positive floating-point numbers with the numeric dtype expected by the specification.

**Score:** 5 — Mandatory  

**Classification:** Code Correctness

---

### 4. Hourly series, gaps, and fill order

**Criterion:** The loaded series uses a regular hourly grid from the minimum to the maximum timestamp; internal missing hours are filled from the immediately preceding valid value, and any remaining leading missing values are filled from the nearest subsequent valid value, following the order required by the specification.

**Score:** 5 — Mandatory  

**Classification:** Code Correctness

---

### 5. Output integrity for downstream use

**Criterion:** The returned table has a monotonically increasing time index, no missing consumption values where the specification requires a complete series, and dtypes appropriate for the rest of the pipeline.

**Score:** 3 — Important  

**Classification:** Code Correctness

---

### 6. Clear structure

**Criterion:** The loading and transformation steps are organized so that data flow from file read through reindexing and filling is easy to follow.

**Score:** 3 — Important  

**Classification:** Code Quality

---

### 7. Documented behavior

**Criterion:** The public interface (for example the docstring) briefly describes how data is loaded, reindexed, and filled.

**Score:** 1 — Nice to have (optional)  

**Classification:** Code Clarity

---

## Related files

- Tests: `tests/test_load_data.py`
- Spec: `prompt.md` §1 and Expected Interface `load_data`
