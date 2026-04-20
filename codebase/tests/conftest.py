"""
Shared pytest fixtures for the Energy Consumption Analytics Pipeline test suite.

All fixtures write two-column CSVs: `timestamp`, `consumption_kwh` (hourly).
That schema is the test harness contract; prompt.md only requires hourly data.

Related prompt.md coverage elsewhere:
  tests/test_load_data.py — §1 ingestion (`load_data`).
  tests/test_tech_stack.py — Tech Stack version pins.
  tests/test_run_pipeline.py — run_pipeline outputs and HTML §9.
"""

import datetime
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

# Ensure pipeline.py is importable: Docker uses PIPELINE_REPO_ROOT=/app; local runs use repo root.
_pipeline_root = os.environ.get("PIPELINE_REPO_ROOT")
if _pipeline_root:
    sys.path.insert(0, _pipeline_root)
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pipeline  # noqa: F401
except ImportError:
    # Docker "before" run: /app has no pipeline.py yet — register a stub so tests
    # collect and fail (instead of ERROR during import), populating before.json.
    _stub = types.ModuleType("pipeline")
    _root = os.environ.get("PIPELINE_REPO_ROOT", "")
    _stub.__file__ = (
        os.path.join(_root, "pipeline.py")
        if _root
        else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "pipeline.py",
        )
    )

    def _missing(*_a, **_kw):
        raise NotImplementedError("pipeline module not available")

    # Real parameter names so inspect.signature matches prompt.md (avoids FAILED on stub).
    def _load_data_missing(filepath):
        raise NotImplementedError("pipeline module not available")

    def _run_pipeline_missing(input_path, output_path, cost_per_kwh):
        raise NotImplementedError("pipeline module not available")

    _stub.load_data = _load_data_missing
    _stub.run_pipeline = _run_pipeline_missing
    for _fn in (
        "compute_daily_summary",
        "compute_monthly_summary",
        "identify_usage_patterns",
        "detect_anomalies",
        "detect_seasonal_trends",
        "estimate_baseline",
        "project_next_month",
        "generate_report",
    ):
        setattr(_stub, _fn, _missing)
    sys.modules["pipeline"] = _stub


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _write_csv(path, index, values):
    """Write a two-column timestamp + consumption_kwh CSV and return its path."""
    df = pd.DataFrame(
        {
            "timestamp": index.strftime("%Y-%m-%d %H:%M:%S"),
            "consumption_kwh": values,
        }
    )
    df.to_csv(str(path), index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def business_weekend_pattern_csv(tmp_path):
    """
    Hourly series where weekday business hours (Mon–Fri 09–17) are much higher
    than nights and weekends — matches prompt.md problem-context wording for a
    'typical' commercial profile (no particular detection algorithm assumed).
    """
    idx = pd.date_range("2024-01-01", periods=14 * 24, freq="h")
    vals = []
    for ts in idx:
        h, dow = ts.hour, ts.dayofweek
        if dow >= 5:
            vals.append(2.0)
        elif 9 <= h <= 17:
            vals.append(100.0)
        else:
            vals.append(2.0)
    return _write_csv(tmp_path / "business_weekend.csv", idx, np.array(vals, dtype=float))


@pytest.fixture
def anomaly_csv(tmp_path):
    """
    28 days (Mon 2021-01-04 – Sun 2021-01-31).
    All hours are 10.0 kWh except one calendar day in the range, which has much
    higher hourly use (single outlier for anomaly detection tests).
    """
    idx = pd.date_range("2021-01-04", periods=24 * 28, freq="h")
    anomaly_date = datetime.date(2021, 1, 18)
    values = np.where(
        np.array([ts.date() for ts in idx]) == anomaly_date,
        50.0,
        10.0,
    )
    return _write_csv(tmp_path / "anomaly.csv", idx, values)


@pytest.fixture
def zscore_outlier_heatmap_csv(tmp_path):
    """
    Four weeks (Mon 2022-01-03 – Sun 2022-01-30) with one weekday (2022-01-17)
    at 100.0 kWh/h and all other weekdays at 10.0 kWh/h; weekends alternate
    8.0 / 12.0 kWh/h per day so both strata have σ > 0. The outlier Monday has
    daily total 2400 vs 240 for normal weekdays, so |z| > 2 in the weekday
    stratum under prompt.md §5 — a correct pipeline MUST flag at least that day.
    Used for heatmap red-annotation tests so they never skip for “no anomalies”.
    """
    start = pd.Timestamp("2022-01-03 00:00:00")
    end = pd.Timestamp("2022-01-30 23:00:00")
    idx = pd.date_range(start, end, freq="h")
    outlier_date = pd.Timestamp("2022-01-17").date()
    kwh_values = []
    for ts in idx:
        if ts.dayofweek < 5:
            kwh_values.append(100.0 if ts.date() == outlier_date else 10.0)
        else:
            day_num = (ts.date() - start.date()).days
            kwh_values.append(8.0 if (day_num % 2 == 0) else 12.0)
    return _write_csv(
        tmp_path / "zscore_outlier_heatmap.csv", idx, np.array(kwh_values, dtype=float)
    )


@pytest.fixture
def multi_year_csv(tmp_path):
    """
    Two full calendar years 2021-01-01 to 2022-12-31, constant 10.0 kWh/h.
    Last date = 2022-12-31 → target month = January 2023 (31 days).
    Prior January data (2021 + 2022): every day totals 240.0 kWh.
    Expected projection = 240.0 × 31 = 7 440.0 kWh.
    """
    idx = pd.date_range("2021-01-01", "2022-12-31 23:00:00", freq="h")
    return _write_csv(tmp_path / "multiyear.csv", idx, 10.0)


@pytest.fixture
def seasonal_csv(tmp_path):
    """
    Two full calendar years 2021-01-01 to 2022-12-31 with pronounced seasonal
    variation — summer months (Jun–Aug) are 5× higher than winter months
    (Dec–Feb) — so a pipeline that detects seasonal effects must expose at
    least two distinct numeric metrics in seasonal_trends.

    Per-hour values:
      Jun–Aug  → 50.0 kWh  (peak cooling season)
      Dec–Feb  → 10.0 kWh  (low heating / base load)
      all other months → 25.0 kWh
    """
    idx = pd.date_range("2021-01-01", "2022-12-31 23:00:00", freq="h")
    vals = np.where(
        idx.month.isin([6, 7, 8]),
        50.0,
        np.where(idx.month.isin([12, 1, 2]), 10.0, 25.0),
    )
    return _write_csv(tmp_path / "seasonal.csv", idx, vals)
