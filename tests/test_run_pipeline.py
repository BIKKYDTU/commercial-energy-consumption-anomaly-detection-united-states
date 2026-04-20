"""
Tests for run_pipeline aligned to explicit requirements in prompt.md.

Requirement map (prompt.md → tests):

  Problem context — Analytics outputs MUST reflect:
    • daily and monthly summaries
    • typical usage pattern (business-hours vs night/weekend profile)
    • anomalous days
    • seasonal trends
    • baseline always-on (nights/weekends)
    • next-month projection

  Key Requirements — Analytics outputs (typed dict keys):
    • daily_summary, monthly_summary, usage_patterns, anomalies,
      seasonal_trends, baseline_kwh, projected_next_month_kwh

  HTML Report:
    • Daily heat map: Plotly heatmap, 24 rows (hours 0–23), one column per
      calendar day, colour encodes consumption_kwh
    • Anomaly highlights: distinct red annotations on the heat map for
      anomalous days
    • Cost estimate: literal text "Total Estimated Cost = $X.XX" using
      sum(consumption_kwh) × cost_per_kwh rounded to two decimal places
    • Embed complete Plotly JS inline (no external network)

  Expected Interface:
    • pipeline.py, run_pipeline(input_path, output_path, cost_per_kwh)
    • Side effect: self-contained HTML; cost line per prompt.md §9
      (Total Estimated Cost = $X.XX from sum(consumption_kwh) × cost_per_kwh, two decimals)

usage_patterns and seasonal_trends: tests require the four keys named in
prompt.md Expected Interface, each a scalar positive float, with values checked
against independent segment/season means computed from load_data (extra keys
allowed if present).

Anomaly outputs (columns, dtypes, z-scores, σ=0 behaviour, thresholds) are
asserted directly against prompt.md. For HTML red-annotation tests,
prompt_helpers.normalize_anomaly_dates and red_marking_dates resolve flagged
dates and red markup without assuming a single Plotly embedding layout. The
default multi-year fixture allows an empty anomalies table when a solution finds
no outliers.
Heatmap assertions use primary_heatmap_trace to avoid depending on trace
order when multiple heatmaps exist. baseline_kwh / projected_next_month_kwh
use prompt float typing (float | numpy.floating). For multi_year_csv only,
projected_next_month_kwh is checked against the fixture contract (constant
hourly kWh; last timestamp in Dec 2022 → next month Jan 2023 total kWh).
HTML tests require non-empty output, no external script src= URLs, inline
<script> blocks with non-empty embedded bodies and a plotly reference, heatmap trace
detection (plotly_report_includes_heatmap), and heatmap z aligned to hourly
consumption_kwh.

Suite-level coverage outside this file (prompt.md):
  • §1 Data Ingestion (`load_data`) — `tests/test_load_data.py` (forward-fill,
    back-fill, schema, dtypes). Not duplicated here to avoid double maintenance.
  • Tech Stack (Python 3.10+, pandas 2.x, numpy 1.x, plotly 5.x) —
    `tests/test_tech_stack.py`.
  • Problem description (“three years of hourly records”) is narrative context;
    Key Requirements do not fix a minimum calendar span — fixtures use
    multi-year hourly ranges where a test needs them.

Changes from original:
  T7a — σ=0 weekday stratum only (removed non-normative weekend z≠0 check).
  T7b — includes total_kwh vs daily hourly-sum verification.
  T10 — <table> for anomalies; rows must cover every flagged date (extras allowed);
        cost literal $X.XX; self-contained Plotly heatmap + inline JS (no CDN).
  T12 — heatmap z equals hourly consumption_kwh on the 24×N grid; y-axis rows are
        hours 0–23 (assert_heatmap_y_axis_rows_are_hours_0_to_23).
  T13 — 24×days heatmap grid, finite z, y-axis hours 0–23; z equality to kWh is T12.
  T15 — heatmap presence instead of mandating Plotly.newPlot/react.
  T17 — zscore_outlier_heatmap_csv (guaranteed ≥1 flagged day); red on flagged days;
        no red on non-flagged; red x dates ⊆ dataset days.
  §1 / Tech Stack — see tests/test_load_data.py and tests/test_tech_stack.py.
"""

from __future__ import annotations

import datetime
import inspect
import os
import pathlib
import re
import numpy as np
import pandas as pd
import pytest

from pipeline import load_data, run_pipeline

from prompt_helpers import (
    assert_heatmap_y_axis_rows_are_hours_0_to_23,
    heatmap_traces_from_html,
    html_includes_inline_plotly_bundle_definition,
    layout_annotations_use_red_on_heatmap,
    normalize_anomaly_dates,
    plotly_report_includes_heatmap,
    primary_heatmap_trace,
    red_marking_dates,
)


def _is_prompt_float(x: object) -> bool:
    """prompt.md specifies float for baseline_kwh and projected_next_month_kwh."""
    return isinstance(x, (float, np.floating)) and not isinstance(x, bool)


def _total_inline_script_body_chars(html: str) -> int:
    """Sum lengths of <script> bodies with no src= attribute (JS embedded in the HTML)."""
    total = 0
    for m in re.finditer(r"<script\b([^>]*)>([\s\S]*?)</script>", html, re.I):
        attrs = m.group(1)
        if re.search(r"\bsrc\s*=", attrs, re.I):
            continue
        total += len(m.group(2))
    return total


def _assert_self_contained_plotly_html(html: str) -> None:
    """
    prompt.md §9: self-contained report — no external script URLs; JavaScript and
    Plotly-related content embedded inline; heatmap figure detectable (no CDN).

    Does not require a specific Plotly API token (e.g. `Plotly.`) or a minimum
    script byte count (prompt.md does not specify one); requires non-empty
    embedded script bodies to reject empty stubs.
    """
    assert not re.search(r"<script[^>]+src=[\"'](https?:|//)", html, re.I), (
        "HTML must not load scripts from an external URL (prompt.md: no external network)"
    )
    assert re.search(r"<script\b", html, re.I), (
        "HTML must contain inline <script> block(s) (prompt.md: JavaScript embedded in the file)"
    )
    assert "plotly" in html.lower(), (
        "Expected Plotly embedded in the HTML file (prompt.md: inline Plotly JS)."
    )
    inline_js = _total_inline_script_body_chars(html)
    assert inline_js > 0, (
        "prompt.md §9: JavaScript must be embedded in the file — expected non-empty "
        f"inline <script> body(ies) without src=; total chars={inline_js}."
    )
    assert plotly_report_includes_heatmap(html), (
        "HTML must embed a Plotly heatmap (prompt.md §9: daily heat map)."
    )
    assert html_includes_inline_plotly_bundle_definition(html), (
        "prompt.md §9: Plotly JavaScript library must be fully included within the file"
    )


def _visible_plain_text(html: str) -> str:
    """Body-visible text only: strip <script> blocks and HTML tags."""
    no_scripts = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    return re.sub(r"<[^>]+>", " ", no_scripts)


def _assert_heatmap_z_encodes_consumption(z_arr: np.ndarray, expected_kwh: np.ndarray) -> None:
    """
    prompt.md §9: cell color encodes consumption_kwh on a 24×N hour×day grid.

    Plotly uses trace z as the value that drives the color scale; for this report,
    those values must be the hourly consumption_kwh cells (same layout as the
    hourly series).
    """
    assert z_arr.shape == expected_kwh.shape, "z grid shape must match hour×day layout"
    assert np.all(np.isfinite(z_arr)), "heatmap z must contain only finite values (prompt.md §9)"
    np.testing.assert_allclose(
        z_arr,
        expected_kwh,
        rtol=1e-5,
        atol=1e-6,
        err_msg=(
            "heatmap z must match hourly consumption_kwh for each hour×day cell "
            "(prompt.md §9: color encodes consumption_kwh)"
        ),
    )


def _parse_three_column_table_rows(html: str) -> list[tuple[datetime.date, float, float]]:
    """
    Scan all HTML tables for data rows with exactly three <td> cells that parse as
    date, float, float (prompt.md §9 anomaly table shape: date, total_kwh, z_score).
    """
    parsed: list[tuple[datetime.date, float, float]] = []
    for table_match in re.finditer(r"<table[^>]*>([\s\S]*?)</table>", html, re.I):
        inner = table_match.group(1)
        tb = re.search(r"<tbody[^>]*>([\s\S]*?)</tbody>", inner, re.I)
        body = tb.group(1) if tb else inner
        for tr in re.finditer(r"<tr[^>]*>([\s\S]*?)</tr>", body, re.I):
            tds = re.findall(r"<td[^>]*>([\s\S]*?)</td>", tr.group(1), re.I)
            if len(tds) != 3:
                continue
            d_raw = re.sub(r"<[^>]+>", "", tds[0]).strip()
            t_raw = re.sub(r"<[^>]+>", "", tds[1]).strip()
            z_raw = re.sub(r"<[^>]+>", "", tds[2]).strip()
            d_ts = pd.to_datetime(d_raw, errors="coerce")
            if pd.isna(d_ts):
                continue
            try:
                parsed.append((d_ts.date(), float(t_raw), float(z_raw)))
            except ValueError:
                continue
    return parsed


def _expected_pipeline_py() -> pathlib.Path:
    """Submission root per evaluation: PIPELINE_REPO_ROOT, else repo root (tests/..)."""
    root = os.environ.get("PIPELINE_REPO_ROOT")
    if root:
        return pathlib.Path(root) / "pipeline.py"
    return pathlib.Path(__file__).resolve().parent.parent / "pipeline.py"


_REQUIRED_KEYS = {
    "daily_summary",
    "monthly_summary",
    "usage_patterns",
    "anomalies",
    "seasonal_trends",
    "baseline_kwh",
    "projected_next_month_kwh",
}


@pytest.fixture
def multi_year_run(multi_year_csv, tmp_path):
    """
    Factory: call ``multi_year_run()`` inside a test (not during fixture setup).

    If ``run_pipeline`` raises (e.g. stub NotImplementedError before pipeline.py
    exists), the failure happens in the test phase so pytest reports FAILED, not
    ERROR from a fixture teardown/setup failure.
    """
    def _run(cost_per_kwh: float = 0.12):
        output_path = str(tmp_path / "report.html")
        result = run_pipeline(multi_year_csv, output_path, cost_per_kwh=cost_per_kwh)
        return result, output_path, multi_year_csv

    return _run


# ---------------------------------------------------------------------------
# TestPromptInterface
# ---------------------------------------------------------------------------

class TestPromptInterface:
    """prompt.md — Expected Interface: run_pipeline name, inputs, outputs."""

    def test_run_pipeline_is_callable_with_documented_signature(self):
        """prompt.md: run_pipeline(input_path, output_path, cost_per_kwh) — names per Expected Interface."""
        # Require pipeline.py on disk at the submission root. Do not use ``pipeline.__file__``:
        # Python may import a different ``pipeline`` from another ``sys.path`` entry.
        pp = _expected_pipeline_py()
        assert pp.is_file() and pp.stat().st_size > 0, (
            "Expected non-empty pipeline.py at repository root (prompt.md); "
            "missing or stub-only baseline must not pass this test."
        )
        assert callable(run_pipeline)
        sig = inspect.signature(run_pipeline)
        params = list(sig.parameters.keys())
        assert params == ["input_path", "output_path", "cost_per_kwh"]
        sig.bind("x.csv", "out.html", 0.1)
        # Return dict, HTML side effect, and analytics: test_returns_dict_with_seven_typed_keys
        # and TestPromptAnalyticsOutputs.


# ---------------------------------------------------------------------------
# TestPromptAnalyticsOutputs
# ---------------------------------------------------------------------------

class TestPromptAnalyticsOutputs:
    """
    prompt.md — Key Requirements / Problem context:
    return value MUST reflect daily & monthly summaries, usage_patterns,
    anomalies, seasonal_trends, baseline_kwh, projected_next_month_kwh.
    """

    def test_returns_dict_with_seven_typed_keys(self, multi_year_run):
        """
        prompt.md Expected Interface — run_pipeline:
        return dict with exactly seven keys and types; HTML at output_path exists with
        non-zero byte size on return.
        """
        result, output_path, _ = multi_year_run()
        assert isinstance(result, dict)
        missing = _REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing required keys: {missing}"
        assert set(result.keys()) == _REQUIRED_KEYS, (
            "run_pipeline must return exactly the seven keys in prompt.md (no extras)"
        )
        assert isinstance(result["daily_summary"], pd.DataFrame)
        assert isinstance(result["monthly_summary"], pd.DataFrame)
        assert isinstance(result["usage_patterns"], dict), (
            "usage_patterns must be a dict (prompt.md Expected Interface)"
        )
        assert isinstance(result["anomalies"], pd.DataFrame)
        assert isinstance(result["seasonal_trends"], dict), (
            "seasonal_trends must be a dict (prompt.md Expected Interface)"
        )
        assert _is_prompt_float(result["baseline_kwh"])
        assert _is_prompt_float(result["projected_next_month_kwh"])

        assert os.path.isfile(output_path), (
            "run_pipeline must write the HTML report to output_path (prompt.md)"
        )
        assert os.path.getsize(output_path) > 0, (
            "HTML file at output_path must be non-zero size on return (prompt.md)"
        )

    # ------------------------------------------------------------------
    # daily_summary — verify all four required columns, column dtypes,
    # and that total/mean/min/max match hourly aggregates per calendar day
    # (prompt.md §2 + compute_daily_summary spec).
    # ------------------------------------------------------------------
    def test_daily_summary_covers_each_calendar_day_in_input(self, multi_year_run):
        """
        prompt.md §2 + compute_daily_summary:
        - one row per calendar day
        - columns: date (datetime.date), total_kwh, mean_kwh, min_kwh, max_kwh (float64)
        - total_kwh, mean_kwh, min_kwh, max_kwh MUST match hourly aggregates per day
        """
        result, _, csv_path = multi_year_run()
        df = load_data(csv_path)
        expected_days = int(df.index.normalize().nunique())
        daily = result["daily_summary"]

        assert isinstance(daily, pd.DataFrame)
        assert not daily.empty, "daily_summary must not be empty for non-empty hourly input (prompt.md)"
        assert len(daily) == expected_days, (
            "daily_summary must contain exactly one row per calendar day in the loaded hourly range (prompt.md)"
        )

        # Required column names per prompt.md compute_daily_summary spec
        for col in ("date", "total_kwh", "mean_kwh", "min_kwh", "max_kwh"):
            assert col in daily.columns, (
                f"daily_summary must contain column '{col}' (prompt.md compute_daily_summary)"
            )

        # date column must be datetime.date dtype
        assert all(isinstance(d, datetime.date) for d in daily["date"]), (
            "daily_summary 'date' column must contain datetime.date values (prompt.md)"
        )

        # float64 for numeric columns
        for col in ("total_kwh", "mean_kwh", "min_kwh", "max_kwh"):
            assert daily[col].dtype == np.float64, (
                f"daily_summary column '{col}' must be float64 (prompt.md)"
            )

        # total/mean/min/max must match hourly consumption_kwh for each calendar day
        daily_indexed = daily.set_index("date")
        for day_ts, group in df.groupby(df.index.normalize()):
            day_date = day_ts.date()
            kwh = group["consumption_kwh"]
            expected_total = float(kwh.sum())
            expected_mean = float(kwh.mean())
            expected_min = float(kwh.min())
            expected_max = float(kwh.max())
            row = daily_indexed.loc[day_date]
            np.testing.assert_allclose(
                float(row["total_kwh"]),
                expected_total,
                rtol=1e-9,
                err_msg=(
                    f"daily_summary total_kwh on {day_date} must equal sum of hourly "
                    "consumption_kwh for that day (prompt.md §2)"
                ),
            )
            np.testing.assert_allclose(
                float(row["mean_kwh"]),
                expected_mean,
                rtol=1e-9,
                err_msg=(
                    f"daily_summary mean_kwh on {day_date} must equal mean hourly "
                    "consumption_kwh for that day (prompt.md §2)"
                ),
            )
            np.testing.assert_allclose(
                float(row["min_kwh"]),
                expected_min,
                rtol=1e-9,
                err_msg=(
                    f"daily_summary min_kwh on {day_date} must equal minimum hourly "
                    "consumption_kwh for that day (prompt.md §2)"
                ),
            )
            np.testing.assert_allclose(
                float(row["max_kwh"]),
                expected_max,
                rtol=1e-9,
                err_msg=(
                    f"daily_summary max_kwh on {day_date} must equal maximum hourly "
                    "consumption_kwh for that day (prompt.md §2)"
                ),
            )

    # ------------------------------------------------------------------
    # monthly_summary — verify required column names, 'YYYY-MM' format,
    # and aggregation math against daily totals
    # (prompt.md §3 + compute_monthly_summary spec).
    # ------------------------------------------------------------------
    def test_monthly_summary_aggregates_calendar_months(self, multi_year_run):
        """
        prompt.md §3 + compute_monthly_summary:
        - one row per calendar month
        - columns: year_month (str 'YYYY-MM'), total_kwh, mean_daily_kwh,
          min_daily_kwh, max_daily_kwh (float64)
        - total_kwh is sum of all daily totals in that month
        - mean/min/max_daily_kwh are derived from daily total_kwh values
        """
        result, _, csv_path = multi_year_run()
        hourly = load_data(csv_path)
        expected_months = (
            pd.to_datetime(hourly.index.normalize().unique()).to_period("M").nunique()
        )
        monthly = result["monthly_summary"]
        daily = result["daily_summary"]

        assert isinstance(monthly, pd.DataFrame)
        assert not monthly.empty, "monthly_summary must not be empty for multi-month input (prompt.md)"
        assert len(monthly) == expected_months, (
            "monthly_summary must contain exactly one row per calendar month "
            "present in the input range (prompt.md)"
        )

        # Required column names per prompt.md compute_monthly_summary spec
        for col in ("year_month", "total_kwh", "mean_daily_kwh", "min_daily_kwh", "max_daily_kwh"):
            assert col in monthly.columns, (
                f"monthly_summary must contain column '{col}' (prompt.md compute_monthly_summary)"
            )

        # year_month must be formatted as 'YYYY-MM'
        ym_pattern = re.compile(r"^\d{4}-\d{2}$")
        for ym in monthly["year_month"]:
            assert isinstance(ym, str) and ym_pattern.match(ym), (
                f"year_month value '{ym}' must match 'YYYY-MM' format (prompt.md)"
            )

        # float64 for numeric columns
        for col in ("total_kwh", "mean_daily_kwh", "min_daily_kwh", "max_daily_kwh"):
            assert monthly[col].dtype == np.float64, (
                f"monthly_summary column '{col}' must be float64 (prompt.md)"
            )

        # Verify aggregation math: total_kwh = sum of daily total_kwh per month;
        # mean/min/max_daily_kwh derived from those daily totals.
        daily_copy = daily.copy()
        daily_copy["year_month"] = pd.to_datetime(daily_copy["date"]).dt.strftime("%Y-%m")
        monthly_indexed = monthly.set_index("year_month")

        for ym, grp in daily_copy.groupby("year_month"):
            exp_total = float(grp["total_kwh"].sum())
            exp_mean = float(grp["total_kwh"].mean())
            exp_min = float(grp["total_kwh"].min())
            exp_max = float(grp["total_kwh"].max())
            row = monthly_indexed.loc[ym]
            np.testing.assert_allclose(float(row["total_kwh"]), exp_total, rtol=1e-9,
                err_msg=f"monthly_summary total_kwh for {ym} must be sum of daily totals (prompt.md §3)")
            np.testing.assert_allclose(float(row["mean_daily_kwh"]), exp_mean, rtol=1e-9,
                err_msg=f"monthly_summary mean_daily_kwh for {ym} must be mean of daily totals (prompt.md §3)")
            np.testing.assert_allclose(float(row["min_daily_kwh"]), exp_min, rtol=1e-9,
                err_msg=f"monthly_summary min_daily_kwh for {ym} must be min of daily totals (prompt.md §3)")
            np.testing.assert_allclose(float(row["max_daily_kwh"]), exp_max, rtol=1e-9,
                err_msg=f"monthly_summary max_daily_kwh for {ym} must be max of daily totals (prompt.md §3)")

    # ------------------------------------------------------------------
    # usage_patterns — verify required keys, float dtype, positivity, and
    # independent computation of each segment mean using exact §4
    # hour/day definitions.
    # (prompt.md §4 + identify_usage_patterns spec)
    # ------------------------------------------------------------------
    def test_usage_patterns_has_required_keys_with_correct_values(
        self, business_weekend_pattern_csv, tmp_path
    ):
        """
        prompt.md §4 + identify_usage_patterns:
        Must return dict with at least these keys, each a positive float:
          business_hours_mean_kwh  — mean over hours 09–17 inclusive, Mon–Fri
          off_hours_mean_kwh       — mean over hours {20–23}∪{00–06}, any day
          weekday_mean_kwh         — mean over all Mon–Fri hours
          weekend_mean_kwh         — mean over all Sat–Sun hours
        Values must be positive floats and match independent computation from
        the loaded hourly data using the exact segment definitions in §4.
        (prompt.md lists required keys; it does not forbid additional keys.)
        """
        out = str(tmp_path / "pattern.html")
        result = run_pipeline(business_weekend_pattern_csv, out, cost_per_kwh=0.1)
        up = result["usage_patterns"]

        assert isinstance(up, dict), "usage_patterns must be a dict (prompt.md §4)"

        required_keys = {
            "business_hours_mean_kwh",
            "off_hours_mean_kwh",
            "weekday_mean_kwh",
            "weekend_mean_kwh",
        }
        missing = required_keys - set(up.keys())
        assert not missing, (
            f"usage_patterns missing required keys: {missing} (prompt.md §4)"
        )

        # Type must be float — prompt.md §4: "every value MUST be a positive float"
        for key in required_keys:
            val = up[key]
            assert _is_prompt_float(val), (
                f"usage_patterns['{key}'] must be a float, not {type(val).__name__} "
                "(prompt.md §4: every value MUST be a positive float)"
            )
            assert np.isfinite(float(val)), (
                f"usage_patterns['{key}'] must be finite (prompt.md §4)"
            )
            assert float(val) > 0, (
                f"usage_patterns['{key}'] must be positive (prompt.md §4)"
            )

        # Independently compute each segment mean using exact §4 definitions
        hourly = load_data(business_weekend_pattern_csv)
        kwh = hourly["consumption_kwh"]
        hour = hourly.index.hour
        dow = hourly.index.dayofweek  # 0=Mon … 6=Sun

        # (a) Business hours: 09–17 inclusive on Mon–Fri
        business_mask = (hour >= 9) & (hour <= 17) & (dow <= 4)
        expected_business = float(kwh[business_mask].mean())

        # (b) Off-hours: {20,21,22,23,0,1,2,3,4,5,6} on any day
        off_hour_set = set(range(0, 7)) | set(range(20, 24))
        off_mask = hour.isin(off_hour_set)
        expected_off = float(kwh[off_mask].mean())

        # (c) All weekday hours: Mon–Fri
        weekday_mask = dow <= 4
        expected_weekday = float(kwh[weekday_mask].mean())

        # (d) All weekend hours: Sat–Sun
        weekend_mask = dow >= 5
        expected_weekend = float(kwh[weekend_mask].mean())

        np.testing.assert_allclose(
            float(up["business_hours_mean_kwh"]), expected_business, rtol=1e-9,
            err_msg="business_hours_mean_kwh must be mean of hours 09–17 Mon–Fri (prompt.md §4)",
        )
        np.testing.assert_allclose(
            float(up["off_hours_mean_kwh"]), expected_off, rtol=1e-9,
            err_msg="off_hours_mean_kwh must be mean of hours {20–23}∪{00–06} any day (prompt.md §4)",
        )
        np.testing.assert_allclose(
            float(up["weekday_mean_kwh"]), expected_weekday, rtol=1e-9,
            err_msg="weekday_mean_kwh must be mean of all Mon–Fri hours (prompt.md §4)",
        )
        np.testing.assert_allclose(
            float(up["weekend_mean_kwh"]), expected_weekend, rtol=1e-9,
            err_msg="weekend_mean_kwh must be mean of all Sat–Sun hours (prompt.md §4)",
        )

    # ------------------------------------------------------------------
    # seasonal_trends — verify required keys, float dtype, positivity,
    # and independent computation of each seasonal mean using exact §6
    # month-to-season mapping.
    # (prompt.md §6 + detect_seasonal_trends spec)
    # ------------------------------------------------------------------
    def test_seasonal_trends_has_required_keys_with_correct_values(
        self, seasonal_csv, tmp_path
    ):
        """
        prompt.md §6 + detect_seasonal_trends:
        Must return dict with at least these keys, each a positive float:
          winter_mean_kwh  — mean over Dec, Jan, Feb
          spring_mean_kwh  — mean over Mar, Apr, May
          summer_mean_kwh  — mean over Jun, Jul, Aug
          fall_mean_kwh    — mean over Sep, Oct, Nov
        Values must be positive floats and match independent computation
        from the loaded hourly data using the exact §6 season definitions.
        (prompt.md lists required keys; it does not forbid additional keys.)
        """
        out = str(tmp_path / "seasonal_check.html")
        result = run_pipeline(seasonal_csv, out, cost_per_kwh=0.1)
        st = result["seasonal_trends"]

        assert isinstance(st, dict), "seasonal_trends must be a dict (prompt.md §6)"

        required_keys = {
            "winter_mean_kwh",
            "spring_mean_kwh",
            "summer_mean_kwh",
            "fall_mean_kwh",
        }
        missing = required_keys - set(st.keys())
        assert not missing, (
            f"seasonal_trends missing required keys: {missing} (prompt.md §6)"
        )

        # Type must be float — prompt.md §6: "every value MUST be a positive float"
        for key in required_keys:
            val = st[key]
            assert _is_prompt_float(val), (
                f"seasonal_trends['{key}'] must be a float, not {type(val).__name__} "
                "(prompt.md §6: every value MUST be a positive float)"
            )
            assert np.isfinite(float(val)), (
                f"seasonal_trends['{key}'] must be finite (prompt.md §6)"
            )
            assert float(val) > 0, (
                f"seasonal_trends['{key}'] must be positive (prompt.md §6)"
            )

        # Independently compute each seasonal mean using exact §6 definitions:
        # Winter=DJF, Spring=MAM, Summer=JJA, Fall=SON
        hourly = load_data(seasonal_csv)
        kwh = hourly["consumption_kwh"]
        month = hourly.index.month

        season_months = {
            "winter_mean_kwh": {12, 1, 2},
            "spring_mean_kwh": {3, 4, 5},
            "summer_mean_kwh": {6, 7, 8},
            "fall_mean_kwh":   {9, 10, 11},
        }
        for key, months in season_months.items():
            expected = float(kwh[month.isin(months)].mean())
            np.testing.assert_allclose(
                float(st[key]), expected, rtol=1e-9,
                err_msg=(
                    f"{key} must be the mean of consumption_kwh for months "
                    f"{sorted(months)} (prompt.md §6)"
                ),
            )

    # ------------------------------------------------------------------
    # T7a — σ=0 edge case (UNCHANGED from original)
    # prompt.md §5: when σ_stratum == 0, z_score=0.0 and is_anomaly=False
    # for every day in that stratum.
    # ------------------------------------------------------------------
    def test_anomalies_zero_variance_stratum_behavior(self, tmp_path):
        """
        prompt.md §5: when σ_stratum == 0 for a given stratum, every day in
        that stratum MUST have z_score = 0.0 and is_anomaly = False.

        Uses a synthetic CSV where all weekday daily totals are identical
        (σ_weekday = 0) while weekend days have natural variance, so the
        σ=0 rule is exercised in isolation without depending on the shared
        anomaly_csv fixture.
        """
        # Build a 4-week synthetic dataset:
        # - All weekday hours: constant 10.0 kWh  → σ_weekday = 0
        # - All weekend hours: alternating 5.0 / 15.0 kWh → σ_weekend > 0
        # Span: Monday 2022-01-03 through Sunday 2022-01-30 (4 full weeks)
        start = pd.Timestamp("2022-01-03 00:00:00")  # Monday
        end   = pd.Timestamp("2022-01-30 23:00:00")  # Sunday
        idx   = pd.date_range(start, end, freq="h")

        kwh_values = []
        for ts in idx:
            if ts.dayofweek < 5:          # weekday — constant
                kwh_values.append(10.0)
            else:                          # weekend — alternating per day
                day_num = (ts.date() - start.date()).days
                kwh_values.append(5.0 if (day_num % 2 == 0) else 15.0)

        csv_path = str(tmp_path / "zero_var.csv")
        pd.DataFrame({
            "timestamp":       idx.strftime("%Y-%m-%d %H:%M:%S"),
            "consumption_kwh": kwh_values,
        }).to_csv(csv_path, index=False)

        out = str(tmp_path / "zero_var_report.html")
        result = run_pipeline(csv_path, out, cost_per_kwh=0.1)
        anomalies = result["anomalies"]

        assert isinstance(anomalies, pd.DataFrame)
        for col in ("date", "total_kwh", "z_score", "is_anomaly"):
            assert col in anomalies.columns, (
                f"anomalies must include '{col}' (prompt.md detect_anomalies)"
            )

        anomalies_copy = anomalies.copy()
        anomalies_copy["dayofweek"] = pd.to_datetime(anomalies_copy["date"]).dt.dayofweek

        # --- σ_weekday = 0: all weekday rows must have z_score=0.0, is_anomaly=False ---
        weekday_rows = anomalies_copy[anomalies_copy["dayofweek"] <= 4]
        assert len(weekday_rows) > 0, (
            "Synthetic fixture must contain weekday rows (internal fixture check)"
        )
        for _, row in weekday_rows.iterrows():
            assert float(row["z_score"]) == 0.0, (
                f"Weekday {row['date']}: z_score must be 0.0 when σ_weekday=0 "
                "(prompt.md §5: if σ_stratum equals 0, z-score set to 0.0)"
            )
            assert row["is_anomaly"] == False, (
                f"Weekday {row['date']}: is_anomaly must be False when σ_weekday=0 "
                "(prompt.md §5: is_anomaly MUST be False for all rows in that stratum)"
            )

    def test_anomalies_zero_variance_weekend_stratum_behavior(self, tmp_path):
        """
        prompt.md §5: same σ_stratum == 0 rule for the weekend stratum — all weekend
        daily totals identical (σ_weekend = 0) while weekdays have variance (σ_weekday > 0).
        """
        start = pd.Timestamp("2022-01-03 00:00:00")
        end = pd.Timestamp("2022-01-30 23:00:00")
        idx = pd.date_range(start, end, freq="h")

        kwh_values = []
        for ts in idx:
            if ts.dayofweek < 5:
                # Weekday: two different daily totals → σ_weekday > 0
                kwh_values.append(10.0 if ts.dayofweek == 0 else 12.0)
            else:
                # Weekend: constant hourly → every weekend day total identical → σ_weekend = 0
                kwh_values.append(5.0)

        csv_path = str(tmp_path / "zero_var_weekend.csv")
        pd.DataFrame({
            "timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"),
            "consumption_kwh": kwh_values,
        }).to_csv(csv_path, index=False)

        out = str(tmp_path / "zero_var_weekend_report.html")
        result = run_pipeline(csv_path, out, cost_per_kwh=0.1)
        anomalies = result["anomalies"].copy()
        for col in ("date", "total_kwh", "z_score", "is_anomaly"):
            assert col in anomalies.columns, (
                f"anomalies must include '{col}' (prompt.md detect_anomalies)"
            )
        anomalies["dayofweek"] = pd.to_datetime(anomalies["date"]).dt.dayofweek

        weekend_rows = anomalies[anomalies["dayofweek"] >= 5]
        assert len(weekend_rows) > 0, (
            "Synthetic fixture must contain weekend rows (internal fixture check)"
        )
        for _, row in weekend_rows.iterrows():
            assert float(row["z_score"]) == 0.0, (
                f"Weekend {row['date']}: z_score must be 0.0 when σ_weekend=0 "
                "(prompt.md §5: if σ_stratum equals 0, z-score set to 0.0)"
            )
            assert row["is_anomaly"] is False, (
                f"Weekend {row['date']}: is_anomaly must be False when σ_weekend=0 "
                "(prompt.md §5: is_anomaly MUST be False for all rows in that stratum)"
            )

    # ------------------------------------------------------------------
    # T7b — NEW: comprehensive normal-stratum test covering all remaining
    # §5 requirements not addressed by T7a:
    #   (a) z-score formula: (total - μ_stratum) / σ_stratum (ddof=0)
    #   (b) is_anomaly=True when abs(z_score) > 2.0
    #   (c) mean of all z_score values ≈ 0.0 (abs < 1e-10), detect_anomalies spec
    #   (d) exactly one row per calendar day
    #   (e) is_anomaly column is boolean dtype
    #   (f) required columns: date, total_kwh, z_score, is_anomaly
    #   (g) date column contains datetime.date values
    #   (h) total_kwh matches daily sum of hourly consumption_kwh
    # (prompt.md §5 + detect_anomalies spec)
    # ------------------------------------------------------------------
    def test_anomalies_normal_stratum_full_requirements(self, tmp_path):
        """
        prompt.md §5 + detect_anomalies spec — normal stratum (σ > 0):

        Validates all §5 requirements NOT covered by the σ=0 test (T7a):
        (a) z-score formula independently verified: (total - μ) / σ (ddof=0)
        (b) when σ_stratum > 0: every row has is_anomaly == (abs(z_score) > 2.0)
        (c) arithmetic mean of all z_score values is approximately 0.0 (abs < 1e-10)
            — detect_anomalies spec
        (d) exactly one row per calendar day in the input
        (e) is_anomaly column is boolean dtype
        (f) required columns present: date, total_kwh, z_score, is_anomaly
        (g) date column contains datetime.date values
        (h) total_kwh matches daily total from hourly data

        Synthetic fixture (4 weeks, Mon 2022-01-03 – Sun 2022-01-30):
        - All weekday hours: 10.0 kWh, except 2022-01-17 (Monday) = 100.0 kWh
          → daily total 2400 vs typical 240, guarantees abs(z) > 2.0 for outlier
          → σ_weekday > 0 (normal stratum behaviour exercised)
        - All weekend hours: alternating 8.0 / 12.0 kWh per day
          → σ_weekend > 0
        """
        start = pd.Timestamp("2022-01-03 00:00:00")  # Monday
        end   = pd.Timestamp("2022-01-30 23:00:00")  # Sunday
        idx   = pd.date_range(start, end, freq="h")

        # Outlier: Monday 2022-01-17 → 100.0 kWh/hr (daily total = 2400)
        # Normal weekday: 10.0 kWh/hr (daily total = 240)
        outlier_date = pd.Timestamp("2022-01-17").date()

        kwh_values = []
        for ts in idx:
            if ts.dayofweek < 5:  # weekday
                if ts.date() == outlier_date:
                    kwh_values.append(100.0)
                else:
                    kwh_values.append(10.0)
            else:  # weekend — alternating to ensure σ_weekend > 0
                day_num = (ts.date() - start.date()).days
                kwh_values.append(8.0 if (day_num % 2 == 0) else 12.0)

        csv_path = str(tmp_path / "normal_var.csv")
        pd.DataFrame({
            "timestamp":       idx.strftime("%Y-%m-%d %H:%M:%S"),
            "consumption_kwh": kwh_values,
        }).to_csv(csv_path, index=False)

        out = str(tmp_path / "normal_var_report.html")
        result = run_pipeline(csv_path, out, cost_per_kwh=0.1)
        anomalies = result["anomalies"]

        # (f) Required columns
        for col in ("date", "total_kwh", "z_score", "is_anomaly"):
            assert col in anomalies.columns, (
                f"anomalies must contain column '{col}' "
                "(prompt.md §5 + detect_anomalies spec)"
            )

        # (g) date column must contain datetime.date values
        assert all(isinstance(d, datetime.date) for d in anomalies["date"]), (
            "anomalies 'date' column must contain datetime.date values (prompt.md §5)"
        )

        # (e) is_anomaly must be boolean dtype
        assert anomalies["is_anomaly"].dtype == bool, (
            "is_anomaly column must be boolean dtype "
            "(prompt.md §5 + detect_anomalies spec)"
        )

        # (d) Exactly one row per calendar day
        hourly = load_data(csv_path)
        expected_days = int(hourly.index.normalize().nunique())
        assert len(anomalies) == expected_days, (
            f"anomalies must contain exactly one row per calendar day "
            f"(expected {expected_days}, got {len(anomalies)}) (prompt.md §5)"
        )

        # (c) detect_anomalies: mean of ALL z_score values ≈ 0.0
        z_mean_all = float(anomalies["z_score"].mean())
        assert abs(z_mean_all) < 1e-10, (
            "detect_anomalies spec: arithmetic mean of all z_score values MUST be "
            f"approximately 0.0 (abs < 1e-10); got {z_mean_all}"
        )

        # Build daily totals for independent z-score verification
        daily_totals = (
            hourly["consumption_kwh"]
            .resample("D")
            .sum()
            .rename("total_kwh")
            .to_frame()
        )
        daily_totals["dayofweek"] = daily_totals.index.dayofweek
        anomalies_indexed = anomalies.set_index("date")

        # total_kwh must match calendar-day sum of hourly consumption (prompt.md §5)
        for day_ts, grp_row in daily_totals.iterrows():
            day_date = day_ts.date()
            expected_total = float(grp_row["total_kwh"])
            actual_total = float(anomalies_indexed.loc[day_date, "total_kwh"])
            np.testing.assert_allclose(
                actual_total,
                expected_total,
                rtol=1e-9,
                err_msg=(
                    f"anomalies total_kwh on {day_date} must equal daily sum of "
                    "hourly consumption_kwh (prompt.md §5)"
                ),
            )

        for stratum_name, dow_filter in [
            ("weekday", lambda d: d <= 4),
            ("weekend", lambda d: d >= 5),
        ]:
            stratum_rows = daily_totals[daily_totals["dayofweek"].apply(dow_filter)]
            if len(stratum_rows) == 0:
                continue
            stratum_totals = stratum_rows["total_kwh"]
            mu    = float(stratum_totals.mean())
            sigma = float(stratum_totals.std(ddof=0))

            if sigma == 0.0:
                # σ=0 behaviour is covered by T7a; skip formula check here
                continue

            # (a) Verify z-score formula independently for each day in stratum
            for day_ts, grp_row in stratum_rows.iterrows():
                day_date = day_ts.date()
                expected_z = (float(grp_row["total_kwh"]) - mu) / sigma
                actual_z   = float(anomalies_indexed.loc[day_date, "z_score"])
                np.testing.assert_allclose(
                    actual_z,
                    expected_z,
                    rtol=1e-8,
                    err_msg=(
                        f"z_score on {day_date} must equal "
                        f"(total - μ_{stratum_name}) / σ_{stratum_name} (ddof=0) "
                        "(prompt.md §5)"
                    ),
                )

        # (b) σ_stratum > 0: anomaly flag must match §5 threshold for every day
        # (σ=0 stratum is covered by test_anomalies_zero_variance_stratum_behavior)
        wt = daily_totals[daily_totals["dayofweek"] <= 4]["total_kwh"]
        wknd = daily_totals[daily_totals["dayofweek"] >= 5]["total_kwh"]
        sigma_weekday = float(wt.std(ddof=0))
        sigma_weekend = float(wknd.std(ddof=0))

        for _, row in anomalies.iterrows():
            dow = row["date"].weekday()
            sigma = sigma_weekday if dow <= 4 else sigma_weekend
            if sigma == 0.0:
                continue
            z = float(row["z_score"])
            expected = abs(z) > 2.0
            assert row["is_anomaly"] == expected, (
                "prompt.md §5: when σ_stratum > 0, is_anomaly must be True iff "
                f"abs(z_score) > 2.0 (date={row['date']}, z={z}, "
                f"is_anomaly={row['is_anomaly']}, expected={expected})"
            )

    # ------------------------------------------------------------------
    # baseline_kwh — verify actual value against the set B definition from
    # prompt.md §7: union of off-hours {0–6, 20–23} on all days + all
    # Saturday + all Sunday hours.
    # ------------------------------------------------------------------
    def test_baseline_kwh_matches_set_b_definition(
        self, business_weekend_pattern_csv, tmp_path
    ):
        """
        prompt.md §7 + estimate_baseline:
        baseline_kwh must equal the arithmetic mean of consumption_kwh over
        set B: off-hours (hour in {0,1,2,3,4,5,6,20,21,22,23} on any day)
        UNION all hours on Saturday (dayofweek==5) UNION all hours on Sunday
        (dayofweek==6). Return type must be float and value must be positive.
        """
        out = str(tmp_path / "baseline_check.html")
        result = run_pipeline(business_weekend_pattern_csv, out, cost_per_kwh=0.1)
        v = result["baseline_kwh"]

        assert _is_prompt_float(v), (
            "baseline_kwh must be a float (prompt.md Expected Interface)"
        )
        assert np.isfinite(v), "baseline_kwh must be finite"
        assert v > 0, "baseline_kwh must be positive (prompt.md §7)"

        # Independently compute set B from the loaded data
        hourly = load_data(business_weekend_pattern_csv)
        off_hours = {0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23}
        mask_off_hours = hourly.index.hour.isin(off_hours)
        mask_saturday = hourly.index.dayofweek == 5
        mask_sunday = hourly.index.dayofweek == 6
        mask_b = mask_off_hours | mask_saturday | mask_sunday
        expected_baseline = float(hourly.loc[mask_b, "consumption_kwh"].mean())

        np.testing.assert_allclose(
            v,
            expected_baseline,
            rtol=1e-9,
            err_msg=(
                "baseline_kwh must equal the arithmetic mean of consumption_kwh "
                "over set B (off-hours ∪ all Saturday ∪ all Sunday) (prompt.md §7)"
            ),
        )

    def test_projected_next_month_kwh_finite_float(self, multi_year_csv, tmp_path):
        """
        prompt.md §8 + project_next_month:
        projected_next_month_kwh — next calendar month's usage; type float.

        Validates:
        - Return type is float (prompt.md Expected Interface)
        - Value is finite
        - Value matches the exact prompt.md §8 formula:
            mean_daily_kwh_same_calendar_month × days_in_target_month
          where mean_daily_kwh_same_calendar_month is the arithmetic mean of all
          daily totals whose calendar month number matches the target month,
          taken from all years available prior to the target month.
        - Falls back to overall mean daily kWh when no prior data exists for the
          target calendar month (prompt.md §8 fallback requirement).
        Float comparison below uses test-only tolerance (prompt.md §8 does not
        specify numeric precision).
        """
        import calendar

        out = str(tmp_path / "proj.html")
        result = run_pipeline(multi_year_csv, out, cost_per_kwh=0.12)
        v = result["projected_next_month_kwh"]

        # Type check — prompt.md Expected Interface: projected_next_month_kwh is float
        assert _is_prompt_float(v), (
            "projected_next_month_kwh must be a float (prompt.md Expected Interface); "
            "int or numpy integer types must not pass"
        )
        assert np.isfinite(v), (
            "projected_next_month_kwh must be finite (prompt.md §8)"
        )

        # --- Derive expected projection using the exact prompt.md §8 formula ---
        hourly = load_data(multi_year_csv)

        daily_totals = (
            hourly["consumption_kwh"]
            .resample("D")
            .sum()
            .rename("total_kwh")
            .to_frame()
        )
        daily_totals["month"] = daily_totals.index.month

        last_date = hourly.index.max()
        if last_date.month == 12:
            target_year = last_date.year + 1
            target_month = 1
        else:
            target_year = last_date.year
            target_month = last_date.month + 1

        days_in_target_month = calendar.monthrange(target_year, target_month)[1]

        target_month_start = pd.Timestamp(year=target_year, month=target_month, day=1)
        prior_same_month_totals = daily_totals[
            (daily_totals.index < target_month_start)
            & (daily_totals["month"] == target_month)
        ]["total_kwh"]

        if len(prior_same_month_totals) > 0:
            mean_daily = float(prior_same_month_totals.mean())
        else:
            mean_daily = float(daily_totals["total_kwh"].mean())

        expected = mean_daily * days_in_target_month

        # FP tolerance for test stability only; prompt.md §8 does not specify numeric precision.
        np.testing.assert_allclose(
            v,
            expected,
            rtol=1e-5,
            atol=0.05,
            err_msg=(
                f"projected_next_month_kwh must equal "
                f"mean_daily_kwh_same_calendar_month × days_in_target_month "
                f"(expected {expected:.4f} for target month "
                f"{target_year}-{target_month:02d} with {days_in_target_month} days; "
                "prompt.md §8)"
            ),
        )

    def test_projected_next_month_kwh_fallback_no_prior_same_month(self, tmp_path):
        """
        prompt.md §8: if no prior daily totals exist for the target calendar month,
        fall back to overall mean daily kWh × days_in_target_month.

        Dataset: hourly rows only in Jan–Feb 2020; last timestamp end of Feb →
        target March 2020. No March daily totals occur before 2020-03-01, so the
        primary branch (same calendar month in prior years) is empty.

        Validates:
        - Return type is float (prompt.md Expected Interface)
        - Value is finite
        - Value matches fallback: overall mean daily kWh × days_in_target_month
        Float comparison below uses test-only tolerance (prompt.md §8 does not
        specify numeric precision).
        """
        import calendar

        start = pd.Timestamp("2020-01-01 00:00:00")
        end = pd.Timestamp("2020-02-29 23:00:00")
        idx = pd.date_range(start, end, freq="h")
        kwh = 4.0
        csv_path = str(tmp_path / "jan_feb_2020_only.csv")
        pd.DataFrame({
            "timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"),
            "consumption_kwh": [kwh] * len(idx),
        }).to_csv(csv_path, index=False)

        out = str(tmp_path / "proj_fallback.html")
        result = run_pipeline(csv_path, out, cost_per_kwh=0.12)
        v = result["projected_next_month_kwh"]

        assert _is_prompt_float(v), (
            "projected_next_month_kwh must be a float (prompt.md Expected Interface)"
        )
        assert np.isfinite(v), (
            "projected_next_month_kwh must be finite (prompt.md §8)"
        )

        hourly = load_data(csv_path)
        last_date = hourly.index.max()
        assert last_date.year == 2020 and last_date.month == 2, (
            "fixture must end in February 2020 so target month is March (internal check)"
        )

        daily_totals = (
            hourly["consumption_kwh"]
            .resample("D")
            .sum()
            .rename("total_kwh")
            .to_frame()
        )
        daily_totals["month"] = daily_totals.index.month

        if last_date.month == 12:
            target_year = last_date.year + 1
            target_month = 1
        else:
            target_year = last_date.year
            target_month = last_date.month + 1

        assert (target_year, target_month) == (2020, 3), (
            "target month must be March 2020 (internal check)"
        )

        days_in_target_month = calendar.monthrange(target_year, target_month)[1]
        target_month_start = pd.Timestamp(year=target_year, month=target_month, day=1)
        prior_same_month_totals = daily_totals[
            (daily_totals.index < target_month_start)
            & (daily_totals["month"] == target_month)
        ]["total_kwh"]
        assert len(prior_same_month_totals) == 0, (
            "fallback fixture must have no prior daily totals for the target month "
            "(internal check)"
        )

        mean_daily = float(daily_totals["total_kwh"].mean())
        expected = mean_daily * days_in_target_month

        # FP tolerance for test stability only; prompt.md §8 does not specify numeric precision.
        np.testing.assert_allclose(
            v,
            expected,
            rtol=1e-5,
            atol=0.05,
            err_msg=(
                "prompt.md §8 fallback: projected_next_month_kwh must equal "
                "overall mean daily kWh × days_in_target_month when no prior "
                f"same-calendar-month days exist (expected {expected:.4f})"
            ),
        )


# ---------------------------------------------------------------------------
# TestRunPipelineHtmlAndHeatmap
# ---------------------------------------------------------------------------

class TestRunPipelineHtmlAndHeatmap:
    """prompt.md — HTML Report + side effect."""

    # ------------------------------------------------------------------
    # T10 — EXPANDED: self-containment checks + presence of all three
    # mandatory §9 sections (heatmap div, anomaly table, cost estimate).
    # Original only verified self-containment; §9 requires all three
    # sections to be present in every valid report.
    # ------------------------------------------------------------------
    def test_writes_nonempty_html_document(self, multi_year_run, multi_year_csv):
        """
        prompt.md §9: writes self-contained HTML report to output_path
        containing all three mandatory sections:
          1. Daily heat map: Plotly heatmap embedded inline (no external CDN)
          2. HTML <table> with rows (three <td>: date, total_kwh, z_score) that cover
             every pipeline-flagged anomaly; extra non-anomaly rows are allowed.
          3. Literal 'Total Estimated Cost = $X.XX' with X.XX =
             round(sum(consumption_kwh) * cost_per_kwh, 2) for the run's rate.

        Plotly is considered inline via _assert_self_contained_plotly_html (no CDN,
        inline <script> with substantive JS, plotly string + heatmap trace).
        """
        result, output_path, _ = multi_year_run()
        cost_rate = 0.12
        assert os.path.exists(output_path), (
            "run_pipeline must write the HTML report to the supplied output_path (prompt.md §9)"
        )
        text = open(output_path, "r", encoding="utf-8").read()
        assert text.strip(), "HTML report must not be empty (prompt.md §9)"

        # Self-containment: no external script URLs; inline script + Plotly + heatmap
        _assert_self_contained_plotly_html(text)

        # §9: y-axis rows are hours 0–23 (row i = hour i)
        hourly_for_hm = load_data(multi_year_csv)
        hm_days = int(hourly_for_hm.index.normalize().nunique())
        traces_hm = heatmap_traces_from_html(text)
        assert traces_hm
        trace_hm = primary_heatmap_trace(traces_hm, hm_days)
        assert trace_hm is not None
        assert_heatmap_y_axis_rows_are_hours_0_to_23(trace_hm)

        # Section 1: daily heat map — typical Plotly embed uses a div
        assert "<div" in text.lower(), (
            "HTML must contain a <div> for embedded Plotly output (prompt.md §9 section 1)"
        )

        # Section 2: anomaly table — prompt requires <table>, not <tbody>
        assert re.search(r"<table\b", text, re.I), (
            "HTML must contain an anomaly highlights <table> (prompt.md §9 section 2)"
        )

        flagged = result["anomalies"][result["anomalies"]["is_anomaly"]].copy()
        parsed = _parse_three_column_table_rows(text)

        # First matching row per date; prompt.md does not constrain duplicate rows.
        parsed_by_date: dict[datetime.date, tuple[float, float]] = {}
        for d, tot, z in parsed:
            if d not in parsed_by_date:
                parsed_by_date[d] = (tot, z)

        for _, row in flagged.iterrows():
            d = pd.Timestamp(row["date"]).date()
            assert d in parsed_by_date, (
                f"Anomaly table must list every anomalous date; missing {d} (prompt.md §9)"
            )
            got_tot, got_z = parsed_by_date[d]
            np.testing.assert_allclose(
                [got_tot, got_z],
                [float(row["total_kwh"]), float(row["z_score"])],
                rtol=1e-4,
                atol=1e-3,
                err_msg=(
                    f"Anomaly table total_kwh/z_score for {d} must match "
                    "run_pipeline anomalies (prompt.md §9)"
                ),
            )

        # Section 3: cost literal Total Estimated Cost = $X.XX (same formula as test_html_report_cost_estimate_format_and_value)
        hourly = load_data(multi_year_csv)
        total_kwh = float(hourly["consumption_kwh"].sum())
        expected_cost = round(total_kwh * cost_rate, 2)
        visible = _visible_plain_text(text)
        assert "Total Estimated Cost = $" in visible, (
            "HTML must contain the literal prefix 'Total Estimated Cost = $' "
            "(prompt.md §9 section 3)"
        )
        cost_line = re.compile(
            r"Total Estimated Cost\s*=\s*\$\s*([\d,]+\.\d{2})"
        )
        match = cost_line.search(visible)
        assert match, (
            "HTML must display cost as 'Total Estimated Cost = $X.XX' with two decimal "
            "places (prompt.md §9 section 3)"
        )
        got_cost = float(match.group(1).replace(",", ""))
        assert got_cost == pytest.approx(expected_cost, abs=0.005), (
            f"Displayed cost must equal round(sum(consumption_kwh) × cost_per_kwh, 2) "
            f"= {expected_cost} (prompt.md §9)"
        )

    # ------------------------------------------------------------------
    # T11 — DELETED (was test_heatmap_trace_type_and_y_axis_hours); superseded
    # by T12/T13 heatmap checks.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # T12 — heatmap z matches hourly consumption matrix (aligned to prompt §9).
    # Prior directional argmax/argmin checks removed — not required by prompt.md.
    # ------------------------------------------------------------------
    def test_heatmap_color_encodes_consumption(self, anomaly_csv, tmp_path):
        """
        prompt.md §9: cell color encodes consumption_kwh.

        Validates:
        - z shape is 24 rows × N days (prompt.md §9: 24 rows, one col per day)
        - z cells are finite numeric values
        - z is non-constant when input consumption varies (prompt.md does not
          specify a correlation threshold or that z equals raw kWh)

        Heatmap data is read via prompt_helpers.heatmap_traces_from_html, which
        prefers Plotly.newPlot/react JSON when parseable and otherwise scans for
        inline JSON heatmap traces (prompt.md does not mandate a specific API).
        """
        out = str(tmp_path / "color_check.html")
        run_pipeline(anomaly_csv, out, cost_per_kwh=0.12)
        html = open(out, "r", encoding="utf-8").read()
        traces = heatmap_traces_from_html(html)
        assert traces
        df_loaded = load_data(anomaly_csv)
        expected_days = int(df_loaded.index.normalize().nunique())
        trace = primary_heatmap_trace(traces, expected_days)
        assert trace is not None
        assert_heatmap_y_axis_rows_are_hours_0_to_23(trace)

        series = df_loaded["consumption_kwh"].astype(float)
        assert not np.allclose(series.std(ddof=0), 0.0), (
            "Fixture must contain variable consumption data (internal fixture check)"
        )

        z = trace.get("z")
        assert isinstance(z, list) and z, "heatmap z must be a non-empty list"
        assert len(z) == 24, (
            f"heatmap z must have 24 rows (hours 0–23); got {len(z)} (prompt.md §9)"
        )
        x = trace.get("x")
        assert isinstance(x, list) and len(x) == expected_days, (
            "heatmap x must list one value per calendar day (prompt.md §9)"
        )
        for row in z:
            assert isinstance(row, list), "each z row must be a list"
            assert len(row) == expected_days, (
                f"each z row must have {expected_days} columns (one per calendar day); "
                f"got {len(row)} (prompt.md §9)"
            )

        # Build expected consumption array aligned to heatmap x/y
        expected = np.zeros((24, expected_days), dtype=float)
        for j, xval in enumerate(x):
            day = pd.to_datetime(xval).normalize()
            for hour in range(24):
                mask = (df_loaded.index.normalize() == day) & (
                    df_loaded.index.hour == hour
                )
                expected[hour, j] = float(
                    df_loaded.loc[mask, "consumption_kwh"].iloc[0]
                )

        z_arr = np.array(
            [[float(z[h][j]) for j in range(expected_days)] for h in range(24)],
            dtype=float,
        )

        _assert_heatmap_z_encodes_consumption(z_arr, expected)

    # ------------------------------------------------------------------
    # T13 — 24 hourly rows × N days: layout only (encoding is test_heatmap_color_encodes_consumption / T12).
    # ------------------------------------------------------------------
    def test_heatmap_has_24_hourly_rows_and_finite_z(self, multi_year_run, multi_year_csv):
        """
        prompt.md §9: daily heat map with 24 rows (hours 0–23) and one column per
        calendar day.

        This test asserts grid shape and finite numeric heatmap values only.
        That cell color *encodes* consumption_kwh (finite z varying with varying kWh)
        is covered by test_heatmap_color_encodes_consumption (T12), not duplicated here.

        Y-axis tick labels are not asserted (prompt.md does not prescribe label format).

        §1 (`load_data`, positivity, fills) is tested in test_load_data.py — not duplicated here.
        """
        _, output_path, _ = multi_year_run()
        with open(output_path, "r", encoding="utf-8") as f:
            pipeline_html = f.read()
        traces = heatmap_traces_from_html(pipeline_html)
        assert traces
        hourly = load_data(multi_year_csv)
        expected_days = int(hourly.index.normalize().nunique())
        primary = primary_heatmap_trace(traces, expected_days)
        assert primary is not None, (
            "Could not identify primary heatmap trace (prompt.md §9)"
        )
        assert_heatmap_y_axis_rows_are_hours_0_to_23(primary)

        z = primary.get("z")
        assert isinstance(z, list) and len(z) == 24, (
            "Primary heatmap must have 24 z rows (hours 0–23) (prompt.md §9)"
        )
        x = primary.get("x")
        assert isinstance(x, list) and len(x) == expected_days, (
            "heatmap x must list one value per calendar day (prompt.md §9)"
        )
        for row in z:
            assert isinstance(row, list) and len(row) == expected_days, (
                f"each z row must have {expected_days} columns (one per calendar day) "
                "(prompt.md §9)"
            )

        z_arr = np.array(
            [[float(z[h][j]) for j in range(expected_days)] for h in range(24)],
            dtype=float,
        )
        assert np.all(np.isfinite(z_arr)), (
            "heatmap z must contain only finite values (prompt.md §9)"
        )
        # When hourly consumption is not constant, displayed heatmap intensities must vary.
        cons_std = float(hourly["consumption_kwh"].astype(float).std(ddof=0))
        if cons_std > 1e-12:
            assert float(z_arr.std()) > 1e-12, (
                "when hourly consumption varies, heatmap z must vary (prompt.md §9)"
            )

    def test_heatmap_x_axis_one_column_per_calendar_day(
        self, multi_year_run, multi_year_csv
    ):
        """prompt.md §9: one column per calendar day (dates on the x-axis); order not prescribed."""
        _, output_path, _ = multi_year_run()
        with open(output_path, "r", encoding="utf-8") as f:
            pipeline_html = f.read()
        traces = heatmap_traces_from_html(pipeline_html)
        assert traces
        hourly = load_data(multi_year_csv)
        expected_days = {ts.date() for ts in hourly.index.normalize().unique()}
        n_days = len(expected_days)
        trace = primary_heatmap_trace(traces, n_days)
        assert trace is not None
        assert_heatmap_y_axis_rows_are_hours_0_to_23(trace)
        x = trace.get("x")
        assert isinstance(x, list) and x
        try:
            x_dates = [pd.to_datetime(d).date() for d in x]
        except Exception:
            pytest.fail("Heatmap x values could not be parsed as dates")
        assert len(x) == n_days, (
            "prompt.md §9: one column per calendar day — x length must match distinct "
            f"days in the input (expected {n_days}, got {len(x)})"
        )
        assert set(x_dates) == expected_days, (
            "prompt.md §9: x-axis must list exactly the calendar days present in the "
            f"hourly range (expected set {sorted(expected_days)}, got {sorted(set(x_dates))})"
        )

    # ------------------------------------------------------------------
    # T15 — Self-contained HTML: no external script URLs; inline JS + heatmap.
    # ------------------------------------------------------------------
    def test_html_embeds_plotly_js_inline(self, multi_year_run):
        """
        prompt.md §9: embed Plotly JavaScript inline; no external CDN script loads;
        Plotly library fully included in the file.

        Delegates to _assert_self_contained_plotly_html:
        - No external script src= URLs
        - Inline <script> block(s) with non-empty embedded bodies (prompt.md does
          not mandate a minimum bundle size)
        - "plotly" appears in the file
        - Heatmap figure detectable (prompt.md §9: daily heat map)

        Does not require a specific Plotly global name (e.g. Plotly.) or a
        particular minified bundle character count; those are not in prompt.md.
        """
        _, output_path, _ = multi_year_run()
        with open(output_path, "r", encoding="utf-8") as f:
            pipeline_html = f.read()
        _assert_self_contained_plotly_html(pipeline_html)

    # ------------------------------------------------------------------
    # T16 — UNCHANGED
    # ------------------------------------------------------------------
    def test_html_report_cost_estimate_format_and_value(
        self, multi_year_csv, tmp_path
    ):
        """
        prompt.md §9: the report MUST display the literal text
        'Total Estimated Cost = $X.XX' where X.XX is
        sum(consumption_kwh) × cost_per_kwh rounded to two decimal places.

        Validates:
        - Exact literal prefix 'Total Estimated Cost = $' is present
        - The displayed dollar amount equals round(total_kwh × rate, 2) for each rate
        """
        hourly = load_data(multi_year_csv)
        total_kwh = float(hourly["consumption_kwh"].sum())

        rate_low, rate_high = 0.10, 0.50
        expected_low = round(total_kwh * rate_low, 2)
        expected_high = round(total_kwh * rate_high, 2)

        out_low = str(tmp_path / "cost_low.html")
        out_high = str(tmp_path / "cost_high.html")

        run_pipeline(multi_year_csv, out_low, cost_per_kwh=rate_low)
        run_pipeline(multi_year_csv, out_high, cost_per_kwh=rate_high)

        html_low = open(out_low, "r", encoding="utf-8").read()
        html_high = open(out_high, "r", encoding="utf-8").read()

        vis_low = _visible_plain_text(html_low)
        vis_high = _visible_plain_text(html_high)

        # prompt.md §9 requires the exact literal prefix
        assert "Total Estimated Cost = $" in vis_low, (
            "HTML report must contain the literal text 'Total Estimated Cost = $' "
            "(prompt.md §9)"
        )
        assert "Total Estimated Cost = $" in vis_high, (
            "HTML report must contain the literal text 'Total Estimated Cost = $' "
            "(prompt.md §9)"
        )

        # Extract the dollar value that follows the required prefix
        pattern = re.compile(r"Total Estimated Cost\s*=\s*\$\s*([\d,]+\.\d{2})")

        match_low = pattern.search(vis_low)
        assert match_low, (
            "Could not extract 'Total Estimated Cost = $X.XX' from report "
            "(prompt.md §9 requires this exact format)"
        )
        match_high = pattern.search(vis_high)
        assert match_high, (
            "Could not extract 'Total Estimated Cost = $X.XX' from report "
            "(prompt.md §9 requires this exact format)"
        )

        got_low = float(match_low.group(1).replace(",", ""))
        got_high = float(match_high.group(1).replace(",", ""))

        # Value must equal sum(consumption_kwh) × cost_per_kwh rounded to 2 d.p.
        assert got_low == pytest.approx(expected_low, abs=0.005), (
            f"Displayed cost {got_low} must equal round(total_kwh × {rate_low}, 2) "
            f"= {expected_low} (prompt.md §9)"
        )
        assert got_high == pytest.approx(expected_high, abs=0.005), (
            f"Displayed cost {got_high} must equal round(total_kwh × {rate_high}, 2) "
            f"= {expected_high} (prompt.md §9)"
        )

    # ------------------------------------------------------------------
    # T17 — red annotations: cover every flagged day; none on non-flagged days
    # ------------------------------------------------------------------
    def test_anomaly_heatmap_uses_red_markings_for_pipeline_flagged_days(
        self, zscore_outlier_heatmap_csv, tmp_path
    ):
        """
        prompt.md §9: Anomaly highlights — anomalous days MUST be marked with
        distinct red annotations on the heat map.

        Uses zscore_outlier_heatmap_csv (conftest): data constructed so a correct
        §5 implementation MUST flag at least one day — this test always exercises
        red markings without pytest.skip.

        Verifies (1) every pipeline-flagged anomaly date has a red heatmap
        annotation, and (2) red annotations do not target non-anomalous calendar
        days (no red column for normal days).

        Red annotation dates are resolved via prompt_helpers.red_marking_dates, which
        aggregates layout annotations/shapes from every embedded heatmap figure it
        can find (Plotly.newPlot/react, inline {data, layout} JSON, and paired JS
        data/layout literals), matching prompt.md's lack of a required Plotly API.
        """
        output_path = str(tmp_path / "anomaly_report.html")
        result = run_pipeline(zscore_outlier_heatmap_csv, output_path, cost_per_kwh=0.10)
        html = open(output_path, "r", encoding="utf-8").read()

        assert layout_annotations_use_red_on_heatmap(html), (
            "prompt.md §9: anomalous days must be marked with red annotations on the heat map "
            "(layout annotations or shapes using a red colour)"
        )

        anomalies = result["anomalies"]
        flagged_dates = normalize_anomaly_dates(anomalies)
        assert flagged_dates, (
            "Fixture zscore_outlier_heatmap_csv must yield at least one anomalous day "
            "under prompt.md §5 (weekday outlier 2022-01-17). Empty flagged set means "
            "detect_anomalies / run_pipeline does not meet §5."
        )

        all_dates = set(pd.to_datetime(anomalies["date"]).dt.date)
        normal_dates = all_dates - flagged_dates

        red_dates = red_marking_dates(html)
        missing = flagged_dates - red_dates
        assert not missing, (
            "prompt.md §9: each anomalous day must have a distinct red annotation on "
            "the heatmap — missing red for dates: "
            f"{sorted(missing)}; flagged={sorted(flagged_dates)}; "
            f"red-marked={sorted(red_dates)}"
        )

        # Distinct from non-anomalous days: red markings must not apply to normal columns.
        spurious = red_dates & normal_dates
        assert not spurious, (
            "prompt.md §9: red anomaly annotations must be distinct from non-anomalous "
            "days — red markings on non-flagged dates: "
            f"{sorted(spurious)}; normal_days={sorted(normal_dates)}; "
            f"red-marked={sorted(red_dates)}"
        )

        # Red annotation x-positions should correspond to heatmap columns (dataset days).
        stray = red_dates - all_dates
        assert not stray, (
            "prompt.md §9: red heatmap annotations should target calendar days present "
            f"in the data — unexpected red x for: {sorted(stray)}; "
            f"dataset_days={sorted(all_dates)}"
        )