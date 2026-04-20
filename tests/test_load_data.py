"""
Tests for pipeline.load_data against prompt.md §1 (Data Ingestion) and Expected Interface.

prompt.md: CSV path input; two columns timestamp + consumption_kwh; timezone-naive
hourly DatetimeIndex; missing hours forward-filled then leading NaNs back-filled;
float64 consumption_kwh with no NaNs after load.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_pipeline_root = os.environ.get("PIPELINE_REPO_ROOT")
if _pipeline_root:
    sys.path.insert(0, _pipeline_root)
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import load_data


def test_load_data_two_columns_float64_timezone_naive_no_nan(tmp_path):
    """prompt.md §1 + load_data: schema, dtypes, complete hourly series without NaNs."""
    idx = pd.date_range("2024-06-01", periods=48, freq="h")
    path = tmp_path / "continuous.csv"
    pd.DataFrame(
        {
            "timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"),
            "consumption_kwh": np.linspace(1.0, 2.0, len(idx)),
        }
    ).to_csv(path, index=False)

    raw = pd.read_csv(path)
    assert list(raw.columns) == ["timestamp", "consumption_kwh"], (
        "prompt.md §1: CSV MUST contain exactly two columns named timestamp and consumption_kwh"
    )

    df = load_data(str(path))
    assert list(df.columns) == ["consumption_kwh"]
    assert df["consumption_kwh"].dtype == np.float64
    assert (df["consumption_kwh"] > 0).all(), (
        "prompt.md §1: consumption_kwh must be positive for all hourly records"
    )
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is None
    assert not df["consumption_kwh"].isna().any()
    assert len(df) == 48
    assert df.index.min() == idx.min()
    assert df.index.max() == idx.max()


def test_load_data_forward_fills_internal_hourly_gaps(tmp_path):
    """prompt.md §1: missing hourly records forward-filled from preceding valid value."""
    t0 = pd.Timestamp("2024-03-01 00:00:00")
    t1 = pd.Timestamp("2024-03-01 12:00:00")
    t2 = pd.Timestamp("2024-03-01 23:00:00")
    path = tmp_path / "gap.csv"
    pd.DataFrame(
        {
            "timestamp": [t0, t1, t2],
            "consumption_kwh": [10.0, 20.0, 30.0],
        }
    ).to_csv(path, index=False)

    raw = pd.read_csv(path)
    assert len(raw.columns) == 2, "prompt.md §1: CSV MUST contain exactly two columns"

    df = load_data(str(path))
    assert len(df) == 24
    assert df.index.min() == t0
    assert df.index.max() == t2
    assert float(df.loc[pd.Timestamp("2024-03-01 00:00:00"), "consumption_kwh"]) == 10.0
    assert float(df.loc[pd.Timestamp("2024-03-01 11:00:00"), "consumption_kwh"]) == 10.0
    assert float(df.loc[pd.Timestamp("2024-03-01 12:00:00"), "consumption_kwh"]) == 20.0
    assert float(df.loc[pd.Timestamp("2024-03-01 22:00:00"), "consumption_kwh"]) == 20.0
    assert float(df.loc[pd.Timestamp("2024-03-01 23:00:00"), "consumption_kwh"]) == 30.0


def test_load_data_backfills_leading_na_after_reindex(tmp_path):
    """prompt.md §1: leading NaNs (after reindex/ffill) back-filled from nearest subsequent value."""
    idx = pd.date_range("2024-04-01 00:00:00", "2024-04-01 23:00:00", freq="h")
    vals = [np.nan] * 6 + [4.25] * 18
    path = tmp_path / "leading.csv"
    pd.DataFrame(
        {
            "timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"),
            "consumption_kwh": vals,
        }
    ).to_csv(path, index=False)

    raw = pd.read_csv(path)
    assert len(raw.columns) == 2, "prompt.md §1: CSV MUST contain exactly two columns"

    df = load_data(str(path))
    assert len(df) == 24
    assert df.index.min() == idx.min()
    assert df.index.max() == idx.max()
    assert float(df.loc[pd.Timestamp("2024-04-01 00:00:00"), "consumption_kwh"]) == 4.25
    assert float(df.loc[pd.Timestamp("2024-04-01 05:00:00"), "consumption_kwh"]) == 4.25
    assert float(df.loc[pd.Timestamp("2024-04-01 06:00:00"), "consumption_kwh"]) == 4.25
