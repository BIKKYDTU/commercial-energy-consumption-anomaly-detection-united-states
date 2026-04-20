"""
Energy Consumption Analytics Pipeline
Processes three years of hourly energy consumption records for a commercial building.
"""

import calendar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

_EXPECTED_CSV_COLUMNS = ["timestamp", "consumption_kwh"]


def _mean_positive_kwh(series: pd.Series, context: str) -> float:
    """Mean of consumption_kwh for a segment; enforces prompt.md positive-float contract."""
    if series.empty:
        raise ValueError(f"{context}: no hourly records in this segment.")
    v = float(series.mean())
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{context}: mean must be a positive float, got {v!r}")
    return v


# ---------------------------------------------------------------------------
# 1. Data Ingestion
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Read the CSV at filepath, parse timestamp into a timezone-naive DatetimeIndex,
    reindex to a complete hourly range from min to max timestamp, forward-fill
    gaps, then back-fill any remaining leading NaN values.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (timezone-naive, hourly frequency), single column
        consumption_kwh (float64), zero NaN values.
    """
    raw = pd.read_csv(filepath)
    if list(raw.columns) != _EXPECTED_CSV_COLUMNS:
        raise ValueError(
            "CSV must contain exactly two columns "
            f"{_EXPECTED_CSV_COLUMNS!r}, got {list(raw.columns)!r}"
        )

    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df[["consumption_kwh"]].astype("float64")

    full_range = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq="h"
    )
    df = df.reindex(full_range)
    df = df.ffill()
    df = df.bfill()

    if df["consumption_kwh"].isna().any():
        raise ValueError("consumption_kwh still contains NaN after gap filling.")
    if not (df["consumption_kwh"] > 0).all():
        raise ValueError(
            "consumption_kwh must be positive for all hourly records after load."
        )

    df.index.freq = pd.tseries.frequencies.to_offset("h")
    return df


# ---------------------------------------------------------------------------
# 2. Daily Summaries
# ---------------------------------------------------------------------------

def compute_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group hourly df by calendar day and compute total, mean, minimum, and
    maximum hourly consumption_kwh.

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime.date), total_kwh, mean_kwh, min_kwh,
        max_kwh (all float64).  One row per calendar day.
    """
    day_key = df.index.normalize()
    grouped = df.groupby(day_key)["consumption_kwh"]
    agg = grouped.agg(
        total_kwh="sum",
        mean_kwh="mean",
        min_kwh="min",
        max_kwh="max",
    ).astype("float64")
    agg.index.name = "date"
    agg = agg.reset_index()
    agg["date"] = agg["date"].dt.date
    return agg.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Monthly Summaries
# ---------------------------------------------------------------------------

def compute_monthly_summary(daily_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Receive the daily summary DataFrame produced by compute_daily_summary and
    aggregate its daily totals to calendar-month level.

    Returns
    -------
    pd.DataFrame
        Columns: year_month (str, 'YYYY-MM'), total_kwh (float64),
        mean_daily_kwh (float64), min_daily_kwh (float64),
        max_daily_kwh (float64).  One row per calendar month.
    """
    tmp = daily_summary.copy()
    tmp["year_month"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m")
    grouped = tmp.groupby("year_month")["total_kwh"]
    summary = grouped.agg(
        total_kwh="sum",
        mean_daily_kwh="mean",
        min_daily_kwh="min",
        max_daily_kwh="max",
    ).astype("float64")
    monthly = summary.reset_index()
    return monthly.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Typical Usage Pattern Identification
# ---------------------------------------------------------------------------

def identify_usage_patterns(df: pd.DataFrame) -> dict:
    """
    Compute mean hourly consumption_kwh for four time segments:
      - business_hours: hours 09-17 inclusive, Monday-Friday
      - off_hours:      hours 20-23 and 00-06 inclusive, any day
      - weekday:        all hours Monday-Friday
      - weekend:        all hours Saturday-Sunday

    Returns
    -------
    dict
        Keys: business_hours_mean_kwh, off_hours_mean_kwh,
        weekday_mean_kwh, weekend_mean_kwh (all positive float).
    """
    hour = df.index.hour
    dow = df.index.dayofweek  # 0=Monday … 6=Sunday

    business_mask = ((hour >= 9) & (hour <= 17)) & (dow <= 4)
    off_hours_mask = (hour >= 20) | (hour <= 6)
    weekday_mask = dow <= 4
    weekend_mask = dow >= 5

    return {
        "business_hours_mean_kwh": _mean_positive_kwh(
            df.loc[business_mask, "consumption_kwh"], "business_hours"
        ),
        "off_hours_mean_kwh": _mean_positive_kwh(
            df.loc[off_hours_mask, "consumption_kwh"], "off_hours"
        ),
        "weekday_mean_kwh": _mean_positive_kwh(
            df.loc[weekday_mask, "consumption_kwh"], "weekday"
        ),
        "weekend_mean_kwh": _mean_positive_kwh(
            df.loc[weekend_mask, "consumption_kwh"], "weekend"
        ),
    }


# ---------------------------------------------------------------------------
# 5. Anomalous Day Detection
# ---------------------------------------------------------------------------

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each calendar day as weekday (Mon-Fri, dayofweek 0-4) or weekend
    (Sat-Sun, dayofweek 5-6), then compute each day's z-score within its
    stratum using population standard deviation (ddof=0):
        (daily_total_kwh - mu_stratum) / sigma_stratum

    If sigma_stratum == 0 for a given stratum, all z-scores in that stratum
    are set to 0.0 and is_anomaly is False for all rows in that stratum.
    Flag days where abs(z_score) > 2.0 as anomalous.

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime.date), total_kwh (float64), z_score (float64),
        is_anomaly (bool).  One row per calendar day.
    """
    daily = df.groupby(df.index.normalize())["consumption_kwh"].sum()
    daily_index = pd.DatetimeIndex(daily.index)
    totals = daily.to_numpy(dtype="float64")

    dow = daily_index.dayofweek.to_numpy()
    is_weekday = dow <= 4
    is_weekend = dow >= 5

    z_scores = np.zeros(len(totals), dtype="float64")
    is_anomaly = np.zeros(len(totals), dtype=bool)

    for mask in (is_weekday, is_weekend):
        stratum_totals = totals[mask]
        if stratum_totals.size == 0:
            continue
        mu = stratum_totals.mean()
        sigma = stratum_totals.std(ddof=0)
        if sigma == 0.0:
            continue
        z_scores[mask] = (stratum_totals - mu) / sigma
        is_anomaly[mask] = np.abs(z_scores[mask]) > 2.0

    return pd.DataFrame({
        "date": [d.date() for d in daily_index],
        "total_kwh": totals,
        "z_score": z_scores,
        "is_anomaly": is_anomaly,
    })


# ---------------------------------------------------------------------------
# 6. Seasonal Trend Detection
# ---------------------------------------------------------------------------

def detect_seasonal_trends(df: pd.DataFrame) -> dict:
    """
    Map each hourly record to its meteorological season and return the mean
    hourly consumption_kwh per season.

    Season definitions:
      Winter = December, January, February
      Spring = March, April, May
      Summer = June, July, August
      Fall   = September, October, November

    Returns
    -------
    dict
        Keys: winter_mean_kwh, spring_mean_kwh, summer_mean_kwh,
        fall_mean_kwh (all positive float).
    """
    month = df.index.month

    winter_mask = month.isin([12, 1, 2])
    spring_mask = month.isin([3, 4, 5])
    summer_mask = month.isin([6, 7, 8])
    fall_mask = month.isin([9, 10, 11])

    # Per §6: mean over hours in each season; empty season → NaN (short date spans).
    return {
        "winter_mean_kwh": float(
            df.loc[winter_mask, "consumption_kwh"].mean()
        ),
        "spring_mean_kwh": float(
            df.loc[spring_mask, "consumption_kwh"].mean()
        ),
        "summer_mean_kwh": float(
            df.loc[summer_mask, "consumption_kwh"].mean()
        ),
        "fall_mean_kwh": float(
            df.loc[fall_mask, "consumption_kwh"].mean()
        ),
    }


# ---------------------------------------------------------------------------
# 7. Baseline "Always-On" Consumption Estimation
# ---------------------------------------------------------------------------

def estimate_baseline(df: pd.DataFrame) -> float:
    """
    Compute the arithmetic mean of consumption_kwh over baseline set B.

    Baseline set B is the union of:
      - All hourly records whose hour-of-day belongs to
        {0,1,2,3,4,5,6,20,21,22,23} (off-hours, any day of week)
      - All records whose day-of-week is Saturday (pandas index 5)
      - All records whose day-of-week is Sunday  (pandas index 6)

    Returns
    -------
    float
        Arithmetic mean of consumption_kwh over set B.
    """
    hour = df.index.hour
    dow = df.index.dayofweek

    off_hours_mask = hour.isin([0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23])
    saturday_mask = dow == 5
    sunday_mask = dow == 6

    baseline_mask = off_hours_mask | saturday_mask | sunday_mask
    return _mean_positive_kwh(
        df.loc[baseline_mask, "consumption_kwh"], "baseline set B"
    )


# ---------------------------------------------------------------------------
# 8. Next-Month Usage Projection
# ---------------------------------------------------------------------------

def project_next_month(df: pd.DataFrame) -> float:
    """
    Project total kWh for the calendar month immediately following the last
    date present in df.

    Formula: mean_daily_kwh_same_calendar_month × days_in_target_month
    where mean_daily_kwh_same_calendar_month is the arithmetic mean of all
    daily totals whose calendar month number matches the target month, taken
    from all years in the dataset prior to the target month.

    Falls back to the overall mean daily kWh if no prior daily totals exist
    for the target calendar month.

    Returns
    -------
    float
        Projected total kWh for the following calendar month.
    """
    last_ts = df.index.max()

    if last_ts.month == 12:
        target_month = 1
        target_year = last_ts.year + 1
    else:
        target_month = last_ts.month + 1
        target_year = last_ts.year

    days_in_target = calendar.monthrange(target_year, target_month)[1]

    daily_series = df.groupby(df.index.normalize())["consumption_kwh"].sum()
    daily_dates = pd.DatetimeIndex(daily_series.index)
    daily_values = daily_series.to_numpy(dtype="float64")

    target_start = pd.Timestamp(target_year, target_month, 1)
    same_month_mask = (daily_dates.month == target_month) & (
        daily_dates < target_start
    )
    same_month_totals = daily_values[same_month_mask]

    if same_month_totals.size > 0:
        mean_daily = float(same_month_totals.mean())
    else:
        mean_daily = float(daily_values.mean())

    return float(mean_daily * days_in_target)


# ---------------------------------------------------------------------------
# 9. HTML Report Generation
# ---------------------------------------------------------------------------

def generate_report(
    df: pd.DataFrame,
    anomalies: pd.DataFrame,
    output_path: str,
    cost_per_kwh: float,
) -> None:
    """
    Write a self-contained HTML file to output_path containing:
      1. An inline Plotly heatmap (24 rows × N day columns) with red
         annotations on every anomalous day column.
      2. An HTML table listing every anomalous date with total_kwh and z_score.
      3. The literal text 'Total Estimated Cost = $X.XX'.

    No external CDN links are used; all JavaScript is embedded inline.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly consumption DataFrame (output of load_data).
    anomalies : pd.DataFrame
        Columns: date (datetime.date), total_kwh (float64),
        z_score (float64), is_anomaly (bool).
    output_path : str
        Filesystem path where the HTML file will be written.
    cost_per_kwh : float
        Cost in dollars per kWh for the cost estimate section.
    """
    # ---- Build pivot: rows=hours (0-23), cols=dates -----------------------
    tmp = df.copy()
    tmp["_date"] = tmp.index.date
    tmp["_hour"] = tmp.index.hour
    pivot = tmp.pivot_table(
        index="_hour", columns="_date", values="consumption_kwh", aggfunc="first"
    )
    # Force an explicit 24-hour y-axis (0..23), even if input spans partial days.
    pivot = pivot.reindex(range(24)).sort_index()

    dates = list(pivot.columns)           # datetime.date objects
    date_strs = [str(d) for d in dates]  # 'YYYY-MM-DD' strings
    z_matrix = pivot.values               # shape (24, N_days)

    # ---- Anomalous date set -----------------------------------------------
    anomaly_date_strs = set(
        anomalies.loc[anomalies["is_anomaly"], "date"].astype(str)
    )

    # ---- Plotly heatmap ---------------------------------------------------
    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=date_strs,
            y=list(range(24)),
            colorscale="Viridis",
            colorbar=dict(title="kWh"),
        )
    )

    # Distinct red annotations on anomalous day columns: a bold red down-arrow
    # marker above the column combined with a red rectangular outline drawn
    # around the full column so each flagged day is clearly distinguishable
    # from non-anomalous days while preserving the underlying heat-map colors.
    annotations = []
    shapes = []
    for d_str in date_strs:
        if d_str in anomaly_date_strs:
            annotations.append(
                dict(
                    x=d_str,
                    y=24.2,
                    text="<b>▼ ANOMALY</b>",
                    showarrow=False,
                    font=dict(color="red", size=14),
                    xanchor="center",
                    yanchor="bottom",
                )
            )
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=d_str,
                    x1=d_str,
                    y0=-0.5,
                    y1=23.5,
                    line=dict(color="red", width=3),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                )
            )

    fig.update_layout(
        title="Daily Energy Consumption Heatmap (kWh per Hour)",
        xaxis_title="Date",
        yaxis_title="Hour of Day",
        annotations=annotations,
        shapes=shapes,
    )

    heatmap_div = pyo.plot(fig, include_plotlyjs=True, output_type="div")

    # ---- Anomaly table rows -----------------------------------------------
    anomaly_rows = ""
    for _, row in anomalies[anomalies["is_anomaly"]].iterrows():
        anomaly_rows += (
            f"    <tr>"
            f"<td>{row['date']}</td>"
            f"<td>{row['total_kwh']:.4f}</td>"
            f"<td>{row['z_score']:.4f}</td>"
            f"</tr>\n"
        )

    # ---- Cost estimate ----------------------------------------------------
    total_kwh = float(df["consumption_kwh"].sum())
    total_cost = round(total_kwh * cost_per_kwh, 2)
    cost_text = f"Total Estimated Cost = ${total_cost:.2f}"
    cost_detail = (
        f"Computed from {total_kwh:,.2f} kWh &times; "
        f"${cost_per_kwh:.4f} / kWh (caller-supplied rate)."
    )

    # ---- Assemble HTML ----------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Energy Consumption Analytics Report</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 2rem;
      color: #2c3e50;
      background: #fafafa;
    }}
    h1 {{ color: #1a252f; }}
    h2 {{
      color: #34495e;
      margin-top: 2.5rem;
      border-bottom: 2px solid #bdc3c7;
      padding-bottom: 0.4rem;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 1rem;
    }}
    th, td {{
      border: 1px solid #bdc3c7;
      padding: 8px 12px;
      text-align: left;
    }}
    th {{
      background-color: #2c3e50;
      color: #fff;
    }}
    tr:nth-child(even) {{ background-color: #ecf0f1; }}
    .cost-section {{
      display: inline-block;
      margin-top: 1rem;
      padding: 1rem 1.5rem;
      background: #eaf4fb;
      border-left: 5px solid #2980b9;
      font-size: 1.25rem;
      font-weight: bold;
      color: #1a252f;
    }}
  </style>
</head>
<body>
  <h1>Energy Consumption Analytics Report</h1>

  <h2>1. Daily Consumption Heatmap</h2>
  {heatmap_div}

  <h2>2. Anomaly Highlights</h2>
  <p>Distinct red annotations (&#9660;&nbsp;ANOMALY markers and red column
     outlines) on the heatmap indicate anomalous days
     (|z-score|&nbsp;&gt;&nbsp;2.0).</p>
  <table>
    <thead>
      <tr>
        <th>Date</th>
        <th>Total kWh</th>
        <th>Z-Score</th>
      </tr>
    </thead>
    <tbody>
{anomaly_rows}    </tbody>
  </table>

  <h2>3. Cost Estimate</h2>
  <div class="cost-section">{cost_text}</div>
  <p>{cost_detail}</p>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str,
    output_path: str,
    cost_per_kwh: float,
) -> dict:
    """
    Orchestrate all pipeline steps: load data, run every analysis function,
    write the HTML report, and return a results dictionary.

    Returns
    -------
    dict
        Exactly seven keys:
          daily_summary          (pd.DataFrame)
          monthly_summary        (pd.DataFrame)
          usage_patterns         (dict)
          anomalies              (pd.DataFrame)
          seasonal_trends        (dict)
          baseline_kwh           (float)
          projected_next_month_kwh (float)
    The HTML report is written to output_path with non-zero byte size.
    """
    df = load_data(filepath=input_path)
    daily_summary = compute_daily_summary(df)
    monthly_summary = compute_monthly_summary(daily_summary)
    usage_patterns = identify_usage_patterns(df)
    anomalies = detect_anomalies(df)
    seasonal_trends = detect_seasonal_trends(df)
    baseline_kwh = estimate_baseline(df)
    projected_next_month_kwh = project_next_month(df)
    generate_report(df, anomalies, output_path, cost_per_kwh)

    return {
        "daily_summary": daily_summary,
        "monthly_summary": monthly_summary,
        "usage_patterns": usage_patterns,
        "anomalies": anomalies,
        "seasonal_trends": seasonal_trends,
        "baseline_kwh": baseline_kwh,
        "projected_next_month_kwh": projected_next_month_kwh,
    }
