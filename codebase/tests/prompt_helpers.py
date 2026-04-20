"""
Helpers for inspecting Plotly 5.x HTML reports produced by solutions.

Supports multiple common Plotly export paths (e.g. Plotly.newPlot, Plotly.react)
and a fallback scan for inline JSON heatmap trace objects (prompt.md §9 does not
require a specific Plotly API). Not part of the solution under test.

"""

from __future__ import annotations

import datetime
import json
import numbers
import os
import pathlib
import re
from collections.abc import Iterator


def require_submitted_pipeline() -> None:
    """
    In Docker verification, PIPELINE_REPO_ROOT is /app before codebase.zip is applied.
    Version-only tests would otherwise pass; they must fail until pipeline.py exists.
    """
    root = os.environ.get("PIPELINE_REPO_ROOT")
    if not root:
        return
    p = pathlib.Path(root) / "pipeline.py"
    assert p.is_file() and p.stat().st_size > 0, (
        "Expected pipeline.py at PIPELINE_REPO_ROOT (submission root)"
    )

import numpy as np
import pandas as pd


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\n\r":
        i += 1
    return i


def _parse_json_string_literal(s: str, i: int) -> tuple[str | None, int]:
    if i >= len(s) or s[i] != '"':
        return None, i
    i += 1
    parts: list[str] = []
    while i < len(s):
        c = s[i]
        if c == "\\":
            if i + 1 >= len(s):
                return None, i
            parts.append(s[i + 1])
            i += 2
            continue
        if c == '"':
            return "".join(parts), i + 1
        parts.append(c)
        i += 1
    return None, i


def _try_parse_newplot_or_react_at(
    html: str, marker: str, start_pos: int
) -> tuple[list | None, dict | None, int]:
    """
    Parse Plotly.newPlot / Plotly.react at the first `marker` occurrence >= start_pos.
    Returns (data, layout, next_search_pos) or (None, None, start_pos+1) on failure.
    """
    call = html.find(marker, start_pos)
    if call < 0:
        return None, None, -1
    lp = html.find("(", call)
    if lp < 0:
        return None, None, call + 1
    p = _skip_ws(html, lp + 1)
    _, p = _parse_json_string_literal(html, p)
    if p >= len(html):
        return None, None, call + 1
    p = _skip_ws(html, p)
    if p >= len(html) or html[p] != ",":
        return None, None, call + 1
    p = _skip_ws(html, p + 1)
    dec = json.JSONDecoder()
    try:
        data, p = dec.raw_decode(html, p)
    except json.JSONDecodeError:
        return None, None, call + 1
    p = _skip_ws(html, p)
    if p >= len(html) or html[p] != ",":
        return None, None, call + 1
    p = _skip_ws(html, p + 1)
    try:
        layout, p = dec.raw_decode(html, p)
    except json.JSONDecodeError:
        return None, None, call + 1
    if isinstance(data, list) and isinstance(layout, dict):
        return data, layout, p
    return None, None, call + 1


def _iter_newplot_react_figures(html: str):
    """Yield (data, layout) for every successfully parsed Plotly.newPlot / Plotly.react call."""
    for marker in ("Plotly.newPlot", "Plotly.react"):
        pos = 0
        while True:
            data, layout, advance = _try_parse_newplot_or_react_at(html, marker, pos)
            if advance < 0:
                break
            if data is not None and layout is not None:
                yield data, layout
            pos = advance


def extract_plotly_newplot_data_layout(html: str) -> tuple[list | None, dict | None]:
    """
    Return (data, layout) from Plotly.newPlot or Plotly.react embedding a heatmap figure.
    Tries every parsed figure until one includes a heatmap trace.
    """
    for data, layout in _iter_newplot_react_figures(html):
        if any(isinstance(t, dict) and t.get("type") == "heatmap" for t in data):
            return data, layout
    return None, None


def plotly_html_declares_heatmap(html: str) -> bool:
    """
    True if inline Plotly output includes a heatmap trace JSON (broader than
    full parse success when figures use unusual formatting).
    """
    return bool(
        re.search(r'"type"\s*:\s*"heatmap"', html)
        or re.search(r"'type'\s*:\s*'heatmap'", html)
    )


def _heatmap_traces_from_inline_json_scan(html: str) -> list[dict]:
    """
    Find heatmap trace dicts embedded as JSON anywhere in the HTML.

    Used when Plotly.newPlot/react call sites are not parseable (different arity,
    minified wrappers, or figure JSON inlined without those APIs). Aligned with
    prompt.md §9: self-contained inline figure data, not a specific embed API.
    """
    dec = json.JSONDecoder()
    patterns = (
        re.compile(r'"type"\s*:\s*"heatmap"'),
        re.compile(r"'type'\s*:\s*'heatmap'"),
    )
    seen_sigs: set[str] = set()
    out: list[dict] = []

    def _trace_signature(tr: dict) -> str:
        z = tr.get("z")
        x = tr.get("x")
        if isinstance(z, list) and z:
            z0 = z[0]
            row0 = z0 if isinstance(z0, list) else []
            return f"{len(z)}x{len(row0) if row0 else 0}:{str(row0[:2])}"
        return repr(sorted(tr.keys()))

    for rx in patterns:
        for m in rx.finditer(html):
            # Try json.JSONDecoder.raw_decode from nearby `{` before this "type":"heatmap"
            before = html[: m.start()]
            chunk_start = max(0, len(before) - 500_000)
            chunk = before[chunk_start:]
            brace_idx = [chunk_start + i for i, c in enumerate(chunk) if c == "{"]
            # Prefer opening braces closest to the match (likely the trace object)
            for j in reversed(brace_idx[-400:]):
                try:
                    obj, _end = dec.raw_decode(html, j)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict) or obj.get("type") != "heatmap":
                    continue
                sig = _trace_signature(obj)
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                out.append(obj)
                break

    return out


def heatmap_traces_from_html(html: str) -> list[dict]:
    data, _ = extract_plotly_newplot_data_layout(html)
    if data:
        traces = [
            t
            for t in data
            if isinstance(t, dict) and t.get("type") == "heatmap"
        ]
        if traces:
            return traces
    return _heatmap_traces_from_inline_json_scan(html)


def plotly_report_includes_heatmap(html: str) -> bool:
    """True if the report embeds Plotly and includes a heatmap (prompt.md daily heat map)."""
    if heatmap_traces_from_html(html):
        return True
    if not re.search(r"plotly", html, re.I):
        return False
    return plotly_html_declares_heatmap(html)


def primary_heatmap_trace(traces: list[dict], expected_day_count: int) -> dict | None:
    """
    Prefer the heatmap whose x-axis length matches the number of calendar days
    (avoids picking a secondary heatmap when multiple traces exist).
    """
    if not traces:
        return None
    for t in traces:
        x = t.get("x")
        if isinstance(x, list) and len(x) == expected_day_count:
            return t
    # Fall back: longest x (typical daily matrix is wider than any small inset)
    return max(
        traces,
        key=lambda tr: len(tr["x"]) if isinstance(tr.get("x"), list) else 0,
    )


def _y_tick_to_hour(val: object) -> int | None:
    """
    Map a heatmap y-axis tick to an hour-of-day in 0..23, or None if unparseable.
    Accepts ints, whole floats, digit strings, and parseable date/time strings.
    """
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, np.integer)):
        i = int(val)
        return i if 0 <= i <= 23 else None
    if isinstance(val, (float, np.floating)):
        if not np.isfinite(float(val)):
            return None
        f = float(val)
        if abs(f - round(f)) > 1e-9:
            return None
        i = int(round(f))
        return i if 0 <= i <= 23 else None
    s = str(val).strip()
    if s.isdigit():
        i = int(s)
        return i if 0 <= i <= 23 else None
    ts = pd.to_datetime(s, errors="coerce")
    if pd.notna(ts):
        return int(pd.Timestamp(ts).hour)
    return None


def assert_heatmap_y_axis_rows_are_hours_0_to_23(trace: dict) -> None:
    """
    prompt.md §9: 24 rows with hours 0–23 on the y-axis — row i must represent hour i.

    Requires a y-array of length 24 where the tick at index i encodes hour i
    (Plotly row index matches chronological hour).
    """
    z = trace.get("z")
    if not isinstance(z, list) or len(z) != 24:
        raise AssertionError(
            "heatmap z must have 24 rows (hours 0–23) (prompt.md §9); "
            f"got z with length {len(z) if isinstance(z, list) else 'n/a'}"
        )
    y = trace.get("y")
    if not isinstance(y, list) or len(y) != 24:
        raise AssertionError(
            "prompt.md §9: y-axis must list 24 hour positions (hours 0–23); "
            f"expected len(y)==24, got {y!r}"
        )
    for i, val in enumerate(y):
        h = _y_tick_to_hour(val)
        if h is None:
            raise AssertionError(
                f"prompt.md §9: y-axis tick at row {i} is not a valid hour 0–23: {val!r}"
            )
        if h != i:
            raise AssertionError(
                f"prompt.md §9: row {i} must correspond to hour {i} on the y-axis; "
                f"y[{i}]={val!r} encodes hour {h}"
            )


def iter_numeric_leaves(obj) -> Iterator[float]:
    """Collect finite numeric leaves from nested dict/list structures."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from iter_numeric_leaves(v)
    elif isinstance(obj, np.ndarray):
        for v in obj.flat:
            yield from iter_numeric_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_numeric_leaves(v)
    elif isinstance(obj, numbers.Real) and not isinstance(obj, bool):
        v = float(obj)
        if pd.notna(v):
            yield v


_CSS_RED_NAMES: frozenset[str] = frozenset({
    "red", "crimson", "firebrick", "darkred", "tomato", "orangered",
    "indianred", "lightcoral", "salmon", "darksalmon",
})


def _parse_rgb_tuple(s_nospace: str) -> tuple[int, int, int] | None:
    """Parse hex, rgb(), or rgba() string to (r, g, b) ints 0-255. None on failure."""
    # #rgb shorthand
    m = re.fullmatch(r"#([0-9a-f])([0-9a-f])([0-9a-f])", s_nospace)
    if m:
        return (int(m.group(1) * 2, 16), int(m.group(2) * 2, 16), int(m.group(3) * 2, 16))
    # #rrggbb
    m = re.fullmatch(r"#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})", s_nospace)
    if m:
        return (int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16))
    # rgb(r,g,b) or rgba(r,g,b,a)
    m = re.fullmatch(r"rgba?\((\d+),(\d+),(\d+)(?:,[^)]+)?\)", s_nospace)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _rgb_hue_is_red(r: int, g: int, b: int) -> bool:
    """True if colour has a red hue (0–20° or 340–360°) with meaningful saturation."""
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    max_c = max(rf, gf, bf)
    min_c = min(rf, gf, bf)
    delta = max_c - min_c
    # Reject near-black, near-white, or desaturated colours.
    if delta < 0.15 or max_c < 0.2:
        return False
    # Red must be the dominant channel.
    if max_c != rf:
        return False
    hue = 60.0 * (((gf - bf) / delta) % 6)
    return hue <= 20.0 or hue >= 340.0


def _color_value_is_red(value) -> bool:
    """Accept any distinctly red CSS colour (hue 0–20° or 340–360°, meaningful saturation).

    Covers named reds (crimson, firebrick, tomato, …), hex (#f00, #ff0000, #c00000, …),
    rgb/rgba, and hsl/hsv — consistent with prompt.md's 'distinct red annotations'.
    """
    if not isinstance(value, str):
        return False
    s = value.strip().lower()
    s_nospace = re.sub(r"\s+", "", s)

    if s_nospace in _CSS_RED_NAMES:
        return True

    rgb = _parse_rgb_tuple(s_nospace)
    if rgb is not None:
        return _rgb_hue_is_red(*rgb)

    # hsl(hue, …) or hsv(hue, …) — accept red-range hues.
    m = re.match(r"hs[lv]\((\d+(?:\.\d+)?)[,.]", s_nospace)
    if m:
        hue = float(m.group(1))
        return hue <= 20.0 or hue >= 340.0

    return False


def _annotation_is_red(ann: dict) -> bool:
    font = ann.get("font") or {}
    if _color_value_is_red(font.get("color")):
        return True
    for prop in ("bgcolor", "bordercolor", "arrowcolor"):
        if _color_value_is_red(ann.get(prop)):
            return True
    return False


def _shape_is_red(sh: dict) -> bool:
    line = sh.get("line") or {}
    if _color_value_is_red(line.get("color")):
        return True
    if _color_value_is_red(sh.get("fillcolor")):
        return True
    return False


def layout_annotations_use_red_on_heatmap(html: str) -> bool:
    for data, layout in _iter_all_heatmap_figure_data_layout_pairs(html):
        for ann in layout.get("annotations") or []:
            if isinstance(ann, dict) and _annotation_is_red(ann):
                return True
        for sh in layout.get("shapes") or []:
            if isinstance(sh, dict) and _shape_is_red(sh):
                return True
    return False


def _iter_inline_json_figure_objects(html: str):
    """
    Yield (data, layout) for JSON figure dicts embedded as `{ "data": [...], "layout": {...} }`
    anywhere in the HTML (not only Plotly.newPlot/react call sites).

    Covers write_html / inline figure blobs when newPlot/react parsing fails.
    """
    dec = json.JSONDecoder()
    seen_starts: set[int] = set()
    for m in re.finditer(r"\{\s*\"data\"\s*:", html):
        start = m.start()
        if start in seen_starts:
            continue
        try:
            obj, _end = dec.raw_decode(html, start)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        data, layout = obj.get("data"), obj.get("layout")
        if not isinstance(data, list) or not isinstance(layout, dict):
            continue
        if not any(isinstance(t, dict) and t.get("type") == "heatmap" for t in data):
            continue
        seen_starts.add(start)
        yield data, layout


def _iter_js_literal_data_layout_pairs(html: str):
    """
    Pair `var data = [...];` / `var layout = {...};` when both are inline JSON literals
    and data includes a heatmap trace (order may be data-first or layout-first).
    """
    dec = json.JSONDecoder()

    def _try_pair_data_then_layout(data_start: int, data_obj: list, data_end: int) -> (
        tuple[list, dict] | None
    ):
        if not any(isinstance(t, dict) and t.get("type") == "heatmap" for t in data_obj):
            return None
        tail = html[data_end : data_end + 100_000]
        lm = re.search(r"(?:var|let|const)\s+layout\s*=\s*(\{)", tail, re.I)
        if not lm:
            return None
        lstart = data_end + lm.start(1)
        try:
            layout, _ = dec.raw_decode(html, lstart)
        except json.JSONDecodeError:
            return None
        if isinstance(layout, dict):
            return data_obj, layout
        return None

    def _try_pair_layout_then_data(layout_start: int, layout_obj: dict, layout_end: int) -> (
        tuple[list, dict] | None
    ):
        tail = html[layout_end : layout_end + 100_000]
        dm = re.search(r"(?:var|let|const)\s+data\s*=\s*(\[)", tail, re.I)
        if not dm:
            return None
        dstart = layout_end + dm.start(1)
        try:
            data_obj, _ = dec.raw_decode(html, dstart)
        except json.JSONDecodeError:
            return None
        if not isinstance(data_obj, list):
            return None
        if not any(isinstance(t, dict) and t.get("type") == "heatmap" for t in data_obj):
            return None
        return data_obj, layout_obj

    for m in re.finditer(r"(?:var|let|const)\s+data\s*=\s*(\[)", html, re.I):
        dstart = m.start(1)
        try:
            data_obj, dend = dec.raw_decode(html, dstart)
        except json.JSONDecodeError:
            continue
        if not isinstance(data_obj, list):
            continue
        pair = _try_pair_data_then_layout(dstart, data_obj, dend)
        if pair:
            yield pair

    for m in re.finditer(r"(?:var|let|const)\s+layout\s*=\s*(\{)", html, re.I):
        lstart = m.start(1)
        try:
            layout_obj, lend = dec.raw_decode(html, lstart)
        except json.JSONDecodeError:
            continue
        if not isinstance(layout_obj, dict):
            continue
        pair = _try_pair_layout_then_data(lstart, layout_obj, lend)
        if pair:
            yield pair


def _iter_all_heatmap_figure_data_layout_pairs(html: str):
    """
    All (data, layout) pairs that include a heatmap trace — newPlot/react, inline JSON
    figures, and common JS literal patterns. prompt.md does not mandate a specific embed API.
    """
    yielded: set[tuple[int, int]] = set()

    def _key(data: list, layout: dict) -> tuple[int, int]:
        return id(data), id(layout)

    for data, layout in _iter_newplot_react_figures(html):
        if any(isinstance(t, dict) and t.get("type") == "heatmap" for t in data):
            k = _key(data, layout)
            if k not in yielded:
                yielded.add(k)
                yield data, layout

    for data, layout in _iter_inline_json_figure_objects(html):
        k = _key(data, layout)
        if k not in yielded:
            yielded.add(k)
            yield data, layout

    for data, layout in _iter_js_literal_data_layout_pairs(html):
        k = _key(data, layout)
        if k not in yielded:
            yielded.add(k)
            yield data, layout


def _collect_red_marking_dates_from_figure(data: list, layout: dict) -> set:
    """Dates referenced by red layout annotations/shapes for a heatmap figure."""
    if not any(isinstance(t, dict) and t.get("type") == "heatmap" for t in data):
        return set()
    dates: set = set()

    def _try_add(xval):
        try:
            dates.add(pd.to_datetime(xval).date())
        except Exception:
            pass

    for ann in layout.get("annotations") or []:
        if isinstance(ann, dict) and _annotation_is_red(ann):
            _try_add(ann.get("x"))
    for sh in layout.get("shapes") or []:
        if isinstance(sh, dict) and _shape_is_red(sh):
            _try_add(sh.get("x0"))
            _try_add(sh.get("x1"))
    return dates


def red_marking_dates(html: str) -> set:
    """
    Calendar dates with distinct red anomaly markings on the heatmap figure.

    Inspects every embedded figure that includes a heatmap: Plotly.newPlot/react,
    inline `{data, layout}` JSON, and paired JS `data`/`layout` literals — not only
    one parsing strategy (prompt.md §9 does not require a specific Plotly API).
    """
    out: set = set()
    for data, layout in _iter_all_heatmap_figure_data_layout_pairs(html):
        out |= _collect_red_marking_dates_from_figure(data, layout)
    return out


def extract_all_numbers_from_html(html: str) -> set[float]:
    text = re.sub(r"<[^>]+>", " ", html)
    result: set[float] = set()
    for m in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text):
        try:
            v = float(m.replace(",", ""))
            if v > 0:
                result.add(v)
        except ValueError:
            pass
    return result


def heatmap_z_matrix_from_loaded_hourly_df(df: pd.DataFrame) -> np.ndarray:
    """
    Same pivot as pipeline.generate_report: 24 rows × one column per calendar day,
    cell values are hourly consumption_kwh (prompt.md: color encodes consumption_kwh).

    Expects load_data-style df: DatetimeIndex, column consumption_kwh.
    """
    tmp = df.copy()
    tmp["_date"] = tmp.index.date
    tmp["_hour"] = tmp.index.hour
    pivot = tmp.pivot_table(
        index="_hour", columns="_date", values="consumption_kwh", aggfunc="first"
    )
    pivot = pivot.reindex(range(24)).sort_index()
    return pivot.values.astype(np.float64)


def html_includes_inline_plotly_bundle_definition(html: str) -> bool:
    """
    prompt.md: embed the *complete* Plotly JavaScript library inline.

    Heuristic: inline <script> contains a Plotly bundle or figure init — not a
    fixed list of API entrypoints (newPlot/react are common but not required
    by prompt.md wording).
    """
    # Strip HTML comments to avoid false positives in documentation strings.
    no_comments = re.sub(r"<!--[\s\S]*?-->", " ", html)
    if not re.search(r"<script[\s\S]*?</script>", no_comments, flags=re.I):
        return False
    if re.search(r"\bPlotly\.newPlot\b", no_comments) or re.search(
        r"\bPlotly\.react\b", no_comments
    ):
        return True
    if re.search(r"window\.Plotly\s*=", no_comments):
        return True
    if re.search(r"function\s+Plotly\b", no_comments):
        return True
    if re.search(r"Plotly\.version\s*=", no_comments):
        return True
    # Any Plotly namespace use (e.g. Plotly.addFrames, minified bundles)
    if re.search(r"\bPlotly\.", no_comments):
        return True
    # Heatmap figure JSON + large inline script strongly implies full Plotly runtime
    if plotly_html_declares_heatmap(no_comments) and len(no_comments) > 20_000:
        return True
    return False


def html_suggests_non_plotly_interactive_viz_library(html: str) -> bool:
    """
    True if HTML appears to load or invoke another major interactive charting library.

    prompt.md: all interactive visualizations in the HTML report MUST use plotly 5.x.

    Note: the bundled plotly.min.js source may contain literal strings such as
    https://cdn.plot.ly/ inside default config — that is not an extra library and
    must not be flagged here. External *loads* are handled by script-src checks.
    """
    lowered = html.lower()
    needles = (
        "chart.js",
        "chartjs",
        "d3js.org",
        "echarts",
        "bokeh.min",
        "highcharts",
        "apexcharts",
        "vega-lite",
        "vega.embed",
    )
    if any(n in lowered for n in needles):
        return True
    # Standalone d3 loader URLs (plotly bundle may mention "d3" in unrelated strings).
    if re.search(r"d3\.js[\"']?\s*\)|unpkg\.com/d3", lowered):
        return True
    return False


def _date_col_candidates(df: pd.DataFrame) -> str | None:
    """prompt.md does not name columns; pick a plausible date column."""
    for name in (
        "date",
        "Date",
        "day",
        "Day",
        "calendar_date",
        "timestamp",
        "Timestamp",
        "anomaly_date",
        "AnomalyDate",
    ):
        if name in df.columns:
            return name
    for c in df.columns:
        if "date" in str(c).lower():
            return c
    # first column with datetime-like dtype or parseable values
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def _bool_flag_candidates(df: pd.DataFrame) -> str | None:
    for name in ("is_anomaly", "anomaly", "flagged", "is_outlier", "outlier"):
        if name not in df.columns:
            continue
        s = df[name]
        if pd.api.types.is_bool_dtype(s):
            return name
        if pd.api.types.is_integer_dtype(s) and set(pd.unique(s.dropna())) <= {0, 1}:
            return name
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            return c
        if pd.api.types.is_integer_dtype(s) and set(pd.unique(s.dropna())) <= {0, 1}:
            return c
    return None


def normalize_anomaly_dates(anomalies_df: pd.DataFrame) -> set:
    """
    Dates of anomalous days. Works with common column names; prompt.md does
    not fix schema beyond pd.DataFrame.

    Supports:
    - rows with a boolean / 0-1 flag column (only True/1 rows)
    - anomaly-only tables (one row per anomaly, no flag column): every row
      counts as an anomalous day
    - DatetimeIndex or date-like index when there is no date column
    """
    if anomalies_df.empty:
        return set()

    dc = _date_col_candidates(anomalies_df)
    fc = _bool_flag_candidates(anomalies_df)

    if dc is None:
        idx = anomalies_df.index
        if isinstance(idx, pd.DatetimeIndex):
            return {pd.Timestamp(ts).date() for ts in idx}
        return set()

    out: set = set()
    if fc is not None:
        sub = anomalies_df.loc[anomalies_df[fc]]
    else:
        # Common pattern: DataFrame lists only anomalous days (no all-days grid).
        sub = anomalies_df

    for d in sub[dc]:
        if hasattr(d, "date"):
            d = d.date()
        out.add(pd.Timestamp(d).date())
    return out
