"""
Tests for prompt.md — Tech Stack + single-file rule + plotly-only interactive HTML.

Explicit prompt.md mappings:

  • Language: Python 3.10+
  • Data processing: pandas 2.x, numpy 1.x
  • Visualization: plotly 5.x; all interactive visualizations in the HTML report
    MUST use plotly 5.x
  • Solution MUST be a single file named pipeline.py at the repository root
"""

from __future__ import annotations

import ast
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import plotly

import pipeline

from prompt_helpers import (
    heatmap_traces_from_html,
    plotly_report_includes_heatmap,
    require_submitted_pipeline,
)


def _repo_root() -> pathlib.Path:
    """Repository root: env PIPELINE_REPO_ROOT (Docker /app) or parent of tests/."""
    env = os.environ.get("PIPELINE_REPO_ROOT")
    if env:
        return pathlib.Path(env)
    return pathlib.Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
PIPELINE_PATH = REPO_ROOT / "pipeline.py"


def _major(version: str) -> int:
    match = re.match(r"^(\d+)", version)
    assert match is not None, f"Could not parse major version from {version!r}"
    return int(match.group(1))


class TestPromptTechStackAndLayout:
    """prompt.md — Tech Stack + 'single file pipeline.py at repository root'."""

    def test_solution_is_single_pipeline_py_at_repo_root(self):
        """prompt.md: single file named pipeline.py at the repository root."""
        assert PIPELINE_PATH.is_file(), "Expected pipeline.py at repository root"
        body = PIPELINE_PATH.read_text(encoding="utf-8")
        assert body.strip(), "pipeline.py must not be empty"
        resolved = pathlib.Path(pipeline.__file__).resolve()
        assert resolved == PIPELINE_PATH.resolve(), (
            "Imported pipeline module must be loaded from repository-root pipeline.py"
        )
        assert callable(getattr(pipeline, "run_pipeline", None))

        # Single-file rule: no imports from other *.py modules in this repo root.
        other_repo_modules = {p.stem for p in REPO_ROOT.glob("*.py")} - {"pipeline"}
        tree = ast.parse(body)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in other_repo_modules, (
                        f"prompt.md requires a single-file solution; found import {alias.name!r}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    assert root not in other_repo_modules, (
                        "prompt.md requires a single-file solution; "
                        f"found from-import of {node.module!r}"
                    )

    def test_python_version_is_3_10_or_higher(self):
        """prompt.md: Language Python 3.10+."""
        require_submitted_pipeline()
        assert sys.version_info >= (3, 10)

    def test_uses_pandas_2x(self):
        """prompt.md: Data processing — pandas 2.x."""
        require_submitted_pipeline()
        assert _major(pd.__version__) == 2, (
            f"Expected pandas 2.x per prompt.md, found {pd.__version__}"
        )

    def test_uses_numpy_1x(self):
        """prompt.md: Data processing — numpy 1.x."""
        require_submitted_pipeline()
        assert _major(np.__version__) == 1, (
            f"Expected numpy 1.x per prompt.md, found {np.__version__}"
        )

    def test_uses_plotly_5x(self):
        """prompt.md: Visualization — plotly 5.x."""
        require_submitted_pipeline()
        assert _major(plotly.__version__) == 5, (
            f"Expected plotly 5.x per prompt.md, found {plotly.__version__}"
        )

    def test_interactive_html_report_uses_plotly_heatmap(self, multi_year_csv, tmp_path):
        """
        prompt.md Tech Stack: plotly 5.x; §9 daily heat map is a Plotly heatmap.

        Asserts runtime plotly major version 5 and that the generated report contains
        a parseable Plotly heatmap trace. Inline JS / no CDN are covered in
        test_run_pipeline.py (prompt.md §9), not duplicated with extra heuristics here.
        """
        assert _major(plotly.__version__) == 5
        from pipeline import run_pipeline

        out = tmp_path / "plotly5_check.html"
        run_pipeline(str(multi_year_csv), str(out), cost_per_kwh=0.1)
        html = out.read_text(encoding="utf-8")
        assert plotly_report_includes_heatmap(html), (
            "HTML report must include a Plotly heatmap (interactive viz via plotly)"
        )
        assert heatmap_traces_from_html(html), (
            "Heatmap trace JSON must be parseable from the report (primary figure)"
        )
