"""
prompt.md — Tech Stack (normative for the solution environment):

- Language: Python 3.10+
- Data processing: pandas 2.x, numpy 1.x
- Visualization: plotly 5.x

These checks enforce the stack line from prompt.md; they are independent of
`test_run_pipeline.py` (analytics + HTML) and `test_load_data.py` (§1 ingestion).
"""

from __future__ import annotations

import sys

from prompt_helpers import require_submitted_pipeline


def test_python_version_is_3_10_or_newer():
    """prompt.md Tech Stack: Language Python 3.10+"""
    require_submitted_pipeline()
    assert sys.version_info >= (3, 10), (
        f"prompt.md requires Python 3.10+; this interpreter is {sys.version.split()[0]}"
    )


def test_pandas_major_version_is_2():
    """prompt.md Tech Stack: pandas 2.x"""
    require_submitted_pipeline()
    import pandas as pd

    major = int(pd.__version__.split(".")[0])
    assert major == 2, (
        f"prompt.md requires pandas 2.x; got pandas {pd.__version__}"
    )


def test_numpy_major_version_is_1():
    """prompt.md Tech Stack: numpy 1.x"""
    require_submitted_pipeline()
    import numpy as np

    major = int(np.__version__.split(".")[0])
    assert major == 1, (
        f"prompt.md requires numpy 1.x; got numpy {np.__version__}"
    )


def test_plotly_major_version_is_5():
    """prompt.md Tech Stack: plotly 5.x"""
    require_submitted_pipeline()
    import plotly

    major = int(plotly.__version__.split(".")[0])
    assert major == 5, (
        f"prompt.md requires plotly 5.x; got plotly {plotly.__version__}"
    )
