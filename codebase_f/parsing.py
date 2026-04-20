#!/usr/bin/env python3
import dataclasses
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List


class TestStatus(Enum):
    """The test status enum."""
    PASSED = 1
    FAILED = 2
    SKIPPED = 3
    ERROR = 4


@dataclasses.dataclass
class TestResult:
    """The test result dataclass."""
    name: str
    status: TestStatus


### DO NOT MODIFY THE CODE ABOVE ###
### Implement the parsing logic below ###

def parse_test_output(stdout_content: str, stderr_content: str) -> List[TestResult]:
    """
    Parse pytest verbose (``-v``) result lines and extract test results.

    Matches lines like::
        tests/test_run_pipeline.py::TestPromptInterface::test_foo PASSED [ 10%]

    Class names, parametrized ``[...]`` ids, and optional progress suffixes are
    allowed. Stdout and stderr are combined so lines on stderr are included.

    **F2P / QC:** Pytest reports ``ERROR`` for setup/teardown failures. For empty
    baselines, QC requires **FAILED**, not ERROR. Any ``ERROR`` status is therefore
    stored as **FAILED**.

    If the same node id appears more than once (duplicate lines across streams),
    the **last** occurrence wins.
    """
    import re
    combined = stdout_content + "\n" + stderr_content
    # Pytest --color=yes embeds ANSI sequences between the node id and PASSED/FAILED/ERROR.
    combined = re.sub(r"\x1b\[[0-9;]*m", "", combined)
    # Node id: file.py::Class::method or file.py::function (each segment after .py starts with ::)
    pattern = re.compile(
        r'^((?:[\w/.-]+/)?[\w.-]+\.py(?:::[\w\[\]-]+)+)\s+(PASSED|FAILED|SKIPPED|ERROR)\b',
        re.MULTILINE,
    )
    by_name: Dict[str, TestResult] = {}
    for match in pattern.finditer(combined):
        name = match.group(1)
        status_str = match.group(2)
        if status_str == "ERROR":
            status_str = "FAILED"
        status = TestStatus[status_str]
        by_name[name] = TestResult(name=name, status=status)
    return list(by_name.values())

### Implement the parsing logic above ###
### DO NOT MODIFY THE CODE BELOW ###

def export_to_json(results: List[TestResult], output_path: Path) -> None:
    json_results = {
        'tests': [
            {'name': result.name, 'status': result.status.name} for result in results
        ]
    }
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)


def main(stdout_path: Path, stderr_path: Path, output_path: Path) -> None:
    with open(stdout_path) as f:
        stdout_content = f.read()
    with open(stderr_path) as f:
        stderr_content = f.read()
    results = parse_test_output(stdout_content, stderr_content)
    export_to_json(results, output_path)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python parsing.py <stdout_file> <stderr_file> <output_json>')
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
