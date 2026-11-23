"""The project grades with the wormbench package: one copy, the same modules, current records.

Covers the benchmark-package spec's requirements "One grader for the project and for outsiders"
(scenario "The old path is the same module") and "The package's default version and its records"
(scenario "A stale copy or record is caught"), and "Reference data resolved when used" (scenario
"The project reads its vendored files").
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[5]
ALIASES = {
    "quantumnematode.benchmark.readouts": "wormbench.readouts",
    "quantumnematode.benchmark.grading": "wormbench.grading",
    "quantumnematode.benchmark.measurement": "wormbench.measurement",
    "quantumnematode.benchmark.chemotaxis": "wormbench.chemotaxis",
    "quantumnematode.benchmark.track_import": "wormbench.track_import",
    "quantumnematode.benchmark.references": "wormbench.references",
    "quantumnematode.validation.roaming_dwelling": "wormbench.classifier",
    "quantumnematode.report.behaviour": "wormbench.records",
    "quantumnematode.evaluation.odour_field": "wormbench.odour_field",
}


@pytest.mark.parametrize(("old", "new"), sorted(ALIASES.items()))
def test_each_old_path_is_the_moved_module_whichever_is_imported_first(old: str, new: str) -> None:
    """Imported first in a fresh interpreter, each old path is the very wormbench module."""
    code = (
        f"import importlib; old = importlib.import_module({old!r}); "
        f"new = importlib.import_module({new!r}); "
        "assert old is new, (old, new); "
        "assert all(getattr(old, k) is getattr(new, k) for k in vars(new))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603 - fixed argv


def test_the_project_reads_its_vendored_files() -> None:
    """With the project imported, every reference file resolves under the repository's data/."""
    import quantumnematode  # noqa: F401 - sets the data root
    from wormbench import data

    assert data.get_root() == (_REPO / "data").resolve()
    listed = data.manifest()["files"]
    for key in listed:
        namespace, name = key.split("/", 1)
        assert data.path(namespace, name) == (_REPO / "data" / namespace / name).resolve()


def test_the_shipped_records_and_copies_are_current() -> None:
    """The package's records, data copies and manifest match the committed sources."""
    script = _REPO / "scripts" / "analysis" / "wormbench_records.py"
    result = subprocess.run(  # noqa: S603 - fixed argv
        [sys.executable, str(script), "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
