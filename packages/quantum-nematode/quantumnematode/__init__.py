"""The quantum nematode package.

In a source checkout, the benchmark's reference data are read from the repository's ``data/``:
it is set as the ``wormbench`` data root here, before any module that reads it is used, so the
project always grades on its vendored files. An installed copy without that directory leaves the
root unset, and ``wormbench`` resolves its files from its own sources.
"""

from pathlib import Path

from wormbench import data as _benchmark_data

_REPOSITORY_DATA = Path(__file__).resolve().parents[3] / "data"
if (_REPOSITORY_DATA / "benchmark").is_dir():
    _benchmark_data.set_root(_REPOSITORY_DATA)
