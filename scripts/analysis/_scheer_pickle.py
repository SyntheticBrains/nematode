"""A restricted unpickler for Scheer & Bargmann 2023's deposited pickles.

The deposit's pickles reference a fixed set of numpy and pandas classes and reconstructors,
``builtins.slice``, ``datetime.date`` and the ``ssm`` model classes. This unpickler admits exactly
those (module, name) pairs and replaces every ``ssm`` class with an empty stand-in, so none of its
code runs. Any other pair is refused before it is resolved, including dotted names, which the pickle
protocol would resolve attribute by attribute, and other callables in an admitted module.

The genotype file was written with pandas 1.x, whose numeric index classes pandas 2 removed. They are
read as ``pandas.Index``, the same renaming pandas applies in ``read_pickle``. The deposit was written
by numpy 1.x (``numpy.core``); the same reconstructors under numpy 2's ``numpy._core`` are admitted
too, so arrays pickled by this project's numpy read the same way.
"""

from __future__ import annotations

import pickle
from typing import Any

_ALLOWED = frozenset(
    {
        ("builtins", "slice"),
        ("datetime", "date"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.ma", "MaskedArray"),
        ("numpy.ma.core", "MaskedArray"),
        ("numpy.ma.core", "_mareconstruct"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.series", "Series"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
        ("pandas.core.internals.managers", "SingleBlockManager"),
    },
)
_RENAMED = {
    ("pandas.core.indexes.numeric", name): ("pandas.core.indexes.base", "Index")
    for name in ("Int64Index", "UInt64Index", "Float64Index")
}


class _Stub:
    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


class SafeUnpickler(pickle.Unpickler):
    """Admit the deposit's listed classes; stand in for ``ssm``; refuse all else."""

    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401 - a class of any type
        if module.startswith("ssm.") and "." not in name:
            return type(name, (_Stub,), {})
        module, name = _RENAMED.get((module, name), (module, name))
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        msg = f"refused {module}.{name}"
        raise pickle.UnpicklingError(msg)
