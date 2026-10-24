"""A restricted unpickler for Scheer & Bargmann 2023's deposited pickles.

The deposit's pickles reference only numpy, pandas, ``builtins.slice``, ``datetime.date`` and the
``ssm`` model classes. This unpickler admits the first four and replaces every ``ssm`` class with an
empty stand-in, so none of its code runs. Any other class is refused before it is constructed.

The genotype file was written with pandas 1.x, whose numeric index classes pandas 2 removed. They are
read as ``pandas.Index``, the same renaming pandas applies in ``read_pickle``.
"""

from __future__ import annotations

import pickle
from typing import Any

_ALLOWED_PREFIXES = ("numpy", "pandas")
_ALLOWED = {("builtins", "slice"), ("datetime", "date")}
_RENAMED = {
    ("pandas.core.indexes.numeric", name): ("pandas.core.indexes.base", "Index")
    for name in ("Int64Index", "UInt64Index", "Float64Index")
}


class _Stub:
    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


class SafeUnpickler(pickle.Unpickler):
    """Allow numpy and pandas, ``slice`` and ``datetime.date``; stand in for ``ssm``; refuse all else."""

    def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401 - a class of any type
        if module.startswith("ssm."):
            return type(name, (_Stub,), {})
        module, name = _RENAMED.get((module, name), (module, name))
        if module.split(".", maxsplit=1)[0] in _ALLOWED_PREFIXES or (module, name) in _ALLOWED:
            return super().find_class(module, name)
        msg = f"refused {module}.{name}"
        raise pickle.UnpicklingError(msg)
