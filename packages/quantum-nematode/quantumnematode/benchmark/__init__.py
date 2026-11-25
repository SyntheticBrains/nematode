"""Real-worm reference data for grading simulated worms.

The grader's modules live in the ``wormbench`` package. Each is registered here under its old name
as the same module object, so ``quantumnematode.benchmark.grading`` and ``wormbench.grading`` are
one module, private names and classes included.
"""

import sys

from wormbench import chemotaxis as chemotaxis
from wormbench import grading as grading
from wormbench import measurement as measurement
from wormbench import readouts as readouts
from wormbench import references as references
from wormbench import track_import as track_import

for _module in (chemotaxis, grading, measurement, readouts, references, track_import):
    sys.modules[f"{__name__}.{_module.__name__.rsplit('.', 1)[1]}"] = _module
