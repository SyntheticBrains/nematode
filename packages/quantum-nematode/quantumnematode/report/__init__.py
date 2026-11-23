"""Report module for Quantum Nematode.

The per-step behavioural record lives in the ``wormbench`` package. It is registered here under its
old name, ``quantumnematode.report.behaviour``, as the same module object.
"""

import sys

from wormbench import records as behaviour

sys.modules[f"{__name__}.behaviour"] = behaviour
