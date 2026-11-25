"""Batched evaluation of frozen controllers over many simulated animals.

Chen et al.'s measured odour field lives in the ``wormbench`` package. It is registered here under
its old name, ``quantumnematode.evaluation.odour_field``, as the same module object.
"""

import sys

from wormbench import odour_field as odour_field

sys.modules[f"{__name__}.odour_field"] = odour_field
