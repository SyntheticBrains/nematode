"""End-to-end check that the brain plugin registry matches BrainType.

Lives in a separate module from ``test_registry.py`` because the latter
has an autouse fixture that snapshots-and-clears the module-level
registry per test. The consistency check needs the production
post-import-time state.
"""

from __future__ import annotations


def test_registry_matches_brain_type_at_import_time() -> None:
    """Importing ``quantumnematode.brain.arch`` raises if registry ↔ enum disagree.

    The package ``__init__.py`` invokes ``assert_registry_matches_enum``
    after every architecture module has been imported. If we got here,
    the assertion already passed at import time. Re-invoke it to verify
    the invariant remains true at test time.
    """
    import quantumnematode.brain.arch
    from quantumnematode.brain.arch._registry import (
        assert_registry_matches_enum,
        list_registered_brains,
    )
    from quantumnematode.brain.arch.dtypes import BrainType

    # The package import is what triggers the registry population; if it
    # imported successfully, the consistency check at the bottom of
    # brain/arch/__init__.py already passed once. Confirm explicitly here.
    _ = quantumnematode.brain.arch
    assert_registry_matches_enum()

    # Spot-check the two sets are equal in this process (no surprise
    # mutations since import).
    assert {bt.value for bt in BrainType} == list_registered_brains()
