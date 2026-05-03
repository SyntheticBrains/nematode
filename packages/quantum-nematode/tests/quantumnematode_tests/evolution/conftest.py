"""Evolution-test conftest: force GC after each test to break leak cycles.

The ``cma`` library wires its ``CMAEvolutionStrategy`` to a
``CMADataLogger`` via mutual references, forming a reference cycle
that CPython's reference-counting GC alone cannot break.  When the
test scope's local ``loop`` reference is dropped, the strategy +
encoder + brain stay alive until the cyclic GC runs — which on this
test suite's serial runs accumulates dozens of stale instances and
their tensor allocations, climbing RSS several GB above baseline.

Forcing ``gc.collect()`` after each test breaks the cycles
immediately.  Confirmed via local profiling: without this fixture,
``CMAEvolutionStrategy`` instance count climbs 1 → 14 across
``test_loop_smoke.py``'s 15 tests; with this fixture, it stays at 0
between tests.

Without this fixture, under pytest-xdist parallelism multiple xdist
workers each accumulating stale evolution loops can produce enough
memory pressure to starve the GitHub Actions runner agent and
trigger control-plane heartbeat timeouts ("operation was canceled").
"""

from __future__ import annotations

import gc

import pytest


@pytest.fixture(autouse=True)
def _gc_after_each_test():  # pyright: ignore[reportUnusedFunction]
    yield
    gc.collect()
