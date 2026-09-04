# pyright: reportPrivateUsage=false
"""Runtime-conformance tests for the BrainTopology / LearningRule Protocols.

Covers the `Focal implementation conforms at runtime` scenario of
``openspec/changes/add-l4-trace-substrate/specs/brain-architecture/spec.md``
— the conformance check the L1 architecture-plugin change promised
(``archive/2026-05-24-.../tasks.md:85``) but never shipped.
"""

from __future__ import annotations

from quantumnematode.brain.arch import BrainTopology, LearningRule
from quantumnematode.brain.arch.connectome_ppo import (
    ConnectomePPOBrain,
    ConnectomePPOBrainConfig,
)
from quantumnematode.brain.arch.dtypes import DeviceType

_SEED = 2026


def _make_brain() -> ConnectomePPOBrain:
    cfg = ConnectomePPOBrainConfig(seed=_SEED)
    return ConnectomePPOBrain(config=cfg, device=DeviceType.CPU)


class TestProtocolConformance:
    def test_connectome_topology_satisfies_brain_topology(self) -> None:
        brain = _make_brain()
        assert isinstance(brain.topology, BrainTopology)

    def test_connectome_ppo_rule_satisfies_learning_rule(self) -> None:
        brain = _make_brain()
        assert isinstance(brain._rule, LearningRule)

    def test_learnable_parameters_is_a_property(self) -> None:
        assert isinstance(
            type(_make_brain().topology).__dict__["learnable_parameters"],
            property,
        )
