# Brain Architecture Plugin Developer Guide

This guide walks through how to add a new brain architecture to
`quantum-nematode`. The codebase uses a decorator-registration plugin
registry so that adding an architecture is a contained edit — no per-arch
branches survive in the simulation entrypoint or YAML loader.

This guide is the canonical reference for what files to touch and in what
order. If you're adding a brain that re-uses an existing topology /
learning-rule combination, you may not even need to touch all of them.

## Files-touched budget

Adding a vanilla new architecture should touch **at most six files**:

1. `packages/quantum-nematode/quantumnematode/brain/arch/<name>.py` —
   the new module (Brain class + Config class + decorator).
2. `packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py` —
   add a `BrainType` enum member matching the registered name.
3. `packages/quantum-nematode/quantumnematode/brain/arch/__init__.py` —
   import the new module so its decorator runs at package-import time,
   and re-export the public symbols.
4. `packages/quantum-nematode/quantumnematode/utils/config_loader.py` —
   add the new config class to the `BrainConfigType` union so
   Pydantic's smart-union picks the right class when loading YAML.
5. `packages/quantum-nematode/quantumnematode/utils/brain_factory.py` —
   add a `_build_infra_kwargs` branch only if the new brain consumes a
   different subset of infrastructure kwargs than the default
   (`{num_actions: 4, device}`).
6. A test file under
   `packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/`.

Plus optionally a YAML config under `configs/scenarios/` if you're
shipping a runnable baseline alongside the brain.

If you find yourself touching more than six files for a vanilla
architecture, the dispatcher / loader factoring has regressed — surface
that as a design issue rather than working around it.

## Step-by-step: add a `TinyMLPBrain`

Worked example: a hypothetical 1-hidden-layer MLP that consumes
food-chemotaxis features and emits 4 action logits. Use this as a
template; the same shape applies to any architecture you add.

### 1. Write the Brain module

`packages/quantum-nematode/quantumnematode/brain/arch/tinymlp.py`:

```python
"""Tiny MLP brain — single hidden layer, REINFORCE update rule."""

from __future__ import annotations

import torch
from torch import nn

from quantumnematode.brain.arch._brain import BrainParams, ClassicalBrain
from quantumnematode.brain.arch._registry import register_brain
from quantumnematode.brain.arch.dtypes import BrainConfig, BrainType


class TinyMLPBrainConfig(BrainConfig):
    hidden_dim: int = 16
    learning_rate: float = 1e-3


@register_brain(
    name="tinymlp",
    config_cls=TinyMLPBrainConfig,
    brain_type=BrainType.TINY_MLP,
    families=("classical",),
)
class TinyMLPBrain(ClassicalBrain):
    def __init__(
        self,
        config: TinyMLPBrainConfig,
        num_actions: int = 4,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self.net = nn.Sequential(
            nn.Linear(2, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, num_actions),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=config.learning_rate,
        )
        # ... rest of the Brain Protocol surface: preprocess, run_brain,
        # update_memory, prepare_episode, post_process_episode, copy.
```

Required keyword arguments to `@register_brain`:

- `name`: the registry key. Must match `brain_type.value` exactly — the
  startup-time consistency check in `brain/arch/__init__.py` will fail
  fast if they drift.
- `config_cls`: the Pydantic `BrainConfig` subclass the brain accepts.
- `brain_type`: the enum member; see § 2.
- `families`: a tuple of family tags. Common values: `("classical",)`,
  `("quantum",)`, `("spiking",)`, or any combination (e.g. a spiking
  quantum architecture uses `("quantum", "spiking")`). The family sets
  exposed from `dtypes.py` (`QUANTUM_BRAIN_TYPES`,
  `CLASSICAL_BRAIN_TYPES`, `SPIKING_BRAIN_TYPES`) are derived from these
  tags at module-import time, so the family set membership tracks
  registrations automatically.

### 2. Add the `BrainType` enum member

`packages/quantum-nematode/quantumnematode/brain/arch/dtypes.py`:

```python
class BrainType(StrEnum):
    # ... existing members ...
    TINY_MLP = "tinymlp"
```

The string value must match the `name=` argument passed to
`@register_brain` in step 1. The startup-time consistency check
guarantees you can't forget either side.

`BrainType` is a `StrEnum`, so `BrainType.TINY_MLP == "tinymlp"` holds
directly. You can use enum members as dict keys, in match-case
patterns, and in tests interchangeably with the string form.

### 3. Wire the module into `brain/arch/__init__.py`

Add the import + the public re-exports:

```python
from .tinymlp import TinyMLPBrain, TinyMLPBrainConfig

__all__ = [
    # ... existing entries ...
    "TinyMLPBrain",
    "TinyMLPBrainConfig",
]
```

The import is what triggers the `@register_brain` decorator to run.
Without it, the registry will not know about the new arch and the
consistency check will fail at import time because the enum member
exists with no matching registration.

### 4. Add the config class to `BrainConfigType` (config_loader.py)

`packages/quantum-nematode/quantumnematode/utils/config_loader.py`:

```python
from quantumnematode.brain.arch import (
    # ...
    TinyMLPBrainConfig,
)

BrainConfigType = (
    # ...
    | TinyMLPBrainConfig
)
```

This is what lets Pydantic's smart-union pick the right config class
when loading a YAML file with `brain.name: tinymlp`. If you forget
this step, the YAML will silently parse to the closest-shaped existing
config class and drop any fields specific to your architecture.

### 5. (Optional) Add a branch to `_build_infra_kwargs`

`packages/quantum-nematode/quantumnematode/utils/brain_factory.py`:

If your brain's `__init__` accepts only `config`, `num_actions`, and
`device`, you don't need to do anything — the default-shape branch
already handles you.

If your brain needs something different (e.g. an explicit `input_dim`,
a `learning_rate` scheduler, a `parameter_initializer`), add a single
branch:

```python
if brain_type is BrainType.TINY_MLP:
    return {
        "num_actions": 4,
        "device": device,
    }
```

The branch is the only place where the per-arch `__init__` signature
shape is reflected. Keep it minimal — only the kwargs the brain
actually consumes.

### 6. Write the tests

`packages/quantum-nematode/tests/quantumnematode_tests/brain/arch/test_tinymlp.py`:

- Construction from a default config.
- Registry round-trip: `instantiate_brain("tinymlp", TinyMLPBrainConfig())`
  returns an instance.
- One forward-pass shape assertion (4 logits).
- One training-step finiteness assertion (no NaNs after a fake update).

The registry-equivalence test pattern in
`tests/quantumnematode_tests/brain/arch/test_registration_equivalence.py`
is a good template if you want to assert that
`instantiate_brain(name, cfg)` and the direct constructor produce
identical initial weights (useful when migrating an existing arch
behind the registry).

### 7. (Optional) Ship a YAML config

`configs/scenarios/foraging/tinymlp_klinotaxis.yml`:

```yaml
brain:
  name: tinymlp
  config:
    hidden_dim: 16
    learning_rate: 1e-3
# ... env + reward + satiety blocks as usual
```

The `tests/quantumnematode_tests/utils/test_config_loader_yaml_compat.py`
YAML-compatibility regression walks `configs/scenarios/` and asserts
every YAML loads to a valid `Brain` via the registry. If your YAML is
malformed, that test will fail with a specific error.

## Architecture-specific extensions

### Custom topology

If your brain has a non-trivial topology (e.g. a fixed adjacency mask,
a non-fully-connected layer structure), implement the `BrainTopology`
Protocol declared in
`packages/quantum-nematode/quantumnematode/brain/arch/_topology.py`:

```python
from quantumnematode.brain.arch._topology import BrainTopology

class MyTopology(nn.Module):
    n_inputs: int
    n_outputs: int
    n_hidden: int

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def apply_weight_mask(self) -> None:
        """Re-apply any structural constraints after an optimiser step."""
```

The Protocol is `runtime_checkable`, so
`isinstance(my_topo, BrainTopology)` works as a runtime conformance
check. The connectome-PPO brain at
`packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py`
is the canonical example — its `ConnectomeTopology` enforces a
strict-mask chemical-synapse adjacency and a fixed gap-junction matrix.

### Custom learning rule

If you want to share a learning rule (PPO, REINFORCE, DQN) across
multiple architectures, implement the `LearningRule` Protocol in
`packages/quantum-nematode/quantumnematode/brain/arch/_rule.py`. The
Protocol declares `step(topology, batch) -> RuleStepReport` and
`reset_episode()`.

This factoring is currently consumed by `ConnectomePPOBrain` only;
the legacy 19 brains keep their fused `(topology, rule)` `__init__`
bodies. Use the factored Protocols if you're writing a new brain that
genuinely separates the two concerns.

## What NOT to do

- **Don't reach into `BRAIN_CONFIG_MAP` or the dispatcher directly.**
  Both are now derived from the registry. Edit the registration, not
  the consumer.
- **Don't add `if isinstance(brain, MyBrain)` branches** anywhere
  outside the brain module itself. The Brain Protocol is the public
  surface; if you need new behaviour exposed, add it to the Protocol.
- **Don't skip the `BrainType` enum member.** The startup-time
  consistency check exists precisely so registration-without-enum is a
  hard failure, not a silent drift.
- **Don't bypass `instantiate_brain(...)` in entrypoints.** All
  entrypoints (`scripts/run_simulation.py`, `scripts/run_evolution.py`)
  go through `setup_brain_model(...)` which goes through the registry.
  If a new entrypoint needs to construct a brain, route it through the
  same factory.

## See also

- `packages/quantum-nematode/quantumnematode/brain/arch/_registry.py` —
  the registry implementation.
- `packages/quantum-nematode/quantumnematode/brain/arch/_topology.py` —
  the `BrainTopology` Protocol.
- `packages/quantum-nematode/quantumnematode/brain/arch/_rule.py` —
  the `LearningRule` Protocol + `RuleStepReport`.
- `packages/quantum-nematode/quantumnematode/brain/arch/mlpppo.py` —
  the simplest fully-featured registered Brain (PPO + actor-critic).
- `packages/quantum-nematode/quantumnematode/brain/arch/connectome_ppo.py` —
  the canonical example of a brain that uses the factored
  `BrainTopology` Protocol with a non-trivial topology.
