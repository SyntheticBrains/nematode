"""Frozen controllers for the batched evaluator: brains and hazard rules, vectorised over animals.

Each controller maps one step's :class:`~quantumnematode.evaluation.batched.BatchedSenses` to the
primitives' five raw numbers in ``[-1, 1]`` for every animal: run speed, run curvature and a start
hazard for each of reversal, omega and delta.

- A brain adapter builds the brain's own features from the senses, exactly as its feature extractor
  does for one animal, and runs its batched forward pass.
- Its policy noise comes from one torch generator per animal, five standard normals a step. Torch's
  normal sample under a seeded global generator is that draw scaled and shifted, so an animal draws
  what the per-step runner draws.
- The hazard rules are the non-learning controllers' arithmetic on arrays.
- The open-loop negatives (a two-state worm switching by a fixed chain, and curvature walks) draw
  their randomness from the same per-animal generators, so an animal acts the same alone or in any
  batch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch

from quantumnematode.brain.arch._policy import clamp_continuous_log_std
from quantumnematode.brain.modules import ModuleName
from quantumnematode.env.primitives import load_kinematics

if TYPE_CHECKING:
    from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain
    from quantumnematode.brain.arch.mlpppo import MLPPPOBrain
    from quantumnematode.evaluation.batched import BatchedSenses

# Codes of the locomotor-state one-hot, in the module's order (run, reversal, omega, delta).
_LOCOMOTOR_WIDTH = 4
# The proprioception module's facing angle: the continuous worm's grid direction never changes.
_PROPRIOCEPTION_ANGLE = 0.5
# Modules the MLP adapter builds, each with its feature builder below.
_MLP_MODULES = frozenset(
    {
        ModuleName.FOOD_CHEMOTAXIS_KLINOTAXIS,
        ModuleName.FOOD_CONTACT,
        ModuleName.INTERNAL_STATE,
        ModuleName.LOCOMOTOR_STATE,
        ModuleName.PROPRIOCEPTION,
        ModuleName.STAM,
    },
)


class BatchedController(Protocol):
    """A frozen controller over N animals."""

    def reset(self, n: int) -> None:
        """Start an episode for ``n`` animals."""
        ...

    def act(
        self,
        senses: BatchedSenses,
        noise: list[torch.Generator] | None,
    ) -> np.ndarray:
        """Return the step's raw actions ``(N, 5)``; ``noise`` None acts on the mean."""
        ...


def _refuse(brain: object, reason: str) -> None:
    msg = f"the batched evaluator cannot run {type(brain).__name__}: {reason}"
    raise ValueError(msg)


def _check_policy(brain: object, config: object) -> None:
    if getattr(config, "action_mode", None) != "continuous":
        _refuse(brain, "its action mode is not continuous")
    if getattr(config, "action_space", None) != "primitives":
        _refuse(brain, "its action space is not primitives")
    if getattr(config, "continuous_std_mode", None) != "state_independent":
        _refuse(brain, "its action noise depends on the state")
    if getattr(config, "learning_rule", "ppo") != "ppo" or getattr(
        config,
        "enable_activity_traces",
        False,
    ):
        _refuse(brain, "it has within-lifetime plasticity or activity traces")
    if getattr(config, "plasticity_node_noise", 0.0):
        _refuse(brain, "it injects node noise")


def _sample(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    noise: list[torch.Generator] | None,
) -> np.ndarray:
    """Return ``center + half_range * tanh(pre)``, ``pre`` the mean plus each animal's own noise."""
    if noise is None:
        pre = mean
    else:
        std = torch.exp(clamp_continuous_log_std(log_std))
        eps = torch.stack([torch.randn(mean.shape[1], generator=g) for g in noise])
        pre = mean + std * eps
    center = (high + low) / 2.0
    half = (high - low) / 2.0
    return (center + half * torch.tanh(pre)).double().numpy()


def _food_features(senses: BatchedSenses) -> np.ndarray:
    return np.stack(
        [
            senses.concentration,
            np.tanh(senses.lateral * senses.lateral_scale),
            np.tanh(senses.derivative * senses.derivative_scale),
        ],
        axis=1,
    )


class BatchedMLP:
    """A frozen MLP-PPO brain on the primitives, its features built as its extractor builds them."""

    def __init__(self, brain: MLPPPOBrain) -> None:
        config = brain.config
        _check_policy(brain, config)
        if getattr(config, "feature_expansion", "none") != "none" or getattr(
            config,
            "feature_gating",
            False,
        ):
            _refuse(brain, "it expands or gates its features")
        modules = sorted(brain.sensory_modules, key=lambda m: m.value)
        unknown = [m.value for m in modules if m not in _MLP_MODULES]
        if unknown:
            _refuse(brain, f"it reads modules the evaluator does not build: {unknown}")
        self.brain = brain
        self.modules = modules

    def reset(self, n: int) -> None:
        """No state carries across steps."""
        del n

    def features(self, senses: BatchedSenses) -> np.ndarray:
        """Return the brain's feature vectors ``(N, D)``, in its sorted module order."""
        n = len(senses.concentration)
        parts: list[np.ndarray] = []
        for module in self.modules:
            if module is ModuleName.FOOD_CHEMOTAXIS_KLINOTAXIS:
                parts.append(_food_features(senses))
            elif module is ModuleName.FOOD_CONTACT:
                parts.append(np.clip(senses.food_contact, 0.0, 1.0)[:, None])
            elif module is ModuleName.INTERNAL_STATE:
                parts.append(senses.satiety_fraction[:, None])
            elif module is ModuleName.LOCOMOTOR_STATE:
                onehot = np.zeros((n, _LOCOMOTOR_WIDTH))
                onehot[np.arange(n), senses.locomotor] = 1.0
                parts.append(onehot)
            elif module is ModuleName.PROPRIOCEPTION:
                parts.append(
                    np.column_stack([np.full(n, senses.moved), np.full(n, _PROPRIOCEPTION_ANGLE)]),
                )
            else:
                parts.append(senses.stam.astype(np.float64))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def act(self, senses: BatchedSenses, noise: list[torch.Generator] | None) -> np.ndarray:
        """Run the actor on every animal's features and draw each animal's noise."""
        brain = self.brain
        with torch.no_grad():
            mean = brain.actor(torch.from_numpy(self.features(senses)))
            return _sample(mean, brain.log_std, brain._action_low, brain._action_high, noise)  # noqa: SLF001


class BatchedConnectome:
    """A frozen settling connectome on the primitives, its modulators carried per animal."""

    def __init__(self, brain: ConnectomePPOBrain) -> None:
        config = brain.config
        _check_policy(brain, config)
        if config.dynamics != "settling":
            _refuse(brain, f"its dynamics are {config.dynamics}, not settling")
        if config.sensing_mode != "klinotaxis":
            _refuse(brain, f"its sensing mode is {config.sensing_mode}, not klinotaxis")
        if config.enable_predator_projection or config.enable_thermotaxis_projection:
            _refuse(brain, "it projects predator or thermal senses")
        self.brain = brain
        self.width = brain._n_concentrations  # noqa: SLF001
        self.concentrations = np.zeros((0, self.width), dtype=np.float32)

    def reset(self, n: int) -> None:
        """Start every animal's modulator concentrations at zero."""
        self.concentrations = np.zeros((n, self.width), dtype=np.float32)

    def state(self, senses: BatchedSenses) -> np.ndarray:
        """Return the brain's state vectors: food features, then the modulators if it has them."""
        food = _food_features(senses)
        if not self.brain.config.modulators:
            return food.astype(np.float32)
        drive = senses.intake.copy()
        if senses.max_intake:
            drive = drive / senses.max_intake
        if self.brain.config.modulator_satiety_link:
            drive = drive * (2.0 - np.clip(senses.satiety_fraction, 0.0, 1.0))
        return np.concatenate(
            [food.astype(np.float32), self.concentrations, drive.astype(np.float32)[:, None]],
            axis=1,
        ).astype(np.float32)

    def act(self, senses: BatchedSenses, noise: list[torch.Generator] | None) -> np.ndarray:
        """Settle every animal's network, update its modulators, and draw its noise."""
        brain = self.brain
        topology = brain.topology
        states = torch.from_numpy(self.state(senses))
        with torch.no_grad():
            food = states[:, : topology.n_food_features]
            mean, hidden = topology.forward_with_hidden_batched(
                food,
                modulators=brain._unpack_modulators(states),  # noqa: SLF001
            )
            if brain.config.modulators:
                release = topology.modulator_release_batched(hidden).numpy().astype(np.float32)
                for k in brain._knocked_ligands:  # noqa: SLF001
                    release[:, k] = 0.0
                decay = brain._modulator_decay  # noqa: SLF001
                self.concentrations = (
                    decay * self.concentrations + (1.0 - decay) * release
                ).astype(np.float32)
            return _sample(mean, topology.log_std, brain._action_low, brain._action_high, noise)  # noqa: SLF001


def _raw_hazards(rates: np.ndarray) -> np.ndarray:
    k = load_kinematics()
    log_low = math.log(k.hazard_min_per_s)
    log_span = math.log(k.hazard_max_per_s) - log_low
    clipped = np.clip(rates, k.hazard_min_per_s, k.hazard_max_per_s)
    return 2.0 * (np.log(clipped) - log_low) / log_span - 1.0


def _raw_speed(speed: float) -> float:
    k = load_kinematics()
    return min(1.0, max(-1.0, 2.0 * speed / k.v_max_mm_per_s - 1.0))


class BatchedConstantHazards:
    """Fixed speed, straight runs and fixed start hazards, for every animal."""

    def __init__(self, speed_mm_per_s: float, hazards_per_s: tuple[float, float, float]) -> None:
        self.speed = speed_mm_per_s
        self.hazards = np.asarray(hazards_per_s, dtype=np.float64)

    def reset(self, n: int) -> None:
        """No state."""
        del n

    def act(self, senses: BatchedSenses, noise: list[torch.Generator] | None) -> np.ndarray:
        """Return the same raw action for every animal."""
        del noise
        n = len(senses.concentration)
        raw = np.zeros((n, 5))
        raw[:, 0] = _raw_speed(self.speed)
        raw[:, 2:] = _raw_hazards(np.broadcast_to(self.hazards, (n, 3)))
        return raw


class BatchedPirouetteHazards:
    """Start hazards ``rate * exp(-beta * tanh(dC/dt * scale))`` and straight runs, per animal."""

    def __init__(
        self,
        speed_mm_per_s: float,
        hazards_per_s: tuple[float, float, float],
        beta: float,
    ) -> None:
        self.speed = speed_mm_per_s
        self.hazards = np.asarray(hazards_per_s, dtype=np.float64)
        self.beta = beta

    def reset(self, n: int) -> None:
        """No state."""
        del n

    def act(self, senses: BatchedSenses, noise: list[torch.Generator] | None) -> np.ndarray:
        """Scale every animal's hazards by its own sensed rate of change."""
        del noise
        n = len(senses.concentration)
        scale = np.exp(-self.beta * np.tanh(senses.derivative * senses.derivative_scale))
        raw = np.zeros((n, 5))
        raw[:, 0] = _raw_speed(self.speed)
        raw[:, 2:] = _raw_hazards(self.hazards[None, :] * scale[:, None])
        return raw


def _raw_curvature(deg_per_s: np.ndarray) -> np.ndarray:
    """Map curvature in degrees per second onto the primitives' raw range, clipped."""
    omega_max = math.degrees(load_kinematics().omega_max_rad_per_s)
    return np.clip(np.asarray(deg_per_s, dtype=np.float64) / omega_max, -1.0, 1.0)


def _uniforms(noise: list[torch.Generator], k: int = 1) -> np.ndarray:
    """Draw ``k`` uniforms per animal, each from that animal's own generator: ``(n, k)``."""
    return np.array([torch.rand(k, generator=gen, dtype=torch.float64).numpy() for gen in noise])


def _normals(noise: list[torch.Generator]) -> np.ndarray:
    """Draw one standard normal per animal, each from that animal's own generator."""
    return np.array(
        [float(torch.randn(1, generator=gen, dtype=torch.float64)[0]) for gen in noise],
    )


def _require_noise(noise: list[torch.Generator] | None) -> list[torch.Generator]:
    if noise is None:
        msg = "this controller draws its randomness from each animal's generator; noise is None"
        raise ValueError(msg)
    return noise


class BatchedStateHazards:
    """An open-loop two-state worm: states switch by a fixed chain, each with its own kinematics.

    Every ``window_steps`` steps each animal's state moves by ``transitions[state]``, whatever it
    senses; the first state is drawn from the chain's stationary distribution. In each state the
    animal runs at that state's speed, with curvature drawn each second from a normal of that
    state's SD, and starts bouts at that state's hazards. Every draw comes from the animal's own
    generator.
    """

    def __init__(
        self,
        transitions: np.ndarray,
        speeds_mm_per_s: tuple[float, float],
        curvature_sd_deg_per_s: tuple[float, float],
        hazards_per_s: np.ndarray,
        window_steps: int,
    ) -> None:
        self.transitions = np.asarray(transitions, dtype=np.float64)
        self.speeds = np.asarray(speeds_mm_per_s, dtype=np.float64)
        self.curvature_sd = np.asarray(curvature_sd_deg_per_s, dtype=np.float64)
        self.hazards = np.asarray(hazards_per_s, dtype=np.float64)
        self.window_steps = window_steps
        up, down = self.transitions[0, 1], self.transitions[1, 0]
        self.stationary_second = up / (up + down)
        self.state: np.ndarray | None = None
        self.steps = 0

    def reset(self, n: int) -> None:
        """Forget every animal's state; it is drawn on the first step."""
        del n
        self.state = None
        self.steps = 0

    def act(self, senses: BatchedSenses, noise: list[torch.Generator] | None) -> np.ndarray:
        """Step each animal's chain on its window's boundary, then act in its state."""
        gens = _require_noise(noise)
        n = len(senses.concentration)
        if self.state is None:
            self.state = (_uniforms(gens)[:, 0] < self.stationary_second).astype(np.int64)
        elif self.steps % self.window_steps == 0:
            stay = self.transitions[self.state, self.state]
            self.state = np.where(_uniforms(gens)[:, 0] < stay, self.state, 1 - self.state)
        self.steps += 1
        curvature = _normals(gens) * self.curvature_sd[self.state]
        raw = np.zeros((n, 5))
        raw[:, 0] = [_raw_speed(float(s)) for s in self.speeds[self.state]]
        raw[:, 1] = _raw_curvature(curvature)
        raw[:, 2:] = _raw_hazards(self.hazards[self.state])
        return raw


class BatchedCurvatureWalk:
    """Runs at a fixed speed and fixed hazards, turning by random curvature each second.

    ``mode="uniform"`` draws each second's curvature uniformly over the primitives' whole range: the
    random walk. ``mode="ar1"`` draws it as ``c_t = rho c_{t-1} + sqrt(1 - rho^2) sd e_t``: the
    correlated walk, and with ``rho = 0`` independent turns of SD ``sd``. Every draw comes from the
    animal's own generator.
    """

    def __init__(
        self,
        speed_mm_per_s: float,
        hazards_per_s: tuple[float, float, float],
        mode: str,
        curvature_sd_deg_per_s: float = 0.0,
        persistence: float = 0.0,
    ) -> None:
        if mode not in ("uniform", "ar1"):
            msg = f"mode must be 'uniform' or 'ar1', got {mode!r}"
            raise ValueError(msg)
        self.speed = speed_mm_per_s
        self.hazards = np.asarray(hazards_per_s, dtype=np.float64)
        self.mode = mode
        self.sd = curvature_sd_deg_per_s
        self.rho = persistence
        self.curvature: np.ndarray | None = None

    def reset(self, n: int) -> None:
        """Forget every animal's curvature."""
        del n
        self.curvature = None

    def act(self, senses: BatchedSenses, noise: list[torch.Generator] | None) -> np.ndarray:
        """Return each animal's run at the fixed speed with this second's curvature."""
        gens = _require_noise(noise)
        n = len(senses.concentration)
        raw = np.zeros((n, 5))
        raw[:, 0] = _raw_speed(self.speed)
        raw[:, 2:] = _raw_hazards(np.broadcast_to(self.hazards, (n, 3)))
        if self.mode == "uniform":
            raw[:, 1] = 2.0 * _uniforms(gens)[:, 0] - 1.0
            return raw
        shock = _normals(gens) * self.sd
        if self.curvature is None:
            self.curvature = shock
        else:
            self.curvature = self.rho * self.curvature + math.sqrt(1.0 - self.rho**2) * shock
        raw[:, 1] = _raw_curvature(self.curvature)
        return raw
