"""Equivariant Quantum PPO Brain.

A genuinely-quantum PPO actor whose policy is **exactly equivariant under the
left-right bilateral mirror (Z2)** — the natural symmetry of klinotaxis
navigation and of *C. elegans*' own bilateral body plan. An equivariant
classical pre-encoder compresses the sensory observation into parity-typed latent
features (even / odd under the mirror); a Z2-equivariant parameterised quantum
circuit (angle encoding + data re-uploading + an entangling, ``U_R``-invariant
ansatz) processes them from a ``U_R``-invariant reference state; and an
equivariant Pauli-observable readout produces the four action logits such that the
mirror swaps ``LEFT``/``RIGHT`` and fixes ``FORWARD``/``STAY``. Trained with PPO
via backprop through an in-repo torch statevector simulator (no parameter-shift,
no SNN/quantum-library dependency).

Two ablation siblings share the env interface, readout shape, and PPO loop, so the
comparison can attribute effects to the symmetry prior and to the quantum circuit
separately:

- ``equivariant=False`` — an unstructured PQC actor (no parity structure, free
  measurement->logit head): isolates the symmetry prior.
- ``quantum=False`` — an equivariant *classical* MLP actor (same parity-typed
  readout): isolates the quantum contribution.

This is built for comparison completeness and a principled inductive-bias study —
not as a quantum-advantage claim. At a few simulated qubits no quantum advantage is
expected on a classical-data control task; the reported quantities are the
equivariant-vs-not and quantum-vs-classical-equivariant deltas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import field_validator, model_validator
from torch import nn, optim

if TYPE_CHECKING:
    from quantumnematode.brain.weights import WeightComponent
    from quantumnematode.initializers._initializer import ParameterInitializer

from quantumnematode.brain.actions import DEFAULT_ACTIONS, Action, ActionData
from quantumnematode.brain.arch import BrainData, BrainParams, ClassicalBrain
from quantumnematode.brain.arch import _quantum_statevector as sv
from quantumnematode.brain.arch._brain import BrainHistoryData
from quantumnematode.brain.arch._ppo_buffer import RolloutBuffer
from quantumnematode.brain.arch._registry import register_brain
from quantumnematode.brain.arch.dtypes import BrainConfig, BrainType, DeviceType
from quantumnematode.brain.modules import (
    SENSORY_MODULES,
    ModuleName,
    extract_classical_features,
    get_classical_feature_dimension,
)
from quantumnematode.logging_config import logger
from quantumnematode.utils.seeding import ensure_seed, get_rng, set_global_seed

# Defaults
DEFAULT_NUM_QUBITS = 8
DEFAULT_K_ODD = 3
DEFAULT_NUM_LAYERS = 3
DEFAULT_CRITIC_HIDDEN_DIM = 64
DEFAULT_CRITIC_NUM_LAYERS = 2
DEFAULT_LR = 0.0003
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CLIP_EPSILON = 0.2
DEFAULT_VALUE_LOSS_COEF = 0.5
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_NUM_EPOCHS = 4
DEFAULT_NUM_MINIBATCHES = 4
DEFAULT_ROLLOUT_BUFFER_SIZE = 512
DEFAULT_MAX_GRAD_NORM = 0.5
MAX_QUBITS = 10
MIN_K_EVEN = 3  # readout needs 3 distinct even observables (FORWARD, STAY, LEFT/RIGHT-shared)
PARAM_INIT_STD = 0.1

# Modules whose feature index 1 ("angle") is a left-right LATERAL gradient
# (``tanh(right - left)``) and therefore Z2-ODD under the bilateral mirror. Every
# other feature (strengths, temporal derivatives, the predator-mechano fore-aft
# zone, proprioception, STAM) is Z2-EVEN. Validated empirically by the
# mirror-consistency test.
_LATERAL_GRADIENT_MODULES = frozenset(
    {
        ModuleName.FOOD_CHEMOTAXIS,
        ModuleName.FOOD_CHEMOTAXIS_KLINOTAXIS,
        ModuleName.FOOD_CHEMOTAXIS_TEMPORAL,
        ModuleName.THERMOTAXIS,
        ModuleName.THERMOTAXIS_KLINOTAXIS,
        ModuleName.THERMOTAXIS_TEMPORAL,
        ModuleName.NOCICEPTION,
        ModuleName.NOCICEPTION_KLINOTAXIS,
        ModuleName.NOCICEPTION_TEMPORAL,
        ModuleName.AEROTAXIS,
        ModuleName.AEROTAXIS_KLINOTAXIS,
        ModuleName.AEROTAXIS_TEMPORAL,
        ModuleName.PREDATOR_CHEMOSENSATION_ORACLE,
        ModuleName.PREDATOR_CHEMOSENSATION_KLINOTAXIS,
        ModuleName.PREDATOR_CHEMOSENSATION_TEMPORAL,
    },
)


def module_parity_pattern(module: ModuleName, dim: int) -> list[int]:
    """Z2 parity (+1 even, -1 odd) per classical feature of ``module``.

    Lateral-gradient modules have an odd "angle" at feature index 1; everything
    else is even. Returns a length-``dim`` list.
    """
    pattern = [1] * dim
    if module in _LATERAL_GRADIENT_MODULES and dim >= 2:  # noqa: PLR2004
        pattern[1] = -1
    return pattern


def parity_vector(modules: list[ModuleName]) -> np.ndarray:
    """Build the observation parity vector aligned with ``extract_classical_features``.

    ``extract_classical_features`` sorts modules by ``m.value`` and concatenates each
    module's ``to_classical`` output, so the parity vector is built in that same
    sorted order. Validated against env reflection by the mirror-consistency test.
    """
    sorted_modules = sorted(modules, key=lambda m: m.value)
    parities: list[int] = []
    for module in sorted_modules:
        sensory_module = SENSORY_MODULES.get(module)
        dim = sensory_module.classical_dim if sensory_module is not None else 2
        parities.extend(module_parity_pattern(module, dim))
    return np.array(parities, dtype=np.float32)


class EquivariantQuantumPPOBrainConfig(BrainConfig):
    """Configuration for the EquivariantQuantumPPOBrain architecture."""

    # Circuit
    num_qubits: int = DEFAULT_NUM_QUBITS  # = k_even + k_odd
    k_odd: int = DEFAULT_K_ODD  # number of odd-parity qubits/latents
    num_layers: int = DEFAULT_NUM_LAYERS  # data-re-uploading layers

    # Ablation flags
    equivariant: bool = True
    quantum: bool = True

    # Critic
    critic_hidden_dim: int = DEFAULT_CRITIC_HIDDEN_DIM
    critic_num_layers: int = DEFAULT_CRITIC_NUM_LAYERS

    # PPO
    actor_lr: float = DEFAULT_LR
    critic_lr: float = DEFAULT_LR
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    clip_epsilon: float = DEFAULT_CLIP_EPSILON
    value_loss_coef: float = DEFAULT_VALUE_LOSS_COEF
    entropy_coef: float = DEFAULT_ENTROPY_COEF
    entropy_coef_end: float | None = None
    entropy_decay_episodes: int | None = None
    num_epochs: int = DEFAULT_NUM_EPOCHS
    num_minibatches: int = DEFAULT_NUM_MINIBATCHES
    rollout_buffer_size: int = DEFAULT_ROLLOUT_BUFFER_SIZE
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM

    # Sensory feature extraction (required)
    sensory_modules: list[ModuleName]

    @field_validator("sensory_modules")
    @classmethod
    def _validate_sensory_modules(cls, v: list[ModuleName]) -> list[ModuleName]:
        if not v:
            msg = "sensory_modules must be non-empty"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _validate_config(self) -> EquivariantQuantumPPOBrainConfig:
        if not (0 < self.k_odd < self.num_qubits):
            msg = (
                f"k_odd must satisfy 0 < k_odd < num_qubits, "
                f"got k_odd={self.k_odd}, num_qubits={self.num_qubits}"
            )
            raise ValueError(msg)
        if self.num_qubits > MAX_QUBITS:
            msg = f"num_qubits must be <= {MAX_QUBITS} (statevector budget), got {self.num_qubits}"
            raise ValueError(msg)
        if self.num_qubits - self.k_odd < MIN_K_EVEN:
            msg = (
                f"k_even = num_qubits - k_odd must be >= {MIN_K_EVEN} (the equivariant readout "
                f"needs 3 even observables), got k_even={self.num_qubits - self.k_odd}"
            )
            raise ValueError(msg)
        if self.num_layers < 1:
            msg = f"num_layers must be >= 1, got {self.num_layers}"
            raise ValueError(msg)
        if self.actor_lr <= 0 or self.critic_lr <= 0:
            msg = "actor_lr and critic_lr must be > 0"
            raise ValueError(msg)
        if not (0 < self.clip_epsilon < 1):
            msg = f"clip_epsilon must be in (0, 1), got {self.clip_epsilon}"
            raise ValueError(msg)
        # Paired entropy-schedule fields: set both or neither.
        end, eps = self.entropy_coef_end, self.entropy_decay_episodes
        if (end is None) != (eps is None):
            msg = (
                "entropy_coef_end and entropy_decay_episodes must be set together "
                "(both to schedule the entropy coefficient, or neither for a flat coefficient)"
            )
            raise ValueError(msg)
        return self


class _ParityLinear(nn.Module):
    """Z2-equivariant linear map: even-in -> even-out, odd-in -> odd-out; bias on even outputs only.

    Equivariance ``W·diag(p_in) = diag(p_out)·W`` forces ``W[i,j]=0`` unless
    ``p_in[j]==p_out[i]`` (block-diagonal in parity); the bias must be invariant, so
    it lives on even outputs only.
    """

    mask: torch.Tensor
    even_out: torch.Tensor

    def __init__(self, in_parity: np.ndarray, out_parity: np.ndarray) -> None:
        super().__init__()
        in_p = torch.tensor(in_parity, dtype=torch.float32)
        out_p = torch.tensor(out_parity, dtype=torch.float32)
        self.register_buffer("mask", (out_p[:, None] == in_p[None, :]).float())
        self.register_buffer("even_out", (out_p > 0).float())
        self.weight = nn.Parameter(torch.empty(len(out_parity), len(in_parity)))
        self.bias = nn.Parameter(torch.zeros(len(out_parity)))
        nn.init.normal_(self.weight, std=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ (self.weight * self.mask).t() + self.bias * self.even_out


class EquivariantQuantumActor(nn.Module):
    """Z2-equivariant quantum actor (+ unstructured-quantum and equivariant-classical ablations)."""

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        in_parity: np.ndarray,
        num_qubits: int,
        k_odd: int,
        num_layers: int,
        *,
        equivariant: bool,
        quantum: bool,
        critic_hidden_dim: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.k_even = num_qubits - k_odd
        self.k_odd = k_odd
        self.num_layers = num_layers
        self.equivariant = equivariant
        self.quantum = quantum
        self.device = device
        self.even_qubits = list(range(self.k_even))
        self.odd_qubits = list(range(self.k_even, num_qubits))
        # Readout qubits: 3 even (FORWARD, STAY, LEFT/RIGHT-shared) + 1 odd (LEFT/RIGHT split).
        self.q_forward, self.q_stay, self.q_lr = 0, 1, 2
        self.q_odd = self.k_even

        if equivariant and quantum:
            out_parity = np.array([1] * self.k_even + [-1] * k_odd, dtype=np.float32)
            self.pre = _ParityLinear(in_parity, out_parity)
            self.rx = nn.Parameter(torch.randn(num_layers, num_qubits) * PARAM_INIT_STD)
            self.rz = nn.Parameter(torch.randn(num_layers, self.k_even) * PARAM_INIT_STD)
            self.xx_pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            self.zz_pairs = [(i, i + 1) for i in range(self.k_even - 1)] + [
                (self.k_even + i, self.k_even + i + 1) for i in range(k_odd - 1)
            ]
            self.xx = nn.Parameter(torch.randn(num_layers, len(self.xx_pairs)) * PARAM_INIT_STD)
            self.zz = nn.Parameter(
                torch.randn(num_layers, max(1, len(self.zz_pairs))) * PARAM_INIT_STD,
            )
            self.even_scale = nn.Parameter(torch.ones(3))  # FORWARD, STAY, LEFT/RIGHT-shared
            self.even_bias = nn.Parameter(torch.zeros(3))
            self.odd_scale = nn.Parameter(torch.ones(1))  # no bias -> stays Z2-odd
        elif quantum:  # unstructured PQC ablation
            self.pre = nn.Linear(input_dim, num_qubits)
            self.u_rx = nn.Parameter(torch.randn(num_layers, num_qubits) * PARAM_INIT_STD)
            self.u_ry = nn.Parameter(torch.randn(num_layers, num_qubits) * PARAM_INIT_STD)
            self.u_rz = nn.Parameter(torch.randn(num_layers, num_qubits) * PARAM_INIT_STD)
            self.cz_pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            self.head = nn.Linear(num_qubits, 4)
        else:  # equivariant classical ablation
            out_parity = np.array([1] * self.k_even + [-1] * k_odd, dtype=np.float32)
            self.pre = _ParityLinear(in_parity, out_parity)
            self.even_mlp = nn.Sequential(
                nn.Linear(self.k_even + k_odd, critic_hidden_dim),
                nn.ReLU(),
                nn.Linear(critic_hidden_dim, 3),
            )
            self.odd_lin = nn.Linear(k_odd, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, latent)`` with logits ordered [FORWARD, LEFT, RIGHT, STAY]."""
        latent = self.pre(x)
        if self.equivariant and self.quantum:
            logits = self._equivariant_quantum_logits(latent)
        elif self.quantum:
            logits = self._unstructured_quantum_logits(latent)
        else:
            logits = self._equivariant_classical_logits(latent)
        return logits, latent

    def _equivariant_quantum_logits(self, latent: torch.Tensor) -> torch.Tensor:
        k = self.num_qubits
        state = sv.zero_state(latent.shape[0], k, self.device)
        # U_R-invariant reference state: H on odd qubits (|+>), even qubits |0>.
        h = sv.h_matrix(self.device)
        for q in self.odd_qubits:
            state = sv.apply_1q(state, h, q, k)
        for layer in range(self.num_layers):
            for j in range(k):  # data re-uploading: encode every layer
                state = sv.apply_1q(state, sv.ry(latent[:, j]), j, k)
            for q in range(k):  # RX on all qubits (commutes with U_R)
                state = sv.apply_1q(state, sv.rx(self.rx[layer, q]), q, k)
            for idx, q in enumerate(self.even_qubits):  # RZ on even qubits only
                state = sv.apply_1q(state, sv.rz(self.rz[layer, idx]), q, k)
            for idx, (a, b) in enumerate(self.xx_pairs):  # IsingXX (any pair)
                state = sv.apply_2q(state, sv.rxx(self.xx[layer, idx]), a, b, k)
            for idx, (a, b) in enumerate(self.zz_pairs):  # IsingZZ (same-parity pairs)
                state = sv.apply_2q(state, sv.rzz(self.zz[layer, idx]), a, b, k)
        a_forward = sv.expect_z(state, self.q_forward, k)  # even observable
        a_stay = sv.expect_z(state, self.q_stay, k)  # even observable
        a_lr = sv.expect_z(state, self.q_lr, k)  # even observable
        b_odd = sv.expect_z(state, self.q_odd, k)  # Z on odd qubit -> Z2-odd observable
        l_forward = self.even_scale[0] * a_forward + self.even_bias[0]
        l_stay = self.even_scale[1] * a_stay + self.even_bias[1]
        l_lr = self.even_scale[2] * a_lr + self.even_bias[2]
        odd = self.odd_scale[0] * b_odd
        return torch.stack([l_forward, l_lr + odd, l_lr - odd, l_stay], dim=-1)

    def _unstructured_quantum_logits(self, latent: torch.Tensor) -> torch.Tensor:
        k = self.num_qubits
        state = sv.zero_state(latent.shape[0], k, self.device)  # |0...0> (no symmetric init)
        for layer in range(self.num_layers):
            for j in range(k):
                state = sv.apply_1q(state, sv.ry(latent[:, j]), j, k)
            for q in range(k):
                state = sv.apply_1q(state, sv.rx(self.u_rx[layer, q]), q, k)
                state = sv.apply_1q(state, sv.ry(self.u_ry[layer, q]), q, k)
                state = sv.apply_1q(state, sv.rz(self.u_rz[layer, q]), q, k)
            cz = sv.cz_matrix(self.device)
            for a, b in self.cz_pairs:
                state = sv.apply_2q(state, cz, a, b, k)
        z = torch.stack([sv.expect_z(state, q, k) for q in range(k)], dim=-1)  # (B, k)
        return self.head(z)

    def _equivariant_classical_logits(self, latent: torch.Tensor) -> torch.Tensor:
        e = latent[:, : self.k_even]
        o = latent[:, self.k_even :]
        even_in = torch.cat([e, o * o], dim=-1)  # all Z2-even
        h = self.even_mlp(even_in)  # (B, 3): FORWARD, STAY, LEFT/RIGHT-shared
        odd = torch.tanh(self.odd_lin(o)).squeeze(-1)  # Z2-odd (tanh odd, linear no-bias odd)
        a_forward, a_stay, a_lr = h[:, 0], h[:, 1], h[:, 2]
        return torch.stack([a_forward, a_lr + odd, a_lr - odd, a_stay], dim=-1)


@register_brain(
    name="equivariantquantum",
    config_cls=EquivariantQuantumPPOBrainConfig,
    brain_type=BrainType.EQUIVARIANT_QUANTUM_PPO,
    families=("quantum",),
)
class EquivariantQuantumPPOBrain(ClassicalBrain):
    """PPO brain with a bilateral-Z2-equivariant quantum actor (+ ablation siblings)."""

    def __init__(
        self,
        config: EquivariantQuantumPPOBrainConfig,
        num_actions: int = 4,
        device: DeviceType = DeviceType.CPU,
        action_set: list[Action] = DEFAULT_ACTIONS,
        *,
        parameter_initializer: ParameterInitializer | None = None,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.seed = ensure_seed(config.seed)
        self.rng = get_rng(self.seed)
        set_global_seed(self.seed)

        self.config = config
        self.sensory_modules = config.sensory_modules
        self.input_dim = get_classical_feature_dimension(config.sensory_modules)
        self.device = torch.device(device.to_torch_device_str())
        self.num_actions = num_actions
        self._action_set = action_set

        # Observation parity vector (aligned with extract_classical_features ordering).
        self.parity = parity_vector(config.sensory_modules)
        num_odd_inputs = int((self.parity < 0).sum())
        if config.k_odd > num_odd_inputs:
            logger.warning(
                "k_odd=%d exceeds the %d odd input feature(s); the surplus odd latents are "
                "linearly dependent under the equivariant pre-encoder.",
                config.k_odd,
                num_odd_inputs,
            )
        logger.info(
            "EquivariantQuantumPPOBrain: input_dim=%d (odd=%d), num_qubits=%d (k_odd=%d), "
            "equivariant=%s, quantum=%s, seed=%d",
            self.input_dim,
            num_odd_inputs,
            config.num_qubits,
            config.k_odd,
            config.equivariant,
            config.quantum,
            self.seed,
        )

        self.actor = EquivariantQuantumActor(
            input_dim=self.input_dim,
            in_parity=self.parity,
            num_qubits=config.num_qubits,
            k_odd=config.k_odd,
            num_layers=config.num_layers,
            equivariant=config.equivariant,
            quantum=config.quantum,
            critic_hidden_dim=config.critic_hidden_dim,
            device=self.device,
        ).to(self.device)

        self.critic = self._build_critic(config).to(self.device)

        self.optimizer = optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": config.actor_lr},
                {"params": self.critic.parameters(), "lr": config.critic_lr},
            ],
        )

        # PPO hyperparameters
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_epsilon = config.clip_epsilon
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.entropy_coef_end = config.entropy_coef_end
        self.entropy_decay_episodes = config.entropy_decay_episodes
        self.num_epochs = config.num_epochs
        self.num_minibatches = config.num_minibatches
        self.max_grad_norm = config.max_grad_norm

        self.buffer = RolloutBuffer(config.rollout_buffer_size, self.device, rng=self.rng)

        self.history_data = BrainHistoryData()
        self.latest_data = BrainData()
        self.training = True
        self.current_probabilities = None
        self.last_value: torch.Tensor | None = None
        self._episode_count = 0
        self._current_episode_rewards: list[float] = []

    def _build_critic(self, config: EquivariantQuantumPPOBrainConfig) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Linear(config.num_qubits, config.critic_hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(config.critic_num_layers - 1):
            layers += [nn.Linear(config.critic_hidden_dim, config.critic_hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(config.critic_hidden_dim, 1))
        return nn.Sequential(*layers)

    def preprocess(self, params: BrainParams) -> np.ndarray:
        """Extract classical features for the active sensory modules."""
        return extract_classical_features(params, self.sensory_modules)

    def _get_current_entropy_coef(self) -> float:
        if self.entropy_coef_end is None or self.entropy_decay_episodes is None:
            return self.entropy_coef
        if self._episode_count >= self.entropy_decay_episodes:
            return self.entropy_coef_end
        frac = self._episode_count / self.entropy_decay_episodes
        return self.entropy_coef + (self.entropy_coef_end - self.entropy_coef) * frac

    def get_action_and_value(
        self,
        state: np.ndarray,
        action: int | None = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """Sample an action and estimate value for a single state."""
        xt = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, latent = self.actor(xt)
        value = self.critic(latent.detach()).reshape(1)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = int(dist.sample().item())
        log_prob = dist.log_prob(torch.tensor(action, device=self.device))
        entropy = dist.entropy()
        probs_np = probs.detach().cpu().numpy()
        self.current_probabilities = probs_np
        return action, log_prob, entropy, value, probs_np

    def run_brain(
        self,
        params: BrainParams,
        reward: float | None = None,  # noqa: ARG002
        input_data: list[float] | None = None,  # noqa: ARG002
        *,
        top_only: bool,  # noqa: ARG002
        top_randomize: bool,  # noqa: ARG002
    ) -> list[ActionData]:
        """Run the quantum actor and select an action."""
        x = self.preprocess(params)
        action_idx, log_prob, _entropy, value, probs_np = self.get_action_and_value(x)
        action_name = self.action_set[action_idx]
        self.last_value = value

        self._pending_state = x
        self._pending_action = action_idx
        self._pending_log_prob = log_prob
        self._pending_value = value

        self.latest_data.action = ActionData(
            state=action_name,
            action=action_name,
            probability=float(probs_np[action_idx]),
        )
        self.history_data.actions.append(self.latest_data.action)
        self.history_data.probabilities.append(float(probs_np[action_idx]))
        return [
            ActionData(
                state=action_name,
                action=action_name,
                probability=float(probs_np[action_idx]),
            ),
        ]

    def learn(
        self,
        params: BrainParams,  # noqa: ARG002
        reward: float,
        *,
        episode_done: bool = False,
    ) -> None:
        """Buffer the last transition and run a PPO update when the buffer fills."""
        self._current_episode_rewards.append(reward)
        if hasattr(self, "_pending_state"):
            self.buffer.add(
                state=self._pending_state,
                action=self._pending_action,
                log_prob=self._pending_log_prob,
                value=self._pending_value,
                reward=reward,
                done=episode_done,
            )
        if self.buffer.is_full() or (episode_done and len(self.buffer) >= self.num_minibatches):
            self._perform_ppo_update()
            self.buffer.reset()
        self.history_data.rewards.append(reward)

    def _perform_ppo_update(self) -> None:
        if len(self.buffer) == 0:
            return
        last_value = (
            self.last_value
            if self.last_value is not None
            else torch.tensor([0.0], device=self.device)
        )
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value,
            self.gamma,
            self.gae_lambda,
        )
        entropy_coef = self._get_current_entropy_coef()
        total_policy_loss = total_value_loss = 0.0
        num_updates = 0
        for _ in range(self.num_epochs):
            for batch in self.buffer.get_minibatches(self.num_minibatches, returns, advantages):
                logits, latent = self.actor(batch["states"])
                values = self.critic(latent.detach()).squeeze(-1)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch["actions"])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch["advantages"]
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, batch["returns"])
                loss = policy_loss + self.value_loss_coef * value_loss - entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                params = list(self.actor.parameters()) + list(self.critic.parameters())
                nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.optimizer.step()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_updates += 1
        if num_updates > 0:
            self.latest_data.loss = total_policy_loss / num_updates

    def update_memory(self, reward: float | None = None) -> None:
        """No-op (PPO is on-policy)."""

    def prepare_episode(self) -> None:
        """No persistent per-step state to reset (the actor is feedforward per step)."""

    def post_process_episode(self, *, episode_success: bool | None = None) -> None:  # noqa: ARG002
        """Advance the episode counter (drives the entropy anneal)."""
        self._episode_count += 1
        self._current_episode_rewards.clear()

    # ------------------------------------------------------------------
    # Weight persistence (WeightPersistence protocol)
    # ------------------------------------------------------------------

    def get_weight_components(
        self,
        *,
        components: set[str] | None = None,
    ) -> dict[str, WeightComponent]:
        """Return the actor / critic / optimizer / training-state weight components."""
        from quantumnematode.brain.weights import WeightComponent

        all_components: dict[str, WeightComponent] = {
            "actor": WeightComponent(name="actor", state=self.actor.state_dict()),
            "critic": WeightComponent(name="critic", state=self.critic.state_dict()),
            "optimizer": WeightComponent(name="optimizer", state=self.optimizer.state_dict()),
            "training_state": WeightComponent(
                name="training_state",
                state={"episode_count": self._episode_count},
            ),
        }
        if components is None:
            return all_components
        unknown = components - set(all_components)
        if unknown:
            msg = f"Unknown weight components: {unknown}. Valid: {set(all_components)}"
            raise ValueError(msg)
        return {k: v for k, v in all_components.items() if k in components}

    def load_weight_components(self, components: dict[str, WeightComponent]) -> None:
        """Load actor / critic / optimizer / training-state weights."""
        if "actor" in components:
            self.actor.load_state_dict(components["actor"].state)
        if "critic" in components:
            self.critic.load_state_dict(components["critic"].state)
        if "optimizer" in components:
            self.optimizer.load_state_dict(components["optimizer"].state)
        if "training_state" in components:
            ts = components["training_state"].state
            if "episode_count" in ts:
                self._episode_count = int(ts["episode_count"])
        self.buffer.reset()

    def copy(self) -> EquivariantQuantumPPOBrain:
        """Not supported."""
        msg = "EquivariantQuantumPPOBrain does not support copying. Use deepcopy if needed."
        raise NotImplementedError(msg)

    @property
    def action_set(self) -> list[Action]:
        """The ordered action set (logit order [FORWARD, LEFT, RIGHT, STAY])."""
        return self._action_set

    @action_set.setter
    def action_set(self, actions: list[Action]) -> None:
        self._action_set = actions

    def build_brain(self) -> None:
        """Not applicable (the circuit is built at construction)."""
        msg = "EquivariantQuantumPPOBrain builds its circuit at construction."
        raise NotImplementedError(msg)

    def update_parameters(
        self,
        gradients: list[float],
        reward: float | None = None,
        learning_rate: float = 0.01,
        **kwargs: object,
    ) -> None:
        """Not used (PPO uses its own optimizer)."""
