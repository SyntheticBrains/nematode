"""Unit tests for the bilateral-Z2-equivariant quantum PPO brain.

Covers: the in-repo torch statevector simulator vs Qiskit + autograd; the
observation parity vector / mirror-consistency at the feature-extraction level;
exact end-to-end policy equivariance (LEFT<->RIGHT swap, FORWARD/STAY fixed) for
the equivariant-quantum brain and the equivariant-classical ablation, and its
ABSENCE for the unstructured ablation; load-bearing entanglement; construction /
finite logits + value / DEFAULT_ACTIONS order; non-degenerate logit variance; a
PPO update that runs and leaves parameters finite; WeightPersistence round-trip;
and the config validators.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
import torch
from quantumnematode.brain.actions import Action
from quantumnematode.brain.arch import BrainParams
from quantumnematode.brain.arch import _quantum_statevector as sv
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.brain.arch.equivariant_quantum import (
    EquivariantQuantumActor,
    EquivariantQuantumPPOBrain,
    EquivariantQuantumPPOBrainConfig,
    module_parity_pattern,
    parity_vector,
)
from quantumnematode.brain.modules import ModuleName, extract_classical_features
from quantumnematode.env import Direction

MODS = [ModuleName.FOOD_CHEMOTAXIS, ModuleName.PROPRIOCEPTION]


def make_brain(**overrides: Any) -> EquivariantQuantumPPOBrain:
    """Construct a small equivariant-quantum brain for tests (input_dim = 4)."""
    config = EquivariantQuantumPPOBrainConfig(
        sensory_modules=MODS,
        num_qubits=4,
        k_odd=1,
        num_layers=2,
        seed=0,
        **overrides,
    )
    return EquivariantQuantumPPOBrain(config, device=DeviceType.CPU)


def _max_swap_error(brain: EquivariantQuantumPPOBrain, n: int = 100) -> float:
    """Max deviation from mirror-equivariance over ``n`` random observations.

    Equivariance requires P(LEFT|s) == P(RIGHT|R*s) (and vice versa) with
    P(FORWARD)/P(STAY) unchanged, where ``R = diag(parity)``.
    """
    parity = torch.tensor(brain.parity, dtype=torch.float32)
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(n):
        x = torch.tensor(rng.standard_normal(len(brain.parity)).astype(np.float32))
        lx, _ = brain.actor(x.unsqueeze(0))
        lrx, _ = brain.actor((x * parity).unsqueeze(0))
        px = torch.softmax(lx, dim=-1).squeeze(0)
        prx = torch.softmax(lrx, dim=-1).squeeze(0)
        worst = max(
            worst,
            abs(px[1] - prx[2]).item(),  # LEFT(s) vs RIGHT(R*s)
            abs(px[2] - prx[1]).item(),  # RIGHT(s) vs LEFT(R*s)
            abs(px[0] - prx[0]).item(),  # FORWARD fixed
            abs(px[3] - prx[3]).item(),  # STAY fixed
        )
    return worst


# ---------------------------------------------------------------------------
# Statevector simulator
# ---------------------------------------------------------------------------


def test_simulator_matches_qiskit() -> None:  # noqa: C901, PLR0912 — gate-dispatch
    """The in-repo torch simulator matches Qiskit's Statevector expectations."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli, Statevector

    n = 4
    dev = torch.device("cpu")
    gates = [
        ("h", 1, None),
        ("ry", 0, 0.7),
        ("rx", 2, -1.1),
        ("rz", 3, 0.4),
        ("rxx", (0, 2), 0.9),
        ("rzz", (1, 3), -0.6),
        ("cz", (0, 1), None),
        ("rxx", (2, 3), 0.5),
    ]
    state = sv.zero_state(1, n, dev)
    for name, q, th in gates:
        if name == "h":
            state = sv.apply_1q(state, sv.h_matrix(dev), q, n)
        elif name == "ry":
            state = sv.apply_1q(state, sv.ry(torch.tensor(th)), q, n)
        elif name == "rx":
            state = sv.apply_1q(state, sv.rx(torch.tensor(th)), q, n)
        elif name == "rz":
            state = sv.apply_1q(state, sv.rz(torch.tensor(th)), q, n)
        elif name == "rxx":
            state = sv.apply_2q(state, sv.rxx(torch.tensor(th)), q[0], q[1], n)
        elif name == "rzz":
            state = sv.apply_2q(state, sv.rzz(torch.tensor(th)), q[0], q[1], n)
        elif name == "cz":
            state = sv.apply_2q(state, sv.cz_matrix(dev), q[0], q[1], n)

    qc = QuantumCircuit(n)
    for name, q, th in gates:
        if name == "h":
            qc.h(q)
        elif name == "ry":
            qc.ry(th, q)
        elif name == "rx":
            qc.rx(th, q)
        elif name == "rz":
            qc.rz(th, q)
        elif name == "rxx":
            qc.rxx(th, q[0], q[1])
        elif name == "rzz":
            qc.rzz(th, q[0], q[1])
        elif name == "cz":
            qc.cz(q[0], q[1])
    qsv = Statevector(qc)

    def label(pauli: str, qubit: int) -> str:
        chars = ["I"] * n
        chars[qubit] = pauli
        return "".join(reversed(chars))

    for q in range(n):
        mine_z = sv.expect_z(state, q, n).item()
        mine_x = sv.expect_x(state, q, n).item()
        ref_z = qsv.expectation_value(Pauli(label("Z", q))).real
        ref_x = qsv.expectation_value(Pauli(label("X", q))).real
        assert abs(mine_z - ref_z) < 1e-5
        assert abs(mine_x - ref_x) < 1e-5


def test_simulator_autograd() -> None:
    """Gradients flow through the simulator to circuit parameters."""
    n = 3
    dev = torch.device("cpu")
    theta = torch.tensor(0.5, requires_grad=True)
    state = sv.zero_state(1, n, dev)
    state = sv.apply_1q(state, sv.h_matrix(dev), 1, n)
    state = sv.apply_1q(state, sv.ry(theta), 0, n)
    state = sv.apply_2q(state, sv.rxx(theta), 0, 1, n)
    loss = sv.expect_z(state, 0, n).sum()
    loss.backward()
    assert theta.grad is not None
    assert bool(torch.isfinite(theta.grad).all())


# ---------------------------------------------------------------------------
# Parity vector + mirror-consistency at the feature-extraction level
# ---------------------------------------------------------------------------


def test_parity_vector_food_proprioception() -> None:
    """food_chemotaxis [strength(+), angle(-)] + proprioception [sin(+), cos(+)]."""
    assert parity_vector(MODS).tolist() == [1.0, -1.0, 1.0, 1.0]


def test_lateral_gradient_angle_is_odd() -> None:
    """Lateral-gradient modules have an odd angle at feature index 1."""
    for module in (
        ModuleName.FOOD_CHEMOTAXIS,
        ModuleName.THERMOTAXIS,
        ModuleName.PREDATOR_CHEMOSENSATION_KLINOTAXIS,
    ):
        assert module_parity_pattern(module, 3) == [1, -1, 1]


def test_predator_mechanosensation_zone_is_even() -> None:
    """The predator-mechano zone is a fore-aft axis (anterior/posterior) -> Z2-even."""
    assert module_parity_pattern(ModuleName.PREDATOR_MECHANOSENSATION_KLINOTAXIS, 3) == [1, 1, 1]


def test_temporal_derivative_is_even() -> None:
    """``*_temporal`` modules have a temporal-derivative angle (Z2-even), not a lateral gradient.

    Index 1 of a temporal module is ``tanh(dC/dt)`` — a spatial left-right mirror leaves a scalar
    concentration's time-derivative unchanged, so it must be even, NOT sign-flipping.
    """
    for module in (
        ModuleName.FOOD_CHEMOTAXIS_TEMPORAL,
        ModuleName.THERMOTAXIS_TEMPORAL,
        ModuleName.NOCICEPTION_TEMPORAL,
        ModuleName.PREDATOR_CHEMOSENSATION_TEMPORAL,
    ):
        assert module_parity_pattern(module, 2) == [1, 1]
        assert bool((parity_vector([module]) > 0).all())


def test_mirror_consistency_all_headings() -> None:
    """A left-right mirror flips the food angle with the same parity for every heading.

    It preserves strength, and the parity operator is heading-independent because klinotaxis
    observations are egocentric. For each heading we place the food 90 deg off the
    fore-aft axis (so the angle is off the mirror axis), reflect the bearing across the
    heading axis (``bearing -> 2*agent_angle - bearing``), recompute, and require the
    recomputed observation to equal ``R * obs`` for the single derived parity vector.
    """
    agent_angles = {
        Direction.UP: np.pi / 2,
        Direction.DOWN: -np.pi / 2,
        Direction.LEFT: np.pi,
        Direction.RIGHT: 0.0,
    }
    parity = parity_vector([ModuleName.FOOD_CHEMOTAXIS])  # [+1, -1]
    for heading, agent_angle in agent_angles.items():
        base = BrainParams(
            agent_direction=heading,
            food_gradient_strength=0.7,
            food_gradient_direction=float(agent_angle + np.pi / 2),  # 90 deg off fore-aft axis
        )
        mirrored = BrainParams(
            agent_direction=heading,
            food_gradient_strength=0.7,
            food_gradient_direction=float(agent_angle - np.pi / 2),  # reflected across heading axis
        )
        obs = extract_classical_features(base, [ModuleName.FOOD_CHEMOTAXIS])
        obs_mirror = extract_classical_features(mirrored, [ModuleName.FOOD_CHEMOTAXIS])
        assert np.allclose(obs_mirror, parity * obs, atol=1e-5), f"heading {heading}"


def test_parity_vector_matches_input_dim() -> None:
    """The parity vector length matches the brain's input_dim (guards dim drift)."""
    brain = make_brain()
    assert len(brain.parity) == brain.input_dim


# ---------------------------------------------------------------------------
# End-to-end policy equivariance + ablations
# ---------------------------------------------------------------------------


def test_policy_equivariance_exact() -> None:
    """The equivariant-quantum policy is exactly mirror-equivariant."""
    assert _max_swap_error(make_brain()) < 1e-4


def test_classical_equivariant_ablation_is_equivariant() -> None:
    """The quantum=False (equivariant-classical) ablation is also exactly equivariant."""
    assert _max_swap_error(make_brain(quantum=False)) < 1e-4


def test_unstructured_ablation_is_not_equivariant() -> None:
    """The equivariant=False (unstructured PQC) ablation is genuinely NOT equivariant."""
    assert _max_swap_error(make_brain(equivariant=False)) > 1e-2


# ---------------------------------------------------------------------------
# Construction / forward / entanglement
# ---------------------------------------------------------------------------


def test_construction_finite_logits_and_value() -> None:
    """Forward produces finite 4-logit / value and a valid action over DEFAULT_ACTIONS."""
    brain = make_brain()
    x = np.random.default_rng(0).standard_normal(brain.input_dim).astype(np.float32)
    action, _log_prob, _entropy, value, probs = brain.get_action_and_value(x)
    assert action in range(4)
    assert np.isfinite(probs).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert bool(torch.isfinite(value).all())


def test_logit_order_matches_default_actions() -> None:
    """The action set / logit order is [FORWARD, LEFT, RIGHT, STAY]."""
    brain = make_brain()
    assert brain.action_set[:4] == [Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY]


def test_logit_variance_nondegenerate() -> None:
    """Logits vary across inputs and do not collapse to a single action."""
    brain = make_brain()
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(100):
        x = torch.tensor(rng.standard_normal(brain.input_dim).astype(np.float32)).unsqueeze(0)
        logits, _ = brain.actor(x)
        rows.append(logits.squeeze(0).detach().numpy())
    arr = np.array(rows)
    assert float(arr.var()) > 0.0
    assert len(set(arr.argmax(axis=1).tolist())) > 1


def test_entanglement_is_load_bearing() -> None:
    """Removing the IsingXX/IsingZZ couplings changes the action-logit distribution."""
    brain = make_brain()
    actor: EquivariantQuantumActor = brain.actor
    separable_actor = copy.deepcopy(actor)  # mutate a copy, not the shared actor
    with torch.no_grad():
        separable_actor.xx.zero_()
        separable_actor.zz.zero_()
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((8, brain.input_dim)).astype(np.float32))
    with torch.no_grad():
        full, _ = actor(x)
        separable, _ = separable_actor(x)
    assert (full - separable).abs().max().item() > 1e-3


# ---------------------------------------------------------------------------
# PPO update + persistence
# ---------------------------------------------------------------------------


def test_ppo_update_runs_and_keeps_params_finite() -> None:
    """A PPO update triggers from buffered transitions and leaves parameters finite."""
    brain = make_brain(rollout_buffer_size=64, num_minibatches=4)
    params = BrainParams(agent_direction=Direction.UP, food_gradient_strength=0.5)
    rng = np.random.default_rng(0)
    for i in range(200):
        brain.run_brain(params, reward=None, input_data=None, top_only=False, top_randomize=False)
        brain.learn(params, reward=float(rng.standard_normal()), episode_done=(i % 50 == 49))
    for p in list(brain.actor.parameters()) + list(brain.critic.parameters()):
        assert bool(torch.isfinite(p).all())


def test_weight_persistence_roundtrip() -> None:
    """Loading a brain's weight components into another yields identical logits."""
    source = make_brain()
    target = make_brain()
    target.load_weight_components(source.get_weight_components())
    x = torch.tensor(
        np.random.default_rng(2).standard_normal(source.input_dim).astype(np.float32),
    ).unsqueeze(0)
    logits_source, _ = source.actor(x)
    logits_target, _ = target.actor(x)
    assert torch.allclose(logits_source, logits_target, atol=1e-6)


# ---------------------------------------------------------------------------
# Config validators
# ---------------------------------------------------------------------------


def test_entropy_schedule_paired_validator() -> None:
    """Setting exactly one of the entropy-schedule fields is rejected."""
    with pytest.raises(ValueError, match="entropy_coef_end and entropy_decay_episodes"):
        EquivariantQuantumPPOBrainConfig(sensory_modules=MODS, entropy_coef_end=0.001)


def test_k_even_floor_validator() -> None:
    """k_even = num_qubits - k_odd must be >= 3 for the equivariant readout."""
    with pytest.raises(ValueError, match="k_even"):
        EquivariantQuantumPPOBrainConfig(sensory_modules=MODS, num_qubits=4, k_odd=2)


def test_qubit_budget_validator() -> None:
    """num_qubits is capped at the statevector budget."""
    with pytest.raises(ValueError, match="num_qubits"):
        EquivariantQuantumPPOBrainConfig(sensory_modules=MODS, num_qubits=12, k_odd=1)


def test_both_ablation_flags_false_rejected() -> None:
    """equivariant=False with quantum=False is rejected (it would mismatch the declared flags)."""
    with pytest.raises(ValueError, match="at least one of"):
        EquivariantQuantumPPOBrainConfig(sensory_modules=MODS, equivariant=False, quantum=False)


def test_ppo_field_validators() -> None:
    """Zero/invalid PPO loop fields are rejected so training cannot be silently disabled."""
    with pytest.raises(ValueError, match="num_epochs"):
        EquivariantQuantumPPOBrainConfig(sensory_modules=MODS, num_epochs=0)
    with pytest.raises(ValueError, match="max_grad_norm"):
        EquivariantQuantumPPOBrainConfig(sensory_modules=MODS, max_grad_norm=-1.0)


def test_registry_registration() -> None:
    """The brain is registered under its BrainType and loads from the registry."""
    from quantumnematode.brain.arch._registry import get_registration
    from quantumnematode.brain.arch.dtypes import BrainType

    registration = get_registration(BrainType.EQUIVARIANT_QUANTUM_PPO)
    assert registration.brain_cls is EquivariantQuantumPPOBrain
    assert registration.config_cls is EquivariantQuantumPPOBrainConfig
