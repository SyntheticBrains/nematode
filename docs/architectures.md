# Brain Architectures

Every brain in Quantum Nematode is a plugin: it implements the `Brain` interface in `quantumnematode.brain.arch`, self-registers with `@register_brain`, and is selected from a scenario YAML by `brain.name`. The registry is the source of truth for what exists; this page is the human-readable catalogue — what each architecture is, what role it played in the research programme, and which optimiser to use with it. To add one, see the [plugin developer guide](architecture/plugin-developer-guide.md).

## Which brain should I start with?

| You want… | Use | Why |
|---|---|---|
| The strongest, fastest-training baseline | `mlpppo` | Wins the Phase 6 ranking on all three behaviours ([Logbook 029](experiments/logbooks/029-continuous-architecture-ranking.md)); plateaus in ~3,000 episodes on the integrated cell |
| The biology | `connectomeppo` | PPO on the real Cook et al. 2019 wiring; the project's focal architecture ([Logbooks 022](experiments/logbooks/022-connectome-substrate.md), [023](experiments/logbooks/023-architecture-plugin-interface.md), [034](experiments/logbooks/034-connectome-structure-controls.md)) |
| A task that needs working memory | `cfcppo`, `transformerppo`, `lstmppo`, `mingruppo`, `minlstmppo` | All clear the delayed-match-to-cue and associative-update probes that the memoryless MLP cannot ([Logbooks 030](experiments/logbooks/030-bit-memory-positive-control.md), [031](experiments/logbooks/031-minimal-rnn-candidates.md), [033](experiments/logbooks/033-associative-memory-probe.md)) |
| Scalar-only (temporal) sensing on the grid | `lstmppo` with `rnn_type: gru` | GRU outperformed LSTM across every evaluated environment ([Logbook 009](experiments/logbooks/009-temporal-sensing-evaluation.md)) |
| A quantum circuit | `qvarcircuit` trained with CMA-ES | The reference quantum brain; gradient-free optimisation is far more reliable than the parameter-shift rule here ([Logbook 002](experiments/logbooks/002-evolutionary-parameter-search.md)) |
| Gradient-free weights | `feedforwardga` | Evolved by the GA optimiser; also the floor the ranking is measured against |

## Catalogue

**Role** says where the architecture sits in the programme today: a *Phase 6 MUST arm* is one of the six pre-registered families in the continuous-substrate ranking; a *comparator* was run in a ranking or probe but is not gating; an *ablation control* exists to isolate one component of another brain; *historical* arms were evaluated in a closed campaign and are kept runnable as reference points.

### Feed-forward and value-based

| `brain.name` | Class | Description | Role |
|---|---|---|---|
| `mlpppo` | `MLPPPOBrain` | MLP actor-critic trained with clipped PPO and GAE; discrete or continuous `(speed, turn)` head | **Phase 6 MUST arm** — ranking winner ([029](experiments/logbooks/029-continuous-architecture-ranking.md)) |
| `mlpreinforce` | `MLPReinforceBrain` | MLP policy trained with REINFORCE and a learned baseline | Phase 0 baseline |
| `mlpdqn` | `MLPDQNBrain` | MLP Q-network with experience replay (DQN) | Phase 0 baseline |
| `feedforwardga` | `FeedforwardGABrain` | Feed-forward network whose weights are evolved by the genetic-algorithm optimiser; graded episodic-progress fitness for sparse-reward cells | **Phase 6 MUST arm** — the gradient-free floor (15.0% on the integrated cell, [029](experiments/logbooks/029-continuous-architecture-ranking.md)) |

### Recurrent, liquid and attention

| `brain.name` | Class | Description | Role |
|---|---|---|---|
| `lstmppo` | `LSTMPPOBrain` | LSTM- or GRU-augmented PPO with chunk-based truncated BPTT, separate actor/critic optimisers and entropy decay; built for temporal-sensing tasks. GRU variant recommended | **Phase 6 MUST arm** ([029](experiments/logbooks/029-continuous-architecture-ranking.md)); Phase 3 temporal-sensing brain ([009](experiments/logbooks/009-temporal-sensing-evaluation.md)) |
| `mingruppo` | `MinGRUPPOBrain` | minGRU (Feng et al. 2024, *Were RNNs All We Needed?*) — parallel-form minimal RNN with input-only gating; bounded, saturation-free recurrent core | Memory-axis candidate; beats plain LSTM on the reactive cell and clears the memory probes with a retention-gate init ([031](experiments/logbooks/031-minimal-rnn-candidates.md)) |
| `minlstmppo` | `MinLSTMPPOBrain` | minLSTM — parallel-form minimal RNN with normalised input-only gates and a single recurrent state; stability comparator to minGRU | Memory-axis candidate ([031](experiments/logbooks/031-minimal-rnn-candidates.md)) |
| `cfcppo` | `CfCPPOBrain` | Closed-form Continuous-time (CfC) liquid network with AutoNCP wiring and continuous-time recurrence, PPO-trained | **Phase 6 MUST arm** — tied 2nd ([029](experiments/logbooks/029-continuous-architecture-ranking.md)); best on the bit-memory probe (0.995, [030](experiments/logbooks/030-bit-memory-positive-control.md)) |
| `transformerppo` | `TransformerPPOBrain` | Self-attention encoder over a temporal window of recent sensory features, PPO-trained; the attention-based comparator to the recurrent arms | **Phase 6 MUST arm** — tied 2nd ([029](experiments/logbooks/029-continuous-architecture-ranking.md)); best on the associative-update probe (0.989, [033](experiments/logbooks/033-associative-memory-probe.md)) |

### Spiking

| `brain.name` | Class | Description | Role |
|---|---|---|---|
| `spikingreinforce` | `SpikingReinforceBrain` | Leaky integrate-and-fire network with surrogate-gradient REINFORCE and population-coded inputs | Phase 0 baseline ([003](experiments/logbooks/003-spiking-brain-optimization.md)) |
| `spikingppo` | `SpikingPPOBrain` | Recurrent adaptive-LIF spiking network with a configurable MLP actor head, trained with PPO | Phase 6 grid comparator — top cluster on the T4 grid ranking ([025](experiments/logbooks/025-weight-search-architecture-ranking.md)); folds into the Phase 7 STDP work |

### Reservoir

| `brain.name` | Class | Description | Role |
|---|---|---|---|
| `crh` | `CRHBrain` | Classical reservoir hybrid — echo-state reservoir with configurable feature channels (raw, cos/sin, squared, pairwise) and a PPO-trained readout | Ablation control for `qrh` ([008](experiments/logbooks/008-quantum-brain-evaluation.md)) |
| `qrh` | `QRHBrain` | Quantum reservoir hybrid — *C. elegans*-inspired structured topology, X/Y/Z + ZZ feature extraction, PPO-trained classical readout | Historical — Phase 2 quantum campaign ([008](experiments/logbooks/008-quantum-brain-evaluation.md)) |
| `crhqlstm` | `CRHQLSTMBrain` | CRH classical reservoir with a QLIF-LSTM temporal readout | Ablation control for `qrhqlstm` |
| `qrhqlstm` | `QRHQLSTMBrain` | QRH quantum reservoir with a QLIF-LSTM temporal readout and recurrent PPO | Historical — Phase 2 quantum campaign ([008](experiments/logbooks/008-quantum-brain-evaluation.md)) |

### Connectome-constrained

| `brain.name` | Class | Description | Role |
|---|---|---|---|
| `connectomeppo` | `ConnectomePPOBrain` | PPO on the real *C. elegans* hermaphrodite connectome (Cook et al. 2019: 302 neurons, 3,709 chemical synapses, 1,093 gap junctions) with biologically-faithful sensor → interneuron → motor projections, multi-hop within-step recurrence, a strict chemical-synapse mask and fixed gap junctions. `wiring: rewired_null` swaps in a degree-preserving random rewiring as a control | **Phase 6 MUST arm and focal architecture** — 5th of 6 under PPO, indistinguishable from its rewired null ([029](experiments/logbooks/029-continuous-architecture-ranking.md), [034](experiments/logbooks/034-connectome-structure-controls.md)) |

### Quantum and hybrid

| `brain.name` | Class | Description | Role |
|---|---|---|---|
| `qvarcircuit` | `QVarCircuitBrain` | Variational quantum circuit with modular sensory encoding; trained by CMA-ES or the parameter-shift rule; runs on Aer or IBM hardware | Reference quantum brain ([001](experiments/logbooks/001-quantum-predator-optimization.md), [002](experiments/logbooks/002-evolutionary-parameter-search.md)); first QPU deployment |
| `qrc` | `QRCBrain` | Quantum reservoir computing with data re-uploading and a classical readout | Historical — 0% success across 1,600+ runs; superseded by `qrh` ([008](experiments/logbooks/008-quantum-brain-evaluation.md)) |
| `qef` | `QEFBrain` | Quantum entangled features — configurable cross-modal entanglement topology (modality-paired, ring, random), Z + ZZ + cos/sin features, PPO-trained readout | Historical — competitive, not advantageous ([008](experiments/logbooks/008-quantum-brain-evaluation.md)) |
| `equivariantquantum` | `EquivariantQuantumPPOBrain` | Z₂-equivariant data-re-uploading circuit with an odd/even-parity latent split, PPO-trained; ships classical-equivariant and symmetry-prior ablation controls | Phase 6 grid comparator — numerical #1 on the T4 grid ranking, but tied by its matched-capacity classical-equivariant control ([025](experiments/logbooks/025-weight-search-architecture-ranking.md)) |
| `qsnnreinforce` | `QSNNReinforceBrain` | Quantum spiking network (QLIF neurons) with surrogate-gradient REINFORCE | Historical ([008](experiments/logbooks/008-quantum-brain-evaluation.md)) |
| `qsnnppo` | `QSNNPPOBrain` | QLIF quantum spiking network trained with PPO | Historical — PPO proved incompatible with the surrogate gradients ([008](experiments/logbooks/008-quantum-brain-evaluation.md)) |
| `qliflstm` | `QLIFLSTMBrain` | LSTM with QLIF quantum gates for temporal memory, recurrent PPO with truncated BPTT | Historical ([008](experiments/logbooks/008-quantum-brain-evaluation.md)) |
| `hybridquantum` | `HybridQuantumBrain` | QSNN reflex layer + classical cortex MLP + classical critic with mode-gated fusion and a three-stage curriculum | Historical — best quantum result on the grid (96.9% pursuit), matched by its classical ablation ([008](experiments/logbooks/008-quantum-brain-evaluation.md)) |
| `hybridclassical` | `HybridClassicalBrain` | `hybridquantum` with the QSNN reflex replaced by a small classical MLP | Ablation control for `hybridquantum` (96.3% pursuit) |
| `hybridquantumcortex` | `HybridQuantumCortexBrain` | QSNN reflex + QSNN cortex (grouped sensory QLIF) + classical critic with a four-stage curriculum | Historical — halted at 40.9% on two predators ([008](experiments/logbooks/008-quantum-brain-evaluation.md)) |

## Which optimiser for which brain

| Brains | Learning rule | Notes and evidence |
|---|---|---|
| `mlpppo`, `lstmppo`, `mingruppo`, `minlstmppo`, `cfcppo`, `transformerppo`, `connectomeppo`, `spikingppo`, `qsnnppo`, `qliflstm`, `qef`, `qrh`/`crh`, `qrhqlstm`/`crhqlstm`, `equivariantquantum`, the `hybrid*` family | Clipped PPO with GAE (Adam) | The most stable on-policy rule on the platform ([004](experiments/logbooks/004-ppo-brain-implementation.md)). Continuous-action heads need roughly three times the discrete entropy coefficient (≈ 0.10) or `log_std` collapses ([027](experiments/logbooks/027-platform-refactor-continuous-2d.md)). Recurrent arms train with chunk-based truncated BPTT; chunk length is a critical hyperparameter ([009](experiments/logbooks/009-temporal-sensing-evaluation.md)). The minimal RNNs only learn delay tasks with a memory-friendly retention-gate init, which costs nothing on reactive cells ([031](experiments/logbooks/031-minimal-rnn-candidates.md)) |
| `mlpreinforce`, `spikingreinforce`, `qsnnreinforce` | REINFORCE with baseline (surrogate gradients through spikes) | Phase 0 baselines; higher variance than PPO ([003](experiments/logbooks/003-spiking-brain-optimization.md)) |
| `mlpdqn` | DQN with experience replay | Phase 0 baseline |
| `qvarcircuit` | **CMA-ES** (recommended); parameter-shift gradients supported | Evolution beat gradient learning decisively on the grid — CMA-ES 80–88% success vs ~22–31% for parameter-shift — because shot noise and barren-plateau-like landscapes make circuit gradients unreliable ([001](experiments/logbooks/001-quantum-predator-optimization.md), [002](experiments/logbooks/002-evolutionary-parameter-search.md)) |
| `feedforwardga` | Genetic algorithm on the weights | Collapses on the integrated continuous cell (15.0%); the floor is optimiser-fundamental, not a bug ([029](experiments/logbooks/029-continuous-architecture-ranking.md)) |
| Any brain — hyperparameters | **TPE** (Optuna) preferred over CMA-ES | +79pp vs +47pp on the predator arm; TPE rescued CMA-ES's dead-zone seed ([012](experiments/logbooks/012-hyperparam-evolution-mlpppo-pilot.md)) |
| Across generations | Lamarckian warm-start inheritance | GO — passes the pre-registered speed gate ([013](experiments/logbooks/013-lamarckian-inheritance-pilot.md)). Baldwin-effect and transgenerational-memory inheritance are implemented but closed with STOP verdicts ([015](experiments/logbooks/015-baldwin-iterative-evaluation.md), [018](experiments/logbooks/018-transgenerational-memory.md)–[020](experiments/logbooks/020-tei-prior-on-lamarckian.md)) |

Hardware: the quantum brains run on the Qiskit Aer simulator (`--device cpu`, or `gpu` with the `gpu` extra) or on IBM Quantum hardware (`--device qpu`, with `--optimize` enabling Q-CTRL Fire Opal error suppression). See the [usage guide](usage.md).

## See also

- [Plugin developer guide](architecture/plugin-developer-guide.md) — files to touch, the no-per-architecture-branches rule, and a worked example.
- [Quantum architecture campaign notes](research/quantum-architectures.md) — specifications and the strategic assessment behind the Phase 2 quantum results.
- [Policy-architecture candidates](research/policy-architecture-candidates.md) — the survey that produced the minimal-RNN arms and the Phase 7 candidate list.
- [Roadmap § Architecture-comparison protocol](roadmap.md#architecture-comparison-protocol) — how arms are ranked (paired seeds, Wilcoxon, bootstrap CIs, BH-FDR).
