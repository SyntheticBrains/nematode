# Brain Architectures

Every brain in Quantum Nematode is a plugin: it implements the `Brain` interface in `quantumnematode.brain.arch`, self-registers with `@register_brain`, and is selected from a scenario YAML by `brain.name`. The registry is the source of truth for what exists; this page is the human-readable catalogue — what each architecture is, what role it played in the research programme, and which optimiser to use with it. To add one, see the [plugin developer guide](architecture/plugin-developer-guide.md).

## Which brain should I start with?

| You want… | Use | Why |
|---|---|---|
| The strongest, fastest-training baseline | `mlpppo` | Wins the Phase 6 ranking on all three behaviours ([Logbook 029](experiments/logbooks/029-continuous-architecture-ranking.md)); plateaus in ~3,000 episodes on the integrated cell |
| The biology | `connectomeppo` | PPO on the real Cook et al. 2019 wiring; the project's focal architecture ([Logbooks 022](experiments/logbooks/022-connectome-substrate.md), [023](experiments/logbooks/023-architecture-plugin-interface.md), [034](experiments/logbooks/034-connectome-structure-controls.md)) |
| A task that needs working memory | `cfcppo`, `transformerppo`, `lstmppo`, `mingruppo`, `minlstmppo` | All separate from the memoryless MLP on the delayed-match-to-cue and associative-update probes — the minimal RNNs retain strongly but update less readily ([Logbooks 030](experiments/logbooks/030-bit-memory-positive-control.md), [031](experiments/logbooks/031-minimal-rnn-candidates.md), [033](experiments/logbooks/033-associative-memory-probe.md)) |
| Scalar-only (temporal) sensing on the grid | `lstmppo` with `rnn_type: gru` | GRU outperformed LSTM across every evaluated environment ([Logbook 009](experiments/logbooks/009-temporal-sensing-evaluation.md)) |
| A quantum circuit | `qvarcircuit` | The reference quantum brain, trained with the parameter-shift rule through `run_simulation.py` on the Aer simulator or IBM hardware. CMA-ES beat parameter-shift gradients decisively on the grid ([Logbook 002](experiments/logbooks/002-evolutionary-parameter-search.md)), but the current weight-genome evolution encoders cover `mlpppo`, `lstmppo` and `feedforwardga` only (hyperparameter evolution is brain-agnostic) |
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
| `mingruppo` | `MinGRUPPOBrain` | minGRU (Feng et al. 2024, *Were RNNs All We Needed?*) — parallel-form minimal RNN with input-only gating; bounded, saturation-free recurrent core | Memory-axis candidate; directionally above plain LSTM on the reactive cell (minLSTM's lead is the significant one) and clears the memory probes with a retention-gate init ([031](experiments/logbooks/031-minimal-rnn-candidates.md)) |
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
| `connectomeppo` | `ConnectomePPOBrain` | PPO on the real *C. elegans* hermaphrodite connectome (Cook et al. 2019: 302 neurons, 3,709 chemical synapses, 1,093 gap junctions) with biologically-faithful sensor → interneuron → motor projections, multi-hop within-step recurrence, a strict chemical-synapse mask and fixed gap junctions. `wiring: rewired_degree_preserving` swaps in a degree-preserving random rewiring as a control (the `_rewired_null` config suffix). `learning_rule` selects the update: `ppo` (default), `three_factor` (reward-modulated Hebbian plasticity over the chemical synapses) or `hebbian` (the same update with the neuromodulator removed — the ablation floor that separates "learned something" from "learned something from reward"). Both plastic modes need `enable_activity_traces` and train `w_chem` alone — sensory gains and the motor readout stay frozen, the readout at the anatomical dorsal/ventral and forward/backward contrasts rather than a random draw (the `_plastic` config suffix) | **Phase 6 MUST arm and focal architecture** — 5th of 6 under PPO, indistinguishable from its rewired null ([029](experiments/logbooks/029-continuous-architecture-ranking.md), [034](experiments/logbooks/034-connectome-structure-controls.md)) |

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
| `mlpppo`, `lstmppo`, `mingruppo`, `minlstmppo`, `cfcppo`, `transformerppo`, `connectomeppo`, `spikingppo`, `qsnnppo`, `qliflstm`, `qef`, `qrh`/`crh`, `qrhqlstm`/`crhqlstm`, `equivariantquantum` | Clipped PPO with GAE (Adam) | The most stable on-policy rule on the platform ([004](experiments/logbooks/004-ppo-brain-implementation.md)). Continuous-action heads need roughly three times the discrete entropy coefficient (≈ 0.10) or `log_std` collapses ([027](experiments/logbooks/027-platform-refactor-continuous-2d.md)). Recurrent arms train with chunk-based truncated BPTT; chunk length is a critical hyperparameter ([009](experiments/logbooks/009-temporal-sensing-evaluation.md)). The minimal RNNs only learn delay tasks with a memory-friendly retention-gate init, which costs nothing on reactive cells ([031](experiments/logbooks/031-minimal-rnn-candidates.md)) |
| `connectomeppo`, `mlpppo` (opt-in) | Reward-modulated three-factor plasticity, or the same rule unmodulated | Local rule: pre- and post-synaptic activity via a decaying eligibility trace, gated by a global reward-prediction-error modulator. No backward pass, optimiser, or value head. Updates only what each substrate declares plastic — on `connectomeppo` the chemical synapses alone, on `mlpppo` every Linear weight — so on the connectome a difference between wiring variants is attributable to the wiring rather than to a differently-fitted decoder. Bounded by weight decay and a magnitude clamp, with saturation reported per update. Two sanity floors bound any plastic arm: frozen weights (no updates at all) and unmodulated Hebbian (updates without reward). Both run under the plasticity rule so they share its anatomical readout — a floor that decoded differently would confound decoding with learning. The rule reads a plastic-topology seam and never names a substrate, so `mlpppo` runs it too as the **matched-rule yardstick**: same rule class, same arithmetic, same hyperparameters from one shared definition. On the MLP every Linear weight is plastic including the output layer — a deliberate asymmetry in the MLP's favour, since its output layer is an arbitrary draw with nothing to preserve, which makes it a conservative yardstick for the connectome's ranking result |
| `mlpreinforce`, `spikingreinforce`, `qsnnreinforce`, `qrc` (readout), `hybridquantumcortex` (reflex and cortex) | REINFORCE with baseline (surrogate gradients through spikes) | The Phase 0 baselines and the REINFORCE-trained quantum arms; higher variance than PPO ([003](experiments/logbooks/003-spiking-brain-optimization.md), [008](experiments/logbooks/008-quantum-brain-evaluation.md)) |
| `hybridquantum`, `hybridclassical` | REINFORCE for the reflex layer, clipped PPO for the cortex and critic | Mode-gated fusion with a three-stage curriculum; the classical ablation matches the quantum reflex ([008](experiments/logbooks/008-quantum-brain-evaluation.md)) |
| `mlpdqn` | DQN with experience replay | Phase 0 baseline |
| `qvarcircuit` | Parameter-shift rule — the current code path, via `run_simulation.py` | CMA-ES beat parameter-shift gradients decisively on the grid — 80–88% success vs ~22–31% — because shot noise and barren-plateau-like landscapes make circuit gradients unreliable ([001](experiments/logbooks/001-quantum-predator-optimization.md), [002](experiments/logbooks/002-evolutionary-parameter-search.md)). That pipeline predates the current evolution framework, whose weight-genome encoders cover `mlpppo`, `lstmppo` and `feedforwardga` only (hyperparameter evolution is brain-agnostic), so a `qvarcircuit` genome encoder is the prerequisite for evolving its weights again |
| `feedforwardga` | Genetic algorithm on the weights | Collapses on the integrated continuous cell (15.0%); the floor is optimiser-fundamental, not a bug ([029](experiments/logbooks/029-continuous-architecture-ranking.md)) |
| Any brain — hyperparameters | **TPE** (Optuna) preferred over CMA-ES | +79pp vs +47pp on the predator arm; TPE rescued CMA-ES's dead-zone seed ([012](experiments/logbooks/012-hyperparam-evolution-mlpppo-pilot.md)) |
| Across generations | Lamarckian warm-start inheritance | GO — passes the pre-registered speed gate ([013](experiments/logbooks/013-lamarckian-inheritance-pilot.md)). Baldwin-effect and transgenerational-memory inheritance are implemented but closed with STOP verdicts ([015](experiments/logbooks/015-baldwin-iterative-evaluation.md), [018](experiments/logbooks/018-transgenerational-memory.md)–[020](experiments/logbooks/020-tei-prior-on-lamarckian.md)) |

Hardware: the quantum brains run on the Qiskit Aer simulator (`--device cpu`, or `gpu` with the `gpu` extra) or on IBM Quantum hardware (`--device qpu`, with `--optimize` enabling Q-CTRL Fire Opal error suppression); `--device mps` is PyTorch-only and is rejected for them. The classical and spiking brains additionally accept `--device mps`, though CPU is faster at their current sizes — see [Devices](usage.md#devices). See the [usage guide](usage.md).

## See also

- [Plugin developer guide](architecture/plugin-developer-guide.md) — files to touch, the no-per-architecture-branches rule, and a worked example.
- [Quantum architecture campaign notes](research/quantum-architectures.md) — specifications and the strategic assessment behind the Phase 2 quantum results.
- [Policy-architecture candidates](research/policy-architecture-candidates.md) — the survey that produced the minimal-RNN arms and the Phase 7 candidate list.
- [Roadmap § Architecture-comparison protocol](roadmap.md#architecture-comparison-protocol) — how arms are ranked (paired seeds, Wilcoxon, bootstrap CIs, BH-FDR).
