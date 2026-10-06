## Overview

B.2a replaces the memoryless settling map with a leaky-integrator substrate and asks one question before
any wiring contrast runs on it: does PPO learn each block-V cell on the new substrate at least as well
as on the old one? The maintainer chose four settings before this change was written: a semi-implicit
integrator, chunked truncated backpropagation through time, both block-V cells, and a gate-only τ pilot.

## Decisions

### Decision A: The dynamics, and why the integrator is semi-implicit

Under `dynamics: leaky` each neuron carries a membrane potential `v` across environment steps, reset to
zero at episode start; its rate is `r = tanh(v)`. Within one environment step the brain takes
`K = forward_pass_depth` sub-steps of length `1/K` step, integrating

```text
τ dv/dt = −v − L v + Wᵀ tanh(v) + I
```

where `W` is the chemical matrix as today (masked under the strict mask), `L = D − G` is the Laplacian
of the existing gap matrix `G` (so the gap term is `Σ_j g_ij (v_j − v_i)`, ohmic coupling between
potentials, not an added drive), and `I` is the sensor injection, held as a sustained current through
the step rather than used as the initial state. With `α = 1/(K τ)`, one sub-step is

```text
v ← M⁻¹ ( v + α (Wᵀ tanh(v) + I) ),     M = (1 + α) I + α L
```

Leak and gap coupling are implicit; the chemical drive is explicit. `L` is symmetric positive
semi-definite for any non-negative symmetric `G`, so every eigenvalue of `M` is at least `1 + α` and
`‖M⁻¹‖₂ ≤ 1/(1 + α) < 1`: the linear part contracts at any step size and any gap weights, and with
`tanh` bounded the state stays bounded. **No count is rescaled**, so the stiffness warning is met by the
integrator and A.6's question of whether counts move with a rewiring stays separate. `G` is fixed in
B.2a, so `M⁻¹` is computed once at construction and each sub-step is two matrix products; B.2b, which
makes `G` learnable, recomputes it per update with a differentiable solve, and must keep `G` non-negative and symmetric for this bound to hold.

The critic and the readout read `r` after the last sub-step, as they read the settled state today. The
output after one step is not the settling map's output with memory added: the sensors enter as a current,
gap coupling acts on differences, and signal travels across steps, so the settling budget's hop limit
(A.2's depth mechanism) no longer bounds what reaches the motor layer. The positive control compares two
substrates, and the record says so.

### Decision B: Replay from stored starting states, in chunks

The rollout buffer stores each step's starting potential, detached; the rollout detaches `v` after every
step so the autograd graph never spans an episode. PPO cuts the buffer into contiguous chunks of
`bptt_chunk_length` (16), shuffles chunk order with the buffer's generator, and replays each chunk from
its first step's stored state with gradients through time inside the chunk; a step that begins an episode
starts from its stored zero state. A partial final chunk is padded and masked. An update fires as today, when the buffer is full or at episode end, except that an end-of-episode update needs at least as many chunks as minibatches; otherwise the experience carries forward to the next update, as a too-short buffer does today. This is the LSTM brains'
pattern (no burn-in). At unchanged parameters, the replay reproduces the rollout's log-probabilities and
values within float tolerance, which a test pins. The settling path keeps its shuffled single-step
minibatches and its exact random draws.

### Decision C: The time constant is swept on a pilot, under a rule fixed first

τ is global (one value for every neuron) and expressed in environment steps; at D22's default of about
five worm-seconds per step, the candidates are **0.2, 1 and 5 steps** (about 1, 5 and 25 worm-seconds).
The pilot runs the dynamical wild type, learning and frozen, at each τ on both cells, beside the settling
wild type on the same seeds, on **seeds 1101–1104**, disjoint from every registered band and every
earlier pilot: per cell, the dynamical arm learning and frozen at three τ, and the settling arm learning and frozen, 8 configs × 4 seeds, **64 runs** across both cells. **The rule, fixed before the pilot runs:**

1. A τ is eligible on a cell if the dynamical arm's gate preflight there is `readable`: it beats its
   frozen floor and does not sit within 5 points of the 90% bar.
2. Among τ eligible on both cells, score each by the smaller of the two cells' mean paired difference,
   dynamical minus settling `auc_success`. **τ = 1 step is chosen** unless another eligible τ beats its
   score by more than **0.05**, about one pilot standard error at the proxy spread on four seeds (0.03 on
   hard350, 0.05 on thermal), in which case that τ is chosen. If τ = 1 is not eligible, the eligible τ
   with the higher score is chosen.
3. If no τ is eligible on both cells, no panel launches: B.2a reports the pilot as its diagnosis and the
   maintainer decides.

No wiring gap is read, since only the wild type runs. Selecting the best τ on four seeds inflates its
expected performance on those seeds; the registered reading runs on disjoint seeds, so its verdict is
not biased by the choice.

### Decision D: The positive controls and their reading

**Control 1, the cell is solvable by the strongest method** (protocol principle 3). MLP-PPO runs on
each cell on **seeds 1201–1208** and passes if every seed's plateau success reaches competence (30%,
the episode metric's threshold) within the 3,000 episodes. hard350 uses the existing width-64 MLP-PPO
config, whose environment is the connectome cell's exactly; thermal at target 35 uses a config that is
the connectome cell with the existing thermal MLP-PPO brain. *(Amended before the registration:
hard350 was to be discharged by citing Logbook 060, but 060 committed only per-width means (87.6–97.0%
full clear), not per-seed values, so the per-seed gate could not be re-derived from committed data.)*

**Control 2, the new substrate learns the cell at least as well.** Per cell, the paired difference
`d = auc_success(dynamical) − auc_success(settling)` over 128 seeds, read as non-inferiority against a
margin `δ` fixed from committed data on that cell (the per-cell requirement):

| cell | seeds | δ | source |
|---|---|---|---|
| hard350 | 641–768 | 0.0143 | 2/3 of A.6's committed lead over the chemical-only null (Logbook 079's minimum) |
| thermal, target 35 | 513–640 | 0.0239 | Logbook 078's registered minimum on that cell |

The margin is the smallest wiring effect the programme registers on that cell: a substrate that costs
more than that would swamp the contrasts B.2b reads on it. With an 80% bootstrap interval:

| verdict | condition |
|---|---|
| **non_inferior** | the interval's lower bound is above −δ |
| **inferior** | the interval's upper bound is below −δ |
| **unresolved** | anything else |

Gates first: the dynamical learning arm beats its frozen floor and does not saturate; a cell failing
them reads **unlearnable**. Episodes to competence are reported beside, never read.

**Sizing.** No committed run pairs the two substrates, so the spread comes from a proxy: the paired
wild-type-minus-null gap on the same seeds (sd 0.062 on hard350, Logbook 079; 0.10 on thermal, Logbook
078). Two substrates may decorrelate a seed more than two wirings on one, so the proxy is expected to err
low. At 128 seeds and a true difference of zero, the chance of reading `non_inferior` is about 0.91 on
hard350 and 0.92 on thermal at the proxy spread. The achieved spread is reported beside, never used to
re-read.

**What each verdict licenses, per cell.** `non_inferior`: B.2b runs on the dynamical substrate on that
cell, and **first re-reads the wild type's lead over the chemical-only null on it** with static gaps: a passing substrate is a new reference frame, so no base effect carries over from the settling substrate (protocol principle 6). `inferior` or `unlearnable`: B.2b closes *unreachable-with-reason* on that cell, the diagnosis its
deliverable (D20 as amended). `unresolved`: neither; the maintainer chooses a registered extension on
fresh seeds or closing. An MLP failure on a cell reads **no_positive_control** for that cell and stops
there.

### Decision E: The analysis takes the gate preflight's shape

`gate_preflight.py` reads an analysis module's `STEMS` table, each level carrying four arms: learning
and frozen for two wirings. `across_step_control.py` makes each **cell a level**, puts the dynamical
substrate in the wild-type slots and the settling substrate in the null slots, and scores each level on
its own cell. The preflight then evaluates both substrates' floors and the saturation bar, so the
settling frozen runs are part of the evidence and their reuse is checked too.

### Decision F: The settling runs are reused, after an identity check

The settling wild type's learning and frozen runs on both bands are on disk from Logbooks 078 and 079.
Before the panel, four learning and two frozen settling runs per cell are re-run on this change's code
and must match the committed logs bit for bit (Logbook 075's precedent). A match licenses the reuse and
shows the settling path unchanged in practice; a mismatch stops the launch.

### Decision G: What is held fixed

Block V's committed point on each cell: edge-order draw, pooled readout, `forward_pass_depth` 4 (now the
sub-steps per environment step), `initial_log_std` as committed, the wild-type wiring, Cook 2019, PPO
hyperparameters unchanged. Only `dynamics`, `membrane_tau_steps` and `bptt_chunk_length` differ.

### Decision H: Combinations that are refused

`leaky` is refused at load with any plasticity rule other than PPO, activity traces, e-prop or node
noise. Each assumes one settled state per step, and none is needed by B.2a or B.2b; supporting them is a
separate change.

## Risks

- **The leaky substrate may learn worse for reasons unrelated to memory**, such as the sustained-current
  input or the ohmic gap form. The verdict is about the substrate as built; the logbook names those
  differences rather than attributing a failure to memory.
- **Chunked replay is slower.** Sixteen sequential steps of four sub-steps replace four batched products
  per minibatch. Cost is measured on the pilot, configured as the campaign will be.
- **Stale starting states.** A chunk starts from a state computed under older parameters; within ten
  epochs this is the approximation the LSTM brains already accept.
