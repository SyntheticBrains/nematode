## Why

Every connectome result so far runs on a memoryless substrate. Each environment step starts every
neuron at zero, injects the sensors, and settles through a few `tanh` updates; nothing survives to the
next step, and gap junctions enter as a fixed symmetric matrix added to the chemical drive. B.2 is the
rung that gives neurons state across steps and makes gap junctions ohmic coupling inside that dynamics
(roadmap § Phase 8, item 7). B.2a is that substrate and its positive controls; B.2b, plastic gap
junctions under PPO, is MUST and runs only if B.2a's control passes (D20 as amended).

Two facts shape the design. **The gap matrix is stiff.** ALA's gap input sums to about 232 against at
most 6.2 of chemical input on any neuron, after the existing degree normalisation; with time constants,
an explicit integrator diverges unless the counts are rescaled (the roadmap's stiffness warning, from
the Wormlight review). **PPO replays shuffled single steps from a zero start.** Once state carries
across steps, the replay has to start each step from the state it actually had.

## What Changes

- **Leaky-integrator dynamics, off by default.** `ConnectomePPOBrain` gains `dynamics: settling | leaky` (default `settling`, byte-identical), a global time constant `membrane_tau_steps` in
  environment steps, and `bptt_chunk_length`. Under `leaky`, each neuron carries a membrane potential
  across steps, reset at episode start; within a step, `forward_pass_depth` semi-implicit Euler
  sub-steps integrate leak, ohmic gap coupling and chemical drive, with the sensors as a sustained
  input current. Leak and the gap Laplacian are stepped implicitly, so the step is stable for any
  non-negative symmetric gap weights; the counts are unchanged.
- **Recurrent replay.** Under `leaky`, the rollout buffer stores each step's starting state, and PPO
  replays contiguous chunks from those states with gradients through time inside each chunk — the
  pattern the LSTM brains use. The settling path, its buffer and its random draws are untouched.
- **Positive controls, in the roadmap's order.** (1) MLP-PPO on each block-V cell: hard350 is
  discharged by Logbook 060's committed runs; thermal at target 35 runs 8 seeds. (2) PPO on the
  dynamical wild type against PPO on the settling wild type, paired by seed on 128 seeds per cell,
  read as non-inferiority against a margin fixed from committed data: the dynamical substrate passes
  on a cell if it learns at least as well, within that cell's registered minimum wiring effect.
- **A gate-only τ pilot** on disjoint seeds chooses the time constant under a rule fixed before it
  runs, reading only the wild type's own learning, never a wiring gap.
- **Reuse of committed settling runs**, licensed by an identity check: the settling wild type's runs on
  seeds 641–768 (hard350, Logbook 079) and 513–640 (thermal target 35, Logbook 078) pair with the new
  dynamical runs, after re-runs on current code match them bit for bit.

## Capabilities

**Modified**: `connectome-ppo-brain`, with one added requirement: across-step leaky-integrator
dynamics.

## Impact

- `brain/arch/connectome_ppo.py`: the config keys and validation, the leaky step, the sequence forward,
  the membrane reset; `brain/arch/_ppo_buffer.py`: optional start states and a chunk iterator;
  `learning_rules/ppo.py`: the chunked replay path
- `configs/scenarios/foraging/`, `configs/scenarios/thermal_foraging/`: dynamical wild-type configs per
  τ and cell through a generator; an MLP-PPO thermal target-35 config
- `scripts/analysis/across_step_control.py`: new — the pilot rule, the MLP gate and the
  non-inferiority reading
- Tests: byte-identical off, stability on the raw counts, rollout-replay equivalence, chunking across
  episode boundaries, the reset, the refused combinations, the configs and the readings
- After the readout: Logbook 080; tracker B.2a; the roadmap's B.2 entry

## Breaking Changes

None. `dynamics: settling` is the default and every existing config, run and pinned result is
byte-identical.
