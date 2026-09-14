# Eligibility from the dynamics, not from noise (R.2)

## Why

Node perturbation is closed. [R.1](../../../docs/experiments/logbooks/060-l4-perturbation-scale.md)
found the rule solving a multi-step cell at **8 perturbed units** and collapsing to **3.0% full clear
at 128**; [R.1c](../../../docs/experiments/logbooks/061-l4-reduced-perturbation.md) took the
connectome from **1208 draws per scored decision down to 39** and returned `not_reducible`; and
[R.1d](../../../docs/experiments/logbooks/062-l4-frozen-readout.md) showed the frozen readout is part
of the limit — substituting it more than doubles what the rule reaches, 3.751 → 9.639 foods of 20 —
**without any arm reaching competence**.

The finding those three share is the argument for this change. **Credited-synapse drift sits at
1.37–1.42× the weight's own norm across every one of them**: 302 perturbed units and 39, four readouts
at two norms and two directions. The rule is never starved of signal. It writes a great deal, in a
direction that does not help, and no amount of narrowing what it credits or improving what reads the
result changes that. **The remaining suspect is credit assignment itself** — what the eligibility
carries.

Under node perturbation the eligibility is `E ← decay·E + M ∘ (h_prev ⊗ ξ)`: the per-unit noise `ξ` is
the only thing telling a synapse *which way* its post-synaptic unit moved. Averaged over draws that is
an unbiased reward-gradient estimate whose variance grows with the number of units perturbed — the 1/N
arithmetic R.1 measured, which the connectome's 1208 draws violate by an order of magnitude.

**e-prop is D1's named fallback and it removes the noise entirely**: the eligibility comes from the
network's own settling dynamics, and the sign comes from a broadcast learning signal rather than from a
draw.

## What changes

- **A third eligibility mode, `eprop`**, at the seam the other two already use. On the settling
  recurrence `h ← tanh(Wᵀh)` the trace a synapse `i → j` accumulates is

  ```text
  eps_ij = sum over settling steps s of  psi_j^(s+1) * h_i^(s)        psi = 1 - tanh^2
  E_ij  <- decay * E_ij + M ∘ (eps_ij * L_j)
  ```

  `psi` is the unit's own activation derivative and `eps` is the local part of `∂h_j/∂w_ij` — e-prop's
  approximation is exactly the decision to keep that part and drop the paths through other units.
  **No perturbation is drawn**, and `plasticity_node_noise` must be zero: the two mechanisms mixed
  would give a result attributable to neither.

- **The per-unit learning signal `L_j`, registered as an arm rather than assumed.** `eps_ij` is
  unsigned with respect to the outcome — it says how strongly `w_ij` moves `h_j`, not whether moving it
  was good. Under node perturbation `ξ_j` carried that; under e-prop it is `L_j`, a broadcast
  projection of the policy's output error:

  ```text
  g_k = (u_k - mu_k) / sigma_k^2      u = the PRE-SQUASH Gaussian draw, not the tanh-squashed action
  L_j = sum_k B_jk * g_k              B fixed for the run, persisted with the checkpoint
  ```

  and the scalar modulator `δ = r − b` multiplies at update time, exactly as it does now.

- **What recon found, and what it does to the arm set.** The motor readout is a `(4, 2)` matrix over
  the **mean-pooled** VB/DB/VA/DA classes, so `∂mu_k/∂h_j = readout[k, class(j)] / |class(j)|` for a
  unit in the pool and **exactly zero for the other 263**. Since e-prop drops the multi-hop paths,
  **symmetric feedback delivers no signal at all outside the 39-unit readout pool** — it is not a
  broad arm with true directions, it is the e-prop analogue of R.1c's `motor` set. So the routing
  varies two things at once, and the campaign crosses them:

  | arm | signal source | credited breadth |
  |---|---|---|
  | `symmetric` | the readout's own transpose | the 39-unit pool — **forced** by the truncation, not chosen |
  | `random_motor` | fixed random projection | the same 39 units, masked to match |
  | `random` | fixed random projection | all 302 |
  | `scalar` | none, `L_j = 1` | all 302 |

  The fourth cell of the 2×2 — true directions reaching all 302 units — is **the cell the mechanism
  forbids**, and saying so is part of the result. `random_motor` is what makes `symmetric` vs `random`
  readable: without it, a `symmetric` win is "the true direction helped" and "crediting only the motor
  pool helped" at once, and R.1c already measured the second as the best perturbation set. `scalar` is
  the ablation: if it matches `random`, the per-unit signal did nothing and the result is about the
  dynamics-derived trace alone.

- **Stage 1, the positive control, before any connectome run.** The protocol requires a new component
  on a validated platform to have its own positive control, and this one has a sharp form: on the
  committed one-step contextual association
  ([`positive_control.py`](../../../packages/quantum-nematode/quantumnematode/plasticity/positive_control.py))
  a single forward pass makes `eps_ij = psi_j · h_i`, so for the plastic layer **`symmetric` is the
  exact REINFORCE gradient**. If it does not learn, the implementation is wrong and nothing downstream
  is interpretable. `random` is the arm the plausibility claim rests on, and `scalar` must **not** solve
  a task whose answer only reward reveals through a per-unit signal.

- **Stage 2, the hard-food cell**, at exactly the operating point R.1c and R.1d measured — the
  `hard350` connectome cell with `plasticity_normalise_modulator`, `plasticity_normalise_trace` and
  `plasticity_homeostasis` all on, `plasticity_rate: 0.001`, `trace_decay: 0.9`,
  `initial_log_std: -1.0`, `forward_pass_depth: 4`, and the **committed anatomical readout**. Four
  learning arms and **one** shared frozen floor — a frozen arm makes no updates, so no routing can
  reach it — at **16 seeds**, the count
  [Logbook 059](../../../docs/experiments/logbooks/059-7a-shipment.md) registered for this gate. The
  comparison is against numbers already on the record: node perturbation's **3.751** learning and
  **3.150** frozen on this exact cell, its `motor` arm's **4.361**, R.1d's best substituted arm at
  **9.639**, and PPO's matched **18.945**.

- **The registered outcomes are 059's three, unchanged**: does not learn (the programme stops and 7b
  proceeds under PPO); learns but misses the wiring bar (a result, 7b stays gated); learns and reads
  the wiring. 059's **three-active-week implementation stopping condition** applies and is carried as a
  task.

Out of scope: **the wiring contrast itself** (059's third stage, on the block-V cells against the
registered ≥ 20% time-to-competence bar) — it is gated on stage 2 producing a rule that learns the
cell, which is the gate R.1b is blocked behind, and registering it now would repeat R.1c's mistake of
stating a consequence its verdict condition could not reach. Also out of scope: making the readout
plastic; any perturbation set (there is no perturbation); e-prop on the MLP yardstick beyond the
one-step control, which the config layer **refuses** rather than silently running; and the readout
scale R.1d surfaced, which is named as the pre-registered follow-up if stage 2 lands in the 2.4–4.4
food band rather than a new arm inside it.

## Capabilities

**Modified**: `plasticity-evaluation` — an eligibility derived from a substrate's own dynamics states
where its approximation drops terms; a learning rule whose update needs a per-unit signal registers how
that signal reaches each unit, and tests it against both the ablation that removes it and a control
matched on which units it can reach.

## Impact

- New: the `eprop` eligibility mode and its learning-signal routing; the topology seam that folds the
  signal in after the action is sampled; three stage-1 arms in the positive-control harness;
  `scripts/analysis/l4_eprop.py`; five configs; records under `supporting/063-l4-eprop/`; Logbook 063.
- Edited: `EligibilityMode` and its validators, the two topologies' trace updates, the connectome
  brain's action steps, the experiments index, `CHANGELOG.md`, the tracker (R.2), the roadmap only if
  the reading changes.
- Compute: **80 runs** — four learning arms and one shared frozen floor × 16 seeds at ~1800 s — about
  **3h 10m** at the measured parallelism, plus a **4-run pilot** on disjoint seeds 101–104 and a rate
  check on the same seeds. Stage 1 is a script and runs in minutes.
