# Tasks

## 1. The mechanism

- [ ] 1.1 A declarable perturbation set on the connectome topology: the per-step draw is multiplied by a
  registered unit mask, so a unit outside the set draws nothing and — because the eligibility is
  `h_prev ⊗ perturbation` — every synapse onto it keeps a zero trace and is never written.
- [ ] 1.2 The config surface: the declared set by name (`full`, `causal`, `hop1`, `motor`,
  `motor_last`), validated against the loaded connectome, with `full` the default so **every committed
  plastic connectome result reproduces unchanged**. The plasticity config is shared with the MLP, which
  has no connectome to derive a set from, so anything but `full` **raises** there — the guard
  `third_factor: pathway` already uses this pattern in `mlpppo.py`.
- [ ] 1.2b A config validator refusing a restricted set together with `plasticity_homeostasis: false`:
  the rule's weight decay is unconditional, so without the homeostatic rescale to cancel it the excluded
  synapses shrink across the run and the mask becomes a decay manipulation.
- [ ] 1.3 The hop-distance derivation from the loaded connectome: distance from each unit to the readout
  pool over **directed chemical** edges, with the readout pool taken from the same motor-class
  constants the readout itself uses, so the two cannot drift apart.
- [ ] 1.4 The per-step causal mask: at step `s`, perturb only units within `depth − s` hops. Derived
  from `forward_pass_depth`, not hard-coded to 4.
- [ ] 1.5 Telemetry: the declared set, unit count, adaptable-synapse count and realised draws per
  decision, recorded per run, with a **failure** when realised draws do not match the declaration.
- [ ] 1.6 Tests: the default reproduces the unmasked stream bit-for-bit; the hop distances match the
  recon table (39/109/247/277 cumulative units at 0/1/2/3 hops, and **25** units at four hops or more
  that can never contribute); the causal mask's draw count is **672** at depth 4; a `forward_pass_depth`
  other than 4 changes the mask; a declaration that does not match the derived set raises; and the
  MLP raises on any set but `full`.
- [ ] 1.6b The homeostasis dependency, **measured in both directions**: a masked unit's incoming
  synapses do not move across an episode with `plasticity_homeostasis: true`, and **do** move with it
  false. The second half is what makes the first a finding rather than an assumption, and it pins why
  the validator in 1.2b exists.

## 2. The cell and the arms

- [ ] 2.1 The hard-food connectome cell under the rule — the first plastic connectome config off the
  2400-step C3 cell — at I.1's passing recipe, and ten configs: five masks × (learning, frozen).
- [ ] 2.2 Each frozen control carries the **same σ and the same mask**, freezing only the update.
- [ ] 2.3 Exact-key test: each config differs from its base by the cell keys, the recipe keys and its own
  mask and arm keys, and nothing else.

## 3. Harness

- [ ] 3.1 `scripts/analysis/l4_reduced_perturbation.py`: scan, plateau-tail mean foods through I.2's
  graded family, each arm against its own frozen control, paired one-sided, BH-FDR across the five
  masks.
- [ ] 3.2 Both effect minima together — **1.0 foods** of 20, and **10% of the reachable gap**, which
  against 058's committed reference (PPO 19.31, frozen 3.82 over 32 seeds) is **1.55 foods**. Recomputed
  per arm where that arm's own frozen mean differs, with both reported and a downgrade naming which
  minimum failed.
- [ ] 3.3 Per-arm drift from its own frozen control, with the seed set a parameter (R.1's defect) and
  unavailable reported as unavailable rather than zero.
- [ ] 3.4 The completeness guard: no verdict from a campaign missing any registered cell.
- [ ] 3.5 The three registered verdicts plus a partial reading that states the ordering.
- [ ] 3.6 Tests for 3.1–3.5, including a fixture where `motor_last` wins and `motor` does not.

## 4. Pilot (disjoint seeds 101–104)

- [ ] 4.1 `launch.md` committed before anything runs.
- [ ] 4.2 `full` and `motor`, learning and frozen, 16 runs: is the frozen floor off the ceiling, does
  `full` reproduce the known failure, and what does a run actually cost?
- [ ] 4.3 Measured per-run wall time, and the campaign scheduled from it rather than from the proposal's
  estimate.
- [ ] 4.4 If `full` learns, or the frozen floor is at the ceiling, the campaign does not launch and the
  change is amended under a dated note.

## 5. Campaign

- [ ] 5.1 Launch record committed first; no branch switches while it runs.
- [ ] 5.2 80 runs: ten arms × eight seeds at 3000 episodes, with `--track-experiment` so drift has
  weights to read.
- [ ] 5.3 Per-seed CSV, the per-arm table with its dimension columns, and the verdict under
  `supporting/061-l4-reduced-perturbation/`.

## 6. The record

- [ ] 6.1 Logbook 061: the causal-reach table as a substrate measurement that stands whatever the arms
  do; the per-arm results with units, adaptable synapses and draws per decision beside each; the drift
  column; the verdict against the three registered outcomes.
- [ ] 6.2 The experiments index row.
- [ ] 6.3 State plainly whether **R.1b is unblocked and at which mask**, and what that mask gives up.
- [ ] 6.4 State what the result may not be cited as — in particular that a win at a reduced mask is a
  win for a restricted learner, and that nothing here transfers to the 2400-step C3 cell.

## 7. Close-out

- [ ] 7.1 `CHANGELOG.md`; the tracker (R.1c, and R.1b's block lifted or not); the roadmap only if the
  reading changes.
- [ ] 7.2 Confirm no committed verdict changed, and that `full` reproduces the unmasked stream.
