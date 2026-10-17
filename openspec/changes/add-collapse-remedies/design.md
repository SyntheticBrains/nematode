## Context

D25 (roadmap v4.5 § Phase 9) lists the remedies in order of comparison risk: KL early stopping; weight
clipping relative to the initial scale; Adam's second-moment settings; running-statistic advantage
normalisation. Logbook 096 named their values and a learning-rate schedule last. D25 also fixes the
confirmation (the MLP control on fresh seeds in M.1(c)), the platform rule (applied identically to every
arm, byte-identical when off, frozen for the phase) and the stop rule (three rounds).

M.1(a) indicated no cause and found collapse to be a sudden event: collapsed and intact runs drift alike
until the fall. That changes how a remedy can be selected. Any setting that changes a run's trajectory
re-rolls its collapse, so against the original runs, which were chosen because they collapsed, a remedy
with no effect would "rescue" at least 6 of 8 about two-thirds of the time. **The maintainer amended D25's
selection rule on 2026-10-17**: each remedy is read against a **placebo re-roll** of the same seeds,
paired by seed, on a continuous late-phase statistic. Neither arm is selected on its own outcome, so
regression to the mean cancels. The same day the maintainer set the KL stop **relative to each learner**,
because the registered fixed target would barely act on the connectome and act on almost every MLP update.

## Decisions

### Decision A: The stabilisers

Each is off by default, and the update is then byte-identical: the same minibatch order, the same
optimiser state, the same next draw of every generator, and no new key in the saved training state.

- **KL early stopping, relative to the learner.** Before each minibatch's optimiser step, its approximate
  KL `mean((r − 1) − log r)` is read from the log-probabilities the loss already computed. If it exceeds
  **3 × the median per-update KL over that learner's last 100 updates**, the step is not taken and the
  update's remaining minibatches and epochs are skipped. No stop is possible before 100 updates. The
  rule is the same for every arm and scale-free, so it acts on each learner's own outsized updates: on
  M.1(a)'s telemetry the connectome's median per-update KL is 0.003 and the MLP's 0.030, a tenfold
  difference that a fixed target cannot serve (a 0.015 stop would fire on 5% of connectome updates and
  87% of MLP updates). It is this project's rule, not a literature default, and its factor and window
  are fixed here, not tuned on the selection seeds. The running median is per-run state, not saved with
  the weights. A skipped epoch draws no minibatch permutation, so the buffer's generator advances less
  and later updates' minibatch order changes; action sampling and the environment are unaffected. The
  telemetry records `kl_stopped` and `minibatches_stepped` per update.
- **Weight clipping**, κ = 2. Elsayed et al. 2024 clip each layer to κ times its initialisation
  distribution's bound; here `s` stands in for that bound as each learnable tensor's largest absolute
  value **at construction**, kept as a non-persistent buffer only when the clip is on (so a loaded
  checkpoint is clipped against the brain's own construction, not re-measured). After every optimiser
  step each tensor is clipped elementwise to ±κ·s, before the connectome's mask and sign projection.
  Tensors with `s = 0` — the biases, `log_std`, the body drive's log-gains — are exempt, because a
  bound relative to zero is zero. **Its predicted reach**, from M.1(a)'s checkpoints (each
  tensor's largest magnitude at the end against episode 250's, the earliest checkpoint): the chemical
  weights grew 1.2–2.2× and the food gains 0.9–2.0×, the thermal gains 1.6–5.3× and the critic's
  weights 1.3–6.1×; the exempt `log_std` and log-gains grew most (5–28×). So round 2 acts chiefly on the
  critic and the thermal gains, which is stated before it runs.
- **Adam's second moment**, ε 1e-4 and β₂ 0.99 in place of PyTorch's 1e-8 and 0.999, passed to the
  optimiser at construction and re-applied after a checkpoint's optimiser state is loaded (a loaded
  state would otherwise restore the saved settings).
- **Running advantage normalisation** is registered as D25's fourth remedy but **not built** here; it is
  built only if a dated decision makes it reachable.
- **The placebo** (`rollout_shuffle_stream`): the rollout buffer's minibatch permutations drawn from a
  generator keyed on the run seed and a fixed tag, instead of the brain's generator. It changes the
  minibatch order and nothing else, so it re-rolls a run's trajectory with no stabilising intent.

### Decision B: The selection, placebo-paired, fixed before any round runs

**The statistic.** A run's **drawdown**: its highest trailing-200-episode success rate minus its mean
success over its last 500 episodes. On M.1(a)'s committed re-runs the collapsed runs drew down 0.62–0.92
and the intact runs 0.06–0.27.

**The rounds.** A placebo round first, outside the three-round budget; then the remedies in D25's order:
(1) the relative KL stop; (2) weight clipping; (3) Adam's second moment. Every round re-runs M.9's 16
seeds — the eight collapsers (wild type 2313, 2328; gap-only null 2304, 2305, 2307, 2308, 2319, 2330)
and their eight intact twins on the other wiring — on Cook 2019, the body at target 20, 6,000 episodes,
with telemetry; each remedy round also carries the placebo's shuffle stream, so a remedy and the placebo
differ in the remedy alone.

**A remedy passes** when, over the 16 seeds paired with the placebo, by the instrument's paired test
(`paired_seed_wilcoxon_bootstrap`, 80% interval):

- the drawdown falls: `placebo − remedy` has a mean of at least **0.05** and an interval above zero;
- learning is not slowed: `auc_success` to 3,000 episodes, `remedy − placebo`, has its interval's lower
  end above **−0.05** (every M.1(a) onset is after episode 3,858, so this window precedes any collapse).

Collapse counts under D25, and rescues and harms against the original runs, are reported beside and
never decide. The rounds stop at the first pass. After three rounds without one the **stop rule** fires:
the collapse statistic ships alone, and R.2 and G.5 register at 3,000 episodes with it.

**The MLP preflight.** Each round also runs the MLP control's retry seeds 9721–9728 under the remedy, read
only as a preflight for M.1(c)'s MLP hold (its plateau against M.1(a)'s committed MLP re-runs), never in
the selection. If a passing remedy then fails M.1(c), the next remedy in order is tried in a new round,
within the three-round budget.

**Failed runs.** A run that exits non-zero is re-run once; if it fails again it counts against the remedy
(its drawdown taken as its peak, its early `auc_success` as zero).

### Decision C: Why these rounds, and what they assume

D25's list is ordered by comparison risk, lowest first, and M.1(a) gives no reason to reorder it. The
remedy is selected on the count-weighted gap model, which G.0 found carries an artefact, and six of the
eight collapsers are gap-only nulls; M.1(c) runs its connectome arms on the corrected model chosen by
G.0-A. The remedy is assumed to transfer, and this change's runs cannot preflight the corrected-model arms.
R.1's change ports the chosen remedy to the other PPO families (recurrent, CfC, transformer), off by
default and byte-identical when off. Under amended D25, wild type 2331 also collapses on 092's panel
(onset 3,862); it is not in the set, which keeps M.1(a)'s pairs.

### Decision D: Cost

A round is 16 body runs at 6,000 episodes plus 8 MLP runs: about 2.2 hours for the connectome wave (M.1(a)
measured 7,775 seconds for the same 16) and about 1.1 for the MLP's. The placebo and three rounds at most:
about 13 hours, with the machine to itself; M.2 and M.3 wait or run between rounds. No gate preflight
applies: no round reads a floor or saturation gate, and the configs are 092's.

## Risks / Trade-offs

- **Sixteen pairs resolve only a large effect on drawdown.** That is a screen, not evidence; M.1(c) on
  fresh seeds carries the evidence.
- **Every update is already clipped** by the gradient-norm threshold, so a remedy acting through the step
  size may change little. That is a reason a round might fail, not a reason to tune.
- **A remedy may reduce a registered wiring effect**, if the null's excess collapses carried part of 092's
  lead; that is what D25 intends.
