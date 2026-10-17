## Context

D25 fixes the remedy list's order (KL early stop; weight clipping relative to the initial scale;
Adam ε 1e-4 and β₂ 0.99; EWMA advantage normalisation; a learning-rate schedule last), the selection
rule (rescue of the known collapsers, read by discordant pairs), the confirmation (the MLP control on
fresh seeds in M.1(c)) and the stop rule (three rounds). M.1(a) indicated no cause, so the order is
D25's own. M.1(a) also recorded two platform-wide facts, never read as causes: every update is clipped
by the 0.5 gradient-norm threshold, on every learner, and exploration noise grows through training.

## Decisions

### Decision A: The stabilisers, and their values fixed by convention, not by these seeds

Each value is the literature's default for the method, fixed here before any round runs; none is tuned
on the 16 seeds, which are the selection set.

- **KL early stopping**, `target_kl` 0.01 (OpenAI Spinning Up's PPO default): within an update, after
  each minibatch, if the approximate KL `mean((r − 1) − log r)` exceeds 1.5 × `target_kl`, the update's
  remaining minibatches and epochs are skipped. The stop is counted in the telemetry. M.1(a)'s KL ran at
  about 0.004–0.02 per update, so a 0.015 stop acts on the larger updates and leaves the rest.
- **Weight clipping**, κ = 2: Elsayed et al. 2024 clip each layer to κ times its initialisation
  distribution's bound; here `s` stands in for that bound and κ = 2 is fixed as a moderate value, not
  tuned on these seeds. After every optimiser step, each learnable weight
  tensor is clipped elementwise to ±κ·s, `s` that tensor's largest absolute value at initialisation
  (biases and zero-initialised tensors, whose `s` is 0, are left unclipped). On the connectome the
  strict mask and the sign constraints are applied after the clip, as now after the step.
- **Adam's second moment**, ε 1e-4 and β₂ 0.99, the values D25's list names (in place of PyTorch's
  1e-8 and 0.999): passed to the optimiser at construction.
- **Running advantage normalisation**, decay 0.99: the advantages are normalised by a running mean and
  standard deviation updated once per update from the raw batch, instead of the batch's own. Registered
  for D25's list; it runs only if a round is skipped or added by a dated decision.

Every stabiliser is off by default and the update is then byte-identical: the same minibatch order,
the same optimiser state, the same next draw of every generator.

### Decision B: The rescue rule, fixed before any round runs

Each round re-runs M.9's 16 seeds (the collapsed wild type on 2313 and 2328 and gap-only null on 2304,
2305, 2307, 2308, 2319 and 2330, and each one's intact twin on the other wiring) under the remedy, Cook
2019, the body at target 20, 6,000 episodes, with telemetry. The baseline is M.1(a)'s re-runs, which
reproduce the committed panel.

- A collapser is **rescued** when it does not collapse under the remedy (D25, as amended); an intact
  twin is **harmed** when it collapses under the remedy.
- The **learning-speed guard**: on the eight intact twins, `auc_success` to 3,000 episodes under the
  remedy against the baseline, paired by seed; its 80% interval's lower end must lie above **−0.05**,
  so a remedy cannot pass by slowing learning until nothing collapses.
- A remedy **passes** when it rescues at least 6 of the 8 collapsers, harms at most 1 of the 8 intact
  twins, and passes the guard. The exact McNemar test of rescues against harms is reported beside.
- The rounds run in order and stop at the first pass. A remedy that fails is recorded and not retried.
  After three rounds without a pass, **the stop rule fires**: the collapse statistic ships alone, and R.2
  and G.5 register at 3,000 episodes with it.

Rescue on the known collapsers is the selection; it is not the evidence. The evidence that a remedy
works is M.1(c)'s confirmatory panel on fresh seeds, read against an equivalence margin.

### Decision C: Why these three rounds

D25's list is ordered by comparison risk, lowest first. With no cause indicated there is no reason to
reorder it. The first three fill the three rounds; running-statistic normalisation and a learning-rate
schedule stay registered as the fourth and fifth, reachable only by a dated decision.

### Decision D: Cost

A round is 16 body runs at 6,000 episodes, one wave at 16 workers: about 2.2 hours, from M.1(a)'s
measured 7,775 seconds for the same 16 runs. Three rounds at most: under 7 hours.

## Risks / Trade-offs

- **A remedy may pass by luck on 16 seeds.** That is why the confirmation is M.1(c) on fresh seeds, and
  why the threshold is 6 of 8 with at most one harm.
- **Every update is already clipped**: a remedy that acts through the step size may change little when
  the step is already normalised. That is a reason a round might fail, not a reason to tune.
- **A remedy may reduce a registered wiring effect.** If the gap-only null's excess collapses carried
  part of 092's lead, a remedy removes that part; that is what D25 intends.
