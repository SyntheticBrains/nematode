## MODIFIED Requirements

### Requirement: Convergence-Aware Budget and Plateau-Performance Metric

The episode budget for the C3 primary cells SHALL be set so that every evaluated architecture reaches a plateau (converges) on the cell, so the comparison is plateau-vs-plateau rather than arbitrary-cutoff-vs-arbitrary-cutoff.

Two distinct convergence operations are used (they are NOT the same test):

- **Budget-setting convergence** (pre-flight): a run is converged for budget-selection purposes when its trailing-window success rate is stable — last-25-mean success within ±5 percentage points of last-100-mean success. This test selects the C3 budget (it set 1000 episodes for the grid-substrate ranking: the recurrent architectures plateau by 1000ep but not 500ep).
- **Ranked-metric plateau detection** (analysis): the ranked metric is **`post_convergence_success_rate`** — the full-clear (`COMPLETED_ALL_FOOD`) rate averaged over the post-convergence plateau, where the plateau onset is found by `detect_convergence` ([`packages/quantum-nematode/quantumnematode/benchmark/convergence.py`](../../../../packages/quantum-nematode/quantumnematode/benchmark/convergence.py)). Plateau detection SHALL be **level-agnostic**: convergence (the policy has stopped improving) SHALL be decoupled from the absolute success level, so a converged plateau is detected at any band (e.g. a stable 40% plateau and a stable 95% plateau are both detected). Convergence SHALL be established by **absence of trend** — comparing the final trailing block of runs against the immediately-preceding block of equal size and declaring converged when their mean full-clear rates agree within a band calibrated to block sampling noise (NOT by requiring a low-variance near-homogeneous window, which on binary outcomes is reachable only near 100% and so mis-classifies stable intermediate plateaus as non-converged). The plateau onset SHALL be the start of the final region whose smoothed (rolling-mean) success rate has reached the converged level (excluding the warm-up climb, and not anchoring on a transient touch of the band mid-climb), and the metric averages raw full-clear success from that onset to the end of the run, requiring ≥ 30 total runs. A run still trending at its budget SHALL return no plateau (flagged, not mis-scored).

The ranked metric SHALL be `post_convergence_success_rate`, NOT a fixed last-N window mean. The fixed-window choice was deliberately rejected because the evaluated arms can have very different warm-up lengths — the from-scratch spiking and quantum arms have long dead-exploration warm-ups on this lethal cell, so a fixed last-N (e.g. last-25) window would mis-measure the slow-igniting arms relative to the fast learners; ranking on the detected post-convergence plateau is the fair comparison. The full-window-mean (plateau mean over a post-warmup tail) SHALL be retained alongside as an **agreement cross-check**: for a homogeneous-warm-up arm set it MUST agree with the detected `post_convergence_success_rate` within noise, bounding the risk that plateau detection introduces bias. (Overall `success_rate`, which includes the warm-up, is also retained in the per-seed export for reference.) For GA cells the analogue is the evolved-champion full-clear rate over a frozen eval. Convergence is distinct from success — an architecture MAY converge to a low plateau (its ceiling, a valid finding) or a high one; the ranked metric SHALL measure that plateau level faithfully whether it is high or low.

**Sample-efficiency reporting (realised scope).** The primary cross-architecture ranking is on asymptotic plateau performance (`post_convergence_success_rate`). Sample efficiency / warm-up length is reported **descriptively** — the per-500-episode full-clear-rate trajectory in the logbook illustrates the long-warm-up-then-ignition shape of the slow arms — rather than as a computed per-architecture "episodes-to-90%-of-plateau" metric. This is a deliberate realised simplification: the asymptotic ranking is the load-bearing result, with the warm-up trajectory as qualitative context. A future pass MAY add the computed sample-efficiency dimension.

#### Scenario: C3 budget is set to the slowest-converging architecture

- **GIVEN** the set of architectures to be compared on the C3 cell
- **WHEN** the C3 episode budget is chosen
- **THEN** the budget SHALL be at least the convergence point (per the ±5pp trailing-window test) of the slowest-converging architecture in the set
- **AND** the pre-flight evidence for that budget SHALL be recorded (e.g. the grid-substrate ranking set 1000 episodes based on its pre-flight: recurrent architectures — LSTMPPO, connectome — plateau by 1000ep but not by 500ep)

#### Scenario: Non-plateau triggers a budget extension and rerun

- **WHEN** a C3 cell's run does NOT satisfy the convergence test at its budget (the final trailing block still differs from the preceding block by more than the calibrated band, indicating it is still climbing)
- **THEN** that is a trigger to extend the episode (or generation) budget for that architecture and rerun until it reaches its plateau
- **AND** because the ranked metric is the post-convergence plateau (`post_convergence_success_rate`), the comparison is plateau-vs-plateau even when arms reach their plateaus at different episode budgets — a uniform episode budget across arms is therefore NOT required, provided every arm's run has reached its plateau (the convergence detector confirms this per run; an arm that never converges is excluded / flagged, not mis-compared against a fixed cutoff)
- **AND** when a SHOULD/MAY architecture is added after the initial budget is set, its budget SHALL be set to its own convergence point (extended as needed); the plateau metric normalises across budgets, so the already-converged cells are NOT force-rerun at a uniform budget

#### Scenario: Asymptotic plateau performance is the ranked metric; warm-up is reported descriptively

- **WHEN** the ranking is computed
- **THEN** each architecture's C3 result SHALL report `post_convergence_success_rate` (the detected-plateau full-clear rate) as the ranked asymptotic metric
- **AND** the warm-up / sample-efficiency dimension SHALL be reported descriptively (the per-500-episode clear-rate trajectory) rather than as a computed episodes-to-90%-of-plateau number
- **AND** the ranking narrative MAY note where a "converged to a higher plateau" claim is distinct from a "converged faster" observation

#### Scenario: Level-agnostic plateau detection ranks a sub-saturation learnable band

- **GIVEN** a C3 cell whose difficulty is locked to a learnable sub-saturation band, so converged architectures plateau across a wide range of full-clear rates (e.g. ≈35–80%) rather than near 100%
- **WHEN** `post_convergence_success_rate` is computed per seed
- **THEN** a stable intermediate plateau (e.g. an architecture that completes the full-clear ~45% of episodes from a converged policy) SHALL be detected as converged and SHALL report its plateau rate (≈45%) averaged over the full post-onset region, NOT silently fall back to a fixed last-N-run window nor be mislabeled non-converged
- **AND** an architecture whose run is still trending at its budget SHALL be flagged as non-converged (extend-and-rerun), not ranked on the noisy fallback window
- **AND** the detected per-seed plateau rate SHALL agree, within sampling noise, with the full-window-mean cross-check
