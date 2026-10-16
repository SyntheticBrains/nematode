## Context

D25 fixes the collapse definition's shape (a trailing-200-episode peak; onset when the 100-episode mean
falls below half of it and stays there at least 300 episodes; censored at the horizon), forbids further
6,000-episode body panels before M.1 reads out (M.1(c) excepted), and requires that a remedy be chosen
by rescue of the known collapsers and then hold the MLP control. This change builds what that needs
and runs the diagnosis; it chooses no remedy.

## Decisions

### Decision A: Telemetry reads, never writes

Every statistic is computed from tensors the update already holds — the batch's returns and raw
advantages before the buffer normalises them, the old and new log-probabilities, the values, the
gradients after `backward()` and before clipping, the parameters — under `torch.no_grad()`. Nothing
calls the buffer's minibatch sampler, the policy's sampler or any other generator, and nothing is
written back. The settling residual needs the second-to-last settling iterate, which the replay
already computes: the replay returns it when telemetry is on, and the forward is otherwise unchanged.
The test: one seed with telemetry on and off, compared on every `Run:` line, the final weights and the
next value of each generator.

### Decision B: Per update, not per minibatch

One line per update keeps the file small (≈ 12,000 lines and a few MB for a 6,000-episode body run).
KL and clip fraction are means over the final epoch, the epoch whose weights the next rollout uses;
entropy and losses are means over all epochs, as the existing report is. Saturation and `srank` are
computed once per update on the update's full batch of features.

### Decision C: The diagnosis is a reproduction, so it runs on M.9's own configs

M.1(a) re-runs committed runs to see inside them; it is a diagnostic licensed by an identity check, not
a new registration, so D23's Emmons rule does not apply to it: the re-runs use M.9's Cook 2019 body
configs at target 20, 6,000 episodes, on M.9's seeds. The pairing:

| seed | collapsed learner | intact learner on the same seed |
|---|---|---|
| 2313, 2328 | wild type | gap-only null |
| 2304, 2305, 2307, 2308, 2319, 2330 | gap-only null | wild type |

Each pair shares a seed, a task draw and every initial weight on the chemical graph, so the stacks
compare a collapse against its own intact twin. The chemical-only null's five collapses are left out:
no Phase 9 block reads that null on the gap question, and its seeds would bring no intact twin on the
same wiring pair. Identity: each re-run's plateau must equal the committed
per-seed CSV's to its printed precision (the step-level logs are archived off-repo, so the plateau is
the field compared, and the rest is named *uncompared*).

### Decision D: The cause set, fixed before the logs are read

From the literature review (roadmap § Phase 9 § What the review established), each with the pattern
that identifies it in the 1,500 episodes before onset.

**The envelope.** Each intact twin is aligned on its collapsed twin's onset episode, so the eight
intact runs give, for every statistic and every aligned episode, a mean and a standard deviation; the
envelope is that mean ± 2 sd. A collapsed run **departs** on a statistic when the statistic, smoothed
over 100 episodes, sits outside the envelope in the cause's predicted direction for at least 200
consecutive episodes inside the 1,500 before its onset; the first episode of that stretch, as a lead
before onset, is reported. Each intact twin is tested the same way against the envelope of the other
seven (leave one out), so the intact side is held to the same rule.

1. **Weight growth → saturation → rank loss**: layer-wise weight norm rising from the start, saturated
   fraction rising and `srank` falling before onset, entropy flat or rising, KL and clip fraction
   spiking at onset.
2. **Fixed step against a vanishing signal**: raw advantage std falling toward zero while KL, clip
   fraction and gradient norm stay constant.
3. **Optimiser second-moment staleness**: a pre-clip gradient-norm jump followed within tens of updates
   by a KL spike.
4. **Critic failure**: explained variance falling and value loss rising before the success rate falls;
   on the connectome, critic gradient norm exceeding actor gradient norm.
5. **Variance collapse**: `log_std` falling far below its initial value before onset.
6. **Mean-head saturation**: the `|u| > 2` fraction rising with `log_std` flat.

**As the harness encodes them** (`collapse_diagnosis.CAUSES`), each pattern is a conjunction of
departures, the rest of its description read beside as context: (1) weight norm up, saturated
fraction up and `srank` down; (2) raw advantage std down with the gradient norm departing in neither
direction; (3) gradient norm up and KL up, the gradient's departure beginning no later than KL's;
(4) explained variance down and value loss up, the critic-to-actor gradient ratio reported beside;
(5) `log_std` down; (6) the `|u| > 2` fraction up with `log_std` departing in neither direction. The
weight norm is the norm over every parameter but the critic's.

A cause is **indicated** when its pattern departs in at least 6 of the 8 collapsed runs and in at most
1 of the 8 intact twins; **not indicated** when it departs in at most 2 collapsed runs;
otherwise **mixed**. If a re-run that collapsed on the committed panel does not collapse under the
registered definition, its pair leaves the stacks, it is named in the readout, and the thresholds
apply as the same fractions (three quarters, one eighth, one quarter) of the pairs that remain. More than one cause may be indicated. The readout names which, and M.1(b)'s change
takes its remedy order from it (D25's list, reordered by what is indicated).

### Decision E: The MLP's incidence

The remedy rule needs the MLP control's own collapse rate at 6,000 episodes, which is known only from
M.9's 8 retry seeds. They are re-run with telemetry (seeds 9721–9728, the retry's configs), their identity checked against
the committed plateaus in Logbook 092's `retry/per_seed.csv`, and their incidence is reported with its Wilson interval beside the connectome's. They are not part of the cause
stacks unless they collapse.

### Decision F: The collapse statistic's code is the registered definition

`collapse.py` takes a per-episode success series and returns the onset (or none), the peak, the final
level, and whether it recovered (mean above 0.8 × peak within 1,000 episodes); incidence over a panel
with Wilson intervals; Kaplan–Meier time-to-collapse censored at the horizon. Its sensitivity analysis
varies the trailing window (100, 200, 400), the fraction (0.4, 0.5, 0.6) and the persistence (200, 300,
500\) on the committed panels whose per-episode series exist, and reports how incidence moves. 092's own
rule (500-episode blocks, peak then below half) is computed beside it for continuity.

## Risks / Trade-offs

- **The re-runs may not reproduce the committed runs** (a dependency or platform change since
  2026-10-14). Then the identity check fails, the diagnosis runs on its own runs and says so, and their
  collapses are what the stacks read; the committed incidence is not re-read.
- **Collapse is stochastic in a deterministic run**: if a re-run reproduces its plateau but not its
  collapse, the identity check has already failed on the plateau, so this is caught.
- **Eight collapsed runs is a small sample** for six causes; the rule's 6-of-8 threshold is strict on
  purpose, and a `mixed` reading is a legitimate result that M.1(b) treats as "try the remedy list in
  D25's order".
