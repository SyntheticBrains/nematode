# Panel 2: the Hebbian wiring contrast, the prior over policies, and count-scaled initialisation

## Why

Logbook 040 resolved the 7a-i panel to `sanity_floor_fail` and left one signal on the table. The
reward-modulated arm beat none of its contrasts, but the *reward-free* unmodulated-Hebbian floor
on the wild-type wiring reached 78%, 64% and 67% on three of eight seeds, and beat the same floor
on the degree-preserving rewired-null by +16.5 points on five of eight seeds while the frozen
floors tied (+0.4). That is the wiring claim the project exists to make — that the specific
*C. elegans* wiring is legible to a local Hebbian process where a degree-matched scramble is not —
and it was descriptive, uncorrected, and one panel deep.

The same panel showed *why* it matters: outcomes on this cell are fixed points seeded by the
random initial weights. The frozen wild-type arm alone spans 0.4% to 36.9% across seeds on
identical wiring. We never characterised that landscape — the *prior over policies* a wiring
imposes on random initialisations — and we never used the one piece of wiring data the substrate
ignores: Cook 2019's per-edge synapse counts (1–75, median 3; a third of edges single-synapse),
which today never reach a weight. Every plasticity result so far has been read against a floor
whose distribution we do not know, drawn from an initialisation that discards the counts.

Ratified with Chris (2026-09-06/07) as the next item ahead of the imitation warm start and 7a-ii:
a second panel that (1) tests the Hebbian wiring contrast as a registered primary at a sample size
that can carry it, (2) measures the prior over policies on both wirings over many seeds, and
(3) adds initialisation as a factor — random against synapse-count-scaled magnitudes (linear in
count, the direct physical reading: each contact adds conductance), with signs still random and
each neuron's incoming norm held at the degree-scaled expectation so homeostasis targets stay
comparable and only the structure *within* a neuron's inputs changes.

It is cheap: the Hebbian arms settle within a few hundred episodes and the frozen arms are
constant policies, so the whole design runs in two to three hours on sixteen workers.

## What Changes

- **Count-scaled initialisation** on the connectome brain: `weight_init`, `degree_scaled`
  (default, byte-identical) or `count_scaled`. Under the rewiring the count travels with its
  edge's pre-synaptic endpoint, so the rewired-null is a null of the count structure too.
- **Four configs**: count-initialised versions of the wild-type and rewired frozen and Hebbian
  arms, each one `weight_init` key off its parent.
- **The registration**: eight arms (wiring × initialisation × {frozen, Hebbian}); Hebbian arms on
  seeds 1–16 at 1000 episodes (seeds 1–8 reproducing panel 1's episode streams); a **prior sweep** of the four
  frozen arms on seeds 1–64 at 600 episodes; a four-test BH-FDR family with the wild-type over
  rewired Hebbian contrast under random initialisation as the primary; Logbook 034's verdict map; a launch record before the run; one bounded extension; no pilot,
  since every value the arms run with is panel 1's pin.
- **A harness** (`scripts/analysis/l4_panel2.py`) reusing panel 1's readers and the committed
  statistics layer, adding the prior-sweep analysis (per-arm distributions and the fraction of
  initialisations that are competent with no learning).
- The run, its records under `supporting/041-l4-panel2/`, tests, docs.

Out of scope: the plastic (reward-modulated) arms and the MLP, which return once the prior is
known; any change to the rule; the logbook (the next tracker item).

## Capabilities

**Modified**: `connectome-ppo-brain` — gains the count-scaled initialisation option.
**Modified**: `l4-plasticity-panel` — gains the panel-2 protocol as a further requirement.

## Impact

- Edited: `brain/arch/connectome_ppo.py` (initialisation, config field), four configs, a new
  analysis script and its tests, the config variant tests, `docs/architectures.md`,
  `configs/README.md`, `CHANGELOG.md`.
- Default builds are byte-identical; panel 1's arms and results are untouched.
