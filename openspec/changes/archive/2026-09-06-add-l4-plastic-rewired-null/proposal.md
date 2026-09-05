# Add the plastic degree-preserving rewired-null arm

## Why

The Phase 7 flagship is a 2×2 — learning rule × wiring — and its **primary contrast** is plastic wild-type against plastic rewired-null: under a rule the animal could plausibly host, does the specific wiring become load-bearing, or does a degree-matched scramble learn just as well? Logbook 034 answered that under PPO the wiring is inert. This arm is the other cell of the row the hypothesis is built on.

Every other piece is in place. The three-factor rule exists, is substrate-generic, and is bounded by its floors and its yardstick. The degree-preserving rewiring exists from Logbook 034, byte-identical-when-off. What has not existed is the combination, specified and proved: the null wiring, trained by the plastic rule, holding everything else constant.

## What Changes

### 1. One config, one key

`connectomeppo_small_continuous2d_combined_klinotaxis_plastic_rewired_null.yml` derives from the plastic wild-type arm by a single key, `wiring: rewired_degree_preserving` — exactly the delta the PPO rewired-null already uses against its own parent. No new mechanism: the rewiring is applied to the connectome *before* the topology is built, so the edge mask, the weight support, and therefore the eligibility traces and every plastic update are on the rewired edge set by construction.

### 2. A requirement stating what the two plastic arms share

The contrast is only interpretable if the arms differ in wiring and in nothing else. That is now stated and pinned rather than assumed. At one run seed, plastic wild-type and plastic rewired-null SHALL have byte-identical motor readout, sensory-projection gains, action-noise parameters, trace configuration, and plasticity hyperparameters; SHALL preserve every neuron's chemical in- and out-degree and gap-junction degree, and therefore each weight's initialisation scale; and SHALL differ in the edge mask, the placement of initial weights, and the gap-junction matrix. The rule's updates and the trace's support SHALL lie entirely on the rewired edge set.

The claims were established empirically before being written down. One candidate claim — that each neuron's initial incoming weight *energy* is preserved — is **false** and is deliberately not made: the same normal draws land on different pre/post pairs, so realised per-neuron sums differ even though the per-neuron scale (1/√in-degree) is identical. The requirement states the scale, which holds, not the energy, which does not.

### 3. Pairing by seed

The rewiring draws from a dedicated generator seeded, by default, from the run seed. Two consequences, both inherited from 034 and both relied on here: the brain's own initialisation stream advances identically in both arms (the synapse *count* is preserved, so the same number of draws occur), and seed *k* of the rewired arm pairs with seed *k* of the wild-type arm. The panel's paired-seed protocol needs nothing further.

### 4. Coverage

Parity and confinement tests against the real C3 plastic configs, a minimal-delta test, and a smoke entry so the arm runs end to end through the entry point before the panel depends on it.

## Impact

- **Affected specs**: `connectome-ppo-brain` gains the shared-substrate requirement for the plastic wiring arms.
- **Affected code**: none in the package — one config, tests, docs.
- **Default behaviour unchanged**: the arm is opt-in; `wiring: wild_type` and `learning_rule: ppo` remain the defaults.
- **Deferred, deliberately**: rewired-null versions of the two sanity floors. They are one-key configs and can be added when the panel's analysis plan is pre-registered, which is the right moment to decide whether "does rewiring move the untrained baseline?" is a question the panel will answer and to budget its extra runs. D10 names one rewired arm; this change ships that one.
- **Not in scope**: the panel (A.7), any run, any claim.
