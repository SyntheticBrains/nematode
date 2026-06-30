# Design: Minimal-RNN (minGRU / minLSTM) Policy Arms

## Context

The bit-memory positive control confirmed the comparison resolves working memory
([Logbook 030](../../../docs/experiments/logbooks/030-bit-memory-positive-control.md)), unblocking
the memory-axis candidate arms. minGRU / minLSTM ([Feng et al. 2024, arXiv:2410.01201](https://arxiv.org/abs/2410.01201))
are the cheapest Tier-1 candidate: minimal RNNs whose gates depend **only on the current input**
(not on the previous hidden state), so the recurrence reduces to a convex/affine state update that
is an associative scan — trainable in parallel and free of the saturating hidden-to-hidden matrix
that makes plain GRU/LSTM seed-fragile here.

The existing `lstmppo` arm ([`brain/arch/lstmppo.py`](../../../packages/quantum-nematode/quantumnematode/brain/arch/lstmppo.py))
is a full recurrent-PPO pipeline: a hidden-state rollout buffer, chunk-based truncated BPTT,
separate actor/critic optimizers, **both** discrete-categorical and continuous tanh-Gaussian heads,
entropy/LR scheduling, and weight persistence. Crucially it already contains `LayerNormRecurrent`,
a **hand-rolled drop-in recurrent cell** selected by config — proof that a custom cell is a
first-class citizen of this pipeline. The minimal RNNs are therefore a new *cell* plus two thin
registrations, not a new training stack.

The brain dispatch is registry-based (`@register_brain` → `instantiate_brain(brain_type.value, …)`),
and `lstmppo` consumes the **default** infrastructure-kwargs shape (`{num_actions, device}`), so
same-shape arms need no `brain_factory` branch and no manual `_registry` edit.

## Goals / Non-Goals

**Goals:**

- Add `mingruppo` and `minlstmppo` as registered classical PPO arms that reuse the `lstmppo`
  pipeline with **zero** duplication of the PPO/BPTT/buffer/persistence code.
- Implement one shared, numerically-stable minimal-RNN cell supporting both variants.
- Keep the existing `lstm` / `gru` / LayerNorm-cell behaviour **byte-identical** (the enabling
  refactor is a pure extract-method).
- Evaluate the arms on two prongs — the memory cell (separation vs the memoryless MLP) and the
  reactive cell (stability A/B vs `lstmppo`) — reusing the committed harnesses.

**Non-Goals:**

- No CUDA/associative parallel-scan kernel — we run the recurrence **sequentially** inside the
  existing short BPTT chunks (length 16); the parallel-scan property is exploited for *stability*,
  not throughput. A log-space parallel scan is a possible future optimisation, out of scope here.
- No new action head, no new PPO variant, no new dependency, no new statistics layer.
- Not modified-S5 (the other Tier-1 candidate) — a separate follow-on change.
- Not a re-opening of the frozen MUST set / Gate 3 — these are non-gating memory-axis arms.

## Decisions

### D1 — Reuse via subclass + an extract-method hook (not a standalone copy, not a config flag)

`MinGRUPPOBrain` / `MinLSTMPPOBrain` **subclass `LSTMPPOBrain`** and override only the
recurrent-core construction. To enable that, refactor the core construction + its couplings in
`LSTMPPOBrain.__init__` into overridable hooks:

- `_build_recurrent_core() -> nn.Module` — currently inline in `__init__` (the `nn.LSTM`/`nn.GRU` /
  `LayerNormRecurrent` selection at lstmppo.py:588-603). The extraction MUST also absorb the
  `self._is_gru = config.rnn_type == "gru"` assignment (line 589) **and** set a
  `self._recurrent_core_label` used by the init-log (replacing the direct `config.rnn_type`
  reference at line 744) — these are the two `config.rnn_type` reads in the base `__init__` that
  otherwise run regardless of extraction and would `AttributeError` if a subclass omitted the field.
  The base hook reproduces today's behaviour exactly (same `_is_gru`, same cell, same effective log).
- `_init_recurrent_weights()` — already a method; the subclass overrides it (see D4).

The `MinimalRNN`-building override **pins `self._is_gru = True`** (single recurrent state,
independent of any inherited `rnn_type`), sets the core label, and builds the cell — so the
inherited `_rnn_forward` / `_zero_hidden` / buffer paths take the single-state path unchanged
(see D3). After the refactor the base `__init__` reads `config.rnn_type` **only** inside
`_build_recurrent_core()`, which the override does not call.

- **Alternative A — a standalone `minimal_rnn_ppo.py` that re-implements the PPO loop:** rejected —
  hundreds of lines of duplicated, separately-maintained training code; the whole point is that the
  pipeline is identical.
- **Alternative B — extend `lstmppo`'s `rnn_type` Literal to `"mingru"/"minlstm"` with no new brain
  name:** rejected — the comparison framework identifies arms by **brain name** (`lstmppo`,
  `cfcppo`, …); a config variant under the `lstmppo` name would not be a distinct ranking arm and
  would muddy per-arm provenance. New registered names are the convention (every arm has its own
  capability spec).

### D2 — Two registered arms sharing one cell (not one arm with a variant flag)

Register `mingruppo` and `minlstmppo` as distinct arms (distinct `BrainType`, distinct config
class), both backed by a single `MinimalRNN(nn.Module)` cell parameterised by `variant`. This keeps
each arm a first-class, independently-rankable entry (matching `lstm` vs `gru`'s history) while the
genuinely-shared computation lives in one place.

- **Alternative — one `minimal-rnn` arm with `variant: mingru|minlstm`:** rejected for the same
  arm-identity reason as D1-B; one config flag would collapse two ranking arms into one name.

### D3 — Single recurrent state for both variants; reuse the `(h, c)` buffer with `c = None`

In the minimal forms **both** variants carry a single recurrent state (minLSTM drops the separate
cell state). So the cell behaves like the GRU path: with `self._is_gru` **pinned True** by the
override (D1), `_zero_hidden` returns `c = None`, the buffer stores `c_state = None`, and
`_rnn_forward` takes the GRU branch (`output, h_new = self.rnn(x_seq, h)`). The shared cell's
`forward(x_seq, h)` mirrors `LayerNormRecurrent`'s signature `(output_seq, h_new)` so the inherited
forward/BPTT code is reused without modification.

Recurrences (per step, `x` = LayerNorm'd features, all gates **input-only**):

- **minGRU:** `z = σ(W_z x)`, `h̃ = W_h x`, `h_t = (1 − z) ⊙ h_{t−1} + z ⊙ h̃`.
- **minLSTM:** `f = σ(W_f x)`, `i = σ(W_i x)`, `h̃ = W_h x`, normalise `f' = f/(f+i)`,
  `i' = i/(f+i)`, `h_t = f' ⊙ h_{t−1} + i' ⊙ h̃`.

Both updates are **convex** (`(1−z, z)` and `(f', i')` sum to 1), so the state stays within the
range of `h̃` — which is affine in the bounded LayerNorm'd input — keeping it bounded over arbitrary
episode lengths **without** a squashing nonlinearity. This is why a plain-space sequential
implementation is numerically safe here and the paper's log-space parallel scan is unnecessary at
chunk length 16.

### D4 — Override recurrent-weight init (the minimal RNN has no hidden-to-hidden matrix)

`lstmppo._init_recurrent_weights` orthogonalises `weight_hh` per gate block to fight
recurrent-state saturation. The minimal RNN has **no** `weight_hh` (gates are input-only), so that
pass is inapplicable; the subclass overrides `_init_recurrent_weights` to Xavier-init the input
projections (`W_z/W_f/W_i/W_h`) and to **bias the retention gate toward holding** (minGRU
`bias_z < 0`, minLSTM `bias_f > 0` / `bias_i < 0`) — a memory-friendly prior. **This bias is
load-bearing.** During a zero-input phase (e.g. the bit-memory delay, obs `[0, 0]`) the gate is
bias-only, so a zeroed bias gives `z = f' = 0.5` — a ~1-step retention half-life that washes out
any held signal over the delay. The eval confirmed this empirically: with zeroed biases both
minimal arms sit at chance (never learn the delayed-match task); with the hold bias they reach
~0.96, joining the LSTM/Transformer cluster. The absence of a saturating recurrent matrix is also
the structural reason these arms are a candidate stability fix for the 029 LSTM laggard.

### D5 — Inherit both action heads unchanged

`run_brain` / `_run_brain_continuous_step` / the BPTT re-scoring already branch on
`config.action_mode` and the shared `_policy` helpers (`continuous_sample_tanh_gaussian`,
`categorical_logprob_entropy_torch`, `ppo_clip_policy_loss`). The subclass inherits all of it; the
continuous (speed, turn) head used by the T7 substrate works with no new code.

### D6 — Two-prong evaluation, extending the committed harnesses' arm rosters

Both harnesses pin their arm sets in **hardcoded tuples**, so the new arms must be added there (the
shared statistics layer is unchanged):

- **Memory cell** — the merged bit-memory delayed-match-to-cue control
  (`scripts/analysis/bit_memory_separation.py`). Add `{mingruppo, minlstmppo}_small_bit_memory.yml`,
  **add both arms to `MEMORY_ARMS`** (currently `("lstmppo", "cfcppo", "transformerppo")`, line 42)
  and update the test that pins that set (`test_bit_memory_separation.py:57`). Confirm the minimal
  RNNs clear chance + beat the memoryless MLP, with LSTM / CfC as the yardstick.
- **Reactive-cell stability A/B** — the 029 continuous-2D klinotaxis C3 cell, paired-seed vs
  `lstmppo`. The live orchestrator is `scripts/analysis/t7_continuous_ranking.py` (it imports the
  shared paired-seed Wilcoxon + bootstrap + BH-FDR functions from
  `weight_search_architecture_ranking`); **add both arms to its `PPO_ARCHS`** (line 57), or run a
  bespoke pairwise A/B against `lstmppo` using the same shared statistics functions. Report return +
  seeds-converged vs the `lstmppo` baseline (the hypothesis is about seed-fragility, not just mean
  return).

### D7 — Configs subclass `LSTMPPOBrainConfig`; the plain-RNN fields are pinned, not honoured

`MinGRUPPOBrainConfig` / `MinLSTMPPOBrainConfig` **subclass `LSTMPPOBrainConfig`** to inherit the
full hyperparameter surface (hidden dim, BPTT chunk, LRs, PPO/entropy/LR-schedule knobs,
`sensory_modules` required) so the A/B is a like-for-like swap of only the recurrent cell. The
inherited `rnn_type` / `recurrent_layernorm` fields are **not honoured** by the minimal arms (the
override pins `_is_gru = True` and builds `MinimalRNN`); a `model_validator` on the minimal configs
**rejects a non-default value** of either field (the config loader repopulates every field from its
default, so a `model_fields_set` check is unreliable), so a stray `rnn_type: gru` fails loudly
instead of silently implying an inconsistent two-state path. This avoids both the `AttributeError`
(the fields exist, inherited) and the silent-misconfig footgun, with zero field duplication.

- **Alternative — re-declare the surface on a fresh `BrainConfig` subclass with `extra="forbid"`
  and no `rnn_type`:** a cleaner field set, but duplicates ~20 inherited field declarations;
  rejected in favour of the inherit-and-pin approach (the validator gives the same loud-failure
  guarantee at far lower cost).

## Risks / Trade-offs

- **Continuous-action evidence is thin** — the paper's wins are discrete-action sequence tasks; the
  continuous tanh-Gaussian gap is unvalidated. → Mitigation: the memory-cell prong validates the
  arm on the project's own continuous head before any leaderboard claim; a null there is a reported
  finding, not a silent gap.
- **Subclassing couples the arms to `lstmppo` internals** — a future `lstmppo` refactor could break
  them. → Mitigation: the coupling is one documented extract-method hook + a registry round-trip
  test; the shared pipeline is the intended single source of truth.
- **minLSTM may be near-redundant with minGRU** (both single-state, input-only gates). → Mitigation:
  the second arm is ~10 lines (one extra gate + normalisation); keeping both answers "does the
  two-gate update help?" cheaply, and minLSTM can be dropped if the eval shows no separation.
- **A null reactive-stability A/B** (minGRU no better than the laggard LSTM) is possible. → That is
  itself a reportable result; the memory-cell prong stands independently.
- **`evolution/encoders.py` `ENCODER_REGISTRY` omission** — the new arms are absent from the
  weight-evolution encoder map. → **Deliberate and non-blocking**: both evaluation prongs train via
  PPO through `setup_brain_model` (the registry), not the evolution path, and the existing recurrent
  PPO arms (`cfcppo`, `transformerppo`) are likewise absent from `ENCODER_REGISTRY`. Adding an
  encoder is only needed if these arms are later weight-evolved (a separate follow-on). Noted so a
  reviewer does not read it as a missed touchpoint.

## Migration Plan

Additive and behaviour-preserving. The new arms are opt-in via config (`name: mingruppo` /
`minlstmppo`); nothing runs them by default. The enabling `lstmppo` refactor is a pure
extract-method with the existing arm's behaviour asserted byte-identical by the current `lstmppo`
tests. Rollback = remove the module + the `BrainType` / export / config-union entries; no data or
schema migration.

## Open Questions

- **minLSTM gate normalisation guard** — `f/(f+i)` needs an `eps` floor for the (near-impossible
  with sigmoid, but defensive) `f+i→0` case; settle the exact form in implementation (a small
  additive `eps`, consistent with the paper's stable form).
- **Stability-A/B metric** — *resolved (D6)*: report both the 029 primary return metric and a
  seeds-converged summary, since the hypothesis is specifically about seed-fragility, not just mean
  return.
