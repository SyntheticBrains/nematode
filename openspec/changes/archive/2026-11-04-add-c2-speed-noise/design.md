## Context

Each second, the connectome brain's continuous action is `pre = mean + std·ε`, with ε independent N(0, 1) per output, squashed by tanh into each output's range. Under the primitives action space the outputs are run speed, run curvature and three bout hazards.

The kinematics family grades 10-s window speed medians and 90th percentiles by state. Its bands are narrow: the dwelling median's scale is 0.000727 mm/s, about 6% of its centre.

On the stage-2 answer, the speed output's noise has log-SD +0.116, an SD of 1.12 before the squash. Setting it to −4 leaves every window statistic in place (`supporting/123-c2-speed-noise/diagnosis/stage2_answer_variants.json`, variants A and B). The window statistics are set by the mean's slow course alone.

The existing paths (see the implementation map in this change's tasks):

- the per-step runner samples in `ConnectomePPOBrain._run_brain_continuous` through `continuous_sample_tanh_gaussian`, and stores the pre-squash sample and log-prob for PPO;
- `learning_rules/ppo.py`'s `_evaluate` re-evaluates through `continuous_evaluate_tanh_gaussian`;
- the batched evaluator samples in `evaluation/controllers.py::_sample`, one generator per animal.

## Goals / Non-Goals

**Goals:**

- Speed variation that persists across seconds, at a timescale the fit chooses.
- The same process in both runners, byte-identical when off.
- Unbiased PPO gradients under it.

**Non-Goals:**

- Slow noise on curvature or the hazards.
- Slow noise for the plain connectome brain or the MLP. If C.3 compares architectures under PPO, their noise models are its decision.
- Learning τ under PPO.

## Decisions

### D1. An OU process on the standardised speed noise, reusing each second's draw

x_t = ρ·x\_{t−1} + √(1−ρ²)·ε_t, with ρ = exp(−Δ/τ), where Δ is the configured step (the encoder already refuses a modulator step that differs from it). At an episode's start, x_0 = ε_0. The stationary variance stays 1, so `noise.speed` keeps its meaning as the marginal SD. The draw ε_t is the one the independent model takes, so the random streams, and every other output, are unchanged.

The squash makes the effect depend on state without another parameter. Dwelling sits where tanh is flat near zero speed, so slow noise moves it little; roaming sits where tanh is close to linear.

The alternatives were weighed and declined:

- **A per-state window redraw, as C.1 has.** C.2 has no explicit state, and C.1's redraw is aligned to the grading's windows.
- **A per-second std head.** The diagnosis shows it cannot reach the window statistics.

### D2. The timescale is the encoder's last parameter, log-scaled over 0.25–300 s

At the 1-s step, ρ runs from 0.018 (the independent model, nested) to 0.997. The log-centre is 8.7 s (ρ = 0.891), and 300 s stays below the shortest assay phase.

The entry `timescale.speed_noise` is appended after every existing entry: index 196, or 132 under per-ligand tying. Every stored vector is therefore a prefix of a slow-noise vector. With `encoder_speed_noise: none`, the entry does not exist.

### D3. The exact conditional likelihood under PPO

The speed dimension's pre-squash density, given the history, is N(m_θ(s_t) + ρ·(u\_{t−1} − m_θ(s\_{t−1})), σ_θ²(1−ρ²)), where u\_{t−1} is the previous pre-squash speed.

The rollout buffer stores u\_{t−1}, the previous input s\_{t−1}, and a flag for whether there was a previous step. The update recomputes m_θ(s\_{t−1}) with one extra batched forward of the stored previous inputs; settling carries no state between steps, so the forward is exact. An episode's first step uses N(m_θ(s_t), σ_θ²).

Conditioning instead on the rollout's stored x\_{t−1} would be simpler, but at ρ ≈ 0.9 its gradient drops −ρ·∇m_θ(s\_{t−1}) and is biased. The maintainer chose the exact form.

The other four outputs keep their independent density. The tanh correction is unchanged, since it depends on the sample alone.

### D4. τ is held under PPO

The encoder already holds the two modulator timescales as buffers. Under PPO the entropy term log(σ√(1−ρ²)) would push ρ toward zero. And a learned τ would change what each stored previous step means. τ is a fidelity parameter fitted by CMA, and C.3 takes it from a fitted or registered value.

### D5. The noise state's lifetime

The noise state resets with the episode: at `prepare_episode` in the per-step runner, and at `reset(n)` in `BatchedConnectome`. It carries across an assay's phases in both, as the modulator concentrations do.

### D6. Refusals

`encoder_speed_noise: ou` requires:

- `encoder: compact`;
- continuous primitives;
- `continuous_std_mode: state_independent`, which the encoder already fixes;
- `dynamics: settling`, since chunked recurrent replay would not carry the state;
- no activity traces, since their e-prop score assumes independent noise.

Each is refused in the validator, and again at construction through one shared refusal, since `model_copy` skips validators. Weight files record the mode, and a file from the other mode is refused, as a mismatched std mode is.

### D7. The component's positive control

`c2_encoder_probe.py speed-noise` grades the stage-2 answer on its own wiring at τ ∈ {0.25, 2, 8.7, 30, 100, 300} s, on 200 animals of seed block 88,100,000.

- **The bar:** the roaming speed's 90th percentile rises by at least 3 band scales (0.0122 mm/s) above its value at τ = 0.25, at some τ. The miss is a shortfall, so the bar requires a rise, not any movement. The dwelling speed's 90th percentile at the same τ is reported beside it; the squash should make it rise less.
- **The check:** at τ = 0.25, every kinematics statistic equals the independent model's within twice its band scale.
  - The comparison is paired: the same seeds, with ρ = 0.018 against 0.
  - The 2-scale tolerance sits above the simulator's own noise. On the same seeds, a far larger noise change (diagnosis variants A against B, a speed log-SD of +0.116 against −4) moved every kinematics statistic by at most 0.62 scales.

The sweep shows the mechanism reaches the window statistics. It is not a fit.

**Outcome (2026-11-04, Logbook 123).** On the stage-2 answer the control failed: no timescale moved the roaming speed's 90th percentile by more than 0.06 band scales. That answer's speed output is pinned at zero speed (pre-squash median −3.9) and its curvature at the maximum (+9.5); its worm moves only by reversals and delta turns, so no speed noise reaches it. A descriptive check with both outputs moved mid-range showed the mechanism does reach the window statistics. The control is therefore re-run on the first fit whose speed output responds, before the readiness restarts rely on slow noise, and the requirement says so.

### D8. Parity and identity

- With slow noise off, `connectome_identity_check.py` must show every job identical. `BatchedMLP`'s sampling path is untouched.
- With it on, the batched-evaluation layered parity holds for the encoded wild type on Scheer's small lawn, and an animal alone matches the same animal in a batch.
- A run that crosses an assay phase start matches.
- Negative controls must fail: a controller that resets at the phase, and one with ρ = 0.

## Risks / Trade-offs

- **Gradient scale.** The speed score grows by about 1/(1−ρ²): 4.9× at the centre, 150× near 300 s. PPO's clip and KL stop will act more often. That is C.3's tuning concern, not a fault in the likelihood.
- **The critic does not see the noise state,** which adds advantage variance at long τ.
- **Memory.** The buffer stores two inputs per transition with slow noise on.
- **Slow noise comes from outside the network.** Every W arm gets the same handle on the kinematics family, so that family says less about wiring than it would if the network produced the variation itself. Without it, every arm fails the kinematics alike, which masks a wiring effect anyway. W's registration reads the kinematics family with this in mind; the tracker records it.
- **The probe must thread the option.** `c2_encoder_probe.entries()` must pass the option before any slow-noise stage; otherwise `write_encoder` refuses the vector's length.

## Migration Plan

None: off by default, byte-identical. The readiness restarts (the next change) turn it on.
