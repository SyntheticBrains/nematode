# Design note: Chemosensory associative-memory probe (proposed next T7 naturalistic memory task)

**Status**: proposed (design sketch). Tracked as `T7.separation.associative_memory` in the Phase-6
tracker. Non-gating, own OpenSpec change when scoped.

**Motivation.** The bit-memory positive control ([Logbook 030](../experiments/logbooks/030-bit-memory-positive-control.md))
proved the architecture comparison **can** resolve working memory, but bit-memory is a deliberately
*artificial* cue. ARS-via-depletion — the intended *biological* twin — came back **null**
([Logbook 032](../experiments/logbooks/032-ars-source-depletion.md)): environmental depletion demands
only *short-horizon* integration, which the biological adaptive sensor already supplies, and a true
episode-scale version is an unlearnable blind search. That null is biologically correct (real ARS is
sensory + slow-neuromodulatory-state, not a remembered map), but it leaves the "capability exists,
natural demand limited" conclusion resting on a **single** naturalistic task. This probe adds a
**second, more faithful** naturalistic memory demand.

**Why associative memory.** Of the biologically-grounded candidates, chemosensory associative learning
is the closest real *C. elegans* "remember-and-use" behaviour: worms associate a gustatory/olfactory
cue (salt, or an AWC/AWA odorant such as butanone/isoamyl alcohol) with **food** or **starvation** and
shift approach↔avoidance accordingly (gustatory plasticity; Kunitomo/Iino, Torayama, and the classic
butanone-conditioning literature). Unlike ARS, the memory here is genuinely *used to guide current
action* across a delay — the same computational shape as bit-memory, but grounded in a documented worm
behaviour rather than an arbitrary channel.

**Task shape (within-episode DMTS on the chemosensory channel).** Reuse the bit-memory cue/delay/
response scaffolding, but make the cue a chemosensory identity and the target its learned valence:

1. **Cue/conditioning phase** — present cue A paired with food (reward) and cue B paired with nothing/
   aversive, in a short early window. The pairing (which cue predicts food) is randomised per episode,
   so it cannot be learned into the weights — it must be held in the recurrent **state**.
2. **Delay phase** — cues withheld, no aids (mirrors bit-memory's no-STAM/no-adaptive delay).
3. **Response phase** — present A and B as spatial options (or a go-signalled binary readout) and
   require approach-A / avoid-B on the *remembered* association.

Because the pairing is per-episode-random and held in activation state, this is a true within-episode
working-memory task: a memoryless MLP is pinned near chance, recurrent/attention arms can solve it —
the bit-memory prediction, on a biological substrate.

**What it tests / expected reads.** Per-arm plateau-tail response accuracy vs chance, n paired seeds,
the committed Wilcoxon/bootstrap/BH-FDR layer. Positive = the memory arms separate from the MLP on a
*naturalistic* remember-and-use task (strengthens "capability yes"); null = even a biological
associative demand doesn't separate at this fidelity (strengthens "reactive-dominated worm"). Either
way it is a second data point the ARS null cannot provide alone.

**Implementation sketch (≈ bit-memory-sized).** Extend the bit-memory task scaffolding: two cue
channels (or reuse the single `cue_signal` with a sign/identity), a per-episode random cue→valence
map, the food/aversive coupling in the reward, and the phase controller. Matched **entropy 0.2** and
the **Transformer as the reliable memory detector** (the recurrent arms' PPO-collapse confound from
032 persists). No env-physics changes; chemosensory channels already exist.

**Caveats / honesty.** Real worm associative learning forms over *minutes to repeated trials* (and
consolidates over hours); a single-shot within-episode pairing **compresses** that timescale, so this
is biologically-*inspired-and-engineered* (like bit-memory), not fully faithful. The faithful version
— association that forms slowly via neuromodulator-gated plasticity, across trials/episodes — is
**phase-7** (it is exactly the L4 neuromodulated-STDP deliverable applied to a learning task; see
`docs/roadmap.md` § Phase 7 and § Known Gaps).

**Sequencing.** Proposed next T7 naturalistic memory probe after the ARS logbook; thermal
reference-memory (cultivation-temperature DMTS) is the backup if the associative framing proves
awkward to isolate. Faithful slow-forming memory is deferred to phase 7.
