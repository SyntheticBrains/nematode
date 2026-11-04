## Context

- **The connectome brain today** (`brain/arch/connectome_ppo.py`; batched as `BatchedConnectome`). Its free values:

  - about 3,709 chemical weights, drawn at random. Even `count_scaled` multiplies counts by a random |z| and sign, and edges without a grounded sign keep a random sign;
  - 18 per-neuron food gains onto ASE, AWC and AWA;
  - a 5×4 pooled readout with no bias;
  - 5 action log-SDs;
  - the data layer's 104 signed and 17 fixed-sign strengths.

  Timescales are config values (60 s by default). Food contact and satiety never reach the network. The only slow state is the 55 ligand pools.

- **What reads what:** `BatchedConnectome` reads the brain's tensors live on each step (`food_gains`, `readout`, `log_std`, `w_chem`, `g_gap`, the modulator strengths and `_modulator_decay`). Writing a vector in place therefore takes effect without rebuilding. Knockout maps and the build-time weight scales are fixed when the brain is built.

- **The data:**

  - Emmons 2024 (the configs' source) carries EM serial-section synapse counts;
  - `synapse_signs: per_connection` grounds 3,176 edges (2,634 positive, 542 negative);
  - the 533 others come from presynaptic neurons releasing dopamine, serotonin or octopamine, or with no known transmitter;
  - the data layer tags each ligand `monoamine` or `peptide` (4 and 51).

- **Cost** (Logbook 104, and a timing this planning made on discarded seeds): about 6.5 µs per animal-step on one core at n = 200. C.2 is fitted on D_fit (the small lawn, the uniform well and kinematics), which needs two assays: about 1.44 M animal-steps, roughly 10–13 s per 200-animal evaluation.

- **Lessons carried from C.1** (Logbook 121):

  - check a model against what the benchmark grades before registering a fit;
  - tighten the plateau's guard, since a median near 507 stopped a restart;
  - keep seed blocks at least 2 M animals apart;
  - about 1 box-centre restart in 3 found C.1's passing basin.

- **W, which this encoder must serve** (D40). It fits the wild type and its nulls at equal budget, with a fresh rewiring draw per fit and 8–16 fits per arm. The nulls are degree-preserving, gap-only on the corrected gap model, boundary-held, class-preserving left–right-symmetric, and the receptor-placement null. W reads held-out fidelity and mutant phenocopy against a minimum of the real between-cohort distance: 2.97 on the fitted families and 2.05 on the held-out ones, on v3 (Logbook 120). W is the phase's longest run, so everything it inherits from the encoder is fixed here:

  - **the gap model.** C.2's configs today use counts with the degree normaliser, the model Phase 9 traced its wiring artefact to (Logbook 103). Strength normalisation is the maintainer's choice for every arm;
  - **how a null is encoded.** Rewiring moves each edge's count with it (`connectome/rewiring.py` 317–356), and each null edge takes its origin's per-connection sign (`_per_connection_edge_signs`);
  - **the fit-to-fit spread of held-out predictions,** which sizes W's draws per arm;
  - **the cost per fit and per generation;**
  - **the compute path.**

## Goals / Non-Goals

**Goals:**

- C.2's encoder, about 200 named parameters with a unit box, in both the per-step runner and the batched evaluator;
- every existing path unchanged;
- the evidence F.0b-ii needs before its registration: cost, slow-state feasibility, and a readiness probe whose answers can be its truths.

**Non-Goals:**

- registering F.0b-ii or C.2, and fitting C.2;
- W's nulls. Each existing null must build under the encoder, which this change tests, but the nulls themselves, and the class-preserving left–right-symmetric null (not yet built), are W's;
- per-synapse or per-class-pair synaptic fitting.

## Decisions

01. **The switch:** `encoder: compact` on `ConnectomePPOBrainConfig`, default `none`.

    - **Unset,** no tensor, draw or output changes.

    - **Set,** the brain refuses unless all of these hold:

      - `dynamics: settling`;
      - `sensing_mode: klinotaxis`;
      - `modulators: true` with `modulator_layer: data`;
      - `modulator_satiety_link: true`;
      - `synapse_signs: per_connection`;
      - `gap_normalisation: strength` with additive coupling;
      - `weight_prior` unset;
      - no predator or thermotaxis projection;
      - the configured step and `modulator_step_seconds` equal.

      That last check guards the 5-s default.

02. **The fixed wiring.**

    - **Chemical weights:**
      - an edge i → j with grounded sign s has weight θ_s · s · c_ij / √(Σ_k c_kj²), where c is its synapse count and the sum runs over j's inputs. This is `count_scaled`'s normalisation without the random draw;
      - θ_exc scales the positive edges and θ_inh the negative ones;
      - an unsigned edge from a presynaptic neuron of transmitter group g has weight γ_g · c_ij / √(Σ_k c_kj²), with g ∈ {dopamine, serotonin, octopamine, none} (`atlas` transmitters).
      - The four groups cover every unsigned edge (checked in planning, `read_atlas_transmitters`): no known transmitter 317 edges from 20 neurons, dopamine 163 from 8, octopamine 42 from 2, serotonin 11 from 2.
    - **Gap junctions:**
      - each pair's count divided by √(s_i·s_j), with s a neuron's summed pair counts (`gap_normalisation: strength`, additive coupling), × θ_gap. Every coupling is at most θ_gap, and the matrix's spectral radius is θ_gap;
      - this is the corrected model Phase 9 left (Logbook 103: no wiring effect remains under it). It replaces the degree normaliser that produced the artefact;
      - additive coupling needs no derived inverse, so a write is one scaling.
    - **Determinism:** the weights depend on the data and the vector alone, not the brain's seed.
    - **The composition** is recomputed on every forward, not cached. It is six elementwise passes over the 302 × 302 matrix, under 1% of a settling step's products at the batch sizes the fits use, and a cache keyed on writes would go stale on a write through `.data`, which torch's version counter does not see. The chemical and gap terms are summed into one matrix first, so each settling step is one product instead of two. The composition is written as functions of a parameter tensor that also accept a leading candidate axis, so a whole-generation batched path can be built for W without redesigning the encoder.

03. **Sensory inputs,** replacing `food_gains` when the encoder is on.

    - **The classes:** the 11 amphid chemosensory classes AWC, AWA, ASE, AWB, ASK, ADF, ASI, ASJ, ASG, ADL and ASH.
    - **Their features:**
      - concentration and dC/dt (tanh, as today), each with a gain per class, shared by the class's left and right neurons;
      - lateral (right minus left), with an antisymmetric gain per class: the right neuron gets +g·lateral and the left −g·lateral. That is 33 gains.
    - **Food contact** (the lawn density under the worm, in [0, 1]) goes into CEP (4 neurons), ADE and PDE with one gain each, the dopaminergic mechanosensors of bacteria behind basal slowing (Sawin et al. 2000). That is 3 gains.
    - **Satiety** enters through the existing satiety link on the ingestion drive onto NSM, which adds no parameter. Within an assay it barely moves, since it is a 12-h reserve.
    - **The per-step runner** feeds food contact from the same sense the `food_contact` module reports.

04. **The readout:** the 5×4 pooled readout (VB, DB, VA and DA means) plus 5 biases, so logits = pooled · readoutᵀ + bias. The 5 log-SDs are fitted.

05. **Neuromodulation:**

    - **The strengths:** the 104 signed strengths are written directly, and the 17 fixed-sign ones as strengths, with ρ = softplus⁻¹(s).
    - **Their tying,** `encoder_modulator_tying`:
      - `per_couple` (the default) fits every couple: 121, so 196 parameters in all;
      - `per_ligand` fits one signed gain per fitted ligand (52) and one strength per fixed-sign ligand (4), keeping each couple's fixed sign, plus the drive: 57, so **132** parameters;
      - D_fit has 14 statistics, so most couples are weakly constrained. The readiness restarts measure what that costs W (decision 11), and F.0b-ii's registration picks the tying from that evidence.
    - **The timescales:** τ_mono for dopamine, octopamine, serotonin and tyramine, and τ_pep for the 51 peptides. Decay is exp(−step / τ).

06. **The unit box** (`connectome/encoder.py`). Its centre is C.2's x₀, set without search data (D38).

    | Block | Range | Centre |
    |---|---|---|
    | θ_exc, θ_inh, θ_gap | [0, 1.5] *(first [0, 3])* | 0.75 *(first 1.5)* |
    | γ (4) | [−2, 2] | 0 |
    | sensory gains (36) | [−5, 5] | 0 |
    | readout (20), bias (5) | [−3, 3] | 0 |
    | log-SD (5) | [−4, 1] | −1.5 |
    | signed strengths (104) | [−3, 3] | 0 |
    | fixed-sign strengths (17) | log [0.001, 3] *(first linear)* | 0.055 *(first 1.5)* |
    | τ_mono, τ_pep | log [10, 3000] s | 173 s |

    The ranges are provisional. The readiness probe reports every parameter its answer leaves at a bound, and F.0b-ii's registration fixes the box.

    *(Amended 2026-11-03, the maintainer's decision.)* **The wiring scales are rescaled to [0, 1.5].** The first box's restart 0 stalled: its answer's ranking score sat at about 10.6 (D_fit about 5.8) from generation 50 to 325, with σ flat and the worm off the small lawn (time on the lawn z −116). An operating-point scan by forward passes alone, with no benchmark data, found the first box's centre (1.5) saturating 56–62% of neurons, and its answer's settled network switch-like (one unit of any sense moving the motor pools across the whole tanh range). Scales of 0.5–0.75 leave the network graded (0–7% saturated). Every other range is unchanged. A 150-generation pilot in the new box must beat the stall's 10.6 before the readiness restarts run, and each restart then runs through the cap with its answer re-simulated every 50 generations (decision 11).

    *(Amended 2026-11-04, the maintainer's decision.)* **The pilot in the new box did not beat the stall** (best checkpoint 10.62, the worm on the small lawn 23% of the time), so the readiness restarts did not run. The ranking score compresses each statistic as 10·asinh(z/10), whose slope at z ≈ 120 is 0.09, so progress on the dominant failure is barely rewarded. A second pilot ranks candidates on definedness first (1,000,000 × the statistics undefined in one record) and then the uncapped weighted RMS of z, scoring its checkpoints on both. It passes when its best checkpoint by the registered ranking score beats 10.6 with the worm on the lawn more than the stall's 26%; the readiness restarts then run under that objective, and F.0b-ii recovers under the same one. x₀ stays at the box centre, so every arm of W keeps the same start. A hand-set, biologically designed start was considered and kept in reserve: a start tuned to routes in the wild type's wiring would favour the wild type over its nulls in W.

    *(Amended 2026-11-04, the maintainer's decision.)* **The second pilot failed too** (best 11.01; the worm on the lawn 21–29% at every checkpoint), so the stall is neither the score's compression nor the box's saturation alone. A reachability probe asks whether any vector of the encoder holds the lawn: it fits only the lawn-holding subset (the 3 wiring scales, the 3 food-contact gains, the speed and reversal readout rows and biases, and their two log-SDs; 18 parameters), every other parameter at the box centre, on the uncompressed score for 150 generations. The worm holds the lawn when the best checkpoint's median time on it reaches 0.75, a bar fixed before the probe runs. If it does, the readiness design becomes a staged fit (this subset first, then all 196 from its answer), the same procedure for every arm of W, so x₀ stays the box centre. If it does not, the encoder's sensory-to-motor path needs a design change, which goes to review.

    *(Amended 2026-11-04, the maintainer's decision.)* **The reachability probe did not hold the lawn** (0.18), but the diagnosis found why. Forward passes show food contact can move the speed and reversal outputs strongly (90th percentile 1.9 pre-squash), and the settings where contact slows the worm most hold Scheer's small lawn once the 17 fixed-sign modulator strengths sit at their floor: 0.92 of the assay on it with 0.35 exits per on-lawn hour (real 0.98 and 0.71), against 0.15 at the box centre's 1.5. On food those pools inject currents held through every settling step, while the sensory input is only the initial state, so the centre's strengths swamp the senses. **The fixed-sign strengths, the ingestion drive among them, are now read on a log scale over the same [0.001, 3], centred at 0.055.** A third pilot on the registered ranking score, with the same pass rule, gates the readiness restarts.

    *(Amended 2026-11-04, the maintainer's decision.)* **The third pilot failed** (best 10.69; the worm on the lawn 22–26%), yet random settings of the lawn subset at the new box's centre hold the lawn (0.93 of the assay on it, 0.27 exits per on-lawn hour): the encoder expresses lawn-holding near its start, and the full fit does not find it. **The fit becomes staged.** Stage 1 fits the 18-parameter lawn subset from the box centre on the uncompressed score and must hold the lawn (its final checkpoint's median time on it at least 0.75). Stage 2 fits every parameter from stage 1's answer at σ₀ 0.1 on the registered ranking score. It is one procedure, the same for every arm of W, with x₀ the box centre. A staged pilot (150 generations per stage) under the same pass rule gates the readiness restarts, and F.0b-ii would recover under the same two stages.

    *(Amended 2026-11-04, the maintainer's decision.)* **Stage 1 reached the best answer of any fit** (registered score 10.0, below the stall's 10.6; 0.34 of the assay on the lawn) **without holding the lawn at 0.75**, so stage 2 did not run under the stated rule. The lawn-holders found by random draws never roam: the best (0.925 on the lawn) leaves 7 of D_fit's 14 statistics undefined. The score needs lawn-holding together with roaming and dwelling bouts, and the encoder has no stochastic source of state switching beyond the action noise after its readout. Stage 2 runs from stage 1's answer regardless, recorded as a deviation from the stated rule, and this change then closes with the switching question taken to a plan review before F.0b-ii.

    *(Amended 2026-11-04, the maintainer's decision.)* **Stage 2 held the lawn.** From stage 1's answer, the full fit reached a registered score of 4.1 by generation 125 (D_fit 4.4) with the worm on the lawn 99% of the assay, roaming 15% and dwelling in bouts, every statistic defined: the encoder switches between roaming and dwelling in closed loop, and the switching question is answered. What misses is kinematics (the dwelling speeds high, the roaming speeds low) and the roaming bouts' length. Stage 2 is resumed to 500 generations, its answer read every 25 generations, to see where the staged fit converges; this change then closes, and the staged procedure and the readiness restarts go to the next change.

07. **The vector interface:**

    - `encoder_parameters()` gives the 196 names in a fixed order;
    - `encoder_entries()` gives each name's block, range and kind;
    - `write_encoder(brain, z)` writes a unit-box vector in place, under `no_grad`;
    - `read_encoder(brain)` returns it.

    A write followed by a read round-trips to float32 precision (the brain's tensors are float32). Under PPO, the policy's learnable parameters are the encoder's tensors and nothing else, its two timescales held, and the PPO rule's critic is unchanged. That composition is differentiable, so C.3 trains the same architecture.

    **Brain reuse:** a fit's worker builds one brain per wiring and writes each candidate's vector into it, instead of rebuilding (0.7–1 s of a roughly 13-s simulation). A changed wiring or mutant always rebuilds, as W's fresh draw per fit requires. A test checks that a reused brain's actions equal a fresh build's.

08. **The mutant encodings,** which bind the registered table's C.2 row:

    - **tph-1:** `modulator_knockout_genes: [tph-1]`;
    - **pdfr-1:** `[pdfr-1]`;
    - **tax-4:** the 21 sensory gains of the 7 tax-4 classes among the encoded inputs (AWC, AWB, ASE, ASG, ASI, ASJ, ASK) are zeroed. AFD, URX, AQR, PQR and BAG receive no encoded input, so nothing changes for them. AWA, ADF, ADL and ASH keep theirs.

    A mutant brain is built with its knockout genes, and the wild type's vector is written into it.

09. **Parity and identity:**

    - **Parity:** the encoded brain meets the batched evaluator's layered parity with the per-step runner (state within 1e-5 over 50 steps; bouts and trajectories as registered), on seeds 5001–5032.
    - **Identity:**
      - a baseline is recorded before any edit on the hand-placed and data-layer connectome arms: batched runs on discarded seeds, all four v3 assays. The data-layer arm is built from the modulated assay configs with a `modulator_layer: data` override, since no data-layer assay config exists yet;
      - afterwards every record must be byte-identical with the encoder unset;
      - the existing parity tests also pass unchanged.
    - **The nulls:**
      - the encoder builds on every existing null wiring (`rewired_degree_preserving`, `rewired_chemical_only`, `rewired_gap_junctions_held`, `rewired_boundary_held`, `rewired_gap_junctions_only`, the gap-outlier nulls) and under `modulator_layer_null: receptor_placement`, with the same 196 (or 132) names;
      - on each, a null edge carries its own count and its origin's sign, an unsigned edge's group follows its presynaptic neuron, and every normalisation is recomputed on the null's own counts;
      - the mutant encodings build on a null too;
      - layered parity also holds on one null (`rewired_degree_preserving`).
    - **What differs between the wild type and a null after fitting:** only edge placement (and receptor placement, under the modulator null). Every parameter's name, range, start, σ₀, cap and stop rule is the same.

10. **Configs:**

    - data-layer primitives assay configs for v3's four assays, carrying `encoder: compact`, `synapse_signs: per_connection`, `modulator_layer: data` and `modulator_satiety_link: true`;
    - `generate_assay_configs.py` gains an `encoded` arm, the data-layer arm with the encoder (`generate_lawn_configs.py`'s `..._lawns_encoded_s1_primitives`), on every assay protocol so the generator's per-arm checks stay uniform;
    - `benchmark_grade.ARMS` gains `connectome_c2`.

11. **The probes** (description, discarded seeds; `scripts/analysis/c2_encoder_probe.py`):

    - **Cost pilot:**
      - one generation's pace, 16 simulations on 16 workers, for D_fit's two assays and for all four, with brain reuse;
      - **profiled:** settling, the per-animal noise loop, record conversion, measurement and grading. The byte-identical speed-ups worth making before W are named from it, as Logbook 118 did for C.1;
      - **a compute-path benchmark:** the encoded settling step at a whole generation's 3,200 animals, in one process on CPU and on MPS. An MPS path would not be byte-identical with the CPU's, so it is W's option only, taken for every arm;
      - its projections for F.0b-ii, C.2 and W come with the measured plateaus and spread below.
    - **Feasibility of a ~9-minute decay** (the roadmap's risk row):
      - a hand-set vector (τ_pep 540 s, one peptide coupling from a producer that is active on food onto the motor pools) runs López-Cruz's protocol;
      - the post-removal onset rate's decay constant, read with the benchmark's own `fit_local_search` (τ = 1/γ), must lie within 6–12 min;
      - it reads no López-Cruz data, since local search is held out for C.2.
    - **Readiness probe:**
      - **up to 3 sep-CMA restarts** (CMA 5980–5982) from the box centre, σ₀ 0.2, population 16, at `per_couple` tying. They run in turn and stop once 2 have passed. One restart is not enough: about 1 in 3 found C.1's passing basin, and F.0b-ii needs at least 2 truths. If only one passes, a warm-start existence test runs from it, as C.1's did;
      - the ranking score against v3's band centres on D_fit's families;
      - the plateau rule once σ ≤ σ₀ and **every median in the last 40 generations is below 100**, so that more than half of each generation's candidates leave no statistic undefined (the guard Logbook 121 asked for);
      - a cap of 1,000 generations;
      - its answer is graded with presence on 5 populations on all five families;
      - reported: which families pass, the parameters at a bound, the reversal share, and the slow state's fitted timescales;
      - ready means an answer passes D_fit's three families in at least 3 of 5. The passing answers become F.0b-ii's truths. Local search and chemotaxis are held-out predictions, reported, not required;
      - **the spread W needs:** across the restarts that plateau, the restart-to-restart distance in the held-out families (local search and chemotaxis) and in the three mutants' differences from their wild type, on the grading populations. Set against W's minimum, it gives W's draws per arm, and whether `per_ligand` tying is needed.

12. **Seeds** (all discarded):

    | Seeds | Use |
    |---|---|
    | 40,000,000 + 200·k + i, k = 20,000·r + 16·(n − 1) + m, r < 3 (to 51,199,999); CMA 5980–5982 | the readiness restarts (the 20,000 stride clears the cap's 15,999) |
    | 52,000,000 + 200·(5r + g) + i, g < 5 | their grading, mutants included |
    | 55,000,000 + 200·(16·(n − 1) + m) + i; CMA 5983 | the warm restart, run only if exactly one of the three passes |
    | 53,000,000 | the cost pilot |
    | 54,000,000 | the feasibility check |
    | 54,100,000 + 200·t + i, t < 20 | the feasibility trials that set the hand-set vector's magnitudes |
    | 59,000,000 | the probe's smoke |
    | 5001–5032 | parity, as registered |

## Risks / Trade-offs

- **[196 parameters is a slow fit, and weakly constrained by 14 statistics]** → The readiness restarts measure both the plateaus and the spread of held-out predictions. `per_ligand` tying (132) is built and ready. F.0b-ii picks with that evidence, and W is sized from measurements, not guesses.
- **[W inherits a flaw found too late]** → This change fixes the gap model, null encoding, compute path, spread and cost before F.0b-ii, so nothing W depends on is decided during W.
- **[The encoder favours the wild type]** → Class-tied gains and fixed data weights are defined on neuron identities. The nulls rewire edges, carrying counts and origin signs, and keep identities. Only placement differs after fitting. The class-preserving left–right-symmetric null is W's to build.
- **[The amphid class list and food-contact targets are a choice]** → They are stated with their sources, and tax-4's zeroing follows from the registered table.
- **[Antisymmetric lateral assumes a head-sweep comparison]** → That is the klinotaxis sense the brain already uses: one scalar, right minus left.
- **[Food contact adds an input C.3's per-step runner must supply]** → The runner already reports it, and parity covers it.
- **[The feasibility check could fail]** → Then the claim is scoped to the faster assays, as the roadmap's risk row states.
