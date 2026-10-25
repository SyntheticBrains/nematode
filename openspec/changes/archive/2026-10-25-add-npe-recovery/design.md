## Context

- **The batched evaluator** gives each controller `BatchedSenses` each second: concentration, `lateral` (the head-sweep difference, scaled by `lateral_scale`), `derivative` (dC/dt, scaled by `derivative_scale`), `food_contact` (lawn density under the worm), `satiety_fraction`, `intake` (this step's, against `max_intake`), and `locomotor`. Controllers draw randomness from per-animal generators (`noise`).
- **The grader** (`benchmark_grade.grade_populations`) returns the 11 wild-type statistics, local search's presence, the per-family passes, and D and D_fit. B.4 timed it: about 12 s for a hazard controller on three assays at 200 animals, mostly classification.
- **B.4's matched parameters** (band data): dwelling 0.022 mm/s and roaming 0.067 mm/s; reversal hazards about 0.08/s on the lawn; the authors' HMM switching at 0.021 and 0.102 per 10 s.
- **`sbi` is not installed.** `cma` 4.5 is.
- **The machine** has 18 cores.

## Goals / Non-Goals

**Goals**

- C.1's model, closed-loop, at about 20 parameters, on the primitives every arm uses.
- A registered test that NPE recovers that model's parameters from the benchmark's statistics, with honest calibration.
- A record of which parameters the statistics constrain.

**Non-goals**

- Fitting C.1 to real worms. That is C.1, after F.0.
- The mutant encodings. They are C's table, registered before C.1 fits.
- sep-CMA and C.2's encoder. That is F.0b.

## Decisions

1. **The model (20 parameters).** Each second, for state s ∈ {dwelling, roaming}:

   | Part | Rule | Parameters |
   |---|---|---|
   | switching | h_DR = exp(a₀ + a_I·I + a_S·S), h_RD = exp(b₀ + b_I·I + b_S·S); switch with 1 − exp(−h) | 6 |
   | runs | speed v_s; curvature κ_w·tanh(lateral·lateral_scale) + σ_s·ε | 5: v_D, v_R, σ_D, σ_R, κ_w |
   | reversal hazard | r_s · exp(−β_r·tanh(dC/dt·scale)) · exp(e·loss) · search | 4: r_D, r_R, β_r, e |
   | omega hazard | o_s · exp(−β_o·tanh(dC/dt·scale)) · search | 3: o_D, o_R, β_o |
   | search | 1 + A·exp(−t_off/τ) on reversal and omega, t_off = seconds since food contact | 2: A, τ |

   - I = intake / max_intake and S = the satiety fraction.
   - loss = max(0, previous contact − contact): a falling food reading.
   - The delta hazard is the primitives' minimum.
   - The edge spike lasts the one second contact falls. It moves nothing if a bout is already running, since a bout ignores the controller. That is intended: a head-poke reversal starts from a run.
   - I is 0 where `max_intake` is undefined.
   - **One generator call per animal per second** draws every random number that second needs: `torch.rand(3, generator=…)` for the switch, the curvature (by inverse normal) and a spare. The loop over animals is the cost; one call keeps it to about 2.5 million calls per simulation.
   - The first state is drawn from the chain's stationary share at the first second's inputs.

2. **Priors**, fixed before any simulation, none from search data:

   | Parameters | Range |
   |---|---|
   | switching intercepts a₀, b₀ | log-uniform rates [10⁻⁴, 10⁻¹] /s |
   | switching slopes a_I, a_S, b_I, b_S | uniform [−5, 5] |
   | speeds v_D, v_R | uniform [0.005, 0.20] mm/s |
   | curvature SDs σ_D, σ_R | uniform [5, 60] °/s |
   | weathervane κ_w | uniform [−50, 50] °/s |
   | base hazards r_s, o_s | log-uniform [10⁻³, 0.5] /s |
   | klinokinesis β_r, β_o | uniform [0, 5] |
   | edge e | uniform [0, 5] |
   | search amplitude A | uniform [0, 20] |
   | search time τ | log-uniform [10, 3,000] s |

   The ranges span the band data's on-lawn magnitudes with margin, and published off-food values. The search terms' priors are wide and uninformed by López-Cruz.

3. **The simulator and its summary.**

   - θ → `BatchedMechanistic(θ)` → the batched evaluator at 200 animals on `scheer_lawn`, `uniform_lawn_scheer` and `food_removal_lopezcruz` → the grader.
   - The summary x is the 11 wild-type statistics and presence's lower bound: 12 numbers.
   - Each is z-scored over the training simulations where it is defined. An undefined entry is then set to −5, with a 0/1 flag appended per statistic, since `sbi` needs finite inputs. Choosing the sentinel after z-scoring keeps it from distorting the scale.
   - C.1 is fitted on these 12 numbers later. Its priors carry no search data (D37).

4. **Simulation seeds.** Simulation k uses animals `1_000_000 + 200·k + i`. Every simulation draws fresh noise, so the noise model is the real sampling noise of 200 animals, and the seeds overlap no band.

5. **NPE.**

   - `sbi`'s NPE with a neural-spline flow (its defaults), trained on 10,000 simulations, with a z-scored summary.
   - Two flows, trained with seeds 5402 and 5403.
   - Each posterior is sampled 1,000 times per truth.

6. **The gates (registered).**

   - **Truths:** 100 θ drawn from the prior (seed 5401), simulated afresh (simulations 10,000–10,099).

   - **Calibrated:** for each flow, the pooled coverage of the central 90% posterior intervals over the 100 × 20 (truth, parameter) pairs lies in **[0.80, 0.97]**.

   - **Informative:**

     - for each flow, take the first 20 truths;
     - simulate each at 20 posterior draws (simulations 10,100 onward, 400 per flow);
     - for each truth, compute D on the 11 statistics between its own simulation and each draw's, with z on the bands' scales at n = 200, and take the median over its draws;
     - the median of those 20 medians must be at most **1.42**, W's minimum.

     A posterior equal to the prior passes calibration by construction, so this gate is what makes the control a test.

   - Either gate failing, for either flow, fails F.0a.

   - **Reported, not gated:**

     - the same D at 20 prior draws per truth (400 more simulations), for the gain over the prior;
     - each parameter's coverage;
     - contraction, 1 − posterior SD / prior SD;
     - coverage on the 25 truths whose own simulation has the lowest D to the bands, since C.1 is fitted near the wild type.

7. **The constraint carried to C.1.** Each simulation summarises 200 animals with no date-to-date variation. The real wild type pools about 1,000 band animals over 41 dates. F.0a is unaffected, since its truths are simulated the same way as training. C.1's real fit must either simulate at the observation's size, or add an explicit between-date noise term matched to the bands. It is noted in the tracker.

8. **The gate preflight.** `gate_preflight.py` does not apply: there are no panel levels, and the informativeness gate cannot be read before a flow is trained. The pilot measures cost, and the launch record says so.

9. **Parallelism.** Simulations run in a process pool (16 workers, one thread each). Each simulation's result is written as it completes, so an interrupted run resumes.

## Risks / Trade-offs

- **[NPE is miscalibrated, and the gate fails]** → F.0a fails. That is the result the control exists to give, and NPE is not trusted for C.1 until it is fixed: more simulations, a different flow, or sequential NPE. The registered gate is not re-read.
- **[Many parameters are unconstrained]** → expected for some, for example the switching slopes, which intake barely moves on a full lawn. Contraction names them. A parameter the benchmark cannot see is reported, not tuned away.
- **[Per-animal draws are slow]** → one generator call per animal per second (decision 1). The pilot measures the whole simulation.
- **[Cost]** → the pilot measures a simulation. The 800 posterior- and prior-predictive simulations add about 8% to the 10,100. If 10,100 simulations would exceed about 6 hours on 16 workers, the pilot's record says so and the maintainer decides before the launch.
- **[The edge and search terms make C.1 more than Roberts']** → stated. Without an edge response no worm stays on a 3 mm lawn (B.4's negatives). The search term is the minimal slow variable, and its prior is uninformed.
