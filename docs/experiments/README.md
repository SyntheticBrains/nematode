# Experiment Logbooks

Human-written analysis and documentation of experiment series.

## Relationship to Other Systems

This project has complementary systems for tracking experiments:

| System | Location | Git Tracked | Purpose |
|--------|----------|-------------|---------|
| **Auto-tracking** | `experiments/*.json` | No | Raw metadata from every simulation run |
| **Evolution results** | `evolution_results/` | No | All evolution run outputs |
| **Artifacts** | `artifacts/` | Yes | Curated outputs referenced in logbooks |
| **Logbooks** (this) | `docs/experiments/logbooks/` | Yes | Human analysis, insights, narrative |
| **Benchmarks** | `benchmarks/` | Yes | Curated best results for leaderboards |

### Workflow

```markdown
1. Run simulations/evolution
   └── Auto-saved to experiments/*.json or evolution_results/

2. Query results
   └── python scripts/experiment_query.py list
   └── python scripts/experiment_query.py show <id>

3. Preserve notable results
   └── Copy to artifacts/ for git tracking
   └── e.g., cp -r evolution_results/20251209_205950 artifacts/evolutions/

4. Document findings
   └── Write logbook in docs/experiments/logbooks/
   └── Reference artifacts: "See artifacts/evolutions/20251209_205950/"

5. Promote best results
   └── python scripts/benchmark_submit.py submit <id>
   └── Saved to benchmarks/ for leaderboards
```

## Active Experiments

| # | Title | Status | Summary |
|---|-------|--------|---------|
| 001 | [Quantum Predator Optimization](logbooks/001-quantum-predator-optimization.md) | completed | Gradient-based learning approaches |
| 002 | [Evolutionary Parameter Search](logbooks/002-evolutionary-parameter-search.md) | completed | CMA-ES and GA optimization |
| 003 | [Spiking Brain Optimization](logbooks/003-spiking-brain-optimization.md) | completed | Surrogate gradients, LIF neurons, decay schedules |
| 004 | [PPO Brain Implementation](logbooks/004-ppo-brain-implementation.md) | completed | SOTA RL baseline, actor-critic, 98.5% foraging, 93% predators |
| 005 | [Health System Predator Scaling](logbooks/005-health-system-predator-scaling.md) | completed | Health system enables better evasion learning via survival + richer rewards |
| 006 | [Unified Sensory Modules](logbooks/006-unified-sensory-modules.md) | completed | Biologically-inspired 4-feature architecture matches legacy 2-feature performance |
| 007 | [PPO Thermotaxis Baselines](logbooks/007-ppo-thermotaxis-baselines.md) | completed | PPO baselines for 9 thermotaxis configs (3 sizes × 3 tasks), 84-98% post-conv |
| 008 | [Quantum Brain Evaluation](logbooks/008-quantum-brain-evaluation.md) | completed | 300+ sessions, 11+ quantum architectures (QA-1 to QA-7). QRH genuine quantum advantage on pursuit (+9.4pp). HybridQuantum 96.9% but classical ablation matches. QEF competitive, not advantageous. QA-7 plasticity: classical shows zero forgetting — hypothesis untestable at current complexity. Pivot to environment enrichment. |
| 009 | [Temporal Sensing Evaluation](logbooks/009-temporal-sensing-evaluation.md) | completed | Phase 3: Temporal/derivative sensing replacing oracle gradients. New GRU PPO brain (19th architecture). Temporal Mode A achieves L100=95% on hardest environment (matching oracle L100=95%; L500=94% vs oracle 97%, within 3pp). GRU outperforms LSTM. Chunk length is critical hyperparameter. |
| 010 | [Aerotaxis Baselines](logbooks/010-aerotaxis-baselines.md) | completed | Oracle + derivative + temporal evaluation across 7 scenarios + 3 controls. Derivative exceeds oracle on all 6 scenarios (+4pp to +28pp). Temporal: 99% L100 single-modality, 89% L100 triple-modality (12k episodes, all seeds converge). Training time scales with modality count. |
| 011 | [Multi-Agent Evaluation](logbooks/011-multi-agent-evaluation.md) | completed | 7 pre-klinotaxis campaigns (A-H) plus 7 Klinotaxis Era campaigns (K1-K7, 104 sessions). Key positive findings: food-marking pheromones +77pp on single-cluster (K1); +47.8% social feeding under scarcity (K6); phenotype frequency-dependent fitness +40-45pp (K3) — meets Phase 4 emergent-behaviour exit criterion. Key negative findings: alarm pheromones inert across 5 conditions including biologically faithful no-nociception baseline; aggregation pheromone informationally inert; single-cluster benefit collapses to +2.7pp on multi-cluster. Three bugs (#112, #115, STAM heterogeneous-dim) found and fixed. |
| 012 | [Hyperparameter Evolution — M2 (MLPPPO + LSTMPPO+klinotaxis + LSTMPPO+klinotaxis+predator)](logbooks/012-hyperparam-evolution-mlpppo-pilot.md) | completed | Hyperparameter-evolution framework across 4 arms / 2 optimisers. TPE wins the optimiser portfolio: +79pp on the predator arm vs CMA-ES's +47pp, rescuing CMA-ES's dead-zone seed. RQ1 closed — TPE is the default for M3. |
| 013 | [Lamarckian Inheritance — M3 (LSTMPPO+klinotaxis+predator, TPE)](logbooks/013-lamarckian-inheritance-pilot.md) | completed | Per-genome Lamarckian inheritance via warm-start from prior-gen elite. **GO** ✅: speed gate +5.25 gens (4.50 vs 9.75), all 4 seeds reach 1.00 best fitness, rescues TPE-unlucky seed 42. Cross-schema check rules out simplification confounder. M4 starts here. |
| 014 | [Baldwin Inheritance — M4 (LSTMPPO+klinotaxis+predator, TPE)](logbooks/014-baldwin-inheritance-pilot.md) | completed (framework shipped) — INCONCLUSIVE | Trait-only Baldwin pilot on a 6-field schema. **INCONCLUSIVE** ⚠️: post-pilot audit found 3 blocking design flaws (schema-shift confounder, F1 test biologically incoherent, apples-to-oranges baseline) plus n=4 underpowered. Framework changes ship; science deferred to M4.5. |
| 015 | [Baldwin Effect — iterative evaluation (M4.5 + M4.6)](logbooks/015-baldwin-iterative-evaluation.md) | completed — **STOP** | Three-iteration arc (M4 → M4.5 → M4.6) closes the Baldwin question for Phase 5. M4.5 closed all five audit findings; structural finding was that the current abstraction is mechanically null vs Control. M4.6 pre-flight smoke ruled out three selection-feedback abstractions and diagnosed the real blocker as substrate-level: single-task K=50 PPO has no Baldwin axis. Deferred to potential M4.7 post-M5 if co-evolution surfaces a Baldwin signal serendipitously. |
| 016 | [Predator-Brain Refactor — M1 (PredatorBrain Protocol + heuristic adapter + per-predator metrics)](logbooks/016-predator-brain-refactor.md) | completed | M5 prerequisite: `PredatorBrain` Protocol seam + `HeuristicPredatorBrain` adapter + per-predator metrics. **GO** ✅: byte-equivalent at both trajectory level (23 unit tests) and campaign-metric level (80/80 deltas exactly 0.0 across 20 cells × 4 metrics). Pluggable predator brains land as pure substrate with zero behavioural cost. |
| 017 | [Co-Evolution Arms Race — M5 (CoevolutionLoop + Red Queen primitives + screen-sweep pilot)](logbooks/017-coevolution-arms-race.md) | completed — **STOP** | 13 single-seed lever screens + R1 re-audit decisively falsify strict Red Queen entanglement at this substrate: own-vs-cross fitness lag delta landed at +0.017 to +0.024 across every candidate that produced a full champion-archive snapshot. Methodology contributions (lag-matrix, cell-grid, fair-test instruments) ship intact and motivate M7 NEAT as the natural next Red Queen attempt. |
| 018 | [Transgenerational Memory — M6 (TEI substrate + cascade + paired-arm ablation)](logbooks/018-transgenerational-memory.md) | completed (framework shipped) — INCONCLUSIVE | TEI framework shipped (substrate dataclass + cascade + LSTMPPO `tei_prior` hook + per-gen `lawn_schedule` + paired-arm aggregator + ~37 tests). Literal output STOP (choice-index gate, geometry-dominated) / PIVOT TEI-on 1/4 (survival-rate gate with F0 training-time override). **INCONCLUSIVE** ⚠️: post-pilot audit found 4 blocking design issues (substrate gradient-unconditional, training-reward + env geometry produces motion-bias attractor, F0 probes have no pathogen context, asymmetric F1+ compute) that mean the gates compared a substrate which cannot encode pathogen-conditional avoidance. M6.9+ deferred to follow-up OpenSpec. |
| 019 | [Transgenerational Memory Redesign — M6.9+ PR-A (sensory-conditional substrate + env-derived probes + audit-B reward/env)](logbooks/019-transgenerational-memory-redesign.md) | completed (framework + M3 validation shipped) — **STOP** | M6 audits A/B/C addressed: sensory-conditional bias-network MLP substrate + Manhattan-ring F0 probes + gradient_proximity reward + min_food_predator_distance env. M3 weights_only reproduces on new env at **+17.5pp** vs control. Three pilots × three substrate variants (basic / +safe_probes / +clamp 2.0→6.0) all collapse tei_on F1+ to ~0 at K=0; cross-arm `tei_on − control` = **-49pp**. Pure-TEI K=0 hypothesis falsified, corroborated by 2024-2026 deep-RL distillation + 2025 wet-lab Kaletsky framing. M6.13 reframe = TEI as a prior on Lamarckian (K>0). |
| 020 | [TEI-as-Prior-on-Lamarckian — M6.13 (composed inheritance + K-sensitivity + F0-confound disambiguation)](logbooks/020-tei-prior-on-lamarckian.md) | completed (framework + GC fix shipped) — **STOP** | `LamarckianTransgenerationalInheritance` strategy + main-loop GC `.tei.pt` preservation fix. Four pilots at K ∈ {1000, 500, 200, 200-F0-matched}. Under fair-F0 comparison, cross-arm `tei_weights − weights_only` F1-F3 delta is **+0.00pp at K=1000** (substrate inert) and **−9.33pp at K=200** (substrate INTERFERES). K-sensitivity sweep apparent dose-response (+0.00→+4.00→+5.33pp) was the F0 confound. Substrate-shape diagnosis: bias-network logit-prior ≠ wet-lab single-circuit excitability shift. M6.14 NOT triggered. |
| 021 | [Phase 5 Synthesis — M8 (Evolution & Adaptation close-out)](logbooks/021-phase5-synthesis.md) | completed — **Phase 5 COMPLETE** | Scoped M8 synthesis (M8.2 exit-criteria walkthrough + M8.3 negative-findings + M8.4 Phase 6 trigger). All 5 Phase 5 exit criteria MET (two with substrate-grounded STOP caveats). STOPs (M4 Baldwin / M5 Red Queen / M6 TEI) are field-consistent substrate diagnoses corroborated by Resendez Prado 2026, Mougi 2026, Chen 2025, Kaletsky 2025. Methodology contributions (lag-matrix + cell-grid fair-test + per-gen reaggregation) ship unscooped. M7 NEAT remains OPTIONAL; M5 architecture-asymmetry + M6 substrate-redesign inherit to Phase 6. |
| 022 | [Connectome Substrate L0 Ingestion](logbooks/022-connectome-substrate.md) | completed | Cook et al. 2019 hermaphrodite connectome (302 neurons, 3709 chemical synapses, 1093 gap junctions) loaded via direct *Nature* SI parsing (no third-party connectome-package dep); cross-validated against Witvliet et al. 2021 adult nerve-ring (180 shared neurons; right-skewed weight-ratio distribution as expected for whole-animal vs single-animal datasets); vendored under `data/connectome/` with PROVENANCE.md; forward-pass smoke ships. T1↔T2 handshake API (`Connectome` dataclass + iteration patterns for chemical/gap-junction tensors) settled for downstream plugin work. |
| 023 | [Architecture Plugin Interface + First Connectome-Constrained Brain](logbooks/023-architecture-plugin-interface.md) | completed — **Phase 6 Gate 1 GO** | L1 plugin parity (decorator-registration registry, 19-architecture migration with byte-equivalence on MLPPPO + LSTMPPO, dispatcher/loader collapse, StrEnum migration) + first connectome-constrained PPO brain (`ConnectomePPOBrain`). PPO-on-wild-type-Cook-2019-connectome learns klinotaxis foraging end-to-end: R2b reference run hits 100% sustained success rate on last-100 episodes, all three Gate 1 G1.c pass-criteria literally satisfied (no NaN/Inf over 500 ep, 16.1× last-25 reward margin over frozen-random-weights control, monotonic improvement 76% → 100%). Within 6 points of MLPPPO + LSTMPPO baselines on the same task / env / seed. Late-training drift at `entropy_coef=0.02` diagnosed as hyperparameter polish, resolved by `entropy_coef=0.005`. Gate 1 closes GO across all four sub-criteria (G1.a–G1.d). |
| 024 | [L2 First Pass — T3 corrected predator-sensing biology prerequisite section (T4 body TBD)](logbooks/024-l2-first-pass.md) | stub — T3 section complete | T3 ships the biologically-grounded two-channel predator-sensing replacement for the legacy single-channel `nociception_klinotaxis` flagged in Logbook 011. Contact-mechanosensory (ASH/ALM/AVM/PVD/PLM with anterior/posterior/lateral zone) + distal-chemosensory (ASH+ASI sulfolipid per Liu et al. 2018) shipped as six new sensor modules with the `_oracle` suffix convention, two new STAM channels, four new BrainParams fields, and the `ContactZone` enum. 38 zone tests + 24 module tests + 45 legacy-config regression tests all green. 100-ep head-to-head smoke vs legacy: new biology learns slower at matched compute (MLPPPO 3% vs 51%; LSTMPPO 0% vs 7%) — recorded as a known T3 finding + carried forward to T4 as T4.0g convergence-rate study + four T4.\*\_reward_ablation sub-tasks. Full L2 evaluation body lands here when T4 begins. |

## How to Use Logbooks

### Reading

Each logbook follows a consistent structure:

- **Objective**: What we're trying to achieve
- **Hypothesis**: What we expected
- **Results**: What actually happened
- **Analysis**: Why it happened
- **Next Steps**: Where to go from here

### Creating New Logbooks

1. Copy `templates/experiment.md` to `logbooks/NNN-descriptive-name.md`
2. Use the next sequential number
3. Update the index table above
4. Reference session IDs from `experiments/*.json` for reproducibility

### Linking to Auto-Tracked Data

Reference specific experiments by session ID:

```markdown
- Session: `20251209_205950` (80% success with CMA-ES)
- Query: `python scripts/experiment_query.py show 20251209_205950`
```

## Key Findings Summary

### Experiment 001: Quantum Circuit Limitations

- 2-qubit circuits max ~31% success with gradient learning
- Learning actively degrades good initializations
- Combined gradient (chemotaxis) works; separated gradients fail

### Experiment 002: Evolutionary Approach

- CMA-ES achieved 80% success on foraging-only
- GA achieved 70% with more stable convergence
- Evolution bypasses gradient noise problem

## Directory Structure

```markdown
docs/experiments/
├── README.md                    # This file
├── templates/
│   └── experiment.md            # Template for new logbooks
└── logbooks/
    ├── 001-quantum-predator-optimization.md
    ├── 002-evolutionary-parameter-search.md
    └── ...

experiments/                     # Auto-generated (gitignored)
├── 20251207_123456.json
└── ...

benchmarks/                      # Curated results (git tracked)
├── foraging_small/classical/
└── ...
```
