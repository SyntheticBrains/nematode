K_test calibration smoke (1 seed × pop 6 × 2 gens × weights_only × K=1000)

F0 mean survival_rate: 0.407 (pop 6)
F1 mean survival_rate: 0.600 (pop 6)

T1' F0 envelope \[0.30, 0.70\]: PASS (0.407)
T3' Lamarckian-headroom (per design.md § D5 procedural rule: F1 ≥ 0.95×F0 → saturated FAIL; F1 < 0.80×F0 → under-trained FAIL; F1 > F0 + 0.05 → clear headroom PASS): PASS (F1=0.600 > F0+0.05=0.457 — Lamarckian retraining grew F1 well past F0, neither saturated at 0.387 nor under-trained below 0.326)
T2' substrate diversity (min pairwise CoV ≥ 5%): PASS (0.9633, 19× over)
T4' substrate magnitude (mean abs bias_network output ≥ 0.1): PASS (1.7954, 18× over)
