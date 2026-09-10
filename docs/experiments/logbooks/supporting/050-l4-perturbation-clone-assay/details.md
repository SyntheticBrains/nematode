# Clone-assay details: the node-perturbation eligibility

Analysis by `scripts/analysis/l4_consolidation_screen.py` over 16 runs (2 arms × seeds 1–8 × 2000
episodes, all exit 0). The assay is the one registered with the clone-destruction diagnostic,
unchanged in arms, comparator, budget, metric and pass rule.

## Result: **fails** — and the frozen control says the rule is not what failed

| arm | mean | Δ vs frozen clone | within hold | at or above | cosine | trajectory (first → final quarter) |
|---|---|---|---|---|---|---|
| `node_perturbation` | 12.0 | −26.7 | 1/8 | 1/8 | 0.70 | 15.2 → 12.0 (−3.2) |
| `perturbation_frozen` | 8.9 | −29.8 | 1/8 | 1/8 | **1.00** | 9.1 → 8.9 (−0.2) |

The committed frozen clone is **38.7**. Under the registered rule the variant **fails**, and the
panel stays gated. That verdict stands as written.

**But the frozen control changes what the failure means.** Its weights are untouched — cosine 1.00,
by construction, since `freeze_updates` writes nothing — and it still scores **8.9**. Perturbation
*alone*, with no learning whatsoever, takes a competent 38.7 policy to 8.9. The plastic arm scores
**12.0**, which is **3.1 points better than the frozen baseline**, not worse.

**The rule did not take the clone apart. The perturbation did, before the rule acted.**

## The trajectory annotation confirms the mechanism

The frozen control is already at **9.1 in its first quarter** and ends at 8.9: flat, because nothing
is learning and nothing further degrades. The competent policy is gone within the first 500
episodes, purely from σ = 0.2 jitter on activities bounded in (−1, 1).

The plastic arm starts at **15.2** — above the frozen baseline, so the rule is already recovering
ground in the first quarter — and settles at 12.0. Its cosine of 0.70 says the weights stayed
comparatively near the clone throughout; it is not rewriting the policy the way the unbraked rules
did at 0.2–0.45.

## What this actually establishes

- **The assay at this σ could not test retention.** There was no competent policy left to retain:
  the perturbation removes it before the rule does anything. A test of "does this rule hold a good
  policy" requires a good policy to survive the mechanism's own exploration, and here it does not.
- **The tension is real and it is the finding.** The σ that makes the rule *learn* — 0.2 was the
  only value in the declared grid that passed the positive control — is the same σ that makes a
  competent policy *unrunnable*. Exploration large enough to give a well-conditioned gradient
  estimate is large enough to wreck the behaviour being estimated for. The control's own
  dose-response already hinted at it: σ = 0.05 reached 7 of 8 seeds above the floor but did not
  clear the learning bar.
- **The registered fail is honest and uninformative about the rule.** Reporting it without the
  frozen control would have said "the variant destroys competent policies", which the evidence
  contradicts.

## What it does not establish

- Not that the variant cannot hold a policy. That question is untested, and testing it needs a σ at
  which a competent policy still functions — which may or may not also learn.
- Not that σ should be re-tuned here. The launch record fixed σ at the control's pinned value
  precisely so this gate could not become a search, and it did not become one.
- Not that annealing works. Large-early, small-late is the obvious next mechanism and is entirely
  untested.

## Limitations

- One σ, by design. The assay was registered as a gate at the control's value, not as a sweep.
- n = 8 and a screen, not a confirmatory test: it reuses seeds Logbook 043 reported.
- The frozen control shares the plastic arm's σ and seed stream but not its weight trajectory, so
  it bounds the immediate cost of perturbation rather than the cost at every point along a run.
- The trajectory annotation is a coarse instrument — quarter means over a 2000-episode run — and
  was registered to separate two gross outcomes, not to characterise a curve.

## Facts

- 16 runs (2 arms × 8 seeds × 2000 episodes), all exit 0, no extensions.
- The launch record fixing the arms, the pinned σ, the unchanged assay and what each outcome
  licenses was committed before the runs; the frozen control and the trajectory annotation were
  registered by spec review while the plastic arm was still running, and neither changes the
  verdict.
