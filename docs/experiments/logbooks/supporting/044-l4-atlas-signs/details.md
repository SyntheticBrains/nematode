# Sign-grounding details

Analysis by `scripts/analysis/l4_atlas_signs.py` from the manifest beside this file (192 logs;
the eight registered extensions replace their shorter logs). `panel.json` is the full output;
`per-seed.csv` and `curves.csv` are the per-run table and learning curves. Comparators are
panel 2's committed per-seed table; its random-sign arms were not re-run.

## Verdict: `degree_statistics`

The wiring contrast is not confirmable once the signs are real. The substrate gate did not
fire: an overwhelmingly excitatory network does not saturate, and the prior over untrained
policies is essentially unchanged.

| test | contrast | mean Δ | 80% CI | q | +seeds | result |
|---|---|---|---|---|---|---|
| G1 | grounded wt frozen − random wt frozen | +0.77 | -2.06 … +3.60 | 0.260 | 41/64 | fail |
| G2 | grounded wt − rn Hebbian (primary) | +4.87 | -1.70 … +11.80 | 0.551 | 8/16 | fail |
| G3 | the same under Dale's law | -2.47 | -8.51 … +3.56 | 0.986 | 7/16 | fail |
| G4 | enforced − unenforced wt Hebbian | -4.75 | -8.89 … -1.27 | 0.986 | 4/16 | **reverse** |

Annotations: `prior_changed` false, `prior_worsened` false, `contrast_holds_under_dale` false, `enforcement_helps` false.

## What grounding did to each arm

| arm | grounded | panel 2, random signs | paired Δ (descriptive) |
|---|---|---|---|
| wt_frozen_atlas | mean 12.0, median 7.3, competent 0.23 | mean 11.3, median 3.7, competent 0.22 | +0.8 (CI -2.1 … +3.6, 41/64) |
| rn_frozen_atlas | mean 8.7, median 3.3, competent 0.12 | mean 8.0, median 2.3, competent 0.16 | +0.8 (CI -2.0 … +3.4, 32/64) |
| wt_hebbian_atlas | mean 14.0, median 6.6, competent 0.25 | mean 31.5, median 24.6, competent 0.56 | -17.5 (CI -29.3 … -4.9, 6/16) |
| rn_hebbian_atlas | mean 9.1, median 4.4, competent 0.12 | mean 17.4, median 15.2, competent 0.31 | -8.3 (CI -14.9 … -1.0, 6/16) |
| wt_hebbian_dale | mean 9.2, median 2.4, competent 0.12 | mean 31.5, median 24.6, competent 0.56 | -22.3 (CI -33.6 … -10.3, 5/16) |
| rn_hebbian_dale | mean 11.7, median 6.8, competent 0.19 | mean 17.4, median 15.2, competent 0.31 | -5.7 (CI -11.1 … -0.4, 7/16) |

## Reading

- **The substrate is fine; the prior barely moves.** Grounding 3,176 of 3,709 synapses takes the
  network from 48% inhibitory to 13%, and the distribution of untrained policies is unchanged:
  the wild-type competent fraction is 0.23 against the committed 0.22, G1 is +0.8 with the
  interval spanning zero. The registered `substrate_fail` outcome did not occur. What did change
  is the shape: the wild-type median doubles (7.3 against 3.7) while the mean holds, and the
  rewired null's competent fraction falls (0.12 against 0.16) — real signs leave the animal's own
  wiring where it was and cost its scramble something. Descriptive, and the direction the wiring
  hypothesis would predict.
- **Reward-free Hebbian learning gets substantially worse.** The wild-type Hebbian arm falls from
  31.5 to 14.0 and the rewired from 17.4 to 9.1, both intervals clear of zero. This is the
  panel's real finding, and it has a mechanism: a purely potentiating co-activity rule on a
  network that is 80% excitatory has no inhibitory brake. Random signs handed the rule a balanced
  substrate in which drift could settle on useful fixed points; the animal's sign structure
  removes that balance, and the rule has nothing to replace it with. Biological circuits pair
  excitatory wiring with anti-Hebbian and inhibitory plasticity (Perks et al., *Nature*
  2026-09-02); this rule has neither.
- **Dale's law makes it worse again on the wild-type.** Enforcing signs through plasticity takes
  the wild-type Hebbian arm from 14.0 to 9.2 (G4 reverses, −4.8, interval clear of zero) while
  the rewired arm rises slightly. Constraining the drift to the sign structure does not help it;
  it removes what little freedom the rule was using.
- **The wiring contrast is not rescued.** G2 is +4.9 with the interval spanning zero, against
  panel 2's +14.1 at the same seeds and also unconfirmed. Grounded signs neither confirm nor
  reverse the contrast; the verdict is `degree_statistics`.

## What this says about the ladder

The first rung's answer is that **the substrate's random signs were not what limited the rule** —
grounding them leaves the prior alone and makes the learning worse. The rule's failure is the
rule's. That moves the anti-Hebbian and decorrelating variant ahead of structured routing in
7a-ii's queue, and it gives that variant a sharp prediction to be tested against: on a grounded,
mostly-excitatory substrate, a rule with an inhibitory or anti-Hebbian term should recover what
the purely potentiating rule loses here. It also means the receptor layer (B.3) is worth having
for fidelity but should not be expected to rescue learning on its own.

## Campaign facts

- Prior sweep: 128 runs (2 arms × seeds 1–64 × 600 episodes), 00:22–01:07 on 16 workers.
- Hebbian contrast: 64 runs (4 arms × seeds 1–16 × 1000 episodes), 01:07–01:35.
- Extensions: 8 runs at 1.5×, all applied; the family and the verdict were unchanged by them.
- Every runner exited 0; no tracebacks in any of the 200 logs.
- Grounding coverage: 280 of 302 neurons carry a release identity; 3,176 of 3,709 synapses
  are signed (2,962 excitatory, 214 inhibitory); 533 keep the sign they drew.
