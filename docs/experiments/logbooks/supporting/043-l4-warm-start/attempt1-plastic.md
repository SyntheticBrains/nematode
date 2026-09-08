# Attempt 1 of the plastic-set learning arms (superseded by the amendment)

The four plastic-set learning arms — Hebbian and three-factor from the plastic-set clone on both
wirings — first ran 15:07–16:07 on 2026-09-08 against a load defect: the rule's homeostatic norm
targets were the incoming norms at construction (random initialisation, median 0.95 per unit)
and were not refreshed on load, while the loaded clone's norms sat six times higher (median
5.4). The first plastic step rescaled every unit back to its random-init norm and undid the
saturated switch policy the clone encodes. Verified directly before the amendment: targets equal
the construction norms, loaded norms are 6.06× the targets at the median, and the targets were
not refreshed on load.

## What that run measured

| arm | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 |
|---|---|---|---|---|---|---|---|---|
| wt_clone_frozen (unaffected, 600 ep) | 37.7 | 36.5 | 37.4 | | | | | |
| wt_clone_hebbian | 29.9 | 30.6 | 31.5 | 31.4 | 28.9 | 30.6 | 29.6 | 32.2 |
| wt_clone_plastic | 7.7 | 2.9 | 1.1 | 1.1 | 1.2 | 1.1 | 0.9 | 1.1 |
| rn_clone_hebbian | 31.7 | 32.8 | 31.8 | 33.6 | 33.0 | 31.8 | 31.1 | 33.5 |
| rn_clone_plastic | 17.6 | 18.4 | 15.3 | 15.7 | 14.2 | 14.6 | 14.2 | 13.5 |

Means over seeds 1–8 at the plateau tail: wt_clone_hebbian 30.9, rn_clone_hebbian 32.3,
wt_clone_plastic 1.0, rn_clone_plastic 13.9. Interim family on these arms: W3 −12.9
(CI [−19.8, −5.4], 2/8), W4 −36.1 (CI [−41.6, −30.9], 0/8), W5 −29.9 (CI [−42.4, −18.2], 1/8);
the interim verdict would have been `sanity_floor_fail` with `rule_destroys_clone`.

## How to read it

The Hebbian arm's immediate drop from 37 to 30 is the rescaling alone: the clone survives it as
a weaker policy, and the unmodulated rule holds a fixed point near it. The three-factor arm's
collapse to 1% on the wild-type is the rescaling plus the modulated rule wandering from a
policy the rescaling had already broken; the rewired null lost less (14%). None of this is a
property of the rule acting on the clone as loaded, which is what the registration asked; it is
the property of the rule acting on the clone with every unit's norm cut sixfold on step one.
The logs are kept under `campaigns/l4-warm-start-panel/attempt1-plastic/`; the amended run's
logs replace them in the analysis.
