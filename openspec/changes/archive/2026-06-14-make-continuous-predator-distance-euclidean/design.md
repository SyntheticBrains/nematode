# Design

A small substrate-coherence fix completing the continuous predator-geometry set. Two decisions:

## D1 — Env override, not a reward edit

The incoherence is that `get_nearest_predator_distance_for` (an `env.py` method) returns Manhattan,
and the reward calculator queries it. The cleanest fix makes the **continuous env method** Euclidean
(an override on `Continuous2DEnvironment`, exactly like the existing
`is_agent_in_danger_for` / `is_agent_in_damage_radius_for` / `get_agent_predator_contact_zone_for`
overrides), leaving `reward_calculator.py` **byte-identical**. This is strictly better than editing the
reward: the reward *formula* doesn't change at all — it just receives a coherent distance on continuous —
so the "reward formula frozen, coherence-only" principle is satisfied by construction, and grid stays
byte-stable (the override is confined to the subclass).

Implementation mirrors the existing overrides:

```python
def get_nearest_predator_distance_for(self, agent_id: str) -> float | None:
    if not self.predator.enabled or not self.predators:
        return None
    ax, ay = self._agent_xy(agent_id)
    return min(math.hypot(ax - px, ay - py)
               for px, py in (self._predator_xy(pred) for pred in self.predators))
```

The single-agent convenience `get_nearest_predator_distance` delegates to the `_for` variant with
`DEFAULT_AGENT_ID` (the grid base reads `self.agent_pos`, i.e. the default agent), removing the latent
Manhattan inconsistency on continuous even though it is currently unused.

## D2 — `default`-mode `prev_pred_dist` left out of scope

`reward_calculator.py` also computes a Manhattan `prev_pred_dist` inline (`abs(Δx) + abs(Δy)` against
`pred.position`), but only under `reward_mode == "default"`. No T7 config uses `default` (all use
`distal_chemo_contact_trigger`, which skips that term), and the agent `path` is integer-typed
(`list[tuple[int, ...]]`), so making `prev_pred_dist` Euclidean would require threading continuous
position history into `path` — a larger change for a code path T7 never exercises. Deferred and
documented rather than half-fixed.
