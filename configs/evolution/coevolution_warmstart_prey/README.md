# Coevolution prey warmstart bundle

Each `seed_{N}.json` is the prey gen-0 warm-start anchor (`CMAESOptimizer(x0=...)`) for the matching co-evolution run seed. `CoevolutionLoop.__init__` reads `coevolution.prey_gen0_seed_path` from the YAML — typically templated as `seed_<run_seed>.json` so each campaign seed loads its own warmstart anchor.

## Provenance

Source campaign: `lamarckian_lstmppo_klinotaxis_predator_pilot.yml` driven by the matching campaign
script. Per-seed elite LSTMPPO weight checkpoints under
`artifacts/logbooks/013/m3_lamarckian_pilot/lamarckian/seed-{42..45}/inheritance/genome-*.pt` (one elite saved
per seed; intermediate checkpoints GC'd by the source campaign).

Seeds covered: [42, 43, 44, 45].

## Format

Each JSON file matches the `CoevolutionLoop._load_prey_warmstart`
schema:

```json
{
  "genome_id": "<uuid>",
  "generation": <int>,
  "fitness": <float>,
  "params": [<float>, ...],
  "brain_config": {
    "name": "lstmppo",
    "comment": "..."
  }
}
```

The `params` array is the flattened weight vector matching the
encoder's `genome_dim` for the source-campaign brain shape (LSTMPPO +
klinotaxis sensing as configured in the source YAML). Co-evolution YAMLs
that target this bundle MUST use the same brain shape (or a compatible
encoder that produces the same `genome_dim`).

## Regenerate

```sh
uv run python scripts/campaigns/curate_coevolution_prey_bundles.py \
    --source-config configs/evolution/lamarckian_lstmppo_klinotaxis_predator_pilot.yml \
    --source-root artifacts/logbooks/013/m3_lamarckian_pilot/lamarckian \
    --out-root configs/evolution/
```
