# Coevolution prey reference bundle

Each `seed_{N}.json` is one prey LSTMPPO elite genome with TWO roles
in the co-evolution loop, both consumed from this single bundle:

1. **Gen-0 warm-start anchor** for the prey CMA-ES (per-seed
   `coevolution.prey_gen0_seed_path` in the YAML, consumed by
   `CoevolutionLoop._load_prey_warmstart`). Each campaign seed loads
   its own anchor (template is `seed_<run_seed>.json`).
2. **Held-out probe opponents** for the co-evolution generality
   probe. `CoevolutionLoop._load_held_out_prey_bundle` walks the same
   directory and loads ALL JSONs (one per seed) into the held-out
   set, then samples `held_out_size` of them per probe firing.

The two roles share the same source genomes by design — one bundle
on disk avoids ~5 MB of byte-identical duplication between separate
warmstart/ and held-out/ directories.

Files committed to repo (vs `artifacts/`) so a fresh checkout can run
the campaign reproducibly. Routed through Git LFS via the repo's
`configs/**/*.json` rule in `.gitattributes`.

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
