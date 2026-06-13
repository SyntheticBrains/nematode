## 1. Shared mode-aware kernel helper (food byte-identical)

- [x] 1.1 Add a module-level pure helper in env.py — `field_magnitude(distance, *, mode, decay, strength, fick_length)` → float — encoding the exp-vs-fick dispatch once (`exponential`: `strength·exp(−distance/decay)`; `fick`: `strength·exp(−(distance/fick_length)²)`; both give `strength` at `distance == 0`).
- [x] 1.2 Refactor `_food_field_magnitude` to delegate to the helper with the foraging params. Confirm it returns **numerically identical** values (the existing `test_fick_*` food tests are the regression).

## 2. Predator Fick fields + config (additive, exponential default)

- [x] 2.1 Add to `PredatorParams`: `gradient_field_mode: str = "exponential"`, `diffusion_coefficient: float | None = None`, `assay_time: float = 1.0`, and a `fick_length()` method (identical in form to `ForagingParams.fick_length()` — `√(4·D·assay_time)`, falling back to `gradient_decay_constant` when `D` is None).
- [x] 2.2 Add the matching fields to `PredatorConfig` (pydantic) — `gradient_field_mode: Literal["exponential","fick"] = "exponential"`, `diffusion_coefficient: float | None = Field(default=None, gt=0.0)`, `assay_time: float = Field(default=1.0, gt=0.0)` — and thread them through `to_params`.

## 3. Mode-aware predator field

- [x] 3.1 Add `_predator_field_magnitude(distance)` on the env, calling the shared helper with the predator params (mode / decay / strength / `fick_length()`).
- [x] 3.2 Route `get_predator_concentration` through `_predator_field_magnitude` (replace the inline `exp(−distance/decay)`). *As-built: the explicit `distance == 0 → strength` special-case was **removed** — the kernel returns `strength` at distance 0, so the boundary is preserved implicitly (byte-identical).* `get_predator_sulfolipid_concentration` (its alias) inherits Fick for free.
- [x] 3.3 Route `_compute_predator_gradient_vector` through `_predator_field_magnitude` for the per-source magnitude (keep the negative/repulsive sign + the `arctan2` direction; skip `distance == 0`).
- [x] 3.4 Confirm temperature / oxygen / pheromone fields are **untouched** (out of scope).

## 4. Tests

- [x] 4.1 Predator Fick kernel: at the `fick` mode the predator field at distance `r` equals `strength·exp(−(r/L)²)` (Gaussian), differs from the exponential kernel, and equals `strength` at `r=0`.
- [x] 4.2 Per-signal `D`: predator field with two different `D` values yields different spatial spread; predator `D` is independent of food `D` (food + predator can run different modes/`D` simultaneously).
- [x] 4.3 `fick_length()`: `√(4·D·assay_time)` when `D` set; falls back to `gradient_decay_constant` when `D` is None.
- [x] 4.4 Byte-stability: predator field at the default (`exponential`) equals the pre-change inline formula; the existing food `test_fick_*` tests still pass (shared-helper equivalence).

## 5. Validation + gates

- [x] 5.1 `openspec validate extend-fick-chemical-fields --strict` passes.
- [x] 5.2 Targeted `pre-commit run --files <changed>` green during iteration; full `pre-commit run -a` before push. *(Full `-a` green: ruff/pyright + full suite pass; ruff-format applied one PEP8 blank line.)*
- [x] 5.3 Full `uv run pytest -m "not nightly"` green. *(3885 passed, 1 skipped, 2 xfailed. Plus an end-to-end headless smoke — `tmp/verify_predator_fick.yml`: predators with `gradient_field_mode: fick` + `diffusion_coefficient: 3.0` + `predator_chemosensation_klinotaxis` sensing — runs clean, exercising the config→env→Fick-predator-field→sensing path.)*
- [x] 5.4 Tick `phase6-tracking` `T7.prep.fick_chemical_fields` (predator-sulfolipid done; pheromone / CO₂ + per-signal-`D` calibration remain noted follow-ups).
