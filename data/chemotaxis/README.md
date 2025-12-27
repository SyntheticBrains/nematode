# Chemotaxis Validation Data

This directory contains published C. elegans chemotaxis data for validating simulated agent behavior against real biological observations.

## Files

- `literature_ci_values.json` - Published chemotaxis index (CI) values from peer-reviewed literature

## Chemotaxis Index (CI)

The chemotaxis index is the standard metric in C. elegans research for quantifying attraction/avoidance behavior:

```
CI = (N_attractant - N_control) / N_total
```

Where:
- **N_attractant** = steps/time spent in the attractant zone (near food/chemical)
- **N_control** = steps/time spent in the control zone (away from attractant)
- **N_total** = total episode steps/time

### Interpretation

| CI Value | Interpretation |
|----------|----------------|
| +1.0 | Perfect attraction (always at food) |
| +0.75 | Excellent attraction (wild-type level) |
| +0.6 | Good attraction (target for simulation) |
| +0.4 | Weak attraction (minimum biological match) |
| 0.0 | No preference (random movement) |
| -1.0 | Perfect avoidance |

## Literature Sources

### Primary References

1. **Bargmann & Horvitz (1991)** - Cell 65(5):837-847
   - Established standard chemotaxis assay paradigm
   - Diacetyl chemotaxis: CI = 0.75 (range: 0.6-0.9)

2. **Bargmann et al. (1993)** - Cell 74(3):515-527
   - Bacterial (food) chemotaxis: CI = 0.7 (range: 0.5-0.85)
   - Most relevant to our foraging simulation

3. **Saeki et al. (2001)** - Neuron 32(2):249-259
   - Salt (NaCl) chemotaxis: CI = 0.6 (range: 0.4-0.8)

4. **Pierce-Shimomura et al. (1999)** - J Neuroscience 19(21):9557-9569
   - Describes biased random walk navigation strategy

## Usage

The validation module (`quantumnematode.validation`) uses this data to:

1. Calculate chemotaxis metrics from simulated trajectories
2. Compare agent CI against biological ranges
3. Report validation level (none/minimum/target/excellent)

### Example

```python
from quantumnematode.validation import (
    calculate_chemotaxis_metrics,
    ChemotaxisValidationBenchmark,
)

# Calculate metrics from trajectory
metrics = calculate_chemotaxis_metrics(
    positions=agent_positions,
    food_positions=food_locations,
    attractant_zone_radius=5.0,
)

# Validate against literature
benchmark = ChemotaxisValidationBenchmark()
result = benchmark.validate_agent(metrics)

print(f"Agent CI: {result.agent_ci:.2f}")
print(f"Biological range: {result.biological_ci_range}")
print(f"Matches biology: {result.matches_biology}")
print(f"Validation level: {result.validation_level.value}")
```

## Validation Thresholds

| Level | CI Threshold | Description |
|-------|-------------|-------------|
| None | < 0.4 | Not biologically plausible |
| Minimum | >= 0.4 | Minimally plausible |
| Target | >= 0.6 | Good biological match |
| Excellent | >= 0.75 | Wild-type level performance |

## Contributing

To add new literature sources:

1. Locate peer-reviewed publications with quantitative CI data
2. Add entry to `literature_ci_values.json` with:
   - Full citation
   - Attractant type
   - Wild-type CI value
   - CI range (min, max)
   - Experimental conditions
3. Document source in this README
