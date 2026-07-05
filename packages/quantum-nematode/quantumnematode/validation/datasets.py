"""Dataset loading and validation benchmark for chemotaxis data.

This module provides functionality to load published C. elegans chemotaxis data
from peer-reviewed literature and compare simulated agent behavior against
biological baselines.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .chemotaxis import (
    ChemotaxisMetrics,
    ValidationResult,
    get_validation_level,
)

# Project root and default dataset path for clearer path resolution
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_DATASET_PATH = _PROJECT_ROOT / "data" / "chemotaxis" / "literature_ci_values.json"
_DEFAULT_BIAS_PATH = _PROJECT_ROOT / "data" / "chemotaxis" / "behavioural_bias_signatures.json"


@dataclass
class LiteratureSource:
    """A literature source for chemotaxis data.

    Attributes
    ----------
        citation: Full citation (e.g., "Bargmann & Horvitz (1991). Cell 65(5):837-847")
        attractant: Type of attractant (e.g., "diacetyl", "bacteria")
        ci_wild_type: Typical CI value for wild-type worms
        ci_range: (min, max) range of CI values observed
        conditions: Experimental conditions description
    """

    citation: str
    attractant: str
    ci_wild_type: float
    ci_range: tuple[float, float]
    conditions: str


@dataclass
class ValidationThresholds:
    """Thresholds for interpreting validation results.

    Attributes
    ----------
        minimum: CI threshold for minimum biological plausibility
        target: CI threshold for target biological match
        excellent: CI threshold for excellent biological match
    """

    minimum: float = 0.4
    target: float = 0.6
    excellent: float = 0.75


@dataclass
class ChemotaxisDataset:
    """Dataset of published chemotaxis values.

    Attributes
    ----------
        version: Dataset version string
        sources: List of literature sources
        thresholds: Validation thresholds
    """

    version: str
    sources: list[LiteratureSource]
    thresholds: ValidationThresholds

    def get_source_by_attractant(self, attractant: str) -> LiteratureSource | None:
        """Find a source by attractant type."""
        for source in self.sources:
            if source.attractant.lower() == attractant.lower():
                return source
        return None

    def get_default_source(self) -> LiteratureSource:
        """Get the default source for validation (food/bacteria chemotaxis)."""
        # Prefer bacteria as it's closest to our food-seeking simulation
        for attractant in ["bacteria", "food", "diacetyl"]:
            source = self.get_source_by_attractant(attractant)
            if source:
                return source
        # Fall back to first source
        if not self.sources:
            error_msg = "ChemotaxisDataset has no sources"
            raise ValueError(error_msg)
        return self.sources[0]


def load_chemotaxis_dataset(
    dataset_path: str | Path | None = None,
) -> ChemotaxisDataset:
    """Load chemotaxis dataset from JSON file.

    Args:
        dataset_path: Path to JSON file. If None, uses default location.

    Returns
    -------
        ChemotaxisDataset loaded from file

    Raises
    ------
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset format is invalid
    """
    dataset_path = _DEFAULT_DATASET_PATH if dataset_path is None else Path(dataset_path)

    if not dataset_path.exists():
        # Return a default dataset if file doesn't exist yet
        return _create_default_dataset()

    with dataset_path.open() as f:
        data = json.load(f)

    # Parse sources with validation
    sources = []
    for i, source_data in enumerate(data.get("sources", [])):
        try:
            ci_range = source_data["ci_range"]
            if not isinstance(ci_range, list) or len(ci_range) != 2:  # noqa: PLR2004
                msg = f"Source {i}: ci_range must be a 2-element list, got: {ci_range}"
                raise ValueError(msg)

            source = LiteratureSource(
                citation=source_data["citation"],
                attractant=source_data["attractant"],
                ci_wild_type=source_data["ci_wild_type"],
                ci_range=(ci_range[0], ci_range[1]),
                conditions=source_data["conditions"],
            )
            sources.append(source)
        except KeyError as e:
            msg = f"Source {i}: missing required field {e}"
            raise ValueError(msg) from e

    # Parse thresholds
    threshold_data = data.get("validation_thresholds", {})
    thresholds = ValidationThresholds(
        minimum=threshold_data.get("biological_match_minimum", 0.4),
        target=threshold_data.get("biological_match_target", 0.6),
        excellent=threshold_data.get("biological_match_excellent", 0.75),
    )

    return ChemotaxisDataset(
        version=data.get("version", "1.0"),
        sources=sources,
        thresholds=thresholds,
    )


def _create_default_dataset() -> ChemotaxisDataset:
    """Create default dataset with key literature values.

    These values are from well-established C. elegans chemotaxis literature.
    """
    sources = [
        LiteratureSource(
            citation="Bargmann & Horvitz (1991). Cell 65(5):837-847",
            attractant="diacetyl",
            ci_wild_type=0.75,
            ci_range=(0.6, 0.9),
            conditions="Standard chemotaxis assay, 1:1000 dilution",
        ),
        LiteratureSource(
            citation="Bargmann et al. (1993). Cell 74(3):515-527",
            attractant="bacteria",
            ci_wild_type=0.7,
            ci_range=(0.5, 0.85),
            conditions="OP50 E. coli lawn, 20°C",
        ),
        LiteratureSource(
            citation="Saeki et al. (2001). Neuron 32(2):249-259",
            attractant="NaCl",
            ci_wild_type=0.6,
            ci_range=(0.4, 0.8),
            conditions="100mM NaCl, standard assay",
        ),
    ]

    return ChemotaxisDataset(
        version="1.0-default",
        sources=sources,
        thresholds=ValidationThresholds(),
    )


class ChemotaxisValidationBenchmark:
    """Benchmark for validating agent behavior against biological data.

    This class compares simulated agent chemotaxis metrics against published
    C. elegans data to determine biological plausibility.
    """

    def __init__(self, dataset: ChemotaxisDataset | None = None) -> None:
        """Initialize the benchmark.

        Args:
            dataset: Chemotaxis dataset to use. If None, loads default.
        """
        self.dataset = dataset or load_chemotaxis_dataset()

    def validate_agent(
        self,
        metrics: ChemotaxisMetrics,
        attractant: str | None = None,
    ) -> ValidationResult:
        """Validate agent chemotaxis metrics against biological data.

        Args:
            metrics: Agent's chemotaxis metrics
            attractant: Type of attractant to compare against (e.g., "bacteria").
                       If None, uses default (bacteria/food).

        Returns
        -------
            ValidationResult with comparison details
        """
        if attractant:
            source = self.dataset.get_source_by_attractant(attractant)
            if source is None:
                source = self.dataset.get_default_source()
        else:
            source = self.dataset.get_default_source()

        agent_ci = metrics.chemotaxis_index
        bio_range = source.ci_range

        # Check if agent falls within biological range
        matches = bio_range[0] <= agent_ci <= bio_range[1]

        # Determine validation level
        validation_level = get_validation_level(agent_ci)

        return ValidationResult(
            agent_ci=agent_ci,
            biological_ci_range=bio_range,
            biological_ci_typical=source.ci_wild_type,
            matches_biology=matches,
            validation_level=validation_level,
            literature_source=source.citation,
            agent_metrics=metrics,
        )

    def validate_multiple_runs(
        self,
        all_metrics: list[ChemotaxisMetrics],
        attractant: str | None = None,
    ) -> dict:
        """Validate metrics from multiple experiment runs.

        Args:
            all_metrics: List of ChemotaxisMetrics from multiple runs
            attractant: Type of attractant to compare against

        Returns
        -------
            Dictionary with aggregated validation results
        """
        if not all_metrics:
            return {
                "num_runs": 0,
                "mean_ci": 0.0,
                "std_ci": 0.0,
                "matches_biology_rate": 0.0,
                "validation_levels": {},
            }

        # Calculate statistics (using sample variance with Bessel's correction)
        cis = [m.chemotaxis_index for m in all_metrics]
        mean_ci = sum(cis) / len(cis)
        variance = sum((ci - mean_ci) ** 2 for ci in cis) / (len(cis) - 1) if len(cis) > 1 else 0.0
        std_ci = variance**0.5

        # Validate each run
        results = [self.validate_agent(m, attractant) for m in all_metrics]
        matches = sum(1 for r in results if r.matches_biology)

        # Count validation levels
        level_counts = {}
        for r in results:
            level = r.validation_level.value
            level_counts[level] = level_counts.get(level, 0) + 1

        return {
            "num_runs": len(all_metrics),
            "mean_ci": mean_ci,
            "std_ci": std_ci,
            "min_ci": min(cis),
            "max_ci": max(cis),
            "matches_biology_rate": matches / len(results),
            "validation_levels": level_counts,
            "individual_results": results,
        }


@dataclass
class BiasCurveReference:
    """A documented behaviour-level signature of a *C. elegans* klinotaxis strategy.

    A behaviour-level reference (bias direction + a reported magnitude range + citation), NOT a
    pixel-digitised curve. Used to grade a model bias statistic REPRODUCED / PARTIAL / ABSENT.

    Attributes
    ----------
        strategy: "klinokinesis" | "klinotaxis".
        statistic: The model bias-statistic name this reference is compared against.
        null_value: The no-bias value of the statistic (1.0 for a rate ratio, 0.0 for a slope).
        sign: +1 if the statistic should EXCEED null_value when the strategy is present.
        magnitude_range: (lo, hi) comparable literature range, or None for a sign-only reference.
        citation: The source paper.
        notes: Precision / unit-comparability caveats (behaviour-level, not figure-exact).
    """

    strategy: str
    statistic: str
    null_value: float
    sign: int
    magnitude_range: tuple[float, float] | None
    citation: str
    notes: str


def _bias_from_dict(d: dict) -> BiasCurveReference:
    mr = d.get("magnitude_range")
    return BiasCurveReference(
        strategy=d["strategy"],
        statistic=d["statistic"],
        null_value=float(d["null_value"]),
        sign=int(d["sign"]),
        magnitude_range=(float(mr[0]), float(mr[1])) if mr is not None else None,
        citation=d["citation"],
        notes=d["notes"],
    )


def _default_bias_signatures() -> dict[str, BiasCurveReference]:
    """Hardcoded fallback mirroring ``behavioural_bias_signatures.json`` (behaviour-level)."""
    return {
        "klinokinesis": BiasCurveReference(
            strategy="klinokinesis",
            statistic="down_up_turn_ratio",
            null_value=1.0,
            sign=1,
            magnitude_range=(1.5, 3.0),
            citation="Pierce-Shimomura, Morse & Lockery (1999). J Neurosci 19(21):9557-9569",
            notes=(
                "Pirouette-initiation rate is elevated heading down-gradient (dC/dt < 0); the "
                "down/up-gradient turn-rate ratio is ~2x. A dimensionless ratio (directly "
                "comparable). Approximate literature signature, not a figure digitization."
            ),
        ),
        "klinotaxis": BiasCurveReference(
            strategy="klinotaxis",
            statistic="weathervane_slope",
            null_value=0.0,
            sign=1,
            magnitude_range=None,
            citation="Iino & Yoshida (2009). J Neurosci 29(17):5370-5380",
            notes=(
                "The weathervane curves the trajectory toward the gradient (positive slope). Its "
                "magnitude in rad/mm-per-bearing is not comparable to the paper's "
                "deg/mm-per-normal-gradient parameterization, so this is a sign-only reference; a "
                "figure-digitized slope is a non-goal."
            ),
        ),
        "klinokinesis_magnitude": BiasCurveReference(
            strategy="klinokinesis",
            statistic="down_up_magnitude_ratio",
            null_value=1.0,
            sign=1,
            magnitude_range=None,
            citation="Pierce-Shimomura, Morse & Lockery (1999). J Neurosci 19(21):9557-9569",
            notes=(
                "Threshold-free companion to down_up_turn_ratio: mean |dtheta| down- vs "
                "up-gradient. Same direction, different unit (magnitude, not rate), so graded "
                "sign-only; a theta_sharp-independent robustness cross-check."
            ),
        ),
        "klinotaxis_all": BiasCurveReference(
            strategy="klinotaxis",
            statistic="weathervane_slope_all",
            null_value=0.0,
            sign=1,
            magnitude_range=None,
            citation="Iino & Yoshida (2009). J Neurosci 29(17):5370-5380",
            notes=(
                "Threshold-free companion to weathervane_slope: the curving-rate-vs-bearing slope "
                "over all usable steps (sharp reorientations not excluded), independent of "
                "theta_sharp. Sign-only reference; the weathervane robustness cross-check."
            ),
        ),
    }


def load_bias_signatures(path: str | Path | None = None) -> dict[str, BiasCurveReference]:
    """Load the behavioural bias-curve reference signatures (JSON, with a hardcoded fallback)."""
    resolved = _DEFAULT_BIAS_PATH if path is None else Path(path)
    if not resolved.exists():
        return _default_bias_signatures()
    with resolved.open() as f:
        data = json.load(f)
    return {key: _bias_from_dict(value) for key, value in data.items()}
