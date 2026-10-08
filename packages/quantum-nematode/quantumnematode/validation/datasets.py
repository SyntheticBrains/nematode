"""Published C. elegans chemotaxis reference values, and the behavioural bias-curve references.

The chemotaxis reference set lists wild-type chemotaxis indices as each cited paper reports them,
with what was assayed and whether the value was stated in the text or read from a figure. Every
published index here is an endpoint count of a population of worms; the simulated index is a
time-in-zone fraction for one animal, so the two are not compared.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Project root and default dataset path for clearer path resolution
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_DATASET_PATH = _PROJECT_ROOT / "data" / "chemotaxis" / "literature_ci_values.json"
_DEFAULT_BIAS_PATH = _PROJECT_ROOT / "data" / "chemotaxis" / "behavioural_bias_signatures.json"
_DEFAULT_BIAS_PATH_THERMOTAXIS = (
    _PROJECT_ROOT / "data" / "thermotaxis" / "behavioural_bias_signatures.json"
)
_BIAS_PATH_BY_MODALITY = {"food": _DEFAULT_BIAS_PATH, "thermotaxis": _DEFAULT_BIAS_PATH_THERMOTAXIS}
_READ_FROM = ("text", "figure")


@dataclass
class LiteratureSource:
    """One published wild-type chemotaxis index, as its paper reports it.

    Attributes
    ----------
        citation: The paper.
        attractant: What the worms were attracted to, with its dilution where one applies.
        ci_wild_type: The reported wild-type chemotaxis index.
        assay: How the paper measured it.
        read_from: ``"text"`` if the paper states the value, ``"figure"`` if it is read from a plot.
        notes: Where in the paper, and any caveat on the value.
    """

    citation: str
    attractant: str
    ci_wild_type: float
    assay: str
    read_from: Literal["text", "figure"]
    notes: str


@dataclass
class ChemotaxisDataset:
    """The chemotaxis reference set.

    Attributes
    ----------
        version: Dataset version string.
        sources: The verified reference values.
    """

    version: str
    sources: list[LiteratureSource]

    def get_source_by_attractant(self, attractant: str) -> LiteratureSource | None:
        """Find the first source for an attractant, matched case-insensitively."""
        for source in self.sources:
            if source.attractant.lower() == attractant.lower():
                return source
        return None


def load_chemotaxis_dataset(dataset_path: str | Path | None = None) -> ChemotaxisDataset:
    """Load the chemotaxis reference set from JSON.

    Args:
        dataset_path: Path to the JSON file. If None, uses the packaged reference set.

    Returns
    -------
        ChemotaxisDataset loaded from file

    Raises
    ------
        FileNotFoundError: If the file does not exist.
        ValueError: If an entry lacks a required field or names an unknown value source.
    """
    dataset_path = _DEFAULT_DATASET_PATH if dataset_path is None else Path(dataset_path)
    with dataset_path.open() as f:
        data = json.load(f)

    sources = []
    for i, source_data in enumerate(data.get("sources", [])):
        try:
            read_from = source_data["read_from"]
            if read_from not in _READ_FROM:
                msg = f"Source {i}: read_from must be one of {_READ_FROM}, got: {read_from!r}"
                raise ValueError(msg)
            sources.append(
                LiteratureSource(
                    citation=source_data["citation"],
                    attractant=source_data["attractant"],
                    ci_wild_type=float(source_data["ci_wild_type"]),
                    assay=source_data["assay"],
                    read_from=read_from,
                    notes=source_data.get("notes", ""),
                ),
            )
        except KeyError as e:
            msg = f"Source {i}: missing required field {e}"
            raise ValueError(msg) from e

    return ChemotaxisDataset(version=data.get("version", "1.0"), sources=sources)


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


def _bias_from_dict(d: dict[str, Any]) -> BiasCurveReference:
    """Build a ``BiasCurveReference`` from one JSON object.

    Parameters
    ----------
    d : dict[str, Any]
        A mapping with keys ``strategy``, ``statistic``, ``null_value``, ``sign``,
        ``magnitude_range`` (a ``[lo, hi]`` list or ``null``), ``citation`` and ``notes``.

    Returns
    -------
    BiasCurveReference
        The parsed reference, with ``magnitude_range`` coerced to a ``(lo, hi)`` tuple or ``None``.
    """
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


def _default_bias_signatures_thermotaxis() -> dict[str, BiasCurveReference]:
    """Hardcoded thermotaxis fallback mirroring ``thermotaxis/behavioural_bias_signatures.json``.

    Thermotaxis is homeostatic: the drive is the setpoint error toward the cultivation temperature
    (``-|T - Tc|``), so all four references are sign-only (direction, not literature-comparable
    magnitude) — turning/curving is biased toward the cultivation temperature.
    """
    kin = "Luo et al. (2014). J Neurosci 34(13):4655-4667; Ryu & Samuel (2002). J Neurosci 22(13):5727-5733"  # noqa: E501
    luo = "Luo et al. (2014). J Neurosci 34(13):4655-4667; Clark et al. (2007). J Neurosci 27(23):6083-6090"  # noqa: E501
    return {
        "klinokinesis": BiasCurveReference(
            strategy="klinokinesis",
            statistic="down_up_turn_ratio",
            null_value=1.0,
            sign=1,
            magnitude_range=None,
            citation=kin,
            notes="Turn-rate elevated when the thermal error worsens (drive decreases). Sign-only.",
        ),
        "klinokinesis_magnitude": BiasCurveReference(
            strategy="klinokinesis",
            statistic="down_up_magnitude_ratio",
            null_value=1.0,
            sign=1,
            magnitude_range=None,
            citation=kin,
            notes="Threshold-free companion: larger |dtheta| heading away from Tc. Sign-only.",
        ),
        "klinotaxis": BiasCurveReference(
            strategy="klinotaxis",
            statistic="weathervane_slope",
            null_value=0.0,
            sign=1,
            magnitude_range=None,
            citation=luo,
            notes="Gradual curving toward the preferred thermal direction (toward Tc). Sign-only.",
        ),
        "klinotaxis_all": BiasCurveReference(
            strategy="klinotaxis",
            statistic="weathervane_slope_all",
            null_value=0.0,
            sign=1,
            magnitude_range=None,
            citation=luo,
            notes="Threshold-free companion to the thermal weathervane slope. Sign-only.",
        ),
    }


def load_bias_signatures(
    path: str | Path | None = None,
    *,
    modality: str = "food",
) -> dict[str, BiasCurveReference]:
    """Load the behavioural bias-curve reference signatures from JSON.

    Parameters
    ----------
    path : str | Path | None
        An explicit signatures file. When omitted, the packaged default for ``modality`` is used,
        falling back to the hardcoded defaults if that file is absent. A *caller-supplied* path that
        does not exist is an error (raised), not silently replaced with the defaults.
    modality : str
        Which reference set the omitted-``path`` default resolves to: ``"food"`` (chemotaxis;
        Pierce-Shimomura / Iino & Yoshida) or ``"thermotaxis"`` (setpoint drive; Ryu & Samuel /
        thermal klinotaxis). Ignored when an explicit ``path`` is given.

    Returns
    -------
    dict[str, BiasCurveReference]
        The reference signatures keyed by their reference key.

    Raises
    ------
    FileNotFoundError
        If an explicit ``path`` is given but does not exist.
    ValueError
        If ``modality`` is not one of the known reference sets (rather than silently grading
        against the food references).
    """
    if modality not in _BIAS_PATH_BY_MODALITY:
        msg = f"Unknown modality {modality!r}; expected one of {sorted(_BIAS_PATH_BY_MODALITY)}."
        raise ValueError(msg)
    default_path = _BIAS_PATH_BY_MODALITY[modality]
    resolved = default_path if path is None else Path(path)
    if not resolved.exists():
        if path is not None:
            msg = f"Bias-signatures file not found: {resolved}"
            raise FileNotFoundError(msg)
        # implicit default missing -> canonical hardcoded values
        return (
            _default_bias_signatures_thermotaxis()
            if modality == "thermotaxis"
            else _default_bias_signatures()
        )
    with resolved.open() as f:
        data = json.load(f)
    return {key: _bias_from_dict(value) for key, value in data.items()}
