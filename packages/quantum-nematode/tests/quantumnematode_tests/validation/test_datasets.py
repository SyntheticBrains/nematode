"""Tests for chemotaxis dataset loading and validation benchmark."""

import json
import tempfile
from pathlib import Path

import pytest
from quantumnematode.validation.chemotaxis import (
    ChemotaxisMetrics,
    ValidationLevel,
)
from quantumnematode.validation.datasets import (
    ChemotaxisDataset,
    ChemotaxisValidationBenchmark,
    LiteratureSource,
    ValidationThresholds,
    load_chemotaxis_dataset,
)


class TestLiteratureSource:
    """Test LiteratureSource dataclass."""

    def test_literature_source_creation(self):
        """Test creating a literature source."""
        source = LiteratureSource(
            citation="Bargmann & Horvitz (1991). Cell 65(5):837-847",
            attractant="diacetyl",
            ci_wild_type=0.75,
            ci_range=(0.6, 0.9),
            conditions="Standard chemotaxis assay",
        )

        assert source.citation == "Bargmann & Horvitz (1991). Cell 65(5):837-847"
        assert source.attractant == "diacetyl"
        assert source.ci_wild_type == 0.75
        assert source.ci_range == (0.6, 0.9)
        assert source.conditions == "Standard chemotaxis assay"


class TestValidationThresholds:
    """Test ValidationThresholds dataclass."""

    def test_default_thresholds(self):
        """Test default validation thresholds."""
        thresholds = ValidationThresholds()

        assert thresholds.minimum == 0.4
        assert thresholds.target == 0.6
        assert thresholds.excellent == 0.75

    def test_custom_thresholds(self):
        """Test custom validation thresholds."""
        thresholds = ValidationThresholds(
            minimum=0.3,
            target=0.5,
            excellent=0.7,
        )

        assert thresholds.minimum == 0.3
        assert thresholds.target == 0.5
        assert thresholds.excellent == 0.7


class TestChemotaxisDataset:
    """Test ChemotaxisDataset class."""

    @pytest.fixture
    def sample_dataset(self) -> ChemotaxisDataset:
        """Create a sample dataset for testing."""
        sources = [
            LiteratureSource(
                citation="Test Citation 1",
                attractant="bacteria",
                ci_wild_type=0.7,
                ci_range=(0.5, 0.85),
                conditions="Test conditions",
            ),
            LiteratureSource(
                citation="Test Citation 2",
                attractant="diacetyl",
                ci_wild_type=0.75,
                ci_range=(0.6, 0.9),
                conditions="Test conditions",
            ),
            LiteratureSource(
                citation="Test Citation 3",
                attractant="NaCl",
                ci_wild_type=0.6,
                ci_range=(0.4, 0.8),
                conditions="Test conditions",
            ),
        ]
        return ChemotaxisDataset(
            version="1.0-test",
            sources=sources,
            thresholds=ValidationThresholds(),
        )

    def test_get_source_by_attractant_found(self, sample_dataset):
        """Test finding source by attractant type."""
        source = sample_dataset.get_source_by_attractant("bacteria")

        assert source is not None
        assert source.attractant == "bacteria"
        assert source.ci_wild_type == 0.7

    def test_get_source_by_attractant_case_insensitive(self, sample_dataset):
        """Test attractant lookup is case-insensitive."""
        source_lower = sample_dataset.get_source_by_attractant("bacteria")
        source_upper = sample_dataset.get_source_by_attractant("BACTERIA")
        source_mixed = sample_dataset.get_source_by_attractant("Bacteria")

        assert source_lower is not None
        assert source_upper is not None
        assert source_mixed is not None
        assert source_lower.citation == source_upper.citation == source_mixed.citation

    def test_get_source_by_attractant_not_found(self, sample_dataset):
        """Test None returned when attractant not found."""
        source = sample_dataset.get_source_by_attractant("unknown_attractant")

        assert source is None

    def test_get_default_source_bacteria(self, sample_dataset):
        """Test default source returns bacteria when available."""
        source = sample_dataset.get_default_source()

        assert source.attractant == "bacteria"

    def test_get_default_source_fallback_diacetyl(self):
        """Test default source falls back to diacetyl if no bacteria."""
        sources = [
            LiteratureSource(
                citation="Test",
                attractant="diacetyl",
                ci_wild_type=0.75,
                ci_range=(0.6, 0.9),
                conditions="Test",
            ),
        ]
        dataset = ChemotaxisDataset(
            version="1.0",
            sources=sources,
            thresholds=ValidationThresholds(),
        )

        source = dataset.get_default_source()
        assert source.attractant == "diacetyl"

    def test_get_default_source_fallback_first(self):
        """Test default source falls back to first if no preferred attractants."""
        sources = [
            LiteratureSource(
                citation="Test",
                attractant="unknown",
                ci_wild_type=0.5,
                ci_range=(0.3, 0.7),
                conditions="Test",
            ),
        ]
        dataset = ChemotaxisDataset(
            version="1.0",
            sources=sources,
            thresholds=ValidationThresholds(),
        )

        source = dataset.get_default_source()
        assert source.attractant == "unknown"


class TestLoadChemotaxisDataset:
    """Test dataset loading from JSON."""

    def test_load_from_valid_json(self):
        """Test loading dataset from a valid JSON file."""
        data = {
            "version": "1.0",
            "sources": [
                {
                    "citation": "Test Citation",
                    "attractant": "bacteria",
                    "ci_wild_type": 0.7,
                    "ci_range": [0.5, 0.85],
                    "conditions": "Test conditions",
                },
            ],
            "validation_thresholds": {
                "biological_match_minimum": 0.35,
                "biological_match_target": 0.55,
                "biological_match_excellent": 0.7,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            dataset = load_chemotaxis_dataset(temp_path)

            assert dataset.version == "1.0"
            assert len(dataset.sources) == 1
            assert dataset.sources[0].attractant == "bacteria"
            assert dataset.sources[0].ci_range == (0.5, 0.85)
            assert dataset.thresholds.minimum == 0.35
            assert dataset.thresholds.target == 0.55
            assert dataset.thresholds.excellent == 0.7
        finally:
            Path(temp_path).unlink()

    def test_load_from_path_object(self):
        """Test loading dataset using Path object."""
        data = {
            "version": "1.0",
            "sources": [
                {
                    "citation": "Test",
                    "attractant": "diacetyl",
                    "ci_wild_type": 0.75,
                    "ci_range": [0.6, 0.9],
                    "conditions": "Test",
                },
            ],
            "validation_thresholds": {},
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            dataset = load_chemotaxis_dataset(temp_path)
            assert len(dataset.sources) == 1
        finally:
            temp_path.unlink()

    def test_load_nonexistent_file_returns_default(self):
        """Test loading from nonexistent file returns default dataset."""
        dataset = load_chemotaxis_dataset("/nonexistent/path/to/file.json")

        # Should return default dataset
        assert dataset.version == "1.0-default"
        assert len(dataset.sources) >= 1
        assert dataset.thresholds is not None

    def test_load_with_missing_thresholds_uses_defaults(self):
        """Test that missing thresholds use default values."""
        data = {
            "version": "1.0",
            "sources": [
                {
                    "citation": "Test",
                    "attractant": "bacteria",
                    "ci_wild_type": 0.7,
                    "ci_range": [0.5, 0.85],
                    "conditions": "Test",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            dataset = load_chemotaxis_dataset(temp_path)

            # Should use default thresholds
            assert dataset.thresholds.minimum == 0.4
            assert dataset.thresholds.target == 0.6
            assert dataset.thresholds.excellent == 0.75
        finally:
            Path(temp_path).unlink()

    def test_load_multiple_sources(self):
        """Test loading dataset with multiple sources."""
        data = {
            "version": "2.0",
            "sources": [
                {
                    "citation": "Citation 1",
                    "attractant": "bacteria",
                    "ci_wild_type": 0.7,
                    "ci_range": [0.5, 0.85],
                    "conditions": "Conditions 1",
                },
                {
                    "citation": "Citation 2",
                    "attractant": "diacetyl",
                    "ci_wild_type": 0.75,
                    "ci_range": [0.6, 0.9],
                    "conditions": "Conditions 2",
                },
            ],
            "validation_thresholds": {},
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            dataset = load_chemotaxis_dataset(temp_path)

            assert len(dataset.sources) == 2
            assert dataset.sources[0].attractant == "bacteria"
            assert dataset.sources[1].attractant == "diacetyl"
        finally:
            Path(temp_path).unlink()


class TestChemotaxisValidationBenchmark:
    """Test the validation benchmark class."""

    @pytest.fixture
    def benchmark(self) -> ChemotaxisValidationBenchmark:
        """Create a benchmark with sample dataset."""
        sources = [
            LiteratureSource(
                citation="Bargmann et al. (1993). Cell 74(3):515-527",
                attractant="bacteria",
                ci_wild_type=0.7,
                ci_range=(0.5, 0.85),
                conditions="OP50 E. coli lawn",
            ),
            LiteratureSource(
                citation="Bargmann & Horvitz (1991). Cell 65(5):837-847",
                attractant="diacetyl",
                ci_wild_type=0.75,
                ci_range=(0.6, 0.9),
                conditions="Standard assay",
            ),
        ]
        dataset = ChemotaxisDataset(
            version="test",
            sources=sources,
            thresholds=ValidationThresholds(),
        )
        return ChemotaxisValidationBenchmark(dataset)

    @pytest.fixture
    def sample_metrics(self) -> ChemotaxisMetrics:
        """Create sample metrics for testing."""
        return ChemotaxisMetrics(
            chemotaxis_index=0.65,
            time_in_attractant=0.7,
            approach_frequency=0.8,
            path_efficiency=0.5,
            total_steps=100,
            steps_in_attractant=70,
            steps_in_control=30,
        )

    def test_validate_agent_matches_biology(self, benchmark, sample_metrics):
        """Test validation when agent matches biological range."""
        result = benchmark.validate_agent(sample_metrics)

        assert result.agent_ci == 0.65
        assert result.matches_biology is True  # 0.65 is within (0.5, 0.85)
        assert result.biological_ci_range == (0.5, 0.85)
        assert result.validation_level == ValidationLevel.TARGET

    def test_validate_agent_below_biological_range(self, benchmark):
        """Test validation when agent is below biological range."""
        metrics = ChemotaxisMetrics(
            chemotaxis_index=0.3,
            time_in_attractant=0.3,
            approach_frequency=0.4,
            path_efficiency=0.3,
            total_steps=100,
            steps_in_attractant=30,
            steps_in_control=70,
        )

        result = benchmark.validate_agent(metrics)

        assert result.agent_ci == 0.3
        assert result.matches_biology is False  # 0.3 is below (0.5, 0.85)
        assert result.validation_level == ValidationLevel.NONE

    def test_validate_agent_above_biological_range(self, benchmark):
        """Test validation when agent exceeds biological range."""
        metrics = ChemotaxisMetrics(
            chemotaxis_index=0.95,
            time_in_attractant=0.95,
            approach_frequency=0.95,
            path_efficiency=0.9,
            total_steps=100,
            steps_in_attractant=95,
            steps_in_control=5,
        )

        result = benchmark.validate_agent(metrics)

        assert result.agent_ci == 0.95
        assert result.matches_biology is False  # 0.95 is above (0.5, 0.85)
        assert result.validation_level == ValidationLevel.EXCELLENT

    def test_validate_agent_specific_attractant(self, benchmark, sample_metrics):
        """Test validation against specific attractant."""
        result = benchmark.validate_agent(sample_metrics, attractant="diacetyl")

        assert result.biological_ci_range == (0.6, 0.9)
        assert "Bargmann & Horvitz" in result.literature_source

    def test_validate_agent_unknown_attractant_uses_default(
        self,
        benchmark,
        sample_metrics,
    ):
        """Test that unknown attractant falls back to default."""
        result = benchmark.validate_agent(sample_metrics, attractant="unknown")

        # Should use default (bacteria)
        assert result.biological_ci_range == (0.5, 0.85)

    def test_validate_agent_includes_metrics(self, benchmark, sample_metrics):
        """Test that result includes the original metrics."""
        result = benchmark.validate_agent(sample_metrics)

        assert result.agent_metrics == sample_metrics
        assert result.agent_metrics.total_steps == 100

    def test_validate_multiple_runs_empty(self, benchmark):
        """Test validating empty list of runs."""
        result = benchmark.validate_multiple_runs([])

        assert result["num_runs"] == 0
        assert result["mean_ci"] == 0.0
        assert result["std_ci"] == 0.0
        assert result["matches_biology_rate"] == 0.0

    def test_validate_multiple_runs_single_run(self, benchmark, sample_metrics):
        """Test validating single run."""
        result = benchmark.validate_multiple_runs([sample_metrics])

        assert result["num_runs"] == 1
        assert result["mean_ci"] == 0.65
        assert result["std_ci"] == 0.0  # No variance with single run
        assert result["matches_biology_rate"] == 1.0

    def test_validate_multiple_runs_statistics(self, benchmark):
        """Test statistics calculation for multiple runs."""
        metrics_list = [
            ChemotaxisMetrics(
                chemotaxis_index=0.6,
                time_in_attractant=0.6,
                approach_frequency=0.6,
                path_efficiency=0.5,
                total_steps=100,
                steps_in_attractant=60,
                steps_in_control=40,
            ),
            ChemotaxisMetrics(
                chemotaxis_index=0.7,
                time_in_attractant=0.7,
                approach_frequency=0.7,
                path_efficiency=0.6,
                total_steps=100,
                steps_in_attractant=70,
                steps_in_control=30,
            ),
            ChemotaxisMetrics(
                chemotaxis_index=0.8,
                time_in_attractant=0.8,
                approach_frequency=0.8,
                path_efficiency=0.7,
                total_steps=100,
                steps_in_attractant=80,
                steps_in_control=20,
            ),
        ]

        result = benchmark.validate_multiple_runs(metrics_list)

        assert result["num_runs"] == 3
        assert abs(result["mean_ci"] - 0.7) < 0.001
        assert result["min_ci"] == 0.6
        assert result["max_ci"] == 0.8
        # All within (0.5, 0.85)
        assert result["matches_biology_rate"] == 1.0
        assert len(result["individual_results"]) == 3

    def test_validate_multiple_runs_mixed_results(self, benchmark):
        """Test validation with mixed passing/failing runs."""
        metrics_list = [
            ChemotaxisMetrics(
                chemotaxis_index=0.7,  # Within range
                time_in_attractant=0.7,
                approach_frequency=0.7,
                path_efficiency=0.6,
                total_steps=100,
                steps_in_attractant=70,
                steps_in_control=30,
            ),
            ChemotaxisMetrics(
                chemotaxis_index=0.3,  # Below range
                time_in_attractant=0.3,
                approach_frequency=0.4,
                path_efficiency=0.3,
                total_steps=100,
                steps_in_attractant=30,
                steps_in_control=70,
            ),
        ]

        result = benchmark.validate_multiple_runs(metrics_list)

        assert result["num_runs"] == 2
        assert result["matches_biology_rate"] == 0.5  # 1 of 2

    def test_validate_multiple_runs_level_counts(self, benchmark):
        """Test validation level counting."""
        metrics_list = [
            ChemotaxisMetrics(
                chemotaxis_index=0.8,  # Excellent
                time_in_attractant=0.8,
                approach_frequency=0.8,
                path_efficiency=0.7,
                total_steps=100,
                steps_in_attractant=80,
                steps_in_control=20,
            ),
            ChemotaxisMetrics(
                chemotaxis_index=0.65,  # Target
                time_in_attractant=0.65,
                approach_frequency=0.65,
                path_efficiency=0.5,
                total_steps=100,
                steps_in_attractant=65,
                steps_in_control=35,
            ),
            ChemotaxisMetrics(
                chemotaxis_index=0.45,  # Minimum
                time_in_attractant=0.45,
                approach_frequency=0.5,
                path_efficiency=0.4,
                total_steps=100,
                steps_in_attractant=45,
                steps_in_control=55,
            ),
            ChemotaxisMetrics(
                chemotaxis_index=0.2,  # None
                time_in_attractant=0.2,
                approach_frequency=0.3,
                path_efficiency=0.2,
                total_steps=100,
                steps_in_attractant=20,
                steps_in_control=80,
            ),
        ]

        result = benchmark.validate_multiple_runs(metrics_list)

        assert result["validation_levels"]["excellent"] == 1
        assert result["validation_levels"]["target"] == 1
        assert result["validation_levels"]["minimum"] == 1
        assert result["validation_levels"]["none"] == 1

    def test_benchmark_default_dataset(self):
        """Test benchmark initializes with default dataset if none provided."""
        benchmark = ChemotaxisValidationBenchmark()

        assert benchmark.dataset is not None
        assert len(benchmark.dataset.sources) > 0


class TestIntegrationWithRealDataset:
    """Integration tests using the actual literature dataset."""

    def test_load_actual_dataset(self):
        """Test loading the actual literature_ci_values.json if it exists."""
        # This tests the real dataset file
        dataset = load_chemotaxis_dataset()

        # Should have some sources
        assert len(dataset.sources) >= 1

        # Should have bacteria source (most relevant for our simulation)
        bacteria_source = dataset.get_source_by_attractant("bacteria")
        if bacteria_source:
            assert bacteria_source.ci_wild_type > 0
            assert bacteria_source.ci_range[0] < bacteria_source.ci_range[1]

    def test_validate_typical_agent_performance(self):
        """Test validating typical agent performance against real dataset."""
        benchmark = ChemotaxisValidationBenchmark()

        # Simulate a moderately successful agent
        metrics = ChemotaxisMetrics(
            chemotaxis_index=0.6,
            time_in_attractant=0.65,
            approach_frequency=0.7,
            path_efficiency=0.4,
            total_steps=500,
            steps_in_attractant=325,
            steps_in_control=175,
        )

        result = benchmark.validate_agent(metrics)

        # Should be target level (CI >= 0.6)
        assert result.validation_level == ValidationLevel.TARGET
        assert result.agent_ci == 0.6
