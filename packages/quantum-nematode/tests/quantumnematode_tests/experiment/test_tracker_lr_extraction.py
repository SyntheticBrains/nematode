"""Test learning rate extraction in experiment tracker."""

from quantumnematode.experiment.tracker import extract_brain_metadata


class TestLearningRateExtraction:
    """Test that brain metadata correctly extracts learning rates."""

    def test_spiking_brain_uses_brain_config_lr(self):
        """Spiking brain should prioritize brain.config.learning_rate over global LR."""
        config = {
            "config": {
                "hidden_size": 128,
                "learning_rate": 0.001,  # Brain-specific LR
                "num_timesteps": 100,
            },
            "learning_rate": {
                "initial_learning_rate": 0.1,  # Global LR (should be ignored)
            },
        }

        metadata = extract_brain_metadata("spiking", config)
        assert metadata.learning_rate == 0.001

    def test_mlp_brain_uses_brain_config_lr(self):
        """MLP brain should prioritize brain.config.learning_rate over global LR."""
        config = {
            "config": {
                "hidden_dim": 64,
                "learning_rate": 0.005,  # Brain-specific LR
            },
            "learning_rate": {
                "initial_learning_rate": 0.1,  # Global LR (should be ignored)
            },
        }

        metadata = extract_brain_metadata("mlp", config)
        assert metadata.learning_rate == 0.005

    def test_fallback_to_global_lr_when_brain_lr_missing(self):
        """Should fall back to global LR when brain config doesn't have learning_rate."""
        config = {
            "config": {
                "hidden_size": 128,
                # No learning_rate in brain config
            },
            "learning_rate": {
                "initial_learning_rate": 0.1,  # Global LR (should be used)
            },
        }

        metadata = extract_brain_metadata("spiking", config)
        assert metadata.learning_rate == 0.1

    def test_none_when_no_lr_anywhere(self):
        """Should return None when no LR is specified anywhere."""
        config = {
            "config": {
                "hidden_size": 128,
            },
            # No global learning_rate section
        }

        metadata = extract_brain_metadata("spiking", config)
        assert metadata.learning_rate is None

    def test_zero_learning_rate_is_valid(self):
        """Zero learning rate should be used when explicitly set (not fall back to global)."""
        config = {
            "config": {
                "hidden_size": 128,
                "learning_rate": 0.0,  # Explicitly zero
            },
            "learning_rate": {
                "initial_learning_rate": 0.1,  # Should be ignored
            },
        }

        metadata = extract_brain_metadata("spiking", config)
        assert metadata.learning_rate == 0.0
