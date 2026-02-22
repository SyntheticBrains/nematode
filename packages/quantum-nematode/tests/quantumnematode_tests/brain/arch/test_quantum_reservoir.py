"""Unit tests for shared quantum reservoir utilities (_quantum_reservoir.py)."""

import torch
from quantumnematode.brain.arch._quantum_reservoir import build_readout_network


class TestBuildReadoutNetwork:
    """Test cases for build_readout_network()."""

    def test_mlp_readout_shape(self):
        """MLP readout should produce correct output shape."""
        network = build_readout_network(
            input_dim=36,
            hidden_dim=64,
            output_dim=4,
            readout_type="mlp",
        )

        x = torch.randn(36)
        output = network(x)
        assert output.shape == (4,)

    def test_linear_readout_shape(self):
        """Linear readout should produce correct output shape."""
        network = build_readout_network(
            input_dim=36,
            hidden_dim=64,  # ignored for linear
            output_dim=4,
            readout_type="linear",
        )

        x = torch.randn(36)
        output = network(x)
        assert output.shape == (4,)

    def test_mlp_readout_is_sequential(self):
        """MLP readout should be an nn.Sequential."""
        network = build_readout_network(
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            readout_type="mlp",
        )
        assert isinstance(network, torch.nn.Sequential)

    def test_linear_readout_is_linear(self):
        """Linear readout should be an nn.Linear."""
        network = build_readout_network(
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            readout_type="linear",
        )
        assert isinstance(network, torch.nn.Linear)

    def test_mlp_single_hidden_layer(self):
        """MLP with num_layers=1 should have [Linear, ReLU, Linear]."""
        network = build_readout_network(
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            readout_type="mlp",
            num_layers=1,
        )
        modules = list(network.children())
        assert len(modules) == 3
        assert isinstance(modules[0], torch.nn.Linear)
        assert isinstance(modules[1], torch.nn.ReLU)
        assert isinstance(modules[2], torch.nn.Linear)

        # Check dimensions
        assert modules[0].in_features == 8
        assert modules[0].out_features == 16
        assert modules[2].in_features == 16
        assert modules[2].out_features == 4

    def test_mlp_two_hidden_layers(self):
        """MLP with num_layers=2 should have [Linear, ReLU, Linear, ReLU, Linear]."""
        network = build_readout_network(
            input_dim=36,
            hidden_dim=64,
            output_dim=4,
            readout_type="mlp",
            num_layers=2,
        )
        modules = list(network.children())
        assert len(modules) == 5
        assert isinstance(modules[0], torch.nn.Linear)  # 36 -> 64
        assert isinstance(modules[1], torch.nn.ReLU)
        assert isinstance(modules[2], torch.nn.Linear)  # 64 -> 64
        assert isinstance(modules[3], torch.nn.ReLU)
        assert isinstance(modules[4], torch.nn.Linear)  # 64 -> 4

        assert modules[0].in_features == 36
        assert modules[0].out_features == 64
        assert modules[2].in_features == 64
        assert modules[2].out_features == 64
        assert modules[4].in_features == 64
        assert modules[4].out_features == 4

    def test_orthogonal_initialization(self):
        """Weights should be orthogonal-initialized (non-trivial check)."""
        network = build_readout_network(
            input_dim=16,
            hidden_dim=16,
            output_dim=4,
            readout_type="mlp",
        )

        for module in network.modules():
            if isinstance(module, torch.nn.Linear):
                w = module.weight.data
                # Orthogonal matrices satisfy W @ W^T ≈ I (scaled)
                # For rectangular matrices, check the smaller dimension
                if w.shape[0] <= w.shape[1]:
                    product = w @ w.T
                    # Should be close to a scaled identity
                    off_diag = product - torch.diag(torch.diag(product))
                    assert off_diag.abs().max() < 0.1, "Off-diagonal elements should be near zero"

                # Bias should be zeros
                if module.bias is not None:
                    assert torch.all(module.bias == 0.0)

    def test_batch_input(self):
        """Network should handle batch inputs."""
        network = build_readout_network(
            input_dim=36,
            hidden_dim=64,
            output_dim=4,
            readout_type="mlp",
        )

        batch = torch.randn(8, 36)
        output = network(batch)
        assert output.shape == (8, 4)

    def test_value_network_single_output(self):
        """Should support single output for critic/value networks."""
        network = build_readout_network(
            input_dim=36,
            hidden_dim=64,
            output_dim=1,
            readout_type="mlp",
            num_layers=2,
        )

        x = torch.randn(36)
        output = network(x)
        assert output.shape == (1,)
