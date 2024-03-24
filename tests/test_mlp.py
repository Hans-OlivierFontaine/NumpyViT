import unittest

import numpy as np
from copy import deepcopy
from pathlib import Path

from model.mlp import ViTMLP


class MLPTests(unittest.TestCase):
    def test_forward_propagation(self):
        mlp_num_hiddens, mlp_num_outputs, dropout = 128, 10, 0.5
        batch_size, input_dim = 4, 512

        vit_mlp = ViTMLP(input_dim, mlp_num_hiddens, mlp_num_outputs, dropout)
        X = np.random.rand(batch_size, input_dim)

        output = vit_mlp.forward(X)

        self.assertTrue(isinstance(output, np.ndarray), "Output should be a numpy array")
        self.assertTrue(output.shape == (
            batch_size, mlp_num_outputs), f"Output shape should be ({batch_size}, {mlp_num_outputs})")

    def test_parameters_update(self):
        mlp_num_hiddens, mlp_num_outputs, dropout = 128, 10, 0.0  # Dropout is disabled to simplify testing
        batch_size, input_dim = 4, 512

        vit_mlp = ViTMLP(input_dim, mlp_num_hiddens, mlp_num_outputs, dropout)

        initial_weights_dense1 = np.copy(vit_mlp.weights1)
        initial_bias_dense1 = np.copy(vit_mlp.bias1)
        initial_weights_dense2 = np.copy(vit_mlp.weights2)
        initial_bias_dense2 = np.copy(vit_mlp.bias2)

        X = np.random.rand(batch_size, input_dim)
        output = vit_mlp.forward(X)

        dOutput = np.random.rand(*output.shape)
        vit_mlp.backward(dOutput)

        self.assertFalse(np.array_equal(initial_weights_dense1,
                                        vit_mlp.weights1), "Weights of dense1 should be updated.")
        self.assertFalse(np.array_equal(initial_bias_dense1, vit_mlp.bias1), "Bias of dense1 should be updated.")
        self.assertFalse(np.array_equal(initial_weights_dense2,
                                        vit_mlp.weights2), "Weights of dense2 should be updated.")
        self.assertFalse(np.array_equal(initial_bias_dense2, vit_mlp.bias2), "Bias of dense2 should be updated.")

    def test_output_value_range(self):
        mlp_num_hiddens, mlp_num_outputs, dropout = 128, 10, 0.5
        batch_size, input_dim = 4, 512

        vit_mlp = ViTMLP(input_dim, mlp_num_hiddens, mlp_num_outputs, dropout)
        X = np.random.rand(batch_size, input_dim) * 2 - 1  # Uniform distribution in [-1, 1]

        output = vit_mlp.forward(X)

        self.assertTrue(np.all(output >= -10) and np.all(output <= 10),
                        "Output values should be within a reasonable range.")

    def test_save_load(self):
        """Check that the save and load functions work."""
        mlp_num_hiddens, mlp_num_outputs, dropout = 128, 10, 0.5
        batch_size, input_dim = 4, 512

        vit_mlp = ViTMLP(input_dim, mlp_num_hiddens, mlp_num_outputs, dropout)

        original_weights1 = deepcopy(vit_mlp.weights1)
        original_bias1 = deepcopy(vit_mlp.bias1)
        original_weights2 = deepcopy(vit_mlp.weights2)
        original_bias2 = deepcopy(vit_mlp.bias2)

        (Path(__file__).parent / "assets").mkdir(exist_ok=True, parents=True)
        vit_mlp.save((Path(__file__).parent / "assets"), "mlp_test")

        vit_mlp.weights1 += 1
        vit_mlp.bias1 += 1
        vit_mlp.weights2 += 1
        vit_mlp.bias2 += 1

        self.assertFalse(np.array_equal(original_weights1, vit_mlp.weights1))
        self.assertFalse(np.array_equal(original_bias1, vit_mlp.bias1))
        self.assertFalse(np.array_equal(original_weights2, vit_mlp.weights2))
        self.assertFalse(np.array_equal(original_bias2, vit_mlp.bias2))

        vit_mlp.load((Path(__file__).parent / "assets"), "mlp_test")

        self.assertTrue(np.array_equal(original_weights1, vit_mlp.weights1))
        self.assertTrue(np.array_equal(original_bias1, vit_mlp.bias1))
        self.assertTrue(np.array_equal(original_weights2, vit_mlp.weights2))
        self.assertTrue(np.array_equal(original_bias2, vit_mlp.bias2))


if __name__ == '__main__':
    unittest.main()
