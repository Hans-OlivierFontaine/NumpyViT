import unittest

import numpy as np

from model.layer_norm import LayerNorm


class MyTestCase(unittest.TestCase):
    def test_forward_output_shape_and_normalization(self):
        normalized_shape = (10,)
        layer_norm = LayerNorm(normalized_shape)
        x = np.random.randn(5, 10)

        output = layer_norm.forward(x)

        self.assertTrue(output.shape == x.shape, "Output shape should match input shape.")
        self.assertTrue(np.allclose(output.mean(axis=-1), 0, atol=1e-6),
                        "Mean of each output vector should be close to 0.")
        self.assertTrue(np.allclose(output.std(axis=-1), 1, atol=1e-4),
                        "Std of each output vector should be close to 1.")

    def test_backward_and_parameter_updates(self):
        normalized_shape = (10,)
        layer_norm = LayerNorm(normalized_shape)
        x = np.random.randn(5, 10)
        original_gamma = np.copy(layer_norm.gamma)
        original_beta = np.copy(layer_norm.beta)

        # Simulate backward pass
        output = layer_norm.forward(x)
        d_out = np.random.randn(*output.shape)
        layer_norm.backward(d_out)

        self.assertFalse(np.array_equal(original_gamma, layer_norm.gamma), "Gamma should be updated.")
        self.assertFalse(np.array_equal(original_beta, layer_norm.beta), "Beta should be updated.")


if __name__ == '__main__':
    unittest.main()
