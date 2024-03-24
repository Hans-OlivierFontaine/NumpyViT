import unittest
import numpy as np
from linear import Linear


class TestLinearLayer(unittest.TestCase):
    def test_forward_output_shape(self):
        """Check that the forward pass outputs the correct shape."""
        input_dim, output_dim, num_samples = 5, 3, 10
        layer = Linear(input_dim, output_dim)
        x = np.random.randn(num_samples, input_dim)

        output = layer.forward(x)

        self.assertEqual(output.shape, (num_samples, output_dim))

    def test_backward_gradients_shape(self):
        """Check that the backward pass outputs gradients with correct shapes."""
        input_dim, output_dim, num_samples = 5, 3, 10
        layer = Linear(input_dim, output_dim)
        x = np.random.randn(num_samples, input_dim)

        _ = layer.forward(x)
        d_out = np.random.randn(num_samples, output_dim)

        d_input, d_weights, d_bias = layer.backward(d_out)

        self.assertEqual(d_input.shape, x.shape)
        self.assertEqual(d_weights.shape, layer.weights.shape)
        self.assertEqual(d_bias.shape, layer.bias.shape)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
