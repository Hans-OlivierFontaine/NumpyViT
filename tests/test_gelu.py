import unittest
import numpy as np

from model.gelu import GELULayer


class MyTestCase(unittest.TestCase):
    def test_gelu_forward(self):
        """
        Test the forward pass of the GELU activation function.
        """
        gelu_layer = GELULayer()
        test_input = np.array([-3, -1, 0, 1, 3])
        expected_output = 0.5 * test_input * (
                1 + np.tanh(np.sqrt(2 / np.pi) * (test_input + 0.044715 * test_input ** 3)))

        output = gelu_layer.forward(test_input)

        self.assertTrue(np.allclose(output, expected_output, atol=1e-6), "Forward pass output is incorrect.")

    def test_gelu_backward(self):
        """
        Test the backward pass of the GELU activation function.
        """
        gelu_layer = GELULayer()
        test_input = np.random.randn(5)
        gelu_layer.forward(test_input)

        # Create a dummy gradient that would come from the next layer
        dummy_gradient = np.random.randn(5)
        gradient_output = gelu_layer.backward(dummy_gradient)

        # Check the gradient shape matches the input shape
        self.assertEqual(gradient_output.shape, test_input.shape, "Backward pass output has incorrect shape.")


if __name__ == '__main__':
    unittest.main()
