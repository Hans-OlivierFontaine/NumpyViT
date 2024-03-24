import unittest
import numpy as np
from model.dropout import Dropout


class MyTestCase(unittest.TestCase):
    def test_forward_training_true(self):
        """Check dropout forward pass in training mode."""
        np.random.seed(0)
        dropout_rate = 0.5
        x = np.ones((10, 10))
        dropout = Dropout(p=dropout_rate)

        x_dropped = dropout.forward(x, training=True)

        self.assertAlmostEqual(np.count_nonzero(x_dropped), 50, delta=5)

        nonzero_elements = x_dropped[x_dropped != 0]
        self.assertTrue(np.allclose(nonzero_elements, 1 / (1 - dropout_rate)))

    def test_forward_training_false(self):
        """Check dropout forward pass in evaluation mode."""
        x = np.ones((10, 10))
        dropout = Dropout(p=0.5)

        x_dropped = dropout.forward(x, training=False)

        self.assertEqual(np.count_nonzero(x_dropped), 100)
        self.assertTrue(np.allclose(x_dropped, x))

    def test_backward(self):
        """Check dropout backward pass."""
        np.random.seed(0)
        dropout = Dropout(p=0.5)
        x = np.ones((10, 10))
        dout = np.ones((10, 10))

        _ = dropout.forward(x, training=True)
        dx = dropout.backward(dout)

        self.assertTrue(np.allclose(dx, dropout.mask))


if __name__ == '__main__':
    unittest.main()
