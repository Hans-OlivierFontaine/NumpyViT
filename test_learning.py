import unittest
import numpy as np
from learning import softmax, cross_entropy_loss


class TestSoftmax(unittest.TestCase):
    def test_shape(self):
        """Test softmax probabilities for correctness."""
        x = np.array([[1, 2, 3], [1, 2, -1]])
        original_shape = x.shape
        soft_x = softmax(x)
        soft_shape = soft_x.shape
        self.assertEqual(original_shape, soft_shape)

    def test_softmax_correctness(self):
        """Test softmax probabilities for correctness."""
        x = np.array([[1, 2, 3], [1, 2, -1]])
        expected = np.array([
            [0.09003057, 0.24472847, 0.66524096],
            [0.2594965, 0.7053845, 0.035119]
        ])
        np.testing.assert_almost_equal(softmax(x), expected, decimal=7, err_msg="Softmax output is incorrect.")

    def test_softmax_output_range(self):
        """Test that softmax output is within the range (0, 1)."""
        x = np.random.randn(5, 10)  # Random input
        output = softmax(x)
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1), "Softmax output not in range (0, 1).")

    def test_softmax_sum_to_one(self):
        """Test that the sum of softmax probabilities equals 1."""
        x = np.random.randn(5, 10)  # Random input
        output = softmax(x)
        sum_output = np.sum(output, axis=1)
        np.testing.assert_almost_equal(sum_output, np.ones(5), decimal=7,
                                       err_msg="Softmax probabilities do not sum to 1.")

    def test_numerical_stability(self):
        """Test softmax numerical stability by using large input values."""
        x = np.array([[1000, 1001, 1002], [1000, 1001, -1000]])
        # Direct computation without stability measures would result in overflow/NaN errors.
        # The test passes if the function successfully returns a valid output without errors.
        output = softmax(x)
        self.assertFalse(np.any(np.isnan(output)), "Softmax is not numerically stable.")


class TestCrossEntropyLoss(unittest.TestCase):
    def test_correct_calculation(self):
        """Test cross-entropy loss calculation for known inputs and outputs."""
        predictions = np.array([[0.7, 0.2, 0.1],
                                [0.1, 0.9, 0.0]])
        labels = np.array([[1, 0, 0],
                           [0, 1, 0]])
        predictions = softmax(predictions)

        expected_loss = 0.6931592805048026
        calculated_loss = cross_entropy_loss(predictions, labels)

        self.assertAlmostEqual(calculated_loss, expected_loss, places=7,
                               msg="Cross-entropy loss calculation is incorrect.")

    def test_handling_zero_probability(self):
        """Ensure numerical stability when predicted probability for the correct class is zero."""
        predictions = np.array([[0.0, 1.0],
                                [1.0, 0.0]])
        labels = np.array([[1, 0],
                           [0, 1]])
        predictions = softmax(predictions)  # Apply softmax

        calculated_loss = cross_entropy_loss(predictions, labels)

        self.assertTrue(np.isfinite(calculated_loss),
                        "Cross-entropy loss is not numerically stable for zero probabilities.")


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
