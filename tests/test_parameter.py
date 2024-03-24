import unittest
import numpy as np

from parameter import Parameter


class TestParameter(unittest.TestCase):
    def test_initialization(self):
        """Test that the Parameter object correctly encapsulates a numpy array."""
        array = np.array([1, 2, 3])
        param = Parameter(array)
        self.assertTrue(np.array_equal(array, param.data), "Parameter data does not match the input array.")

    def test_attribute_access(self):
        """Test that the Parameter's data attribute can be accessed and modified."""
        initial_array = np.array([1, 2, 3])
        new_array = np.array([4, 5, 6])
        param = Parameter(initial_array)

        # Test data access
        self.assertTrue(np.array_equal(param.data, initial_array), "Parameter data cannot be correctly accessed.")

        # Test data modification
        param.data = new_array
        self.assertTrue(np.array_equal(param.data, new_array), "Parameter data cannot be modified.")

    def test_assert_failure(self):
        not_an_array = "not_an_array"
        with self.assertRaises(AssertionError):
            Parameter(not_an_array)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)