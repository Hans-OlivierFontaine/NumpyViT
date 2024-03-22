import unittest
import numpy as np

from img2patch import img_to_patch


class TestImgToPatch(unittest.TestCase):
    def test_output_shape(self):
        """Test that the output shape is correct."""
        B, C, H, W = 2, 3, 4, 4  # Example dimensions.
        patch_size = 2
        x = np.random.rand(B, C, H, W)

        # Expected shape calculations
        num_patches = (H // patch_size) * (W // patch_size)
        expected_shape_flatten = (B, num_patches, C * patch_size * patch_size)
        expected_shape_non_flatten = (B, num_patches, C, patch_size, patch_size)

        # Test with flatten_channels = True
        patches_flatten = img_to_patch(x, patch_size, flatten_channels=True)
        self.assertEqual(patches_flatten.shape, expected_shape_flatten)

        # Test with flatten_channels = False
        patches_non_flatten = img_to_patch(x, patch_size, flatten_channels=False)
        self.assertEqual(patches_non_flatten.shape, expected_shape_non_flatten)

    def test_content_integrity(self):
        """Test that the patch extraction maintains content integrity."""
        B, C, H, W = 1, 1, 4, 4  # Keeping dimensions simple for content check.
        x = np.arange(H * W).reshape(B, C, H, W)
        patch_size = 2

        expected_output = np.array([
            [[0, 1, 4, 5]],
            [[2, 3, 6, 7]],
            [[8, 9, 12, 13]],
            [[10, 11, 14, 15]]
        ]).reshape(B, H * W // (patch_size ** 2), C * (patch_size ** 2))

        patches = img_to_patch(x, patch_size, flatten_channels=True)
        self.assertTrue(np.array_equal(patches, expected_output))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
