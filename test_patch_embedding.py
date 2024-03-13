import numpy as np
import unittest

from patch_embeddings import PatchEmbedding


class TestPatchEmbedding(unittest.TestCase):
    def setUp(self):
        self.patch_height = 2
        self.patch_width = 2
        self.num_channels = 3
        self.dim = 6
        self.embedding = PatchEmbedding(self.patch_height, self.patch_width, self.num_channels, self.dim)

    def test_output_type(self):
        batch_size = 1
        height = 4
        width = 4
        x = np.random.randn(batch_size, height, width, self.num_channels)

        output = self.embedding.forward(x)

        self.assertTrue(isinstance(output, np.ndarray), "Output should be a numpy ndarray")

    def test_output(self):
        img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
        patch_emb_np = PatchEmbedding(img_size, patch_size, num_hiddens)
        X_np = np.zeros((batch_size, img_size, img_size, 3))
        output_np = patch_emb_np.forward(X_np)
        expected_shape = (batch_size, (img_size // patch_size) ** 2, num_hiddens)

        def check_shape(actual, expected):
            assert actual.shape == expected, f"Expected shape {expected}, but got {actual.shape}"

        check_shape(output_np, expected_shape)

    def test_backwards(self):
        img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4

        patch_emb_np = PatchEmbedding(img_size, patch_size, num_hiddens)

        X_np = np.random.rand(batch_size, img_size, img_size, 3)

        initial_weights = np.copy(patch_emb_np.weights)
        initial_bias = np.copy(patch_emb_np.bias)

        output_np = patch_emb_np.forward(X_np)

        dOutput_np = np.random.rand(*output_np.shape)

        patch_emb_np.backward(dOutput_np, X_np)

        weights_changed = not np.array_equal(initial_weights, patch_emb_np.weights)
        bias_changed = not np.array_equal(initial_bias, patch_emb_np.bias)

        assert weights_changed, "Weights did not change after backward pass."
        assert bias_changed, "Biases did not change after backward pass."

        print("Backward pass test passed. Weights and biases have been updated.")


if __name__ == '__main__':
    unittest.main()
