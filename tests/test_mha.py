import unittest

import numpy as np
from pathlib import Path
from copy import deepcopy

from mha import MultiHeadAttention, scaled_dot_product_attention


class TestMHA(unittest.TestCase):
    def test_scaled_dot_product_attention(self):
        seq_len, d_k = 3, 2
        np.random.seed(42)
        q = np.random.randn(seq_len, d_k)
        k = np.random.randn(seq_len, d_k)
        v = np.random.randn(seq_len, d_k)
        values, attention = scaled_dot_product_attention(q, k, v)
        self.assertTrue(values.shape == (3, 2))
        self.assertTrue(attention.shape == (3, 3))

    def test_multi_head_attention_output_shape(self):
        batch_size = 2
        seq_length = 32
        input_dim = 64
        embed_dim = 128
        num_heads = 4

        mha = MultiHeadAttention(input_dim, embed_dim, num_heads)

        x = np.random.rand(batch_size, seq_length, input_dim)

        output = mha.forward(x)
        self.assertTrue(output.shape == (batch_size, seq_length, embed_dim), "Output shape is incorrect.")

    def test_multi_head_attention_parameters_update(self):
        batch_size = 2
        seq_length = 32
        input_dim = 64
        embed_dim = 128
        num_heads = 4

        mha = MultiHeadAttention(input_dim, embed_dim, num_heads)

        initial_qkv_proj_weight = np.copy(mha.qkv_proj_weight)
        initial_o_proj_weight = np.copy(mha.o_proj_weight)

        x = np.random.rand(batch_size, seq_length, input_dim)
        d_output = np.random.rand(batch_size, seq_length, embed_dim)

        mha.forward(x)
        mha.backward(d_output)

        self.assertFalse(np.array_equal(initial_qkv_proj_weight, mha.qkv_proj_weight),
                         "QKV projection weights not updated.")
        self.assertFalse(np.array_equal(initial_o_proj_weight, mha.o_proj_weight),
                         "Output projection weights not updated.")

        initial_qkv_proj_weight = np.copy(mha.qkv_proj_weight)
        initial_o_proj_weight = np.copy(mha.o_proj_weight)

        x = np.random.rand(batch_size, seq_length, input_dim)
        d_output = np.random.rand(batch_size, seq_length, embed_dim)

        mha.forward(x)
        mha.backward(d_output)

        self.assertFalse(np.array_equal(initial_qkv_proj_weight, mha.qkv_proj_weight),
                         "QKV projection weights not updated.")
        self.assertFalse(np.array_equal(initial_o_proj_weight, mha.o_proj_weight),
                         "Output projection weights not updated.")

    def test_save_load(self):
        """Check that the save and load functions work."""
        embed_dim = 256
        num_heads = 8

        mha = MultiHeadAttention(embed_dim, num_heads)

        original_weights1 = deepcopy(mha.qkv_proj_weight)
        original_bias1 = deepcopy(mha.qkv_proj_bias)
        original_weights2 = deepcopy(mha.o_proj_weight)
        original_bias2 = deepcopy(mha.o_proj_bias)

        (Path(__file__).parent / "assets").mkdir(exist_ok=True, parents=True)
        mha.save((Path(__file__).parent / "assets"), "mha_test")

        mha.qkv_proj_weight += 1
        mha.qkv_proj_bias += 1
        mha.o_proj_weight += 1
        mha.o_proj_bias += 1

        self.assertFalse(np.array_equal(original_weights1, mha.qkv_proj_weight))
        self.assertFalse(np.array_equal(original_bias1, mha.qkv_proj_bias))
        self.assertFalse(np.array_equal(original_weights2, mha.o_proj_weight))
        self.assertFalse(np.array_equal(original_bias2, mha.o_proj_bias))

        mha.load((Path(__file__).parent / "assets"), "mha_test")

        self.assertTrue(np.array_equal(original_weights1, mha.qkv_proj_weight))
        self.assertTrue(np.array_equal(original_bias1, mha.qkv_proj_bias))
        self.assertTrue(np.array_equal(original_weights2, mha.o_proj_weight))
        self.assertTrue(np.array_equal(original_bias2, mha.o_proj_bias))


if __name__ == '__main__':
    unittest.main()
