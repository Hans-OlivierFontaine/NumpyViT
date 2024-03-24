import unittest

import numpy as np

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


if __name__ == '__main__':
    unittest.main()
