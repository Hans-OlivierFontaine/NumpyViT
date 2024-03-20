import unittest
import numpy as np
from attention_block import AttentionBlock
from copy import deepcopy


class TestAttentionBlock(unittest.TestCase):

    def setUp(self):
        self.embed_dim = 64
        self.hidden_dim = 128
        self.num_heads = 4
        self.seq_length = 10
        self.batch_size = 2
        self.dropout_rate = 0.1
        self.attention_block = AttentionBlock(self.embed_dim, self.hidden_dim, self.num_heads, self.dropout_rate)

    def test_attention_block_output_shape(self):
        """
        Test that the AttentionBlock forward pass outputs the correct shape.
        """
        x = np.random.randn(self.batch_size, self.seq_length, self.embed_dim)
        output = self.attention_block.forward(x)

        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.embed_dim))

    def test_parameter_updates(self):
        """
        Test that parameters in the AttentionBlock are updated.
        """
        x = np.random.randn(self.batch_size, self.seq_length, self.embed_dim)

        output = self.attention_block.forward(x)

        copied_params = {'qkv_proj_weight': deepcopy(self.attention_block.attn.qkv_proj_weight),
                         'qkv_proj_bias': deepcopy(self.attention_block.attn.qkv_proj_bias),
                         'o_proj_weight': deepcopy(self.attention_block.attn.o_proj_weight),
                         'o_proj_bias': deepcopy(self.attention_block.attn.o_proj_bias),
                         'linear1_weights': deepcopy(self.attention_block.linear_1.weights),
                         'linear1_bias': deepcopy(self.attention_block.linear_1.bias),
                         'linear2_weights': deepcopy(self.attention_block.linear_2.weights),
                         'linear2_bias': deepcopy(self.attention_block.linear_2.bias)}

        fake_grads = np.random.randn(*output.shape)

        self.attention_block.backward(fake_grads)

        self.assertFalse(np.array_equal(copied_params['qkv_proj_weight'], self.attention_block.attn.qkv_proj_weight))
        self.assertFalse(np.array_equal(copied_params['qkv_proj_bias'], self.attention_block.attn.qkv_proj_bias))

        self.assertFalse(np.array_equal(copied_params['o_proj_weight'], self.attention_block.attn.o_proj_weight))
        self.assertFalse(np.array_equal(copied_params['o_proj_bias'], self.attention_block.attn.o_proj_bias))

        self.assertFalse(np.array_equal(copied_params['linear1_weights'], self.attention_block.linear_1.weights))
        self.assertFalse(np.array_equal(copied_params['linear1_bias'], self.attention_block.linear_1.bias))

        self.assertFalse(np.array_equal(copied_params['linear2_weights'], self.attention_block.linear_2.weights))
        self.assertFalse(np.array_equal(copied_params['linear2_bias'], self.attention_block.linear_2.bias))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
