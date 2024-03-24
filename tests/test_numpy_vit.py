import unittest
import numpy as np
from numpy_vit import VisionTransformer
from copy import deepcopy


class TestAttentionBlock(unittest.TestCase):

    def setUp(self):
        self.img_size = 32
        self.embed_dim = 256
        self.hidden_dim = 512
        self.num_channels = 3
        self.num_heads = 8
        self.num_layers = 6
        self.num_classes = 10
        self.batch_size = 2
        self.patch_size = 4
        self.num_patches = 64
        self.dropout_rate = 0.1
        self.vision_transformer = VisionTransformer(image_size=self.img_size, embed_dim=self.embed_dim,
                                                    hidden_dim=self.hidden_dim, num_channels=self.num_channels,
                                                    num_heads=self.num_heads, num_layers=self.num_layers,
                                                    num_classes=self.num_classes, patch_size=self.patch_size,
                                                    num_patches=self.num_patches, dropout=self.dropout_rate)

    def test_attention_block_output_shape(self):
        """
        Test that the AttentionBlock forward pass outputs the correct shape.
        """
        x = np.random.randn(self.batch_size, self.img_size, self.img_size, self.num_channels)
        output = self.vision_transformer.forward(x)

        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_parameter_updates(self):
        """
        Test that parameters in the AttentionBlock are updated.
        """
        x = np.random.randn(self.batch_size, self.img_size, self.img_size, self.num_channels)

        output = self.vision_transformer.forward(x)

        linear_input_params = {'linear_input_weights': deepcopy(self.vision_transformer.input_layer.weights),
                               'linear_input_biases': deepcopy(self.vision_transformer.input_layer.bias)}

        attention_params = [{'qkv_proj_weight': deepcopy(transformer_layer.attn.qkv_proj_weight),
                             'qkv_proj_bias': deepcopy(transformer_layer.attn.qkv_proj_bias),
                             'o_proj_weight': deepcopy(transformer_layer.attn.o_proj_weight),
                             'o_proj_bias': deepcopy(transformer_layer.attn.o_proj_bias),
                             'linear1_weights': deepcopy(transformer_layer.linear_1.weights),
                             'linear1_bias': deepcopy(transformer_layer.linear_1.bias),
                             'linear2_weights': deepcopy(transformer_layer.linear_2.weights),
                             'linear2_bias': deepcopy(transformer_layer.linear_2.bias)} for transformer_layer in
                            self.vision_transformer.transformer_layers]

        fake_grads = np.random.randn(*output.shape)

        self.vision_transformer.backward(fake_grads)

        self.assertFalse(np.array_equal(linear_input_params['linear_input_weights'],
                                        self.vision_transformer.input_layer.weights))
        self.assertFalse(np.array_equal(linear_input_params['linear_input_biases'],
                                        self.vision_transformer.input_layer.bias))

        for i, transformer_layer in enumerate(self.vision_transformer.transformer_layers):
            self.assertFalse(
                np.array_equal(attention_params[i]['qkv_proj_weight'], transformer_layer.attn.qkv_proj_weight))
            self.assertFalse(np.array_equal(attention_params[i]['qkv_proj_bias'], transformer_layer.attn.qkv_proj_bias))

            self.assertFalse(np.array_equal(attention_params[i]['o_proj_weight'], transformer_layer.attn.o_proj_weight))
            self.assertFalse(np.array_equal(attention_params[i]['o_proj_bias'], transformer_layer.attn.o_proj_bias))

            self.assertFalse(np.array_equal(attention_params[i]['linear1_weights'], transformer_layer.linear_1.weights))
            self.assertFalse(np.array_equal(attention_params[i]['linear1_bias'], transformer_layer.linear_1.bias))

            self.assertFalse(np.array_equal(attention_params[i]['linear2_weights'], transformer_layer.linear_2.weights))
            self.assertFalse(np.array_equal(attention_params[i]['linear2_bias'], transformer_layer.linear_2.bias))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
