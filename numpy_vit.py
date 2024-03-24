import numpy as np

from attention_block import AttentionBlock
from linear import Linear
from layer_norm import LayerNorm
from parameter import Parameter
from img2patch import img_to_patch
from mlp import ViTMLP


class VisionTransformer:
    def __init__(self, image_size, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size,
                 num_patches, dropout=0.0, learning_rate=0.01):
        assert all([isinstance(val, int) and val > 0 for val in [image_size, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches]])
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        calculated_num_patches = (image_size // patch_size) ** 2
        assert num_patches == calculated_num_patches, "Number of patches is incorrect."

        assert hidden_dim >= embed_dim, "Hidden dimension should be at least as large as embedding dimension."
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.dropout_rate = dropout
        self.learning_rate = learning_rate

        self.input_layer = Linear(num_channels * (patch_size ** 2), embed_dim, learning_rate=learning_rate)

        self.transformer_layers = [AttentionBlock(embed_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        self.mlp_head = ViTMLP(embed_dim, hidden_dim, num_classes, dropout, learning_rate=learning_rate)

        self.cls_token = Parameter(np.random.randn(1, 1, embed_dim))
        self.pos_embedding = Parameter(np.random.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        x = img_to_patch(x, self.patch_size)
        B, T, X = x.shape
        x = self.input_layer.forward(x)

        cls_tokens = np.repeat(self.cls_token.data, B, axis=0)
        x = np.concatenate([cls_tokens, x], axis=1)
        x += self.pos_embedding.data[:, :T + 1]

        for layer in self.transformer_layers:
            x = layer.forward(x)

        cls = x[:, 0, :]
        cls = self.mlp_head.forward(cls)

        return cls

    def backward(self, d_out):
        d_out = self.mlp_head.backward(d_out)

        d_x = np.zeros((d_out.shape[0], 1 + int((self.image_size / self.patch_size))**2, d_out.shape[1]))
        d_x[:, 0, :] = d_out
        d_x = d_x.transpose((2, 1, 0))
        for layer in reversed(self.transformer_layers):
            d_x = layer.backward(d_x)
        d_x = d_x[:, 1:, :]

        self.input_layer.backward(d_x)
