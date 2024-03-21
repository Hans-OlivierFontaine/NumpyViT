import numpy as np

from attention_block import AttentionBlock
from linear import Linear
from layer_norm import LayerNorm
from parameter import Parameter
from img2patch import img_to_patch


class VisionTransformer:
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches,
                 dropout=0.0):
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.dropout_rate = dropout

        self.input_layer = Linear(num_channels * (patch_size ** 2), embed_dim)

        self.transformer_layers = [AttentionBlock(embed_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)]

        self.mlp_head = [LayerNorm(embed_dim), Linear(embed_dim, num_classes)]

        self.cls_token = Parameter(np.random.randn(1, 1, embed_dim))
        self.pos_embedding = Parameter(np.random.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer.forward(x)

        cls_tokens = np.repeat(self.cls_token.data, B, axis=0)
        x = np.concatenate([cls_tokens, x], axis=1)
        x += self.pos_embedding.data[:, :T + 1]

        for layer in self.transformer_layers:
            x = layer.forward(x)

        cls = x[:, 0, :]
        for layer in self.mlp_head:
            cls = layer.forward(cls)

        return cls

    def backward(self, d_out):
        for layer in reversed(self.mlp_head):
            d_out = layer.backward(d_out)

        d_x = np.zeros_like(d_out)
        for layer in reversed(self.transformer_layers):
            d_x = layer.backward(d_x)

        d_x = self.input_layer.backward(d_x)
