from pathlib import Path
import numpy as np

from model.layer_norm import LayerNorm
from model.mha import MultiHeadAttention
from model.gelu import GELULayer
from model.dropout import Dropout
from model.linear import Linear


class AttentionBlock:
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0, learning_rate=0.01):
        self.layer_norm_1 = LayerNorm(embed_dim, learning_rate=learning_rate)
        self.attn = MultiHeadAttention(embed_dim, num_heads, learning_rate=learning_rate)
        self.layer_norm_2 = LayerNorm(embed_dim, learning_rate=learning_rate)
        self.linear_1 = Linear(embed_dim, hidden_dim, learning_rate=learning_rate)
        self.gelu = GELULayer()
        self.dropout_1 = Dropout(dropout)
        self.linear_2 = Linear(hidden_dim, embed_dim, learning_rate=learning_rate)
        self.dropout_2 = Dropout(dropout)
        self.learning_rate = learning_rate

    def forward(self, x):
        inp_x = self.layer_norm_1.forward(x)
        attn_output = self.attn.forward(inp_x)[0]
        x = x + attn_output

        inp_x2 = self.layer_norm_2.forward(x)
        linear_output = self.linear_1.forward(inp_x2)
        gelu_output = self.gelu.forward(linear_output)
        dropout_output_1 = self.dropout_1.forward(gelu_output)
        linear_output_2 = self.linear_2.forward(dropout_output_1)
        dropout_output_2 = self.dropout_2.forward(linear_output_2)

        x = x + dropout_output_2
        return x

    def backward(self, d_output):
        d_dropout_output_2 = self.dropout_2.backward(d_output)
        d_linear_output_2, d_linear_2_weight, d_linear_2_bias = self.linear_2.backward(d_dropout_output_2)
        d_dropout_output_1 = self.dropout_1.backward(d_linear_output_2)
        d_gelu_output = self.gelu.backward(d_dropout_output_1)
        d_linear_output_1, d_linear_1_weight, d_linear_1_bias = self.linear_1.backward(d_gelu_output)

        d_layer_norm_2, d_gamma_2, d_beta_2 = self.layer_norm_2.backward(d_linear_output_1 + d_output)

        d_attn_output = self.attn.backward(d_layer_norm_2)
        d_layer_norm_1, d_gamma_1, d_beta_1 = self.layer_norm_1.backward(d_attn_output)

        self.grads = {
            'd_gamma_1': d_gamma_1,
            'd_beta_1': d_beta_1,
            'd_linear_1_weight': d_linear_1_weight,
            'd_linear_1_bias': d_linear_1_bias,
            'd_linear_2_weight': d_linear_2_weight,
            'd_linear_2_bias': d_linear_2_bias,
            'd_gamma_2': d_gamma_2,
            'd_beta_2': d_beta_2
        }
        return d_layer_norm_1

    def save(self, save_dir: Path, name: str):
        self.attn.save(save_dir, f"attention_block_mha_{name}")
        self.linear_1.save(save_dir, f"attention_block_linear1_{name}")
        self.linear_2.save(save_dir, f"attention_block_linear2_{name}")

    def load(self, save_dir: Path, name: str):
        self.attn.load(save_dir, f"attention_block_mha_{name}")
        self.linear_1.load(save_dir, f"attention_block_linear1_{name}")
        self.linear_2.load(save_dir, f"attention_block_linear2_{name}")
