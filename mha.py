import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None, eps=1e-6):
    d_k = K.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k + eps)

    if mask is not None:
        scores += (mask * -1e9)

    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)

    output = np.matmul(attention_weights, V)
    return output, attention_weights


def scaled_dot_product_attention_backward(d_output, Q, K, V, attention_weights, mask=None, eps=1e-6):
    d_Q = np.zeros_like(Q)
    d_K = np.zeros_like(K)
    d_V = np.matmul(attention_weights.transpose(0, 1, 3, 2), d_output)

    d_attention_weights = np.matmul(d_output, np.swapaxes(V, -1, -2))

    attention_gradients = attention_weights * (1 - attention_weights)
    d_scores = d_attention_weights * attention_gradients

    d_scores /= np.sqrt(K.shape[-1] + eps)
    if mask is not None:
        d_scores = np.where(mask, 0, d_scores)

    d_Q = np.matmul(d_scores, K)
    d_K = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)

    return d_Q, d_K, d_V


def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = np.expand_dims(mask, axis=1)
    while mask.ndim < 4:
        mask = np.expand_dims(mask, axis=0)
    return mask


def xavier_uniform_(array):
    """
    Applies the Xavier uniform initialization to the provided array in-place.
    This function mimics the nn.init.xavier_uniform_ behavior in PyTorch.
    """
    in_features, out_features = array.shape
    limit = np.sqrt(6.0 / (in_features + out_features))
    array[:] = np.random.uniform(-limit, limit, size=array.shape)


class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj_weight = np.random.rand(embed_dim, 3 * embed_dim)
        self.qkv_proj_bias = np.zeros(3 * embed_dim)
        self.o_proj_weight = np.random.rand(embed_dim, embed_dim)
        self.o_proj_bias = np.zeros(embed_dim)

        xavier_uniform_(self.qkv_proj_weight)
        xavier_uniform_(self.o_proj_weight)

        self.batch_size = None
        self.seq_length = None
        self.x = None
        self.q = None
        self.k = None
        self.v = None
        self.attention = None
        self.attention_weights = None

    def forward(self, x, mask=None, return_attention=False):
        self.batch_size, self.seq_length, _ = x.shape
        self.x = x
        if mask is not None:
            mask = expand_mask(mask)

        qkv = np.dot(x, self.qkv_proj_weight) + self.qkv_proj_bias
        qkv = qkv.reshape(self.batch_size, self.seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.transpose(0, 2, 1, 3)
        self.q, self.k, self.v = np.split(qkv, 3, axis=-1)

        values, attention = scaled_dot_product_attention(self.q, self.k, self.v, mask=mask)
        values = values.transpose(0, 2, 1, 3)
        values = values.reshape(self.batch_size, self.seq_length, self.embed_dim)

        o = np.dot(values, self.o_proj_weight) + self.o_proj_bias

        self.attention = o
        self.attention_weights = attention

        if return_attention:
            return o, attention
        else:
            return o

    def backward(self, d_output, mask=None):
        d_values = np.dot(d_output, self.o_proj_weight.T)
        self.d_o_proj_weight = np.dot(self.attention.reshape(-1, self.embed_dim).T,
                                      d_output.reshape(-1, self.embed_dim))
        self.d_o_proj_bias = np.sum(d_output, axis=(0, 1))

        d_values = d_values.reshape(self.batch_size, self.seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1,
                                                                                                               3)

        d_q, d_k, d_v = scaled_dot_product_attention_backward(d_values, self.q, self.k, self.v, self.attention_weights,
                                                              mask)

        d_qkv = np.concatenate([d_q, d_k, d_v], axis=-1).transpose(0, 2, 1, 3).reshape(self.batch_size, self.seq_length,
                                                                                       -1)
        self.d_qkv_proj_weight = np.dot(self.x.reshape(-1, self.embed_dim).T, d_qkv.reshape(-1, 3 * self.embed_dim))
        self.d_qkv_proj_bias = np.sum(d_qkv, axis=(0, 1))

        self.qkv_proj_weight -= self.d_qkv_proj_weight
        self.qkv_proj_bias -= self.d_qkv_proj_bias
        self.o_proj_weight -= self.d_o_proj_weight
        self.o_proj_bias -= self.d_o_proj_bias

        return d_values.reshape(d_values.shape[0], d_values.shape[2], d_values.shape[1] * d_values.shape[3])
