import numpy as np


class ViTMLP:
    def __init__(self, input_dim, mlp_num_hiddens: int, mlp_num_outputs: int, dropout_rate=0.5, learning_rate=0.01):
        self.mlp_num_hiddens = mlp_num_hiddens
        self.mlp_num_outputs = mlp_num_outputs
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.weights1 = np.random.randn(input_dim, mlp_num_hiddens) * np.sqrt(2. / input_dim)
        self.bias1 = np.zeros(mlp_num_hiddens)
        self.weights2 = np.random.randn(mlp_num_hiddens, mlp_num_outputs) * np.sqrt(2. / mlp_num_hiddens)
        self.bias2 = np.zeros(mlp_num_outputs)

        self.grad_weights1 = None
        self.grad_bias1 = None
        self.grad_weights2 = None
        self.grad_bias2 = None

        self.x = None

        self.x_after_dense1 = None
        self.x_after_gelu = None

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def dropout(self, x, rate):
        mask = np.random.binomial(1, 1 - rate, size=x.shape)
        return x * mask / (1 - rate)

    def forward(self, x):
        self.x = x
        self.x_after_dense1 = np.dot(x, self.weights1) + self.bias1
        self.x_after_gelu = self.gelu(self.x_after_dense1)
        x = self.dropout(self.x_after_gelu, self.dropout_rate)
        x = np.dot(x, self.weights2) + self.bias2
        return self.dropout(x, self.dropout_rate)

    def backward(self, dOut):
        dOut = self.dropout(dOut, self.dropout_rate)
        dOut_dDense2 = dOut

        dDense2_dWeights2 = self.x_after_gelu.T
        self.grad_weights2 = np.dot(dDense2_dWeights2, dOut_dDense2)
        self.grad_bias2 = np.sum(dOut_dDense2, axis=0)

        dGelu = dOut_dDense2.dot(self.weights2.T) * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.x_after_dense1 + 0.044715 * np.power(self.x_after_dense1, 3)))) + \
                0.0356774 * self.x_after_dense1 * np.power(1 - np.tanh(np.sqrt(2 / np.pi) * (self.x_after_dense1 + 0.044715 * np.power(self.x_after_dense1, 3))), 2)

        dDense1_dWeights1 = self.x.T
        self.grad_weights1 = np.dot(dDense1_dWeights1, dGelu)
        self.grad_bias1 = np.sum(dGelu, axis=0)

        self.weights1 -= self.learning_rate * self.grad_weights1
        self.bias1 -= self.learning_rate * self.grad_bias1
        self.weights2 -= self.learning_rate * self.grad_weights2
        self.bias2 -= self.learning_rate * self.grad_bias2

        return dDense1_dWeights1
