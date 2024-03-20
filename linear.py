import numpy as np


class Linear:
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        """
        Initialize the weights and biases of the Linear Layer.

        :param input_dim: The size of the input vectors.
        :param output_dim: The size of the output vectors.
        """
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.bias = np.zeros(output_dim)
        self.learning_rate = learning_rate
        self.input = None

    def forward(self, x):
        """
        Compute the forward pass of the Linear Layer.

        :param x: The input array.
        :return: The output of the layer.
        """
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, d_output):
        """
        Compute the backward pass of the Linear Layer.

        :param d_output: The gradient of the loss with respect to the output of the layer.
        :return: The gradient of the loss with respect to the input of the layer.
        """
        d_input = np.dot(d_output, self.weights.T)
        d_weights = np.dot(self.input.T, d_output)
        d_biases = np.sum(d_output, axis=0)

        self.weights -= self.learning_rate * d_weights

        return d_input, d_weights, d_biases
