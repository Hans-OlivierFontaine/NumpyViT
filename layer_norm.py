import numpy as np


class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5, learning_rate=0.01):
        """
        Initialize the layer normalization module.

        Parameters:
        normalized_shape (int or tuple): The shape of the input tensor expected by the layer.
        eps (float): A value added to the denominator for numerical stability.
        learning_rate (float): The learning rate for updating gamma and beta during backpropagation.
        """
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self.eps = eps
        self.learning_rate = learning_rate

        self.x_centered = None
        self.std_inv = None
        self.x_normalized = None

    def forward(self, x):
        """
        Forward pass for layer normalization.

        Parameters:
        x (numpy.ndarray): Input data of shape (..., normalized_shape).

        Returns:
        numpy.ndarray: The output of layer normalization.
        """
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(variance + self.eps)
        self.x_centered = x - mean
        self.x_normalized = self.x_centered * self.std_inv
        return self.gamma * self.x_normalized + self.beta

    def backward(self, grad_output):
        """
        Backward pass for layer normalization.

        Parameters:
        grad_output (numpy.ndarray): The gradient of the loss with respect to the output.
        """
        dgamma = np.sum(grad_output * self.x_normalized, axis=0)
        dbeta = np.sum(grad_output, axis=0)

        self.gamma -= self.learning_rate * dgamma
        self.beta -= self.learning_rate * dbeta

        N = grad_output.shape[-1]

        dx_normalized = grad_output * self.gamma
        dvar = np.sum(dx_normalized * self.x_centered * -0.5 * np.power(self.std_inv, 3), axis=-1, keepdims=True)
        dmean = np.sum(dx_normalized * -self.std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * self.x_centered,
                                                                                               axis=-1, keepdims=True)
        dx = dx_normalized * self.std_inv + dvar * 2.0 * self.x_centered / N + dmean / N

        return dx
