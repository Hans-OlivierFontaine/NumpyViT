import numpy as np


class GELULayer:
    def __init__(self):
        self.cache = None  # Cache to store forward pass information for the backward calculation.

    def forward(self, x):
        # Using an approximation for the erf function in the GELU formula.
        self.cache = x
        x_cubed = x ** 3
        inner_tanh = np.sqrt(2 / np.pi) * (x + 0.044715 * x_cubed)
        output = 0.5 * x * (1 + np.tanh(inner_tanh))
        return output

    def backward(self, dout):
        x = self.cache
        x_cubed = x ** 3
        inner_tanh_grad = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x_cubed)
        tanh_grad = 1 - np.tanh(inner_tanh_grad) ** 2
        dx = dout * 0.5 * (1 + np.tanh(inner_tanh_grad)) + dout * 0.5 * x * tanh_grad * inner_tanh_grad
        return dx
