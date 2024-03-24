import numpy as np


class Dropout:
    def __init__(self, p=0.0):
        self.p = p

    def forward(self, x, training=True):
        if training and self.p > 0.0:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
            return x * self.mask
        return x

    def backward(self, dout):
        return dout * self.mask
