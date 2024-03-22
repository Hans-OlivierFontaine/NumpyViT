import numpy as np


def softmax(x):
    """
    Compute the softmax of each element along an axis of X.

    Args:
    x: A numpy array of shape (N, C) where N is the number of data points, and C is the number of classes.

    Returns:
    A numpy array of the same shape as X containing the softmax probabilities.
    """
    x_shifted = x - np.max(x, axis=-1, keepdims=True)

    x_exp = np.exp(x_shifted)
    sum_x_exp = np.sum(x_exp, axis=-1, keepdims=True)

    softmax = x_exp / sum_x_exp

    return softmax