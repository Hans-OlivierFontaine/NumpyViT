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


def cross_entropy_loss(predictions, labels):
    """
    Compute the cross-entropy loss given predictions and one-hot encoded labels.

    Args:
    predictions: Numpy array of shape (N, M) containing the predicted probabilities for M classes for N examples.
    labels: Numpy array of shape (N, M) containing the one-hot encoded labels for N examples.

    Returns:
    The average cross-entropy loss as a float.
    """
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)

    log_preds = np.log(predictions)

    loss = -np.sum(labels * log_preds) / labels.shape[0]

    return loss
