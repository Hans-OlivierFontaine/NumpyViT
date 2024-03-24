import numpy as np


class Parameter:
    def __init__(self, data):
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        self.data = data

    def __repr__(self):
        return f'Parameter({self.data})'
