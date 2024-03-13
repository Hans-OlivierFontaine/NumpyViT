import numpy as np


class PatchEmbedding:
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512, learning_rate=0.01):
        """
        A class to implement patch embedding as used in Vision Transformers.

        The PatchEmbedding class simulates extracting fixed-size patches from input images
        and projecting them into a specified embedding space. This implementation mirrors
        the operation of a convolutional layer with kernel size and stride equal to the
        patch size, flattening the patches, and then linearly transforming the patch data
        to the desired embedding dimension. It is a simplified numpy-based implementation
        intended for educational purposes and does not include optimizations for
        high-performance computing.

        Attributes:
            img_size (int or tuple of int): The size of the input images. If an integer is
                provided, it is assumed the image is square. If a tuple is provided, it should
                be in the form (height, width).
            patch_size (int or tuple of int): The size of each patch. If an integer is provided,
                patches are assumed to be square. If a tuple is provided, it should be in the
                form (patch_height, patch_width).
            num_hiddens (int): The dimensionality of the output patch embeddings.
            learning_rate (float): The learning rate used for parameter updates in the backward pass.
            weights (numpy.ndarray): The weights used to linearly transform the patch data.
            bias (numpy.ndarray): The bias added during the linear transformation of the patch data.
        """
        patch_size = self._make_tuple(patch_size)
        img_size = self._make_tuple(img_size)

        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_size = patch_size
        self.img_size = img_size
        self.assert_divisibility()
        self.num_hiddens = num_hiddens
        self.learning_rate = learning_rate

        self.weights = np.random.randn(num_hiddens, *patch_size, 3) * 0.02  # Assuming 3 channels for RGB
        self.bias = np.zeros(num_hiddens)

    def assert_divisibility(self):
        """
        Asserts that the patch size divides the image size evenly in both dimensions.

        Parameters:
        img_size (tuple): A tuple representing the image dimensions (height, width).
        patch_size (tuple): A tuple representing the patch dimensions (height, width).
        """

        assert self.img_size[0] % self.patch_size[0] == 0, "Patch height must evenly divide image height!"
        assert self.img_size[1] % self.patch_size[1] == 0, "Patch width must evenly divide image width!"

    def _make_tuple(self, x):
        """
        Ensures that the input is converted to a tuple format. This method standardizes
        the representation of dimensions in the PatchEmbedding class, facilitating consistent
        processing of image and patch sizes regardless of how these sizes are initially provided.

        Parameters:
        x (int or tuple): The input value representing a dimension size or sizes. If an integer
            is provided, this method converts it to a tuple with duplicated values, implying
            uniform dimensions (e.g., a square shape for image or patch sizes). If a tuple is
            already provided, it returns the tuple unchanged, assuming it correctly represents
            the dimension sizes.

        Returns:
        tuple: A tuple of the dimension sizes, ensuring that even a single integer input is
            transformed into a tuple format for uniform handling within the class. The output
            is either (x, x) if the input is an integer or the original tuple if the input is
            already a tuple.
        """
        if not isinstance(x, (list, tuple)):
            return (x, x)
        return x

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        out_height = height // self.patch_size[0]
        out_width = width // self.patch_size[1]

        out = np.zeros((batch_size, self.num_hiddens, out_height * out_width))

        for i in range(out_height):
            for j in range(out_width):
                patch = X[:, i * self.patch_size[0]: (i + 1) * self.patch_size[0],
                        j * self.patch_size[1]: (j + 1) * self.patch_size[1], :]
                patch = patch.reshape(batch_size, -1)
                for k in range(self.num_hiddens):
                    out[:, k, i * out_width + j] = patch.dot(self.weights[k].reshape(-1)) + self.bias[k]

        return out.transpose(0, 2, 1)

    def backward(self, dOut, X):
        dOut = dOut.transpose(0, 2, 1)
        batch_size, height, width, _ = X.shape
        out_height = height // self.patch_size[0]
        out_width = width // self.patch_size[1]

        dWeights = np.zeros_like(self.weights)
        dBias = np.zeros_like(self.bias)

        for i in range(out_height):
            for j in range(out_width):
                patch = X[:, i * self.patch_size[0]: (i + 1) * self.patch_size[0],
                        j * self.patch_size[1]: (j + 1) * self.patch_size[1], :]
                patch = patch.reshape(batch_size, -1)
                for k in range(self.num_hiddens):
                    dWeights[k] += np.sum(dOut[:, k, i * out_width + j][:, np.newaxis] * patch, axis=0).reshape(
                        self.weights[k].shape)
                    dBias[k] += np.sum(dOut[:, k, i * out_width + j])

        self.weights -= self.learning_rate * dWeights
        self.bias -= self.learning_rate * dBias
