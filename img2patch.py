import numpy as np


def img_to_patch(x: np.ndarray, patch_size: int, flatten_channels=True):
    """
    Inputs:
        x - Numpy array representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of an image grid.
    """
    B, H, W, C = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.transpose(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]

    if flatten_channels:
        x = x.reshape(B, -1, C * patch_size * patch_size)  # [B, H'*W', C*p_H*p_W]
    else:
        x = x.reshape(B, -1, C, patch_size, patch_size)  # [B, H'*W', C, p_H, p_W]

    return x
