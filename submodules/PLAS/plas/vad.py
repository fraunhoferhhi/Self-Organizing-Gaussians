import numpy as np


def compute_vad(image):
    """
    Calculate the variance of absolute differences for a 2D image.

    Args:
    image (numpy.ndarray): A 2D numpy array representing the image.

    Returns:
    float: The variance of absolute differences.
    """

    image = image.astype(np.float32)

    # Calculate absolute differences in x and y directions
    diff_x = np.abs(np.diff(image, axis=1))
    diff_y = np.abs(np.diff(image, axis=0))

    # Calculate the variance for each and then the overall average
    var_diff_x = np.var(diff_x)
    var_diff_y = np.var(diff_y)
    var_diff = (var_diff_x + var_diff_y) / 2

    return var_diff
