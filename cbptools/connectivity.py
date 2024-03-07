from .exceptions import ShapeError
import numpy as np


def seed_based_correlation(x: np.ndarray, y: np.ndarray,
                           standardize: bool = True,
                           ddof: int = 0) -> np.ndarray:
    """ Compute seed-based correlation between x and y.

    Input time-series x and y, with samples (e.g., time-points) on
    axis 0 and features (e.g., voxels) on axis 1 are correlated and
    returned as a 2D array with the shape of the number of samples
    of x by the number of samples of y. The number of features do not
    have to match between input arrays, but the number of samples
    must be equivalent.

    Parameters
    ----------
    x : np.ndarray
        n_samples by n_features input subject time series
    y : np.ndarray
        n_samples by n_features input subject time series
    standardize : bool, optional
        if set to True, the features of input arrays x and y are
        standardized per sample
    ddof : int, optional
        By default set to 0 (n-0 instead of n-1). This most matches the
        matlab norm results

    Returns
    -------
    np.ndarray
        x[n_features] by y[n_features] connectivity matrix

    """

    if x.shape[0] != y.shape[0]:
        raise ShapeError(x.shape[0], y.shape[0])

    if standardize:
        x, y = map(
            lambda z: (z - np.mean(z, axis=0)) / np.std(z, axis=0, ddof=ddof),
            (x, y)
        )

    # Correlation
    r = (y.T.dot(x) / x.shape[0]).T.astype(np.float32)

    # No-variance voxels have 0 std, causing division by 0 for
    # standardization resulting in NaNs. Here we 0 NaNs.
    r[np.isnan(r)] = 0

    return r
