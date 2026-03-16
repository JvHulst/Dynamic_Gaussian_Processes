"""Squared exponential (RBF / Gaussian) kernel matrix."""

import numpy as np


def squexp(U: np.ndarray, V: np.ndarray, sigma: float) -> np.ndarray:
    """Compute the squared exponential kernel matrix.

    Parameters
    ----------
    U : (N, d) or (N,) array
    V : (M, d) or (M,) array
    sigma : length scale (scalar)

    Returns
    -------
    K : (N, M) array with K[i,j] = exp(-||U[i]-V[j]||^2 / (2*sigma^2))
    """
    U = np.atleast_2d(U)
    V = np.atleast_2d(V)

    # Squared pairwise distance: ||u_i - v_j||^2
    D = np.sum(U**2, axis=1, keepdims=True) \
        + np.sum(V**2, axis=1, keepdims=True).T \
        - 2.0 * (U @ V.T)

    return np.exp(-D / (2.0 * sigma**2))
