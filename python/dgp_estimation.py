"""DGP Kalman-filter estimator (update + prediction recursion).

Equivalent of DGP_estimation.m
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from dgp_function_fitting import FitResult

# Type alias: a kernel is any function (ndarray, ndarray) -> ndarray
Kernel = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class EstResult:
    """Output of :func:`dgp_estimation`."""
    z_pred: np.ndarray     # (n_s, N+1)
    z_upd: np.ndarray      # (n_s, N)
    Psi_pred: np.ndarray   # (n_s, n_s, N+1)
    Psi_upd: np.ndarray    # (n_s, n_s, N)
    f_pred: np.ndarray     # (N_test, N+1)
    f_upd: np.ndarray      # (N_test, N)
    c_pred: np.ndarray     # (N_test, N_test, N+1)
    c_upd: np.ndarray      # (N_test, N_test, N)


def dgp_estimation(fit: FitResult, x: np.ndarray, y: np.ndarray,
                    Q_v) -> EstResult:
    """Run the DGP estimator (Python port of DGP_estimation.m).

    Parameters
    ----------
    fit : FitResult from dgp_function_fitting
    x   : (p, N) measurement locations
    y   : (p, N) noisy measurements
    Q_v : callable (X, V) -> (p, p) measurement noise covariance
    """

    Lambda   = fit.Lambda
    Lambda_U = fit.Lambda_U
    Lambda_f = fit.Lambda_f
    Lambda_w = fit.Lambda_w
    z_bar    = fit.z_bar
    u        = fit.u
    U_test   = fit.U_test
    n_s      = fit.n_s

    M = len(u)
    p, N = y.shape
    N_test = U_test.shape[1]

    A = Lambda @ Lambda_U          # state transition matrix

    # allocate
    z_pred   = np.full((n_s, N + 1), np.nan)
    z_upd    = np.full((n_s, N), np.nan)
    Psi_pred = np.full((n_s, n_s, N + 1), np.nan)
    Psi_upd  = np.full((n_s, n_s, N), np.nan)
    f_pred   = np.full((N_test, N + 1), np.nan)
    f_upd    = np.full((N_test, N), np.nan)
    c_pred   = np.full((N_test, N_test, N + 1), np.nan)
    c_upd    = np.full((N_test, N_test, N), np.nan)

    # initial condition
    z_pred[:, 0] = z_bar
    Psi_pred[:, :, 0] = Lambda_f
    f_pred[:, 0] = U_test.T @ z_pred[:, 0]
    c_pred[:, :, 0] = U_test.T @ Psi_pred[:, :, 0] @ U_test

    # estimation loop
    for t in range(N):

        # evaluate basis at measurement locations
        U_samp = np.zeros((n_s, p))
        for i in range(M):
            U_samp[i, :] = u[i](x[:, t])

        # --- DGP update (measurement incorporation) ---
        S = U_samp.T @ Psi_pred[:, :, t] @ U_samp + Q_v(x[:, t], x[:, t])
        Psi_Gamma = Psi_pred[:, :, t] @ U_samp @ np.linalg.inv(S)

        z_upd[:, t] = z_pred[:, t] + Psi_Gamma @ (y[:, t] - U_samp.T @ z_pred[:, t])
        Psi_upd[:, :, t] = Psi_pred[:, :, t] - Psi_Gamma @ U_samp.T @ Psi_pred[:, :, t]

        f_upd[:, t] = U_test.T @ z_upd[:, t]
        c_upd[:, :, t] = U_test.T @ Psi_upd[:, :, t] @ U_test

        # --- DGP prediction (time propagation) ---
        z_pred[:, t + 1] = A @ z_upd[:, t]
        Psi_pred[:, :, t + 1] = A @ Psi_upd[:, :, t] @ A.T + Lambda_w

        f_pred[:, t + 1] = U_test.T @ z_pred[:, t + 1]
        c_pred[:, :, t + 1] = U_test.T @ Psi_pred[:, :, t + 1] @ U_test

    return EstResult(
        z_pred=z_pred, z_upd=z_upd,
        Psi_pred=Psi_pred, Psi_upd=Psi_upd,
        f_pred=f_pred, f_upd=f_upd,
        c_pred=c_pred, c_upd=c_upd,
    )
