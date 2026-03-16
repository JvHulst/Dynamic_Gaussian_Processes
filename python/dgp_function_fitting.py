"""Fit basis functions to DGP system equations via least squares.

Equivalent of DGP_function_fitting.m
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.linalg import eig, solve

from squexp import squexp

# Type aliases for kernel and mean function handles
Kernel = Callable[[np.ndarray, np.ndarray], np.ndarray]
MeanFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class FitResult:
    """Output of :func:`dgp_function_fitting`."""
    Lambda: np.ndarray        # (n_s, n_s)   evolution matrix
    Lambda_U: np.ndarray      # (n_s, n_s)   basis Gram matrix
    Lambda_f: np.ndarray      # (n_s, n_s)   initial covariance
    Lambda_w: np.ndarray      # (n_s, n_s)   disturbance covariance
    z_bar: np.ndarray         # (n_s,)       initial mean
    u: list[Callable]         # [M]          basis function callables
    U_test: np.ndarray        # (n_s, N_test) basis at test points (zero-padded)
    n_states: int
    n_s: int


@dataclass
class Config:
    """Configuration struct (mirrors MATLAB ``cfg``)."""

    # ── grid and domain ──
    basis: str = "Fourier"        # basis type: 'RBF' | 'NRBF' | 'Fourier' | 'Discrete'
    M: int = 31                   # number of basis functions
    x_min: float = -10.0          # spatial domain lower bound
    x_max: float = 10.0           # spatial domain upper bound
    N_test: int = 251             # evaluation grid resolution

    # ── estimation settings ──
    p: int = 3                    # measurements per time step
    N: int = 21                   # number of time steps

    # ── system / kernel selection ──
    system: str = "Wave_equation"
    # Truth source.  Choices:
    #   "Discrete_approximation" — simulate with the chosen kernel (single-state only)
    #   "Heat_equation"          — analytical 1-D heat equation (Dirac i.c.)
    #   "Wave_equation"          — analytical 1-D wave equation (D'Alembert)
    #   "Data"                   — load from .mat file

    kernel: str = "wave_equation"
    # Evolution kernel for the estimator.  Choices:
    #   "heat_equation"  — Gaussian Green's function  (single-state)
    #   "wave_equation"  — D'Alembert 2×2 kernel      (two-state: [f, g])
    #   "smoothing"      — squared-exponential blur    (single-state)
    #   "integrator"     — Kronecker-delta identity    (single-state)

    initial_mean: str = "squexp"
    # Prior mean function m(x).  Choices:
    #   "zero"     — m(x) = 0
    #   "squexp"   — m(x) = 10 * exp(-x²/2)   (Gaussian bump)
    #   "ones"     — m(x) = 1
    #   "parabola" — m(x) = -2x² + 8

    # ── plotting ──
    plot_confidence_bounds: bool = True
    plot_basis_functions: bool = False

    # ── derived (computed in __post_init__) ──
    dx: float = field(init=False)
    x_test: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.dx = (self.x_max - self.x_min) / (self.N_test - 1)
        self.x_test = np.linspace(self.x_min, self.x_max, self.N_test)


# ─────────────────────────────────────────────────────────────────────────
# Basis function constructors
# ─────────────────────────────────────────────────────────────────────────

def _fourier_cos(X: np.ndarray, n: int, period: float) -> np.ndarray:
    c = np.cos(2 * np.pi * n * X.ravel() / period)
    if n == 0:
        c = c / np.sqrt(2)
    return c


def _fourier_sin(X: np.ndarray, n: int, period: float) -> np.ndarray:
    return np.sin(2 * np.pi * n * X.ravel() / period)


def _build_basis(cfg: Config):
    """Return (u_list, orthogonal) where u_list is a list of M callables."""
    M = cfg.M
    x_min, x_max = cfg.x_min, cfg.x_max
    u = []

    if cfg.basis == "RBF":
        orthogonal = False
        if M == 1:
            db = (x_max - x_min) / 2
            centers = np.array([(x_max + x_min) / 2])
        else:
            db = (x_max - x_min) / (M - 1)
            centers = np.linspace(x_min, x_max, M)
        l = db * 0.8
        for i in range(M):
            ci = centers[i]
            u.append(lambda X, c=ci, s=l: squexp(X, np.atleast_2d(c), s).ravel())

    elif cfg.basis == "NRBF":
        orthogonal = False
        if M == 1:
            db = (x_max - x_min) / 2
            centers = np.array([(x_max + x_min) / 2])
        else:
            db = (x_max - x_min) / (M - 1)
            centers = np.linspace(x_min, x_max, M)
        l = db * 0.6
        for i in range(M):
            ci = centers[i]
            def _nrbf(X, c=ci, s=l, ctrs=centers):
                num = squexp(X, np.atleast_2d(c), s).ravel()
                den = np.sum(squexp(X, ctrs.reshape(-1, 1), s), axis=1)
                return num / den
            u.append(_nrbf)

    elif cfg.basis == "Fourier":
        orthogonal = True
        period = x_max - x_min
        scale = np.sqrt(2.0 / period)
        n = 0
        for i in range(M):
            if i % 2 == 0:
                ni = n
                u.append(lambda X, nn=ni, p=period, s=scale: _fourier_cos(X, nn, p) * s)
                n += 1
            else:
                ni = n
                u.append(lambda X, nn=ni, p=period, s=scale: _fourier_sin(X, nn, p) * s)

    elif cfg.basis == "Discrete":
        orthogonal = True
        db = (x_max + 1e-12 - x_min) / M
        bounds = np.arange(M + 1) * db + x_min
        bounds[-1] = x_max + 1e-12
        for i in range(M):
            lo, hi = bounds[i], bounds[i + 1]
            u.append(lambda X, a=lo, b=hi, d=db:
                     ((X.ravel() >= a) & (X.ravel() < b)).astype(float) / d)
    else:
        raise ValueError(f"Unknown basis type: {cfg.basis}")

    return u, orthogonal


# ─────────────────────────────────────────────────────────────────────────
# Main fitting routine
# ─────────────────────────────────────────────────────────────────────────

def dgp_function_fitting(
    cfg: Config,
    kf,      # Kernel or list-of-lists of Kernel
    m: MeanFn,
    Q_f,     # Kernel or list-of-lists
    Q_w,     # Kernel or list-of-lists
) -> FitResult:
    """Fit basis functions to system equations (Python port of DGP_function_fitting.m)."""

    M = cfg.M
    x_min, x_max = cfg.x_min, cfg.x_max
    N_test = cfg.N_test
    x_test = cfg.x_test.reshape(-1, 1)

    # ── normalise to list-of-lists (cell array) ──
    if not isinstance(kf, list):
        kf  = [[kf]]
        Q_f = [[Q_f]]
        Q_w = [[Q_w]]

    n_states = len(kf)
    n_s = n_states * M

    # ── build basis ──
    u, orthogonal = _build_basis(cfg)

    N_fit = N_test
    dx_fit = (x_max - x_min) / (N_fit - 1)
    x_fit = np.linspace(x_min, x_max, N_fit).reshape(-1, 1)

    U_fit = np.zeros((M, N_fit))
    U_test_small = np.zeros((M, N_test))
    for i in range(M):
        U_fit[i, :] = u[i](x_fit)
        U_test_small[i, :] = u[i](x_test)

    if cfg.plot_basis_functions:
        import matplotlib.pyplot as plt
        plt.figure(10)
        plt.clf()
        for i in range(M):
            plt.plot(x_test.ravel(), U_test_small[i, :])
        plt.xlabel("$x$")
        plt.ylabel("$u_i(x)$")
        plt.grid(True)
        plt.show(block=False)

    # ── system projection ──
    t0 = time.time()

    # basis inner-product matrix  (M x M)
    # Build the "U_mat" tensor  (N_fit^2 x M^2)
    U_mat = np.zeros((N_fit, N_fit, M * M))
    Lambda_U_small = np.zeros((M, M))

    for i in range(M):
        for j in range(M):
            U_mat[:, :, i * M + j] = U_fit[i, :, None] * U_fit[j, None, :]
            Lambda_U_small[i, j] = np.sum(U_fit[i, :] * U_fit[j, :]) * dx_fit

    U_mat = U_mat.reshape(N_fit**2, M**2)

    if orthogonal:
        col_norms = np.sum(U_mat**2, axis=0)
        Pinv_U = np.diag(1.0 / col_norms) @ U_mat.T
    else:
        Pinv_U = solve(U_mat.T @ U_mat, U_mat.T)

    # allocate block matrices
    Lambda   = np.zeros((n_s, n_s))
    Lambda_f = np.zeros((n_s, n_s))
    Lambda_w = np.zeros((n_s, n_s))
    z_bar    = np.zeros(n_s)

    for ii in range(n_states):
        # initial mean
        if ii == 0:
            z_bar[:M] = solve(U_fit @ U_fit.T, U_fit @ m(x_fit)).ravel()
        # else: remains zero

        for jj in range(n_states):
            rows = slice(ii * M, (ii + 1) * M)
            cols = slice(jj * M, (jj + 1) * M)

            Kf_ij = kf[ii][jj](x_fit, x_fit).T.ravel()
            Lambda[rows, cols] = (Pinv_U @ Kf_ij).reshape(M, M, order='F')

            Qf_ij = Q_f[ii][jj](x_fit, x_fit).ravel()
            Lambda_f[rows, cols] = (Pinv_U @ Qf_ij).reshape(M, M, order='F')

            Qw_ij = Q_w[ii][jj](x_fit, x_fit).ravel()
            Lambda_w[rows, cols] = (Pinv_U @ Qw_ij).reshape(M, M, order='F')

    # extend Lambda_U to block-diagonal
    Lambda_U = np.kron(np.eye(n_states), Lambda_U_small)

    # enforce symmetry
    Lambda_f = (Lambda_f + Lambda_f.T) / 2
    Lambda_w = (Lambda_w + Lambda_w.T) / 2

    elapsed = time.time() - t0
    print(f"Fitting elapsed: {elapsed:.3f} s")

    # stability check
    eig_max = np.max(np.abs(eig(Lambda @ Lambda_U)[0]))
    if eig_max < 1.0:
        print(f"Approximation is stable (max |eig| = {eig_max:.4f})")
    elif abs(eig_max - 1.0) < 1e-6:
        print(f"Approximation is marginally stable (max |eig| = {eig_max:.6f})")
    else:
        print(f"Approximation is NOT stable (max |eig| = {eig_max:.4f})")

    # ── extend U_test for multi-state ──
    if n_states > 1:
        U_test_full = np.vstack([
            U_test_small,
            np.zeros(((n_states - 1) * M, N_test)),
        ])
    else:
        U_test_full = U_test_small

    return FitResult(
        Lambda=Lambda,
        Lambda_U=Lambda_U,
        Lambda_f=Lambda_f,
        Lambda_w=Lambda_w,
        z_bar=z_bar,
        u=u,
        U_test=U_test_full,
        n_states=n_states,
        n_s=n_s,
    )
