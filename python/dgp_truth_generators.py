"""Ground-truth generators for DGP experiments.

Equivalent of DGP_heat_equation.m, DGP_wave_equation.m, DGP_simulation.m, DGP_data.m
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.linalg import eig, solve

from dgp_function_fitting import Config


@dataclass
class Truth:
    """Ground-truth data."""
    x: np.ndarray       # (p, N)   measurement locations
    y: np.ndarray       # (p, N)   noisy measurements
    f_true: np.ndarray  # (N_test, N)  true function at test points


# ─────────────────────────────────────────────────────────────────────────
# Heat equation
# ─────────────────────────────────────────────────────────────────────────

def dgp_heat_equation(cfg: Config, alpha: float, delta_t: float,
                       sigma2_v: float) -> Truth:
    """D'Alembert heat equation with Dirac initial condition.

    f(x,t) = 1/sqrt(4*pi*alpha*t) * exp(-x^2 / (4*alpha*t))
    """
    p, N = cfg.p, cfg.N
    x_test = cfg.x_test.reshape(-1, 1)

    Sigma2_v = sigma2_v * np.eye(p)

    x      = np.full((p, N), np.nan)
    y      = np.full((p, N), np.nan)
    f_true = np.full((cfg.N_test, N), np.nan)

    for t in range(N):
        x[:, t] = np.random.uniform(cfg.x_min, cfg.x_max, p)

        time = delta_t * (t + 1)
        f_true[:, t] = _heat_eq(x_test, alpha, time).ravel()
        f_samp = _heat_eq(x[:, t].reshape(-1, 1), alpha, time).ravel()

        v = np.random.multivariate_normal(np.zeros(p), Sigma2_v)
        y[:, t] = f_samp + v

    return Truth(x=x, y=y, f_true=f_true)


def _heat_eq(x: np.ndarray, alpha: float, time: float) -> np.ndarray:
    return (1.0 / np.sqrt(4 * np.pi * alpha * time)
            * np.exp(-x**2 / (4 * alpha * time)))


# ─────────────────────────────────────────────────────────────────────────
# Wave equation
# ─────────────────────────────────────────────────────────────────────────

def dgp_wave_equation(cfg: Config, m: Callable, c_wave: float,
                       delta_t_wave: float, sigma2_v: float) -> Truth:
    """D'Alembert wave equation: f(x,t) = [m(x-ct) + m(x+ct)] / 2."""
    p, N = cfg.p, cfg.N
    x_test = cfg.x_test.reshape(-1, 1)

    Sigma2_v = sigma2_v * np.eye(p)

    x      = np.full((p, N), np.nan)
    y      = np.full((p, N), np.nan)
    f_true = np.full((cfg.N_test, N), np.nan)

    for t in range(N):
        x[:, t] = np.random.uniform(cfg.x_min, cfg.x_max, p)

        time = delta_t_wave * t           # t=0 gives the initial condition
        f_true[:, t] = _wave_eq(x_test, m, c_wave, time).ravel()
        f_samp = _wave_eq(x[:, t].reshape(-1, 1), m, c_wave, time).ravel()

        v = np.random.multivariate_normal(np.zeros(p), Sigma2_v)
        y[:, t] = f_samp + v

    return Truth(x=x, y=y, f_true=f_true)


def _wave_eq(x: np.ndarray, m: Callable, c: float, time: float) -> np.ndarray:
    return (m(x - c * time) + m(x + c * time)) / 2.0


# ─────────────────────────────────────────────────────────────────────────
# Discrete-bin simulation
# ─────────────────────────────────────────────────────────────────────────

def dgp_simulation(cfg: Config, kf, m: Callable,
                    Q_f, Q_w, Q_v) -> Truth:
    """Simulate via fine discrete-bin truth model (single-state only)."""
    if isinstance(kf, list):
        raise ValueError(
            "dgp_simulation does not support multi-state (list-of-lists kf).")

    p, N = cfg.p, cfg.N
    N_test = cfg.N_test
    x_test = cfg.x_test.reshape(-1, 1)

    M_true = N_test
    dx_fit = (cfg.x_max - cfg.x_min) / (M_true - 1)
    x_fit  = np.linspace(cfg.x_min, cfg.x_max, M_true).reshape(-1, 1)

    bounds = np.linspace(cfg.x_min, cfg.x_max * (1 + 1e-12), M_true + 1)

    u_true = []
    U_fit  = np.zeros((M_true, M_true))
    U_test = np.zeros((M_true, N_test))

    for i in range(M_true):
        lo, hi = bounds[i], bounds[i + 1]
        fn = (lambda X, a=lo, b=hi, d=dx_fit:
              ((X.ravel() >= a) & (X.ravel() < b)).astype(float) / d)
        u_true.append(fn)
        U_fit[i, :] = fn(x_fit)
        U_test[i, :] = fn(x_test)

    Pinv_U    = solve(U_fit @ U_fit.T, U_fit)
    Lambda_U  = U_fit.T @ U_fit * dx_fit**2

    Lambda_true = Pinv_U @ kf(x_fit, x_fit)
    z_bar_true  = (Pinv_U @ m(x_fit)).ravel()
    Lambda_f    = Pinv_U @ Q_f(x_fit, x_fit)
    Lambda_w    = Pinv_U @ Q_w(x_fit, x_fit)

    Lambda_f = (Lambda_f + Lambda_f.T) / 2
    Lambda_w = (Lambda_w + Lambda_w.T) / 2

    eig_max = np.max(np.abs(eig(Lambda_true @ Lambda_U)[0]))
    if eig_max < 1.0:
        print(f"'True' system is stable (max |eig| = {eig_max:.4f})")
    elif abs(eig_max - 1.0) < 1e-6:
        print(f"'True' system is marginally stable (max |eig| = {eig_max:.6f})")
    else:
        print(f"'True' system is NOT stable (max |eig| = {eig_max:.4f})")

    x      = np.full((p, N), np.nan)
    y      = np.full((p, N), np.nan)
    z      = np.full((M_true, N + 1), np.nan)
    f_true = np.full((N_test, N), np.nan)

    z[:, 0] = np.random.multivariate_normal(z_bar_true, Lambda_f)

    U_samp = np.zeros((M_true, p))

    for t in range(N):
        x[:, t] = np.random.uniform(cfg.x_min, cfg.x_max, p)

        for i in range(M_true):
            U_samp[i, :] = u_true[i](x[:, t])

        f_true[:, t] = U_test.T @ z[:, t]
        f_samp = U_samp.T @ z[:, t]

        v = np.random.multivariate_normal(np.zeros(p), Q_v(x[:, t], x[:, t]))
        y[:, t] = f_samp + v

        xi = np.random.multivariate_normal(np.zeros(M_true), Lambda_w)
        z[:, t + 1] = Lambda_true @ Lambda_U @ z[:, t] + xi

    return Truth(x=x, y=y, f_true=f_true)


# ─────────────────────────────────────────────────────────────────────────
# Load from data
# ─────────────────────────────────────────────────────────────────────────

def dgp_data(cfg: Config, mat_file: str = "heat_equation_simulation.mat"):
    """Load ground-truth from a .mat file. Returns (Truth, cfg) — cfg may be updated."""
    import scipy.io as sio

    data = sio.loadmat(mat_file)
    x_data = data["x"]
    y_data = data["y"]

    cfg.N = y_data.shape[1]
    cfg.p = y_data.shape[0]

    if "f" in data:          # function handle — not directly portable from .mat
        raise NotImplementedError("Function-handle 'f' in .mat not supported in Python.")
    elif "f_true" in data and data["f_true"].shape[0] == cfg.N_test:
        f_true_data = data["f_true"]
        if "x_test" in data:
            cfg.x_min = float(data["x_test"].min())
            cfg.x_max = float(data["x_test"].max())
            cfg.N_test = data["x_test"].shape[0]
            cfg.__post_init__()           # recompute derived fields
    elif "f_true" not in data:
        f_true_data = np.full((cfg.N_test, cfg.N), np.nan)
    else:
        raise ValueError("Dataset does not have the correct format")

    return Truth(x=x_data, y=y_data, f_true=f_true_data), cfg
