"""Dynamic Gaussian Processes — main script.

Python equivalent of Dynamic_Gaussian_Process_Main.m

Jilles van Hulst, 2023 (Python port 2025)

Estimation of Dynamic Gaussian Processes
— J. van Hulst, R. van Zuijlen, D. Antunes, W.P.M.H. Heemels

Usage
-----
Configure the settings in the "Configuration" block below, then run::

    python main.py

The script will:
  1. Build the estimator kernels  (evolution, mean, covariance, noise)
  2. Generate or load ground-truth data
  3. Fit basis functions to the system equations
  4. Run the DGP (Kalman-filter) estimator
  5. Display waterfall and error-norm plots

For your own data, set ``system = "Data"`` and choose any ``kernel`` that
matches the dynamics you expect.  See the Config docstring for all options.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from squexp import squexp
from dgp_function_fitting import Config, dgp_function_fitting
from dgp_estimation import dgp_estimation
from dgp_truth_generators import (
    dgp_heat_equation,
    dgp_wave_equation,
    dgp_simulation,
    dgp_data,
)
from plotting import plot_summary, plot_error_norm


# ═══════════════════════════════════════════════════════════════════════════
# System-kernel builders
# ═══════════════════════════════════════════════════════════════════════════
#
# Each builder returns  (kf, Q_f, Q_w, params)  where
#   kf   — evolution kernel (callable or list-of-lists for multi-state)
#   Q_f  — initial function covariance kernel  (same shape as kf)
#   Q_w  — disturbance covariance kernel       (same shape as kf)
#   params — dict of physics parameters used by the corresponding truth
#            generator (e.g. c_wave, delta_t_wave).  May be empty.
#
# The prior mean m(x) is built separately by _build_mean().
# ═══════════════════════════════════════════════════════════════════════════

def _zero(X, V):
    """Zero kernel block (off-diagonal in multi-state systems)."""
    return np.zeros((np.atleast_1d(X).ravel().size,
                     np.atleast_1d(V).ravel().size))


# ── Wave equation (D'Alembert) ───────────────────────────────────────────

def _build_wave_equation(cfg):
    """D'Alembert 2×2 kernel system for the 1-D wave equation.

    Two function-valued states [f, g] where g = df/dt.
    The kernel entries follow from D'Alembert's formula:
        f(x, t+dt) = 1/2 [f(x-c*dt) + f(x+c*dt)]  +  1/(2c) int g(v) dv

    Parameters used internally
    --------------------------
    c  : float — wave propagation speed             [length / time]
    dt : float — sample time between observations   [time]
    """
    c   = 2.0
    dt  = 0.2
    shift = c * dt
    eps   = cfg.dx          # Gaussian approximation width

    def kf_ff(X, V):
        Xc, Vc = X.reshape(-1, 1), V.reshape(1, -1)
        return (np.exp(-(Xc - Vc - shift)**2 / (2*eps**2))
              + np.exp(-(Xc - Vc + shift)**2 / (2*eps**2))) \
              / (2*eps*np.sqrt(2*np.pi))

    def kf_fg(X, V):
        Xc, Vc = X.reshape(-1, 1), V.reshape(1, -1)
        return (1/(4*c)) * (
            erf((Vc - Xc + shift) / (eps*np.sqrt(2)))
          - erf((Vc - Xc - shift) / (eps*np.sqrt(2))))

    def kf_gf(X, V):
        Xc, Vc = X.reshape(-1, 1), V.reshape(1, -1)
        return c / (2*eps**3*np.sqrt(2*np.pi)) * (
            (Xc - Vc - shift) * np.exp(-(Xc - Vc - shift)**2 / (2*eps**2))
          - (Xc - Vc + shift) * np.exp(-(Xc - Vc + shift)**2 / (2*eps**2)))

    kf = [[kf_ff, kf_fg],
          [kf_gf, kf_ff]]                   # kf_gg = kf_ff by symmetry

    Q_f = [[lambda X, V: 1e1  * squexp(X, V, 1e1), _zero],
           [_zero, lambda X, V: 1e-1 * squexp(X, V, 1e0)]]
    Q_w = [[lambda X, V: 1e-2 * squexp(X, V, 1e0), _zero],
           [_zero, lambda X, V: 1e-2 * squexp(X, V, 1e0)]]

    params = dict(c_wave=c, delta_t_wave=dt)
    return kf, Q_f, Q_w, params


# ── Heat equation (Gaussian Green's function) ────────────────────────────

def _build_heat_equation(cfg):
    """Gaussian Green's-function kernel for the 1-D heat equation.

    Single function-valued state.  The kernel is the fundamental solution:
        kf(x, v) = 1 / sqrt(4*pi*alpha*dt) * exp(-(x-v)^2 / (4*alpha*dt))

    Parameters used internally
    --------------------------
    alpha : float — thermal diffusivity   [length^2 / time]
    dt    : float — sample time           [time]
    """
    alpha = 1.0
    dt    = 1.0

    kf = lambda X, V: (1.0 / np.sqrt(4*np.pi*alpha*dt)
        * np.exp(-(X.reshape(-1, 1) - V.reshape(1, -1))**2 / (4*alpha*dt)))

    Q_f = lambda X, V: 1e0  * squexp(X, V, 1e1)
    Q_w = lambda X, V: 1e-2 * squexp(X, V, 1e0)

    params = dict(alpha=alpha, delta_t=dt)
    return kf, Q_f, Q_w, params


# ── Smoothing kernel (RBF blur) ──────────────────────────────────────────

def _build_smoothing(cfg):
    """Squared-exponential smoothing / blurring kernel (single-state).

    Each time step the function is convolved with a normalised Gaussian,
    causing it to gradually flatten.  Useful for slowly decaying signals.

    Parameters used internally
    --------------------------
    sigma_kf : float — kernel length scale  (larger = more smoothing)
    a_kf     : float — amplitude scaling    (< 1 for strict contraction)
    """
    sigma_kf = 0.5
    a_kf     = 0.8

    kf = lambda X, V: a_kf * squexp(X, V, sigma_kf) \
                       / (sigma_kf * np.sqrt(2*np.pi))

    Q_f = lambda X, V: 1e0  * squexp(X, V, 1e1)
    Q_w = lambda X, V: 1e-2 * squexp(X, V, 1e0)

    params = dict(sigma_kf=sigma_kf, a_kf=a_kf)
    return kf, Q_f, Q_w, params


# ── Integrator (Kronecker-delta / identity) ──────────────────────────────

def _build_integrator(cfg):
    """Kronecker-delta kernel:  f_{t+1}(x) = f_t(x) + w_t(x).

    The function does not evolve deterministically — all change comes from
    the disturbance w_t.  Useful as a baseline / random-walk prior.
    """
    dx = cfg.dx

    kf = lambda X, V: ((np.abs(X.reshape(-1, 1) - V.reshape(1, -1)) <= dx/2)
                        .astype(float) / dx)

    Q_f = lambda X, V: 1e0  * squexp(X, V, 1e1)
    Q_w = lambda X, V: 1e-2 * squexp(X, V, 1e0)

    params = {}
    return kf, Q_f, Q_w, params


# ── Kernel registry ──────────────────────────────────────────────────────

_KERNEL_BUILDERS = {
    "wave_equation":  _build_wave_equation,
    "heat_equation":  _build_heat_equation,
    "smoothing":      _build_smoothing,
    "integrator":     _build_integrator,
}


# ═══════════════════════════════════════════════════════════════════════════
# Prior mean builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_mean(cfg):
    """Return the prior mean function m(x) based on ``cfg.initial_mean``.

    For multi-state systems (e.g. wave equation), m(x) is applied to the
    first state only; additional states are initialised at zero by
    dgp_function_fitting.
    """
    if cfg.initial_mean == "zero":
        return lambda X: np.zeros((np.atleast_1d(X).ravel().size, 1))
    elif cfg.initial_mean == "squexp":
        return lambda X: 10 * squexp(X.reshape(-1, 1), np.array([[0.0]]), 1.0)
    elif cfg.initial_mean == "ones":
        return lambda X: np.ones((np.atleast_1d(X).ravel().size, 1))
    elif cfg.initial_mean == "parabola":
        return lambda X: (-2 * X.reshape(-1, 1)**2 + 8)
    else:
        raise ValueError(f"Unknown initial_mean: {cfg.initial_mean!r}")


# ═══════════════════════════════════════════════════════════════════════════
# Top-level kernel constructor
# ═══════════════════════════════════════════════════════════════════════════

def build_kernels(cfg):
    """Build all estimator kernels from the configuration.

    Returns
    -------
    kf     : evolution kernel  (callable, or list-of-lists for multi-state)
    m      : prior mean function  m(x) -> (N, 1)
    Q_f    : initial covariance kernel
    Q_w    : disturbance covariance kernel
    Q_v    : measurement noise covariance kernel  (diagonal)
    params : dict with physics parameters for truth generators
    """
    builder = _KERNEL_BUILDERS.get(cfg.kernel)
    if builder is None:
        raise ValueError(
            f"Unknown kernel: {cfg.kernel!r}.  "
            f"Choose from {list(_KERNEL_BUILDERS)}")

    kf, Q_f, Q_w, params = builder(cfg)
    m = _build_mean(cfg)

    # measurement noise — diagonal, shared across all kernels
    sigma2_v = 1e-5
    Q_v = lambda X, V: sigma2_v * (X.ravel()[:, None] == V.ravel()[None, :]).astype(float)
    params["sigma2_v"] = sigma2_v

    return kf, m, Q_f, Q_w, Q_v, params


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():

    # ── Configuration ────────────────────────────────────────────────────
    cfg = Config(
        # spatial domain
        x_min=-10.0,
        x_max=10.0,
        N_test=251,               # evaluation grid resolution

        # estimation settings
        p=5,                      # measurement locations per time step
        N=21,                     # number of time steps
        M=31,                     # number of basis functions

        # basis function family
        basis="Fourier",
        #   'RBF'      — radial basis functions (Gaussian bumps)
        #   'NRBF'     — normalised RBF (partition of unity)
        #   'Fourier'  — Fourier cosine / sine basis (orthogonal)
        #   'Discrete' — piecewise-constant bins      (orthogonal)

        # evolution kernel used by the estimator
        kernel="wave_equation",
        #   'heat_equation'  — Gaussian Green's function       (single-state)
        #   'wave_equation'  — D'Alembert 2×2 kernel           (two-state [f, g])
        #   'smoothing'      — squared-exponential blur         (single-state)
        #   'integrator'     — Kronecker-delta / random walk    (single-state)

        # prior mean function m(x)
        initial_mean="squexp",
        #   'zero'     — m(x) = 0
        #   'squexp'   — m(x) = 10·exp(−x²/2)     (Gaussian bump)
        #   'ones'     — m(x) = 1
        #   'parabola' — m(x) = −2x² + 8

        # ground-truth data source
        system="Wave_equation",
        #   'Discrete_approximation'  — simulate from the chosen kernel
        #                                (single-state kernels only)
        #   'Heat_equation'           — analytical heat-equation solution
        #   'Wave_equation'           — analytical wave-equation solution
        #   'Data'                    — load from .mat file (see dgp_data)

        # plotting
        plot_confidence_bounds=True,
        plot_basis_functions=False,
    )

    # ── Build estimator kernels ──────────────────────────────────────────
    kf, m, Q_f, Q_w, Q_v, params = build_kernels(cfg)

    # ── Generate / load ground truth ─────────────────────────────────────
    #
    # The truth source is independent of the estimator kernel.  For the
    # analytical PDE solutions the physics parameters are taken from the
    # kernel builder so that truth and estimator are matched by default.
    # To study model mismatch, override the values below.

    print(f"True system : {cfg.system.replace('_', ' ')}")
    print(f"Kernel      : {cfg.kernel}")
    print(f"Initial mean: {cfg.initial_mean}\n")

    if cfg.system == "Discrete_approximation":
        # Simulate the chosen kernel on a fine discrete-bin grid.
        # NOTE: only works for single-state kernels (not wave_equation).
        truth = dgp_simulation(cfg, kf, m, Q_f, Q_w, Q_v)

    elif cfg.system == "Heat_equation":
        # Analytical Gaussian Green's function (Dirac initial condition).
        truth = dgp_heat_equation(
            cfg,
            alpha=params.get("alpha", 1.0),
            delta_t=params.get("delta_t", 1.0),
            sigma2_v=params["sigma2_v"],
        )

    elif cfg.system == "Wave_equation":
        # D'Alembert solution; the estimator mean m is the initial condition.
        truth = dgp_wave_equation(
            cfg, m,
            c_wave=params.get("c_wave", 2.0),
            delta_t_wave=params.get("delta_t_wave", 0.2),
            sigma2_v=params["sigma2_v"],
        )

    elif cfg.system == "Data":
        # Load from .mat — estimator uses whatever kernel you selected above.
        truth, cfg = dgp_data(cfg)

    else:
        raise ValueError(f"Unknown system: {cfg.system!r}")

    x      = truth.x
    y      = truth.y
    f_true = truth.f_true

    # ── Fit basis functions to system equations ──────────────────────────
    fit = dgp_function_fitting(cfg, kf, m, Q_f, Q_w)

    # ── Run DGP estimator ────────────────────────────────────────────────
    est = dgp_estimation(fit, x, y, Q_v)

    f_pred = est.f_pred
    f_upd  = est.f_upd
    c_pred = est.c_pred
    c_upd  = est.c_upd

    e_upd = f_true - f_upd

    # ── Plots ────────────────────────────────────────────────────────────
    plot_summary(cfg, x, y, f_true, f_pred, f_upd, c_pred, c_upd, e_upd)
    plot_error_norm(cfg, e_upd)

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
