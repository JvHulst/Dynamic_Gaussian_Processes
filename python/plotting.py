"""Plotting utilities for the DGP estimator.

Provides the waterfall and error-norm plots used by main.py.
Separated out so that main.py stays focused on the estimation pipeline
and these helpers can be reused in notebooks or other scripts.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_waterfall(ax, cfg, x, y, f, c, c_map, show_cb,
                   title_str, zlabel_str):
    """3-D waterfall plot of function snapshots over time.

    Parameters
    ----------
    ax         : Axes3D
    cfg        : Config with .x_test, .N_test, .x_min, .x_max
    x, y       : measurement locations and values (may be None for error plots)
    f          : (N_test, N) function values to plot
    c          : (N_test, N_test, N) covariance (may be None)
    c_map      : matplotlib colormap
    show_cb    : bool — draw ±2σ confidence bounds
    title_str  : str
    zlabel_str : str
    """
    N = f.shape[1]
    for t in range(N):
        color = c_map(t / max(N - 1, 1))
        ax.plot(cfg.x_test, np.full(cfg.N_test, t), f[:, t],
                color=color, linewidth=1.5)

        if show_cb and c is not None:
            cb = 2.0 * np.sqrt(np.diag(c[:, :, t]))
            ax.plot(cfg.x_test, np.full(cfg.N_test, t), f[:, t] - cb,
                    color=color, linewidth=0.5)
            ax.plot(cfg.x_test, np.full(cfg.N_test, t), f[:, t] + cb,
                    color=color, linewidth=0.5)

        if x is not None and y is not None:
            ax.scatter(x[:, t], np.full(x.shape[0], t), y[:, t],
                       marker='x', color=color, s=60)

    ax.set_title(title_str, fontsize=15)
    ax.set_xlabel(r'$x$', fontsize=15)
    ax.set_ylabel(r'$t$', fontsize=15)
    ax.set_zlabel(zlabel_str, fontsize=15)
    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(0, N - 1)
    ax.view_init(elev=30, azim=-45)


def plot_summary(cfg, x, y, f_true, f_pred, f_upd, c_pred, c_upd, e_upd):
    """Four-panel waterfall figure: truth, prediction, update, error.

    Parameters
    ----------
    cfg    : Config
    x      : (p, N)           measurement locations
    y      : (p, N)           noisy observations
    f_true : (N_test, N)      true function
    f_pred : (N_test, N+1)    predicted function mean
    f_upd  : (N_test, N)      updated function mean
    c_pred : (N_test, N_test, N+1)
    c_upd  : (N_test, N_test, N)
    e_upd  : (N_test, N)      estimation error after update
    """
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.size": 12,
    })

    fig = plt.figure(figsize=(16, 10))
    c_map = plt.get_cmap('winter')

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    plot_waterfall(ax1, cfg, x, y, f_true, None, c_map, False,
                   'True function', r'$f_{t}(x)$')

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    plot_waterfall(ax2, cfg, x, y, f_pred[:, :cfg.N], c_pred[:, :, :cfg.N],
                   c_map, cfg.plot_confidence_bounds,
                   'Estimate after prediction step', r'$\hat{f}_{t|t-1}(x)$')

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    plot_waterfall(ax3, cfg, x, y, f_upd, c_upd, c_map,
                   cfg.plot_confidence_bounds,
                   'Estimate after update step', r'$\hat{f}_{t|t}(x)$')

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    plot_waterfall(ax4, cfg, None, None, e_upd, c_upd, c_map,
                   cfg.plot_confidence_bounds,
                   'Estimation error after update step',
                   r'$\tilde{f}_{t|t}(x)$')

    fig.tight_layout()
    return fig


def plot_error_norm(cfg, e_upd):
    """Plot the 2-norm of the estimation error over time.

    Parameters
    ----------
    cfg   : Config
    e_upd : (N_test, N)  estimation error after update
    """
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(np.arange(cfg.N), np.sqrt(np.sum(e_upd**2, axis=0)),
            linewidth=1, label=r'$\Vert \tilde{f}_{t|t} \Vert_2$')
    ax.grid(True)
    ax.set_title('Estimation error 2-norm after update step', fontsize=15)
    ax.set_xlabel(r'$t$', fontsize=15)
    ax.set_ylabel(r'$\Vert \tilde{f}_{t|t} \Vert_2$', fontsize=15)
    ax.legend()
    fig.tight_layout()
    return fig
