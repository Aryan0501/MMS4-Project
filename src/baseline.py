"""Non-learned baselines the ML model must beat.

A reconstruction model that cannot outperform a cheap interpolation rule is not
worth its complexity. The reference baseline here is **per-energy-shell angular
mean**: for each occluded (E, theta, phi) bin, predict the mean of the *visible*
bins at the same energy E. This exploits the fact that an electron distribution
is, to first order, fairly isotropic in look-direction at fixed energy — so the
visible angular bins are a reasonable estimator for the hidden ones.

We also report the trivial "predict the dataset-mean constant" floor.
"""

from __future__ import annotations

import numpy as np


def energy_shell_mean_fill(cubes_masked: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill occluded bins with the visible-bin mean at the same energy shell.

    Parameters
    ----------
    cubes_masked : (n, 32, 16, 32) — cubes with occluded bins already zeroed.
    mask : (32, 16, 32) bool — True where occluded.

    Returns
    -------
    (n, 32, 16, 32) — filled cubes.
    """
    visible = ~mask                                  # (32,16,32)
    out = cubes_masked.copy()
    for e in range(cubes_masked.shape[1]):
        vis_e = visible[e]                           # (16,32)
        if not vis_e.any():
            shell_mean = np.zeros(cubes_masked.shape[0])
        else:
            shell_mean = cubes_masked[:, e][:, vis_e].mean(axis=1)   # (n,)
        occ_e = mask[e]                              # (16,32)
        out[:, e][:, occ_e] = shell_mean[:, None]
    return out


def masked_mse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """Mean squared error over occluded bins only (per-bin, averaged over samples)."""
    diff2 = (pred - truth) ** 2
    return float(diff2[:, mask].mean())


def masked_mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    return float(np.abs(pred - truth)[:, mask].mean())


def evaluate_baselines(cubes_full: np.ndarray, mask: np.ndarray) -> dict:
    """Compute baseline masked-MSE/MAE on a set of (n,32,16,32) cubes in model space."""
    masked_in = cubes_full.copy()
    masked_in[:, mask] = 0.0

    shell = energy_shell_mean_fill(masked_in, mask)
    const_val = cubes_full[:, ~mask].mean()          # mean over all visible bins
    const = np.full_like(cubes_full, const_val)

    return {
        "energy_shell_mean": {
            "masked_mse": masked_mse(shell, cubes_full, mask),
            "masked_mae": masked_mae(shell, cubes_full, mask),
        },
        "constant_mean": {
            "masked_mse": masked_mse(const, cubes_full, mask),
            "masked_mae": masked_mae(const, cubes_full, mask),
        },
    }
