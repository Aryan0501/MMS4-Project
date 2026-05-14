"""Data pipeline for MMS/FPI electron distribution-function gap reconstruction.

Responsibilities
----------------
1. Locate and read MMS FPI burst-mode L2 ``des-dist`` CDF files.
2. Extract the 3-D velocity distribution function ``f(E, theta, phi)`` per timestep
   (native CDF shape: ``(n_time, 32 energy, 16 theta, 32 phi)``).
3. Transform the distribution into a network-friendly range:
       x = log10(f + EPS)            # f spans ~10**-32.5 .. 10**-24, plus exact zeros
       x_norm = (x - LOG_MIN) / (LOG_MAX - LOG_MIN)   # -> roughly [0, 1], then clipped
   The inverse transform is provided so reconstructed cubes can be returned to
   physical units (s**3 / cm**6) for moments validation.
4. Build the occlusion mask (which (E, theta, phi) bins are "missing") and apply it.

Design notes
------------
The transform constants are FIXED (not fitted per file). That is deliberate: the
incremental / continual-learning loop processes one file at a time and discards it,
so a global, file-independent normalisation keeps every batch on the same scale and
makes warm-started training numerically consistent across batches.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

try:  # cdflib is only needed when actually reading CDFs
    import cdflib
except ImportError:  # pragma: no cover
    cdflib = None


# --- distribution-function transform constants -------------------------------
# log10(f) for non-zero electron PSD in these files sits in ~[-32.5, -24].
# Exact-zero bins (below one-count level / occluded) map to LOG_MIN.
EPS = 1.0e-33          # additive floor so log10(0) -> log10(EPS) = -33
LOG_MIN = -33.0        # maps to 0.0 after normalisation
LOG_MAX = -23.0        # maps to 1.0 after normalisation

DIST_VAR_TEMPLATE = "mms{sc}_des_dist_brst"   # e.g. mms1_des_dist_brst

N_ENERGY, N_THETA, N_PHI = 32, 16, 32


# --- transforms --------------------------------------------------------------
def to_model_space(f: np.ndarray) -> np.ndarray:
    """Physical PSD -> normalised log space in [0, 1] (float32)."""
    x = np.log10(np.asarray(f, dtype=np.float64) + EPS)
    x = (x - LOG_MIN) / (LOG_MAX - LOG_MIN)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def to_physical_space(x: np.ndarray) -> np.ndarray:
    """Normalised log space -> physical PSD (s**3 / cm**6)."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    log_f = x * (LOG_MAX - LOG_MIN) + LOG_MIN
    f = 10.0 ** log_f - EPS
    return np.clip(f, 0.0, None)


# --- occlusion mask ----------------------------------------------------------
def synthetic_wedge_mask(
    n_energy: int = N_ENERGY,
    n_theta: int = N_THETA,
    n_phi: int = N_PHI,
    theta_frac: float = 0.5,
    phi_frac: float = 0.5,
) -> np.ndarray:
    """A contiguous theta x phi wedge, identical across all energies.

    Mimics a spacecraft-body shadow: a fixed solid-angle region the detector
    cannot see. Returns a boolean array, ``True`` where the bin is OCCLUDED
    (i.e. must be reconstructed). ``theta_frac * phi_frac`` of bins are masked.
    """
    mask = np.zeros((n_energy, n_theta, n_phi), dtype=bool)
    t_hi = int(round(n_theta * theta_frac))
    p_hi = int(round(n_phi * phi_frac))
    mask[:, :t_hi, :p_hi] = True
    return mask


def load_explosion_mask(repo_root: str) -> np.ndarray:
    """Load the project's realistic occlusion mask ``skymaps/explosion.npy``.

    Returns a boolean array of shape (32, 16, 32), ``True`` where occluded.
    The stored file is 0/1 float with mean ~0.5; we treat the 1-valued bins
    as the occluded region.
    """
    path = os.path.join(repo_root, "MMS-FPI-Data-Gaps", "skymaps", "explosion.npy")
    arr = np.load(path)
    if arr.shape != (N_ENERGY, N_THETA, N_PHI):
        raise ValueError(f"explosion.npy has unexpected shape {arr.shape}")
    return arr > 0.5


def apply_mask(cubes: np.ndarray, mask: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Return a copy of ``cubes`` (..., 32, 16, 32) with ``mask`` bins set to ``fill``.

    ``fill=0.0`` in model space corresponds to PSD = 10**LOG_MIN, i.e. the
    "nothing detected" sentinel — the same value real empty bins take.
    """
    out = cubes.copy()
    out[..., mask] = fill
    return out


def apply_mask_shell_mean(cubes: np.ndarray, mask_per_bin: np.ndarray) -> np.ndarray:
    """Replace ``mask_per_bin``-True bins with the per-energy-shell visible-bin mean.

    Used in place of the dumb ``fill=0`` to avoid the model-cheat where the
    network simply outputs 0 wherever it sees 0 in the input. After this fill,
    the input is a smooth, mean-interpolated continuation of the visible data
    at every energy; the model can only reduce loss by learning structure that
    does *better* than the shell-mean.

    Vectorised: handles all timesteps at once.
    Inputs:
        cubes : (n, 32, 16, 32) — model-space PSDs
        mask_per_bin : (32, 16, 32) bool — True where the bin should be replaced
    """
    visible = ~mask_per_bin                               # (32, 16, 32)
    # per-energy: how many visible bins at each E
    n_vis = visible.reshape(cubes.shape[1], -1).sum(axis=1).clip(min=1)   # (32,)
    # per-(time, E) sum over visible bins
    sums = (cubes * visible[None, ...]).reshape(cubes.shape[0], cubes.shape[1], -1).sum(axis=2)
    shell_mean = sums / n_vis[None, :]                    # (n, 32)
    fill_field = shell_mean[:, :, None, None] * np.ones((1, 1, cubes.shape[2], cubes.shape[3]))
    return np.where(visible[None, ...], cubes, fill_field).astype(cubes.dtype)


# --- CDF reading -------------------------------------------------------------
@dataclass
class FileBatch:
    """One CDF file's worth of data, already transformed to model space."""

    path: str
    cubes: np.ndarray          # (n_time, 32, 16, 32) float32, model space
    epoch: np.ndarray          # (n_time,) CDF TT2000 timestamps

    @property
    def n_samples(self) -> int:
        return self.cubes.shape[0]


def find_dist_files(root: str, pattern: str = "**/*des-dist*.cdf") -> list[str]:
    """Recursively find FPI des-dist CDF files under ``root`` (sorted)."""
    return sorted(glob.glob(os.path.join(root, pattern), recursive=True))


def read_dist_file(path: str, spacecraft: int = 1, subsample: int = 1,
                   with_phi: bool = False) -> FileBatch:
    """Read one des-dist CDF, return a :class:`FileBatch` in model space.

    Parameters
    ----------
    subsample : take every ``subsample``-th timestep (>=1). Useful on CPU.
    with_phi  : also load the per-timestep azimuth table ``des_phi_brst`` and
                store it on the returned object as ``.phi`` (n_time, 32) in degrees.
    """
    if cdflib is None:  # pragma: no cover
        raise ImportError("cdflib is required to read CDF files (`pip install cdflib`)")
    cdf = cdflib.CDF(path)
    var = DIST_VAR_TEMPLATE.format(sc=spacecraft)
    raw = np.asarray(cdf.varget(var), dtype=np.float32)   # (n_time, 32, 16, 32)
    epoch = np.asarray(cdf.varget("Epoch"))
    phi = np.asarray(cdf.varget(f"mms{spacecraft}_des_phi_brst")) if with_phi else None
    if subsample > 1:
        raw = raw[::subsample]
        epoch = epoch[::subsample]
        if phi is not None:
            phi = phi[::subsample]
    fb = FileBatch(path=path, cubes=to_model_space(raw), epoch=epoch)
    if phi is not None:
        fb.phi = phi
    return fb


# --- feature channels + temporal windows ------------------------------------
def build_inputs(
    fb: FileBatch,
    mask: np.ndarray,
    use_pitch_angle: bool = False,
    use_logb: bool = False,
    temporal_window: int = 0,
    with_validity: bool = False,
    always_zero: np.ndarray | None = None,
    missing_fill: str = "zero",
):
    """Assemble (X, Y) arrays for one file.

    Y is the unmasked distribution, shape (n, 32, 16, 32, 1).
    X stacks, along the channel axis:
        - the masked distribution at offsets -w..+w (2w+1 channels) if temporal_window=w,
          else just the masked distribution (1 channel);
        - the pitch-angle map (1 channel) if use_pitch_angle;
        - the log|B| map (1 channel) if use_logb.
    Channels of neighbouring timesteps at the file edges are clamped (edge-replicated).

    The masked-distribution channels use ``fill=0.0`` for occluded bins (== "nothing
    detected" in model space). The target Y is the full, unmasked cube.
    """
    cubes = fb.cubes                                   # (n, 32,16,32) model space
    n = cubes.shape[0]
    # Combine the synthetic mask with the always-zero region so we treat both as
    # "unreliable input" -- avoids the cheat where the model copies 0 from input.
    full_unreliable = mask if always_zero is None else (mask | always_zero)
    if missing_fill == "shell_mean":
        masked = apply_mask_shell_mean(cubes, full_unreliable)
    else:
        masked = apply_mask(cubes, full_unreliable)    # legacy: zero-fill

    chans = []
    w = temporal_window
    if w > 0:
        for d in range(-w, w + 1):
            idx = np.clip(np.arange(n) + d, 0, n - 1)
            chans.append(masked[idx])
    else:
        chans.append(masked)

    if use_pitch_angle or use_logb:
        from features import pitch_angle_and_logb
        phi = getattr(fb, "phi", None)
        pa, lb = pitch_angle_and_logb(fb.path, fb.epoch, phi)
        if use_pitch_angle:
            chans.append(pa)
        if use_logb:
            chans.append(lb)

    if with_validity:
        # 1 where the bin is RELIABLE input (visible AND not always-zero); 0 otherwise.
        # Tells the model "trust this bin's input value or not", solving the
        # ambiguity between (a) synthetic-mask bin, (b) naturally-zero bin, and
        # (c) genuinely-low-PSD bin -- all of which would otherwise be 0 in input.
        n = cubes.shape[0]
        v = np.ones((n, *cubes.shape[1:]), dtype=np.float32)
        v[:, mask] = 0.0
        if always_zero is not None:
            v[:, always_zero] = 0.0
        chans.append(v)

    X = np.stack(chans, axis=-1).astype(np.float32)    # (n, 32,16,32, C)
    Y = cubes[..., None]
    return X, Y


def split_files(files: list[str], val_fraction: float = 0.2, seed: int = 0):
    """Split a file list into (train_files, val_files).

    Splitting at the *file* level (not the sample level) avoids temporal leakage:
    consecutive burst timesteps are highly correlated, so a random per-sample split
    would put near-duplicates in both train and validation.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    n_val = max(1, int(round(len(files) * val_fraction)))
    val_idx = set(idx[:n_val].tolist())
    train = [f for i, f in enumerate(files) if i not in val_idx]
    val = [f for i, f in enumerate(files) if i in val_idx]
    return train, val
