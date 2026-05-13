"""Extra physical feature channels for the reconstruction model.

The earlier project notebooks read only the distribution function itself and left almost
every other CDF variable on the table. The two highest-value extras are:

1. **Pitch-angle map** — the angle between each detector look-direction and the local
   magnetic field. This injects the *geometry* of B into the model, which is what makes a
   plasma distribution anisotropic in the first place. Computed from FGM burst data
   (matched ``data_*/fgm/*.cdf``) interpolated onto the FPI timestamps. We use B in the
   DMPA (despun) frame, which is close enough to the FPI instrument frame for a feature
   channel; a publication-grade pitch angle would apply the full attitude transform.

2. **log|B|** — the field magnitude, a per-timestep scalar broadcast over the cube. Cheap,
   and it correlates with plasma regime.

Other CDF variables worth exploring later (kept as notes, not yet wired in): the per-bin
``des_errorflags`` / ``des_compressionloss`` quality words; the time-varying ``des_energy``
step tables; spacecraft position ``fgm_r_gse``; and entirely separate products such as
``des-moms`` (official moments, for ground-truth validation) and EDP electric field.

All feature channels are returned already normalised to roughly [0, 1].
"""

from __future__ import annotations

import glob
import os

import numpy as np

try:
    import cdflib
except ImportError:  # pragma: no cover
    cdflib = None

from data_pipeline import N_ENERGY, N_THETA, N_PHI

# nominal FPI polar-bin centres (deg): 16 contiguous 11.25-deg bins
_THETA_DEG = 11.25 / 2.0 + 11.25 * np.arange(N_THETA)
# nominal azimuth-bin centres (deg) if the per-timestep table is unavailable
_PHI_DEG_NOMINAL = 11.25 / 2.0 + 11.25 * np.arange(N_PHI)

_LOGB_MIN, _LOGB_MAX = -1.0, 3.0   # log10(|B|/nT): ~0.1 nT .. ~1000 nT -> [0,1]


def _look_directions(phi_deg: np.ndarray) -> np.ndarray:
    """Unit velocity-space directions for every (theta, phi) bin.

    Follows the FPI/moments convention v_hat = (-sinθ cosφ, -sinθ sinφ, -cosθ).

    Parameters
    ----------
    phi_deg : (n_phi,) azimuth-bin centres in degrees (per-timestep or nominal).

    Returns
    -------
    (n_theta, n_phi, 3) array of unit vectors.
    """
    th = np.deg2rad(_THETA_DEG)[:, None]            # (16,1)
    ph = np.deg2rad(phi_deg)[None, :]               # (1,32)
    vx = -np.sin(th) * np.cos(ph)
    vy = -np.sin(th) * np.sin(ph)
    vz = -np.cos(th) * np.ones_like(ph)
    v = np.stack([vx * np.ones_like(ph), vy * np.ones_like(ph), vz], axis=-1)  # (16,32,3)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _find_matching_fgm(dist_path: str) -> str | None:
    """Find the FGM burst file whose start-time stamp matches this des-dist file."""
    base = os.path.basename(dist_path)
    # ...des-dist_YYYYMMDDhhmmss_v...
    try:
        stamp = base.split("des-dist_")[1].split("_")[0]
    except IndexError:
        return None
    fgm_dir = os.path.join(os.path.dirname(os.path.dirname(dist_path)), "fgm")
    hits = glob.glob(os.path.join(fgm_dir, f"*_{stamp}_*.cdf"))
    return hits[0] if hits else None


def load_b_dmpa(fgm_path: str):
    """Return (epoch_tt2000, B_dmpa[:, :3], |B|) from an FGM burst file."""
    if cdflib is None:  # pragma: no cover
        raise ImportError("cdflib required")
    c = cdflib.CDF(fgm_path)
    epoch = np.asarray(c.varget("Epoch"))
    b = np.asarray(c.varget("mms1_fgm_b_dmpa_brst_l2"), dtype=np.float64)  # (n,4): Bx,By,Bz,|B|
    return epoch, b[:, :3], b[:, 3]


def pitch_angle_and_logb(
    dist_path: str,
    fpi_epoch: np.ndarray,
    fpi_phi: np.ndarray | None = None,
):
    """Build (pitch_angle_norm, logB_norm) channels aligned to ``fpi_epoch``.

    Parameters
    ----------
    dist_path : path to the des-dist CDF (used to locate the matching FGM file).
    fpi_epoch : (n_time,) TT2000 timestamps of the FPI samples.
    fpi_phi   : (n_time, n_phi) per-timestep azimuth centres in deg, or None to use nominal.

    Returns
    -------
    pa : (n_time, 32, 16, 32) float32 in [0,1]  — arccos(v_hat . B_hat)/pi, broadcast over energy
    lb : (n_time, 32, 16, 32) float32 in [0,1]  — log10(|B|) normalised, broadcast everywhere
    Both are returned as zeros if no matching FGM file is found (graceful degradation).
    """
    n_time = len(fpi_epoch)
    fgm_path = _find_matching_fgm(dist_path)
    if fgm_path is None:
        z = np.zeros((n_time, N_ENERGY, N_THETA, N_PHI), dtype=np.float32)
        return z, z.copy()

    fgm_epoch, b_vec, b_mag = load_b_dmpa(fgm_path)
    t_fgm = fgm_epoch.astype(np.float64)
    t_fpi = np.asarray(fpi_epoch, dtype=np.float64)
    # interpolate each B component (and |B|) onto the FPI clock
    bx = np.interp(t_fpi, t_fgm, b_vec[:, 0])
    by = np.interp(t_fpi, t_fgm, b_vec[:, 1])
    bz = np.interp(t_fpi, t_fgm, b_vec[:, 2])
    bmag = np.interp(t_fpi, t_fgm, b_mag)
    b_hat = np.stack([bx, by, bz], axis=-1)
    b_hat = b_hat / (np.linalg.norm(b_hat, axis=-1, keepdims=True) + 1e-30)   # (n_time,3)

    pa = np.empty((n_time, N_THETA, N_PHI), dtype=np.float32)
    for i in range(n_time):
        phi_deg = fpi_phi[i] if fpi_phi is not None else _PHI_DEG_NOMINAL
        v = _look_directions(phi_deg)                       # (16,32,3)
        cos_a = np.clip(v @ b_hat[i], -1.0, 1.0)            # (16,32)
        pa[i] = (np.arccos(cos_a) / np.pi).astype(np.float32)

    pa = np.repeat(pa[:, None, :, :], N_ENERGY, axis=1)      # (n_time,32,16,32)
    lb = np.clip((np.log10(np.maximum(bmag, 1e-3)) - _LOGB_MIN) / (_LOGB_MAX - _LOGB_MIN), 0, 1)
    lb = lb.astype(np.float32)[:, None, None, None] * np.ones((1, N_ENERGY, N_THETA, N_PHI), np.float32)
    return pa, lb
