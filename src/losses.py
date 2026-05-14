"""Loss functions for distribution-function inpainting.

The headline metric is reconstruction error *inside the occluded region* — the model
gets the visible bins for free as input, so scoring it on those would flatter it. We
therefore weight the masked bins much more heavily than the visible ones.

A small "total-signal" penalty keeps the integrated cube (a crude density proxy)
consistent between prediction and truth. Unlike the earlier project code, this proxy is
computed *per batch from the current ground truth* — never from a stale, hard-coded
training-set tensor.

All losses operate in normalised model space ([0, 1]); convert with
``data_pipeline.to_physical_space`` before computing real plasma moments.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def make_masked_loss(mask, masked_weight: float = 10.0, signal_weight: float = 1e-3,
                     gt_only: np.ndarray | None = None):
    """Build a Keras loss closure for a fixed occlusion ``mask``.

    Parameters
    ----------
    mask : bool array (32, 16, 32) — True where occluded.
    masked_weight : relative weight of occluded-bin MSE vs visible-bin MSE.
    signal_weight : weight of the integrated-signal consistency term.
    gt_only : optional bool array (32, 16, 32) — True for bins to KEEP in the loss.
        If provided, bins where ``gt_only`` is False get **zero gradient**. Use this with
        the always-zero mask (``always_zero`` from data_richness.npz) so the model is never
        penalised for putting non-zero predictions in bins that are zero in the raw data
        only because they're unmeasured -- those zeros are not ground truth.
    """
    # (1, 32, 16, 32, 1) so it broadcasts over (batch, E, theta, phi, channel)
    m = tf.constant(mask.astype("float32")[None, ..., None])
    w = 1.0 + (masked_weight - 1.0) * m   # visible bins weight 1, masked bins weight `masked_weight`
    if gt_only is not None:
        # zero out the weight for bins we shouldn't trust as ground truth
        v = tf.constant(gt_only.astype("float32")[None, ..., None])
        w = w * v

    def loss(y_true, y_pred):
        sq = K.square(y_true - y_pred)
        denom = K.sum(w * tf.ones_like(sq), axis=[1, 2, 3, 4])
        recon = K.sum(w * sq, axis=[1, 2, 3, 4]) / (denom + 1e-9)
        sig_true = K.sum(y_true, axis=[1, 2, 3, 4])
        sig_pred = K.sum(y_pred, axis=[1, 2, 3, 4])
        sig = K.square(sig_true - sig_pred) / (K.square(sig_true) + 1e-6)
        return K.mean(recon + signal_weight * sig)

    return loss


def make_moments_aware_loss(mask, gt_only: np.ndarray | None = None,
                             moments_weight: float = 0.05):
    """Uniform MSE on data-rich bins + a density-consistency penalty.

    The reconstruction MSE alone doesn't directly enforce that the integrated
    density (sum over angles, per energy) of the prediction matches the true
    density. Adding ``|sum(pred * data_rich) - sum(truth * data_rich)|^2``
    nudges the model to preserve the moment that actually matters physically.

    The data-rich constraint matters: integrating over always-zero bins would
    pull the prediction toward zero in those bins, which we explicitly want to
    avoid (that's the same cheat from RESULTS §8i).
    """
    v = (None if gt_only is None
         else tf.constant(gt_only.astype("float32")[None, ..., None]))

    def loss(y_true, y_pred):
        diff_sq = K.square(y_true - y_pred)
        if v is None:
            recon = K.mean(diff_sq)
        else:
            recon = K.sum(v * diff_sq, axis=[1, 2, 3, 4]) / (K.sum(v) + 1e-9)
            recon = K.mean(recon)
        # density per (sample, energy) from the data-rich bins only
        if v is None:
            sum_t = K.sum(y_true, axis=[2, 3, 4])
            sum_p = K.sum(y_pred, axis=[2, 3, 4])
        else:
            sum_t = K.sum(v * y_true, axis=[2, 3, 4])
            sum_p = K.sum(v * y_pred, axis=[2, 3, 4])
        moments = K.mean(K.square(sum_t - sum_p) / (K.square(sum_t) + 1e-6))
        return recon + moments_weight * moments

    return loss


def make_uniform_loss(gt_only: np.ndarray | None = None):
    """Plain MSE over the cube, optionally excluding always-zero bins.

    Use this with --random-mask (where the mask varies per file so a mask-aware
    weighted loss can't be precompiled) combined with --gt-only-loss (where we
    want zero gradient on bins whose "ground truth" of 0 is just unmeasured).
    """
    if gt_only is None:
        return "mse"
    v = tf.constant(gt_only.astype("float32")[None, ..., None])
    n_eff = tf.maximum(tf.reduce_sum(v) * tf.cast(1, tf.float32), 1.0)

    def loss(y_true, y_pred):
        sq = K.square(y_true - y_pred) * v
        per_sample = K.sum(sq, axis=[1, 2, 3, 4]) / n_eff
        return K.mean(per_sample)

    return loss


def jepa_loss(var_weight: float = 1.0, eps: float = 1e-4):
    """Loss for the I-JEPA-lite model.

    The model output is ``concat([z_pred, z_target], axis=-1)`` (each EMBED_DIM wide);
    ``y_true`` is ignored (pass zeros). Loss = MSE(z_pred, z_target) plus a hinge
    variance regulariser on z_pred (à la VICReg) that pushes each embedding dimension
    to keep std ≥ 1, which is what stops the whole thing collapsing to a constant.
    The target branch already has stop-gradient applied inside the model.
    """
    def loss(y_true, y_pred):
        d = tf.shape(y_pred)[-1] // 2
        z_pred = y_pred[:, :d]
        z_tgt = y_pred[:, d:]
        pred_err = tf.reduce_mean(tf.square(z_pred - z_tgt))
        std = tf.sqrt(tf.math.reduce_variance(z_pred, axis=0) + eps)
        var_term = tf.reduce_mean(tf.nn.relu(1.0 - std))
        return pred_err + var_weight * var_term
    return loss


def energy_spectrum_target(y_true_cube):
    """Per-energy integrated signal of a ground-truth cube batch, normalised to ~[0,1].

    y_true_cube : (batch, 32, 16, 32, 1) in model space.
    returns     : (batch, 32) — sum over (theta, phi) at each energy, divided by
                  (n_theta * n_phi) so the result stays in [0, 1].
    Used as the target for the U-Net physics auxiliary head.
    """
    import numpy as np
    arr = np.asarray(y_true_cube)
    return (arr[..., 0].sum(axis=(2, 3)) / (arr.shape[2] * arr.shape[3])).astype("float32")


def make_masked_metrics(mask, gt_only: np.ndarray | None = None):
    """Return [masked_mse, masked_mae] metrics scoped to the occluded region only.

    These are *per-occluded-bin* errors, averaged over the batch — i.e. directly
    comparable across runs and against the interpolation baseline. With ``gt_only``,
    error is only counted on bins where ground truth is trusted (i.e. bins that
    are NOT always-zero in the raw data).
    """
    full = mask.astype("float32")
    if gt_only is not None:
        full = full * gt_only.astype("float32")
    m = tf.constant(full[None, ..., None])
    n_eff = tf.maximum(tf.reduce_sum(m), 1.0)

    def masked_mse(y_true, y_pred):
        per_sample = K.sum(m * K.square(y_true - y_pred), axis=[1, 2, 3, 4]) / n_eff
        return K.mean(per_sample)

    def masked_mae(y_true, y_pred):
        per_sample = K.sum(m * K.abs(y_true - y_pred), axis=[1, 2, 3, 4]) / n_eff
        return K.mean(per_sample)

    return [masked_mse, masked_mae]
