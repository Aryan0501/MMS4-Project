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

import tensorflow as tf
from tensorflow.keras import backend as K


def make_masked_loss(mask, masked_weight: float = 10.0, signal_weight: float = 1e-3):
    """Build a Keras loss closure for a fixed occlusion ``mask``.

    Parameters
    ----------
    mask : bool array (32, 16, 32) — True where occluded.
    masked_weight : relative weight of occluded-bin MSE vs visible-bin MSE.
    signal_weight : weight of the integrated-signal consistency term.
    """
    # (1, 32, 16, 32, 1) so it broadcasts over (batch, E, theta, phi, channel)
    m = tf.constant(mask.astype("float32")[None, ..., None])
    w = 1.0 + (masked_weight - 1.0) * m   # visible bins weight 1, masked bins weight `masked_weight`

    def loss(y_true, y_pred):
        sq = K.square(y_true - y_pred)
        recon = K.sum(w * sq, axis=[1, 2, 3, 4]) / K.sum(w * tf.ones_like(sq), axis=[1, 2, 3, 4])
        sig_true = K.sum(y_true, axis=[1, 2, 3, 4])
        sig_pred = K.sum(y_pred, axis=[1, 2, 3, 4])
        sig = K.square(sig_true - sig_pred) / (K.square(sig_true) + 1e-6)
        return K.mean(recon + signal_weight * sig)

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


def make_masked_metrics(mask):
    """Return [masked_mse, masked_mae] metrics scoped to the occluded region only.

    These are *per-occluded-bin* errors, averaged over the batch — i.e. directly
    comparable across runs and against the interpolation baseline.
    """
    m = tf.constant(mask.astype("float32")[None, ..., None])
    n_masked = tf.reduce_sum(m)   # scalar: number of occluded bins per cube

    def masked_mse(y_true, y_pred):
        per_sample = K.sum(m * K.square(y_true - y_pred), axis=[1, 2, 3, 4]) / n_masked
        return K.mean(per_sample)

    def masked_mae(y_true, y_pred):
        per_sample = K.sum(m * K.abs(y_true - y_pred), axis=[1, 2, 3, 4]) / n_masked
        return K.mean(per_sample)

    return [masked_mse, masked_mae]
