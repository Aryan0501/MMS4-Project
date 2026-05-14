"""3-D convolutional autoencoder with skip connections for distribution-function inpainting.

Why this architecture
---------------------
The task is volumetric inpainting: given a distribution cube ``f(E, theta, phi)`` with a
contiguous region of (theta, phi) bins blanked out by spacecraft occlusion, recover the
missing values. The single most important architectural feature for inpainting is the
**skip connection**: it lets the decoder copy high-resolution structure straight from the
visible part of the *same* cube, instead of forcing everything through a lossy bottleneck.
The earlier project models were plain encoder->bottleneck->decoder stacks with no skips,
which is exactly the wrong shape for this problem.

The cube is small (32 x 16 x 32), so a shallow U-Net (two down/up levels) is plenty and
keeps the parameter count low enough to train on CPU. Time is *not* a model dimension here:
each cube is reconstructed independently. (A temporal variant — stacking neighbouring
timesteps as extra input channels — is a documented next step; it does not change the
incremental-training machinery.)

Output activation is sigmoid so predictions stay in [0, 1] == valid model space, which
maps back to non-negative PSD by construction.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model

# native cube layout (energy, theta, phi); channel axis added as last dim
INPUT_SHAPE = (32, 16, 32, 1)
N_ENERGY = 32   # length of the physics-head energy-spectrum target


# --------------------------------------------------------------------------- #
#  3-D U-Net (the workhorse) — also serves the "temporal channels" variant     #
# --------------------------------------------------------------------------- #
def _conv_block(x, filters, name):
    x = layers.Conv3D(filters, 3, padding="same", name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_a1")(x)
    x = layers.Conv3D(filters, 3, padding="same", name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_a2")(x)
    return x


def build_unet3d(base_filters: int = 16, input_shape=INPUT_SHAPE,
                 physics_head: bool = False) -> Model:
    """Shallow 3-D U-Net for volumetric inpainting.  ~0.13-0.2 M params.

    ``input_shape`` may carry more than one channel — that is exactly the
    "temporal channels" variant: pass ``(32,16,32, 2w+1+extra)`` where the first
    ``2w+1`` channels are the masked cube at timesteps t-w..t+w and the rest are
    feature maps (pitch angle, log|B|, ...). No architectural change is needed.

    Down-sampling is applied only on the energy and phi axes (both size 32), not
    theta (size 16), so feature maps stay sensibly shaped.

    If ``physics_head`` is True the model returns two outputs:
        - ``dist_out``    : the reconstructed cube (32,16,32,1), sigmoid
        - ``spectrum_out``: a 32-vector — the per-energy integrated signal
          (sum over theta, phi), a density-spectrum proxy. Training the bottleneck
          to also predict this nudges the latent toward physically-meaningful
          structure and yields a cheap "what does the spectrum/field look like"
          read-out. The target is computed directly from the ground-truth cube.
    """
    pool = (2, 1, 2)   # (energy, theta, phi)

    inp = layers.Input(shape=input_shape, name="dist_in")

    e1 = _conv_block(inp, base_filters, "enc1")            # (32,16,32, F)
    p1 = layers.MaxPool3D(pool, name="pool1")(e1)          # (16,16,16, F)
    e2 = _conv_block(p1, base_filters * 2, "enc2")         # (16,16,16, 2F)
    p2 = layers.MaxPool3D(pool, name="pool2")(e2)          # (8,16,8, 2F)

    b = _conv_block(p2, base_filters * 4, "bottleneck")    # (8,16,8, 4F)

    u2 = layers.UpSampling3D(pool, name="up2")(b)          # (16,16,16, 4F)
    u2 = layers.Concatenate(name="skip2")([u2, e2])
    d2 = _conv_block(u2, base_filters * 2, "dec2")         # (16,16,16, 2F)

    u1 = layers.UpSampling3D(pool, name="up1")(d2)         # (32,16,32, 2F)
    u1 = layers.Concatenate(name="skip1")([u1, e1])
    d1 = _conv_block(u1, base_filters, "dec1")             # (32,16,32, F)

    dist_out = layers.Conv3D(1, 1, padding="same", activation="sigmoid", name="dist_out")(d1)

    if not physics_head:
        return Model(inp, dist_out, name="unet3d_inpaint")

    g = layers.GlobalAveragePooling3D(name="phys_gap")(b)
    g = layers.Dense(128, activation="relu", name="phys_d1")(g)
    spectrum_out = layers.Dense(N_ENERGY, activation="sigmoid", name="spectrum_out")(g)
    return Model(inp, [dist_out, spectrum_out], name="unet3d_inpaint_physics")


# --------------------------------------------------------------------------- #
#  ConvLSTM "cube through time" — the rigorous form of the original idea        #
# --------------------------------------------------------------------------- #
def build_convlstm(base_filters: int = 16, n_time: int = 3, n_energy: int = 32,
                   n_theta: int = 16, n_phi: int = 32) -> Model:
    """ConvLSTM over a short sequence of distribution cubes.

    Keras ships ``ConvLSTM2D`` but not ``ConvLSTM3D``, so the cube is handled as a
    2-D angular image (theta x phi) with the 32 energy bins folded into the channel
    axis — i.e. convolution is shared across look-direction (where translation
    structure exists) but not across energy (where it doesn't). This keeps spatial
    structure *inside* the recurrence, which is the whole point of "a cube moving
    through time" and the thing the original TimeDistributed-CNN + plain-LSTM model
    failed to do (it flattened the cube before the LSTM).

    Input  : (n_time, n_theta, n_phi, n_energy)  — masked cubes, energy as channels
    Output : (n_energy, n_theta, n_phi, 1)       — reconstructed cube for the LAST frame
    """
    inp = layers.Input(shape=(n_time, n_theta, n_phi, n_energy), name="seq_in")

    x = layers.ConvLSTM2D(base_filters * 2, 3, padding="same", return_sequences=True,
                          name="clstm1")(inp)
    x = layers.BatchNormalization(name="clstm1_bn")(x)
    x = layers.ConvLSTM2D(base_filters * 2, 3, padding="same", return_sequences=False,
                          name="clstm2")(x)                # (theta, phi, 2F)
    x = layers.BatchNormalization(name="clstm2_bn")(x)
    x = layers.Conv2D(base_filters * 2, 3, padding="same", activation="relu", name="dec_c1")(x)
    x = layers.Conv2D(n_energy, 1, padding="same", activation="sigmoid", name="dec_out")(x)  # (theta,phi,E)
    # -> (E, theta, phi, 1)
    x = layers.Permute((3, 1, 2), name="to_etp")(x)
    out = layers.Reshape((n_energy, n_theta, n_phi, 1), name="dist_out")(x)
    return Model(inp, out, name="convlstm_inpaint")


# --------------------------------------------------------------------------- #
#  JEPA-lite "predict in latent space" model                                   #
# --------------------------------------------------------------------------- #
EMBED_DIM = 64


def _cube_encoder(base_filters: int, in_channels: int, name: str) -> Model:
    """Small 3-D conv encoder: cube -> EMBED_DIM vector (this is the 'embedding')."""
    inp = layers.Input(shape=(32, 16, 32, in_channels), name=f"{name}_in")
    x = _conv_block(inp, base_filters, f"{name}_b1")
    x = layers.MaxPool3D((2, 1, 2), name=f"{name}_p1")(x)
    x = _conv_block(x, base_filters * 2, f"{name}_b2")
    x = layers.MaxPool3D((2, 1, 2), name=f"{name}_p2")(x)
    x = _conv_block(x, base_filters * 4, f"{name}_b3")
    x = layers.GlobalAveragePooling3D(name=f"{name}_gap")(x)
    x = layers.Dense(EMBED_DIM, name=f"{name}_embed")(x)
    return Model(inp, x, name=name)


def build_jepa(base_filters: int = 16, in_channels: int = 1) -> tuple[Model, Model]:
    """I-JEPA-lite: predict the embedding of the FULL cube from the MASKED cube.

    Inspired by Joint-Embedding Predictive Architectures (LeCun et al.): instead of
    reconstructing the missing PSD bins pixel-by-pixel, learn a *representation* of
    the distribution such that the masked-cube embedding predicts the full-cube
    embedding. No decoder — cheaper to train; and the embedding becomes a compact
    "fingerprint" of the plasma state that can be projected to 2-D to reveal regime
    structure / outliers (a representation-learning route to "detect changes in B").

    Architecture (single shared encoder, no EMA target — a stop-gradient on the
    target branch is the simplest collapse-resistant variant):
        z_visible = encoder(masked_cube)
        z_pred    = predictor_MLP(z_visible)
        z_target  = stop_gradient(encoder(full_cube))
        loss      = MSE(z_pred, z_target) + variance_regulariser(z_pred)   # see losses.py
    The model returns concat([z_pred, z_target], axis=-1); the loss splits it.

    Returns (training_model, encoder) — `encoder` is reused at eval time to dump
    embeddings of any cube set.
    """
    enc = _cube_encoder(base_filters, in_channels, "jepa_encoder")

    masked_in = layers.Input(shape=(32, 16, 32, in_channels), name="masked_in")
    full_in = layers.Input(shape=(32, 16, 32, in_channels), name="full_in")

    z_vis = enc(masked_in)
    z_pred = layers.Dense(EMBED_DIM, activation="relu", name="pred_h")(z_vis)
    z_pred = layers.Dense(EMBED_DIM, name="z_pred")(z_pred)
    z_tgt = layers.Lambda(lambda t: tf.stop_gradient(t), name="z_target")(enc(full_in))

    out = layers.Concatenate(axis=-1, name="jepa_out")([z_pred, z_tgt])
    train_model = Model([masked_in, full_in], out, name="jepa_lite")
    return train_model, enc


# --------------------------------------------------------------------------- #
#  factory                                                                     #
# --------------------------------------------------------------------------- #
def build_model(name: str, base_filters: int = 16, in_channels: int = 1,
                temporal_window: int = 0, physics_head: bool = False):
    """Dispatch by name.

    name == "unet"     : 3-D U-Net, input (32,16,32, in_channels). With
                         temporal_window>0 the caller stacks 2*tw+1 dist channels
                         into in_channels. Returns a single Model.
    name == "convlstm" : ConvLSTM over (2*tw+1) frames. Returns a single Model.
    name == "jepa"     : I-JEPA-lite. Returns (train_model, encoder).
    """
    if name == "unet":
        return build_unet3d(base_filters, input_shape=(32, 16, 32, in_channels),
                            physics_head=physics_head)
    if name == "unet_residual":
        return build_residual_temporal_unet(base_filters, in_channels=in_channels,
                                             temporal_window=temporal_window)
    if name == "unet_residual_attn":
        return build_residual_temporal_energy_attn_unet(base_filters, in_channels=in_channels,
                                                         temporal_window=temporal_window)
    if name == "unet_energy_attn":
        return build_energy_attention_unet(base_filters, in_channels=in_channels)
    if name == "convlstm":
        return build_convlstm(base_filters, n_time=max(3, 2 * temporal_window + 1))
    if name == "jepa":
        return build_jepa(base_filters, in_channels)
    if name == "ijepa":
        return build_ijepa(base_filters, in_channels)
    raise ValueError(f"unknown model {name!r}")


def bottleneck_encoder(unet_model: Model) -> Model:
    """Wrap a trained U-Net so it outputs the GAP of its bottleneck (its 'embedding')."""
    b = unet_model.get_layer("bottleneck_a2").output
    emb = layers.GlobalAveragePooling3D(name="embed")(b)
    return Model(unet_model.input, emb, name="unet_embedder")


# --------------------------------------------------------------------------- #
#  I-JEPA (proper) — EMA target encoder + variance + covariance regularisation #
# --------------------------------------------------------------------------- #
def build_ijepa(base_filters: int = 16, in_channels: int = 1):
    """Image-style JEPA, fixed up: separate EMA target encoder + invariance + variance +
    covariance regularisation à la VICReg.

    Architecture:
        online_encoder      learnable; produces z_pred from masked cube
        target_encoder      EMA copy of online_encoder; produces z_tgt from full cube
        predictor_MLP       maps z_pred -> z_pred (learned, small)
        loss = invariance(z_pred, sg(z_tgt))               # MSE in latent space
              + var_weight * variance_hinge(z_pred)         # std >= 1 per dim
              + cov_weight * cov_off_diag(z_pred)           # decorrelate dims
        target_encoder weights <- 0.99 * target + 0.01 * online  every step

    Returns (training_model, online_encoder, target_encoder, ema_update_fn).
    The training script calls ema_update_fn() after every optimiser step.
    """
    online = _cube_encoder(base_filters, in_channels, "online_enc")
    target = _cube_encoder(base_filters, in_channels, "target_enc")
    target.set_weights(online.get_weights())
    target.trainable = False    # never updated by gradient

    masked_in = layers.Input(shape=(32, 16, 32, in_channels), name="masked_in")
    full_in   = layers.Input(shape=(32, 16, 32, in_channels), name="full_in")

    z_vis = online(masked_in)
    z_pred = layers.Dense(EMBED_DIM, activation="relu", name="pred_h1")(z_vis)
    z_pred = layers.Dense(EMBED_DIM, name="z_pred")(z_pred)
    z_tgt  = layers.Lambda(lambda t: tf.stop_gradient(t), name="z_target")(target(full_in))

    out = layers.Concatenate(axis=-1, name="ijepa_out")([z_pred, z_tgt])
    train_model = Model([masked_in, full_in], out, name="ijepa_proper")

    EMA = 0.99
    def ema_update():
        for w_t, w_o in zip(target.weights, online.weights):
            w_t.assign(EMA * w_t + (1.0 - EMA) * w_o)
    return train_model, online, target, ema_update


# --------------------------------------------------------------------------- #
#  Energy-axis attention U-Net                                                  #
# --------------------------------------------------------------------------- #
def build_residual_temporal_unet(base_filters: int = 16, in_channels: int = 5,
                                  temporal_window: int = 2) -> Model:
    """Residual U-Net with explicit temporal context.

    The temporal-variance diagnostic in `outputs/stream_week_mar2024/temporal_variance.png`
    showed that the plain U-Net's predictions in the masked region are ~3-4× flatter in
    time than truth — the model's outputs barely vary across the 120-s burst even when
    the truth changes a lot. Two design changes here address that:

    1. **Residual connection from the centre frame to the output.** Instead of mapping
       (visible cube) → (full cube) from scratch, the model maps to a small *correction*
       added on top of the centre-frame's masked input. Whatever frame-to-frame variation
       is in the input is preserved by construction; the U-Net only has to learn the
       inpainting delta.

    2. **Wider temporal window** (default ``temporal_window=2`` → 5-frame stack of t-2..t+2,
       caller stacks them as input channels). Gives the model more frame-to-frame context
       so it has more signal to vary the prediction with.

    Input shape ``(32, 16, 32, in_channels)`` — caller is expected to stack
    ``2*temporal_window+1`` masked-dist channels plus any feature channels (e.g.
    pitch_angle + log|B|), so for ``temporal_window=2`` and pa+lb features
    ``in_channels = 5 + 2 = 7``. The CENTRE temporal channel is taken to be index
    ``temporal_window`` (i.e. channel 2 for window=2).
    """
    pool = (2, 1, 2)
    inp = layers.Input(shape=(32, 16, 32, in_channels), name="dist_in")
    e1 = _conv_block(inp, base_filters, "enc1")
    p1 = layers.MaxPool3D(pool, name="pool1")(e1)
    e2 = _conv_block(p1, base_filters * 2, "enc2")
    p2 = layers.MaxPool3D(pool, name="pool2")(e2)
    b = _conv_block(p2, base_filters * 4, "bottleneck")
    u2 = layers.UpSampling3D(pool, name="up2")(b)
    u2 = layers.Concatenate(name="skip2")([u2, e2])
    d2 = _conv_block(u2, base_filters * 2, "dec2")
    u1 = layers.UpSampling3D(pool, name="up1")(d2)
    u1 = layers.Concatenate(name="skip1")([u1, e1])
    d1 = _conv_block(u1, base_filters, "dec1")
    correction = layers.Conv3D(1, 1, padding="same", activation="tanh", name="correction")(d1)
    # residual: prediction = sigmoid(centre_input + small_correction).
    # Slice the centre temporal channel so this works whether window=1, 2, or 3.
    centre_idx = temporal_window
    centre = layers.Lambda(lambda t: t[..., centre_idx:centre_idx + 1], name="centre_in")(inp)
    summed = layers.Add(name="add_residual")([centre, correction])
    out = layers.Activation("sigmoid", name="dist_out")(summed)
    return Model(inp, out, name="unet3d_residual_temporal")


def build_residual_temporal_energy_attn_unet(
        base_filters: int = 16, in_channels: int = 7, temporal_window: int = 2) -> Model:
    """v9 architecture: residual-temporal U-Net + 4-head self-attention over the
    energy axis at the bottleneck.

    Combines two earlier ideas that should be additive:
      - residual head from the centre temporal frame (the v7 fix for temporal flatness)
      - energy-axis self-attention at the bottleneck (long-range coupling between
        adjacent energy bins that 3-D conv locality misses; an electron beam at one
        energy correlates with adjacent energies, etc.)
    """
    pool = (2, 1, 2)
    inp = layers.Input(shape=(32, 16, 32, in_channels), name="dist_in")
    e1 = _conv_block(inp, base_filters, "enc1")
    p1 = layers.MaxPool3D(pool, name="pool1")(e1)
    e2 = _conv_block(p1, base_filters * 2, "enc2")
    p2 = layers.MaxPool3D(pool, name="pool2")(e2)
    b = _conv_block(p2, base_filters * 4, "bottleneck")
    # energy-axis attention block at the bottleneck (same form as build_energy_attention_unet)
    shape = tf.keras.backend.int_shape(b)
    seq = layers.Reshape((shape[1], shape[2] * shape[3] * shape[4]), name="b_to_seq")(b)
    seq = layers.LayerNormalization(name="b_ln")(seq)
    seq = layers.MultiHeadAttention(num_heads=4, key_dim=max(8, base_filters), name="b_attn")(seq, seq)
    b_attn = layers.Reshape(shape[1:], name="seq_to_b")(seq)
    b = layers.Add(name="b_add")([b, b_attn])
    u2 = layers.UpSampling3D(pool, name="up2")(b)
    u2 = layers.Concatenate(name="skip2")([u2, e2])
    d2 = _conv_block(u2, base_filters * 2, "dec2")
    u1 = layers.UpSampling3D(pool, name="up1")(d2)
    u1 = layers.Concatenate(name="skip1")([u1, e1])
    d1 = _conv_block(u1, base_filters, "dec1")
    correction = layers.Conv3D(1, 1, padding="same", activation="tanh", name="correction")(d1)
    centre = layers.Lambda(lambda t: t[..., temporal_window:temporal_window + 1], name="centre_in")(inp)
    out = layers.Activation("sigmoid", name="dist_out")(layers.Add(name="add_residual")([centre, correction]))
    return Model(inp, out, name="unet3d_v9_residual_energy_attn")


def build_energy_attention_unet(base_filters: int = 16, in_channels: int = 1) -> Model:
    """3-D U-Net with a self-attention block over the ENERGY axis at the bottleneck.

    Plasma-physical motivation: the 32 energy bins carry a smooth spectrum (low-E peak,
    mid-E body, high-E tail), and the relationship between bins matters. Pure 3-D conv
    locality misses long-range energy structure (e.g. an electron beam at one energy
    can correlate with deviations at the adjacent energies). One light multi-head
    self-attention layer over energy at the bottleneck adds that long-range coupling
    without blowing up parameters.

    Same 32x16x32xC input / 32x16x32x1 sigmoid output as build_unet3d, drop-in
    replacement.
    """
    pool = (2, 1, 2)
    inp = layers.Input(shape=(32, 16, 32, in_channels), name="dist_in")
    e1 = _conv_block(inp, base_filters, "enc1")
    p1 = layers.MaxPool3D(pool, name="pool1")(e1)
    e2 = _conv_block(p1, base_filters * 2, "enc2")
    p2 = layers.MaxPool3D(pool, name="pool2")(e2)
    b = _conv_block(p2, base_filters * 4, "bottleneck")          # (8, 16, 8, 4F)

    # energy-axis attention: collapse (theta, phi) -> tokens, attend across the 8 energies
    shape = tf.keras.backend.int_shape(b)                         # (None, 8, 16, 8, 4F)
    seq = layers.Reshape((shape[1], shape[2] * shape[3] * shape[4]), name="b_to_seq")(b)  # (8, 16*8*4F)
    seq = layers.LayerNormalization(name="b_ln")(seq)
    seq = layers.MultiHeadAttention(num_heads=4, key_dim=max(8, base_filters), name="b_attn")(seq, seq)
    b_attn = layers.Reshape(shape[1:], name="seq_to_b")(seq)
    b = layers.Add(name="b_add")([b, b_attn])                     # residual

    u2 = layers.UpSampling3D(pool, name="up2")(b)
    u2 = layers.Concatenate(name="skip2")([u2, e2])
    d2 = _conv_block(u2, base_filters * 2, "dec2")
    u1 = layers.UpSampling3D(pool, name="up1")(d2)
    u1 = layers.Concatenate(name="skip1")([u1, e1])
    d1 = _conv_block(u1, base_filters, "dec1")
    out = layers.Conv3D(1, 1, padding="same", activation="sigmoid", name="dist_out")(d1)
    return Model(inp, out, name="unet3d_energy_attn")
