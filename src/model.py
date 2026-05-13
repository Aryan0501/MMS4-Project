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
    if name == "convlstm":
        return build_convlstm(base_filters, n_time=max(3, 2 * temporal_window + 1))
    if name == "jepa":
        return build_jepa(base_filters, in_channels)
    raise ValueError(f"unknown model {name!r}")


def bottleneck_encoder(unet_model: Model) -> Model:
    """Wrap a trained U-Net so it outputs the GAP of its bottleneck (its 'embedding')."""
    b = unet_model.get_layer("bottleneck_a2").output
    emb = layers.GlobalAveragePooling3D(name="embed")(b)
    return Model(unet_model.input, emb, name="unet_embedder")
