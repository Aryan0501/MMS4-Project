"""Incremental ("streaming") training for FPI distribution-function modelling.

Why incremental
---------------
The MMS/FPI archive is far too large to hold on disk, let alone in RAM (~100 Gbit/day
collected across the constellation; the public archive is petabyte-scale). The only
workable pattern is:

    for each file in the stream:
        load it  ->  train a few epochs on it  ->  free it  ->  next file

i.e. continual / online learning with the model carried (warm-started) from one file
to the next. This script does exactly that and records **training loss and validation
loss after every file**, so the learning curve can be inspected for divergence or
catastrophic forgetting. A small replay buffer (random samples retained from each file
seen so far, mixed into the next file) is the cheap standard defence against forgetting.

Model zoo (``--model``)
-----------------------
- ``unet``     : 3-D U-Net with skip connections (the workhorse for volumetric inpainting).
                 ``--temporal-window W`` adds the masked cube at t-W..t+W as extra input
                 channels ("cube through time", feed-forward form). ``--features
                 pitch_angle,logb`` adds a pitch-angle map + log|B| (from the matched FGM
                 file). ``--physics-head`` adds a second output predicting the per-energy
                 spectrum (regularises the latent; a step toward "detect changes in B").
- ``convlstm`` : ConvLSTM over a short sequence of cubes — the rigorous form of the
                 original "cube evolving through time" idea (convolution *inside* the
                 recurrence, unlike the original TimeDistributed-CNN + plain-LSTM model).
- ``jepa``     : I-JEPA-lite — predict the embedding of the full cube from the masked
                 cube's embedding (loss in latent space, no decoder). The learned
                 embedding is a compact fingerprint of the plasma state.

``--dump-embeddings`` (``unet``/``jepa`` only): after training, run the encoder on the
validation cubes and save ``embeddings.npy`` plus a 2-D PCA scatter — the "what does the
cube look like in latent space" view.

What this script does NOT do
----------------------------
It never deletes your CDF files. "Deleting old data" here means freeing it from memory
between files. A production run against the live archive would add one ``os.remove(path)``
line; it is omitted on purpose so an unattended run can't destroy local data.

Resuming after a crash / machine sleep
--------------------------------------
The trainer saves model + history + state + replay buffer after EVERY file. To pick
up where a previous run stopped (machine slept, OOM, kill -9, whatever), re-run the
SAME command with ``--resume`` appended:

    python src/train_incremental.py ... --out outputs/unet_full --resume

It loads the weights, restores the replay buffer, and skips every (round, file) pair
already in ``history.json``. At most one file's worth of work is lost.

Examples
--------
    python src/train_incremental.py --model unet --rounds 2 --out outputs/unet_plain
    python src/train_incremental.py --model unet --temporal-window 1 \
        --features pitch_angle,logb --dump-embeddings --out outputs/unet_full
    python src/train_incremental.py --model jepa --dump-embeddings --out outputs/jepa
    python src/train_incremental.py --model convlstm --temporal-window 1 --out outputs/clstm
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np
import tensorflow as tf

import data_pipeline as dp
from model import build_model, bottleneck_encoder
from losses import make_masked_loss, make_masked_metrics, energy_spectrum_target, jepa_loss
from baseline import evaluate_baselines


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="MMS-FPI-Data-Gaps")
    p.add_argument("--out", default="outputs/run1")
    p.add_argument("--model", choices=["unet", "convlstm", "jepa"], default="unet")
    p.add_argument("--mask", choices=["wedge", "explosion"], default="wedge")
    p.add_argument("--random-mask", action="store_true",
                   help="sample a new random wedge position for every training file "
                        "(validation set keeps the fixed mask for comparability). Forces a "
                        "uniform MSE loss instead of the mask-aware weighted loss, since the "
                        "mask varies per file. This is the fix for the data-aware-mask "
                        "failure mode documented in RESULTS §8i.")
    p.add_argument("--gt-only-loss", action="store_true",
                   help="exclude always-zero bins (from outputs/data_richness.npz) from the "
                        "training loss and metrics. The raw zeros in the data are NOT ground "
                        "truth (they mean 'no measurement'), so the model should not be "
                        "penalised for putting non-zero predictions there. Whatever it learns "
                        "to predict in those bins is its data-rich-learned prior extrapolated.")
    p.add_argument("--data-aware-mask", action="store_true",
                   help="restrict the synthetic occlusion mask to bins that are always-data "
                        "(P(signal>floor)>=0.95 in outputs/data_richness.npz). The model only "
                        "ever has to inpaint bins where we have ground truth -- and we trust "
                        "it on the truly-missing bins for free, because they share the same "
                        "input statistics. Pairs naturally with --gt-only-loss.")
    p.add_argument("--temporal-window", type=int, default=0)
    p.add_argument("--features", default="", help="comma list from {pitch_angle, logb}; unet only")
    p.add_argument("--physics-head", action="store_true", help="unet only: also predict the per-energy spectrum")
    p.add_argument("--dump-embeddings", action="store_true", help="unet/jepa: save validation-set embeddings + PCA plot")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--inner-epochs", type=int, default=1)
    p.add_argument("--subsample", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--base-filters", type=int, default=12)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--val-cap", type=int, default=800)
    p.add_argument("--replay-per-file", type=int, default=24)
    p.add_argument("--replay-cap", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="if --out exists with a checkpoint, load model + state + replay buffer "
                        "and continue from where the last run stopped (handles machine sleep/crash)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Resume-from-checkpoint helpers                                              #
#  The trainer saves after every file. On --resume it reads:                   #
#    model_latest.h5  -> trained weights so far                                #
#    history.json     -> per-file metrics (used to derive completed (rnd,file))#
#    state.json       -> {round, completed: [[rnd, basename], ...]}            #
#    replay.npz       -> replay buffer arrays (flattened: X_i_j, Y_i_j)        #
#  so the machine can sleep / crash / be killed at any point without losing    #
#  more than the file currently being trained on.                              #
# --------------------------------------------------------------------------- #
def _save_state(out_dir, history, completed_pairs, replay_X, replay_Y):
    import io
    with open(os.path.join(out_dir, "state.json"), "w") as fh:
        json.dump({"completed": list(completed_pairs)}, fh)
    # serialise replay buffer (each entry is either an ndarray or a list of ndarrays)
    data = {}
    for i, (x, y) in enumerate(zip(replay_X, replay_Y)):
        xs = x if isinstance(x, list) else [x]
        ys = y if isinstance(y, list) else [y]
        data[f"meta_{i}"] = np.array([len(xs), len(ys)], dtype=np.int32)
        for j, a in enumerate(xs):
            data[f"X_{i}_{j}"] = a
        for j, a in enumerate(ys):
            data[f"Y_{i}_{j}"] = a
    np.savez(os.path.join(out_dir, "replay.npz"), **data)


def _load_state(out_dir):
    """Return (completed_pairs, replay_X, replay_Y, history) or None if no state saved.

    state.json is optional: if only history.json exists (a run from before the resume
    feature was added), completed-pair set is derived from the history. The replay
    buffer is empty in that case, which just makes the resumed run a bit more
    forget-prone at first — not fatal.
    """
    hp = os.path.join(out_dir, "history.json")
    if not os.path.exists(hp):
        return None
    history = json.load(open(hp))["history"]
    sp = os.path.join(out_dir, "state.json")
    if os.path.exists(sp):
        completed = set(tuple(p) for p in json.load(open(sp))["completed"])
    else:
        completed = set((int(r["round"]), r["file"]) for r in history)
    rX, rY = [], []
    rp = os.path.join(out_dir, "replay.npz")
    if os.path.exists(rp):
        z = np.load(rp)
        i = 0
        while f"meta_{i}" in z:
            nx, ny = z[f"meta_{i}"]
            x = [z[f"X_{i}_{j}"] for j in range(nx)]
            y = [z[f"Y_{i}_{j}"] for j in range(ny)]
            rX.append(x[0] if len(x) == 1 else x)
            rY.append(y[0] if len(y) == 1 else y)
            i += 1
    return completed, rX, rY, history


def build_mask(args) -> np.ndarray:
    base = (dp.load_explosion_mask(os.path.dirname(os.path.abspath(args.data_root)) or ".")
            if args.mask == "explosion"
            else dp.synthetic_wedge_mask(theta_frac=0.5, phi_frac=0.5))
    if args.data_aware_mask:
        richness = _load_richness(args)
        # only mask bins that are confidently data-rich -- i.e. we have ground truth there
        return base & richness["always_data"]
    return base


def _load_richness(args):
    path = os.path.join(os.path.dirname(os.path.abspath(args.data_root)) or ".",
                       "outputs", "data_richness.npz")
    if not os.path.exists(path):
        raise SystemExit(f"need outputs/data_richness.npz for --data-aware-mask / --gt-only-loss; "
                         f"build it with `python _richness.py` first.  Looked at: {path}")
    return dict(np.load(path))


def random_wedge(rng, theta_span: int = 8, phi_span: int = 16,
                 n_energy: int = 32, n_theta: int = 16, n_phi: int = 32) -> np.ndarray:
    """A wedge mask of the same SIZE as the default, placed at a random (θ, φ) origin.

    Used by --random-mask to expose the model to every mask position during training
    (the fix for the data-aware-mask failure documented in RESULTS §8i). Wraps
    around both axes so the wedge is always contiguous in the toroidal sense.
    """
    t0 = int(rng.integers(0, n_theta))
    p0 = int(rng.integers(0, n_phi))
    msk = np.zeros((n_energy, n_theta, n_phi), dtype=bool)
    for t in range(theta_span):
        for p in range(phi_span):
            msk[:, (t0 + t) % n_theta, (p0 + p) % n_phi] = True
    return msk


def _feature_flags(args):
    feats = {f.strip() for f in args.features.split(",") if f.strip()}
    return ("pitch_angle" in feats), ("logb" in feats)


def assemble(fb: dp.FileBatch, mask: np.ndarray, args):
    """Return (X, Y) for one file, in the layout the chosen model expects.

    unet     : X (n,32,16,32,C),  Y (n,32,16,32,1)  [+ spectrum if --physics-head]
    convlstm : X (n,T,16,32,32),  Y (n,32,16,32,1)
    jepa     : X = [masked (n,32,16,32,1), full (n,32,16,32,1)], Y = zeros (n,1)  (dummy)
    """
    if args.model == "jepa":
        Xu, Y = dp.build_inputs(fb, mask, temporal_window=0)        # Xu = masked[...,None], Y = full[...,None]
        return [Xu, Y], np.zeros((Xu.shape[0], 1), np.float32)

    use_pa, use_lb = _feature_flags(args)
    w = args.temporal_window
    Xu, Y = dp.build_inputs(fb, mask, use_pitch_angle=use_pa, use_logb=use_lb, temporal_window=w)
    if args.model == "unet":
        return (Xu, [Y, energy_spectrum_target(Y)]) if args.physics_head else (Xu, Y)
    # convlstm
    n_time = max(3, 2 * w + 1)
    dist_chans = Xu[..., : (2 * w + 1 if w > 0 else 1)]
    if dist_chans.shape[-1] < n_time:
        pad = n_time - dist_chans.shape[-1]
        dist_chans = np.concatenate([dist_chans[..., :1]] * pad + [dist_chans], axis=-1)
    Xc = np.transpose(dist_chans, (0, 4, 2, 3, 1)).astype(np.float32)
    return Xc, Y


def _full_cube(Y):
    """Extract the (n,32,16,32) ground-truth cube from whatever `assemble` returned for Y/X."""
    if isinstance(Y, list):     # unet+physics: [cube, spectrum]
        return Y[0][..., 0]
    return Y[..., 0]


def load_validation(val_files, mask, args):
    use_pa, _ = _feature_flags(args)
    Xs, Ys, cubes = [], [], []
    for f in val_files:
        fb = dp.read_dist_file(f, subsample=args.subsample, with_phi=use_pa)
        X, Y = assemble(fb, mask, args)
        Xs.append(X); Ys.append(Y)
        cubes.append(fb.cubes)                     # raw full cubes, for baselines + embedding dump
        del fb
    cubes = np.concatenate(cubes, 0)
    if args.model == "jepa":
        X = [np.concatenate([x[0] for x in Xs], 0), np.concatenate([x[1] for x in Xs], 0)]
        Y = np.concatenate(Ys, 0)
    elif args.model == "unet" and args.physics_head:
        X = np.concatenate(Xs, 0)
        Y = [np.concatenate([y[0] for y in Ys], 0), np.concatenate([y[1] for y in Ys], 0)]
    else:
        X = np.concatenate(Xs, 0); Y = np.concatenate(Ys, 0)
    n = (X[0].shape[0] if isinstance(X, list) else X.shape[0])
    if n > args.val_cap:
        sel = np.random.default_rng(args.seed).choice(n, args.val_cap, replace=False)
        X = [a[sel] for a in X] if isinstance(X, list) else X[sel]
        Y = [a[sel] for a in Y] if isinstance(Y, list) else Y[sel]
        cubes = cubes[sel]
    return X, Y, cubes


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    tf.random.set_seed(args.seed); np.random.seed(args.seed)

    files = dp.find_dist_files(args.data_root)
    if not files:
        raise SystemExit(f"no *des-dist*.cdf files under {args.data_root!r}")
    train_files, val_files = dp.split_files(files, args.val_fraction, args.seed)
    print(f"[setup] {len(files)} files: {len(train_files)} train, {len(val_files)} val", flush=True)

    mask = build_mask(args)
    print(f"[setup] model={args.model} mask={args.mask} occluded={mask.mean()*100:.1f}% "
          f"temporal_window={args.temporal_window} features={args.features or 'none'} "
          f"physics_head={args.physics_head} dump_embeddings={args.dump_embeddings}", flush=True)

    print("[setup] loading validation set ...", flush=True)
    Xv, Yv, cubes_v = load_validation(val_files, mask, args)
    in_channels = 1
    if args.model == "unet":
        in_channels = Xv.shape[-1]
    elif args.model == "jepa":
        in_channels = Xv[0].shape[-1]
    n_val = (Xv[0].shape[0] if isinstance(Xv, list) else Xv.shape[0])
    print(f"[setup] validation: {n_val} samples; model input channels = {in_channels}", flush=True)

    built = build_model(args.model, base_filters=args.base_filters, in_channels=in_channels,
                        temporal_window=args.temporal_window, physics_head=args.physics_head)
    if args.model == "jepa":
        model, encoder = built
    else:
        model, encoder = built, None

    # if --gt-only-loss, load the always-zero mask and pass its inverse into the loss/metrics
    gt_only = None
    if args.gt_only_loss:
        richness = _load_richness(args)
        gt_only = ~richness["always_zero"]
        print(f"[setup] gt-only-loss: {gt_only.mean()*100:.1f}% of bins kept in loss "
              f"(always-zero bins excluded)", flush=True)
    if args.model == "jepa":
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=jepa_loss())
    elif args.random_mask:
        # mask varies per file -- can't use the mask-aware weighted loss; plain MSE
        # over the whole cube is what teaches the model "fill in whatever is zero".
        # Metrics still scoped to the FIXED default mask so the curve is comparable.
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="mse",
                      metrics=make_masked_metrics(mask, gt_only=gt_only))
    elif args.model == "unet" and args.physics_head:
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                      loss={"dist_out": make_masked_loss(mask, gt_only=gt_only), "spectrum_out": "mse"},
                      loss_weights={"dist_out": 1.0, "spectrum_out": 0.1},
                      metrics={"dist_out": make_masked_metrics(mask, gt_only=gt_only)})
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                      loss=make_masked_loss(mask, gt_only=gt_only),
                      metrics=make_masked_metrics(mask, gt_only=gt_only))
    model.summary(print_fn=lambda s: print("[model] " + s, flush=True))

    def _vlk(h): return "val_loss" if "val_loss" in h.history else "val_dist_out_loss"
    def _vmk(h):
        for k in ("val_masked_mse", "val_dist_out_masked_mse"):
            if k in h.history: return k
        return None

    history, replay_X, replay_Y = [], [], []
    completed_pairs = set()
    if args.resume:
        loaded = _load_state(args.out)
        ckpt = os.path.join(args.out, "model_latest.h5")
        if loaded is not None and os.path.exists(ckpt):
            completed_pairs, replay_X, replay_Y, history = loaded
            model.load_weights(ckpt)
            print(f"[resume] loaded weights + state: {len(completed_pairs)} (round,file) pairs done, "
                  f"{len(history)} history entries, replay buffer = {sum((a[0].shape[0] if isinstance(a,list) else a.shape[0]) for a in replay_X)} samples",
                  flush=True)
        else:
            print("[resume] no checkpoint found — starting fresh", flush=True)

    rng = np.random.default_rng(args.seed)
    step = len(history); t0 = time.time()
    use_pa, _ = _feature_flags(args)

    def _concat_X(Xf, extra):  # Xf and extras are either arrays or [a,b] lists (jepa)
        if isinstance(Xf, list):
            return [np.concatenate([Xf[i]] + [e[i] for e in extra], 0) for i in range(len(Xf))]
        return np.concatenate([Xf] + extra, 0)

    def _concat_Y(Yf, extra):
        if isinstance(Yf, list):
            return [np.concatenate([Yf[i]] + [e[i] for e in extra], 0) for i in range(len(Yf))]
        return np.concatenate([Yf] + extra, 0)

    for rnd in range(args.rounds):
        for k in rng.permutation(len(train_files)):
            path = train_files[k]
            pair = (rnd + 1, os.path.basename(path))
            if pair in completed_pairs:                   # skip files already done in this round on a prior run
                continue
            fb = dp.read_dist_file(path, subsample=args.subsample, with_phi=use_pa)
            file_mask = random_wedge(rng) if args.random_mask else mask
            Xf, Yf = assemble(fb, file_mask, args)
            n_f = (Xf[0].shape[0] if isinstance(Xf, list) else Xf.shape[0])

            if replay_X:
                Xtr = _concat_X(Xf, replay_X)
                Ytr = _concat_Y(Yf, replay_Y)
            else:
                Xtr, Ytr = Xf, Yf

            h = model.fit(Xtr, Ytr, epochs=args.inner_epochs, batch_size=args.batch_size,
                          validation_data=(Xv, Yv), verbose=0)
            train_loss = float(h.history["loss"][-1])
            val_loss = float(h.history[_vlk(h)][-1])
            vmk = _vmk(h)
            val_mmse = float(h.history[vmk][-1]) if vmk else float("nan")
            step += 1; el = time.time() - t0
            print(f"[r{rnd+1}/{args.rounds}] f{step:3d} {os.path.basename(path):44s} n={n_f:5d} "
                  f"train={train_loss:.4e} val={val_loss:.4e} val_mmse={val_mmse:.4e} ({el:.0f}s)", flush=True)
            history.append(dict(step=step, round=rnd + 1, file=os.path.basename(path), n_samples=n_f,
                                train_loss=train_loss, val_loss=val_loss, val_masked_mse=val_mmse, elapsed_s=el))

            if args.replay_per_file > 0:
                m = min(args.replay_per_file, n_f)
                sel = rng.choice(n_f, m, replace=False)
                replay_X.append([a[sel].copy() for a in Xf] if isinstance(Xf, list) else Xf[sel].copy())
                replay_Y.append([a[sel].copy() for a in Yf] if isinstance(Yf, list) else Yf[sel].copy())
                tot = sum((a[0].shape[0] if isinstance(a, list) else a.shape[0]) for a in replay_X)
                while tot > args.replay_cap and len(replay_X) > 1:
                    drop = replay_X.pop(0); replay_Y.pop(0)
                    tot -= (drop[0].shape[0] if isinstance(drop, list) else drop.shape[0])

            del fb, Xf, Yf, Xtr, Ytr, h; gc.collect()
            completed_pairs.add(pair)
            model.save(os.path.join(args.out, "model_latest.h5"))
            with open(os.path.join(args.out, "history.json"), "w") as fh:
                json.dump({"args": vars(args), "history": history}, fh, indent=2)
            _save_state(args.out, history, completed_pairs, replay_X, replay_Y)

    # --- final evaluation ---
    ev = {k: float(v) for k, v in model.evaluate(Xv, Yv, verbose=0, return_dict=True).items()}
    out = {"model": ev}
    if args.model != "jepa":
        out["baselines"] = evaluate_baselines(cubes_v, mask)
        print("[done] baselines:", out["baselines"], flush=True)
        _plot_curves(history, os.path.join(args.out, "loss_curves.png"),
                     baseline_mse=out["baselines"]["energy_shell_mean"]["masked_mse"])
    else:
        _plot_curves(history, os.path.join(args.out, "loss_curves.png"))
    print("[done] final validation (model):", ev, flush=True)
    with open(os.path.join(args.out, "final_metrics.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # --- embedding dump ---
    if args.dump_embeddings:
        try:
            if args.model == "jepa":
                emb = encoder.predict(cubes_v[..., None], verbose=0)
            elif args.model == "unet":
                emb = bottleneck_encoder(model).predict(Xv if not isinstance(Xv, list) else Xv[0], verbose=0)
            else:
                emb = None
            if emb is not None:
                np.save(os.path.join(args.out, "embeddings.npy"), emb)
                _plot_embeddings(emb, os.path.join(args.out, "embeddings_pca.png"))
                print(f"[done] embeddings: {emb.shape} -> embeddings.npy, embeddings_pca.png", flush=True)
        except Exception as e:
            print(f"[warn] embedding dump failed: {e}", flush=True)

    print(f"[done] artifacts in {args.out}/", flush=True)


def _plot_curves(history, path, baseline_mse=None):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; skipping plot"); return
    steps = [r["step"] for r in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(steps, [r["train_loss"] for r in history], "o-", ms=3, label="train loss")
    ax1.plot(steps, [r["val_loss"] for r in history], "s-", ms=3, label="val loss")
    ax1.set_xlabel("file # (incremental step)"); ax1.set_ylabel("loss"); ax1.set_yscale("log")
    ax1.legend(); ax1.set_title("Incremental training: loss per file")
    vm = [r["val_masked_mse"] for r in history]
    if not all(np.isnan(vm)):
        ax2.plot(steps, vm, "s-", ms=3, color="C2", label="val masked MSE")
        if baseline_mse is not None:
            ax2.axhline(baseline_mse, color="red", ls="--", lw=1.2, label="energy-shell-mean baseline")
        ax2.set_yscale("log"); ax2.legend(); ax2.set_title("Reconstruction error in occluded region")
        ax2.set_xlabel("file # (incremental step)"); ax2.set_ylabel("masked-region MSE")
    else:
        ax2.axis("off"); ax2.set_title("(no reconstruction metric for this model)")
    for r in history:
        if r["file"] == history[0]["file"] and r["step"] != 1:
            for ax in (ax1, ax2): ax.axvline(r["step"] - 0.5, color="grey", ls=":", lw=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_embeddings(emb, path):
    """2-D PCA scatter of the embeddings, coloured by sample (time) order."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    X = emb - emb.mean(0, keepdims=True)
    # PCA via SVD (no sklearn dependency)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=np.arange(len(Z)), cmap="viridis", s=8)
    fig.colorbar(sc, label="validation sample index (time order)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Distribution-function embeddings (2-D PCA)\ncolour ~ time; clusters/trajectories = plasma-state structure")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
