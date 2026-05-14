"""Comprehensive evaluation of a trained inpainting model.

Generates a full battery of figures and quantitative metrics for the model
checkpoint at ``--ckpt``, all written under ``--out`` (defaults to the checkpoint's
directory). Designed to be the standard "after training, run this" script.

Outputs
-------
  reconstruct_120s_filmstrip.png      4-col TRUE / MASKED / IMPUTED / FULLY IMPUTED
  reconstruct_120s_compare.png        density: true (raw) + true (data-rich-only) +
                                       imputed + fully-imputed; the methodology
                                       fix ("the raw 'true' line is biased low
                                       because the cube has missing bins as 0s")
  reconstruct_120s.gif                4-col animation
  reconstruct_120s_seeds.png          3 random 120-s samples (different seeds)
                                       side by side, lets us see if results
                                       generalise or if seed 13 was lucky
  temporal_variance.png               per-bin std vs truth diagnostic
  per_energy_density.png              density per energy shell vs time, true vs imp
  spectrogram_compare.png             energy-time spectrogram, true vs imputed vs
                                       fully-imputed
  metrics.json                        all the headline numbers in one file

Usage
-----
    python _evaluate_model.py --ckpt outputs/unet_v7/model_latest.h5 \
        --model unet_residual --temporal-window 2 --base-filters 12
"""
from __future__ import annotations
import os, sys, argparse, warnings, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "MMS-FPI-Data-Gaps"))
import data_pipeline as dp
from model import build_model
from skymaps.skymap import Skymap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="unet")
    p.add_argument("--temporal-window", type=int, default=1)
    p.add_argument("--base-filters", type=int, default=10)
    p.add_argument("--features", default="pitch_angle,logb")
    p.add_argument("--validity-channel", action="store_true",
                   help="match training: include the binary validity channel as last input")
    p.add_argument("--missing-fill", choices=["zero", "shell_mean"], default="zero",
                   help="match training: how unreliable input bins are filled")
    p.add_argument("--out", default=None)
    p.add_argument("--data-root", default=os.path.join(ROOT, "MMS-FPI-Data-Gaps"))
    return p.parse_args()


def feature_flags(args):
    feats = {f.strip() for f in args.features.split(",") if f.strip()}
    return ("pitch_angle" in feats), ("logb" in feats)


def predict_for_window(model, X):
    out = model.predict(X, verbose=0)
    return out[..., 0] if out.ndim == 5 else out


def density(cube, n=None):
    n = cube.shape[0] if n is None else n
    sk = Skymap(n, name="x"); sk.skymap = np.zeros((n, 32, 16, 32), dtype=np.float64)
    sk.skymap[:] = dp.to_physical_space(cube); return np.asarray(sk.momsTS.density)


def density_data_rich(cube, always_data, n=None):
    """Density computed using ONLY the always-data bins (zero everything else first).
    Removes the bias from raw 'always-zero' bins being treated as physical 0s."""
    n = cube.shape[0] if n is None else n
    masked = cube.copy()
    masked[:, ~always_data] = 0.0
    return density(masked, n)


def evaluate_one_sample(model, args, mask_default, fb, sub_idx, w, always_zero=None):
    """Run model on one 120-s sample. Returns dict of arrays."""
    use_pa, use_lb = feature_flags(args)
    fb.cubes = fb.cubes[sub_idx]; fb.epoch = fb.epoch[sub_idx]
    if hasattr(fb, "phi"): fb.phi = fb.phi[sub_idx]
    X, Y = dp.build_inputs(fb, mask_default, use_pitch_angle=use_pa, use_logb=use_lb,
                            temporal_window=w,
                            with_validity=getattr(args, "validity_channel", False),
                            always_zero=always_zero,
                            missing_fill=getattr(args, "missing_fill", "zero"))
    P = predict_for_window(model, X)
    y_cube = Y[..., 0]
    inpaint = y_cube.copy(); inpaint[:, mask_default] = P[:, mask_default]
    measured = (y_cube > 0.05) & (~mask_default[None, ...])
    fully = np.where(measured, y_cube, P)
    return dict(y=y_cube, P=P, imp=inpaint, full=fully, X=X)


def main():
    args = parse_args()
    out_dir = args.out or os.path.dirname(os.path.abspath(args.ckpt))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[eval] ckpt: {args.ckpt}\n[eval] out:  {out_dir}", flush=True)

    # build model + load weights — match the input-channel count used at training
    use_pa, use_lb = feature_flags(args)
    in_channels = (2 * args.temporal_window + 1) + int(use_pa) + int(use_lb) + int(args.validity_channel)
    model = build_model(args.model, base_filters=args.base_filters,
                        in_channels=in_channels, temporal_window=args.temporal_window)
    if isinstance(model, tuple):  # jepa returns a tuple
        model = model[0]
    model.load_weights(args.ckpt)
    print(f"[eval] in_channels={in_channels} (validity={args.validity_channel}, fill={args.missing_fill})", flush=True)

    files = dp.find_dist_files(args.data_root)
    _, val_files = dp.split_files(files, val_fraction=0.2, seed=0)
    richness = dict(np.load(os.path.join(ROOT, "outputs", "data_richness.npz")))
    always_data = richness["always_data"]
    always_zero = richness["always_zero"]
    az_for_input = always_zero if (args.validity_channel or args.missing_fill != "zero") else None
    mask_default = dp.synthetic_wedge_mask()
    SUB = 12; DUR_S = 120.0

    metrics = {"args": vars(args)}

    # --- 1. Multi-seed gallery (3 random 120-s samples) --- #
    print("[eval] multi-seed gallery (3 samples) ...", flush=True)
    seeds = [13, 42, 137]
    seed_panels = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        chosen = val_files[int(rng.integers(0, len(val_files)))]
        fb = dp.read_dist_file(chosen, subsample=SUB, with_phi=True)
        n_needed = int(DUR_S * 1000 / (SUB * 30))
        n_keep = min(n_needed, fb.n_samples)
        t0 = 0 if fb.n_samples < n_needed else int(rng.integers(0, fb.n_samples - n_keep + 1))
        idx = np.arange(t0, t0 + n_keep)
        ev = evaluate_one_sample(model, args, mask_default, fb, idx, args.temporal_window,
                                  always_zero=az_for_input)
        ev["chosen"] = chosen; ev["t0"] = t0; ev["seed"] = seed
        seed_panels.append(ev)
    # plot 3-row gallery, one row per seed, columns = TRUE / MASKED / IMPUTED / FULLY IMPUTED at frame 50
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    E_BIN = 16
    for row, ev in enumerate(seed_panels):
        i = min(50, ev["y"].shape[0] - 1)
        vmax = float(np.percentile(ev["y"][:, E_BIN], 99))
        masked_view = ev["y"][i, E_BIN].copy(); masked_view[mask_default[E_BIN]] = 0.0
        axes[row, 0].imshow(ev["y"][i, E_BIN],     vmin=0, vmax=vmax, origin="lower", aspect="auto")
        axes[row, 1].imshow(masked_view,           vmin=0, vmax=vmax, origin="lower", aspect="auto")
        axes[row, 2].imshow(ev["imp"][i, E_BIN],   vmin=0, vmax=vmax, origin="lower", aspect="auto")
        axes[row, 3].imshow(ev["full"][i, E_BIN],  vmin=0, vmax=vmax, origin="lower", aspect="auto")
        axes[row, 0].set_ylabel(f"seed {ev['seed']}\n{os.path.basename(ev['chosen'])[:25]}\nframe {i}", fontsize=8)
        for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
    for ax, t in zip(axes[0], ("TRUE", "MASKED", "IMPUTED", "FULLY IMPUTED")):
        ax.set_title(t, fontsize=11, fontweight="bold")
    fig.suptitle("Multi-seed reconstruction gallery (3 random 120-s held-out samples)\n"
                 f"model: {args.model}  base_filters={args.base_filters}  temporal_window={args.temporal_window}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reconstruct_120s_seeds.png"), dpi=110); plt.close(fig)
    print(f"[eval] wrote reconstruct_120s_seeds.png", flush=True)

    # --- 2. The headline 120-s sample (seed=13) details --- #
    ev0 = seed_panels[0]
    y_cube, imp, full, X = ev0["y"], ev0["imp"], ev0["full"], ev0["X"]
    n_keep = y_cube.shape[0]
    real_t = SUB * 30 / 1000
    t_axis = np.arange(n_keep) * real_t
    chosen_name = os.path.basename(ev0["chosen"])

    # 4-column filmstrip
    N_KF = 12
    idxs = np.linspace(0, n_keep - 1, N_KF).round().astype(int)
    vmax = float(np.percentile(y_cube[:, E_BIN], 99))
    fig, axes = plt.subplots(N_KF, 4, figsize=(12.5, 1.05 * N_KF))
    for row, i in enumerate(idxs):
        masked_view = y_cube[i, E_BIN].copy(); masked_view[mask_default[E_BIN]] = 0.0
        axes[row, 0].imshow(y_cube[i, E_BIN],   vmin=0, vmax=vmax, origin="lower", aspect="auto")
        axes[row, 1].imshow(masked_view,        vmin=0, vmax=vmax, origin="lower", aspect="auto")
        axes[row, 2].imshow(imp[i, E_BIN],      vmin=0, vmax=vmax, origin="lower", aspect="auto")
        axes[row, 3].imshow(full[i, E_BIN],     vmin=0, vmax=vmax, origin="lower", aspect="auto")
        axes[row, 0].set_ylabel(f"t={t_axis[i]:5.1f}s", fontsize=9)
        for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
    for ax, t in zip(axes[0], ("TRUE", "MASKED", "IMPUTED", "FULLY IMPUTED")):
        ax.set_title(t, fontsize=10, fontweight="bold")
    fig.suptitle(f"120-s reconstruction (seed 13)  •  {chosen_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reconstruct_120s_filmstrip.png"), dpi=110); plt.close(fig)
    print(f"[eval] wrote reconstruct_120s_filmstrip.png", flush=True)

    # --- 3. Density comparison with the methodology fix --- #
    print("[eval] computing density (4 ways) on 120-s sample ...", flush=True)
    MOM_STRIDE = 4
    sel = np.arange(0, n_keep, MOM_STRIDE)
    d_true_raw = density(y_cube[sel], len(sel))                           # biased low
    d_true_dr  = density_data_rich(y_cube[sel], always_data, len(sel))    # unbiased estimate
    d_imp      = density(imp[sel], len(sel))
    d_full     = density(full[sel], len(sel))
    def med(a, ref):
        ok = np.abs(ref) > 1e-30
        return float(np.median(np.abs(ref[ok] - a[ok]) / np.abs(ref[ok])))
    metrics["density_seed13"] = {
        "median_rel_err_imputed_vs_raw_truth":      med(d_imp, d_true_raw),
        "median_rel_err_fully_vs_raw_truth":        med(d_full, d_true_raw),
        "median_rel_err_imputed_vs_data_rich_truth":med(d_imp, d_true_dr),
        "median_rel_err_fully_vs_data_rich_truth":  med(d_full, d_true_dr),
        "ratio_data_rich_to_raw":                   float(np.median(d_true_dr / np.maximum(d_true_raw, 1e-30))),
    }
    fig, ax = plt.subplots(figsize=(11, 4.6))
    t_mom = t_axis[sel]
    ax.plot(t_mom, d_true_raw, "k-",  lw=1.5, label="true (raw cube — biased low: counts missing as 0)")
    ax.plot(t_mom, d_true_dr,  "k--", lw=2.0, label=f"true+missing (data-rich bins only — unbiased; ×{metrics['density_seed13']['ratio_data_rich_to_raw']:.2f})")
    ax.plot(t_mom, d_imp,      "C0-", lw=1.4, label=f"imputed (med err vs data-rich = {med(d_imp, d_true_dr):.2f})")
    ax.plot(t_mom, d_full,     "C2-", lw=1.4, label=f"fully imputed (med err vs data-rich = {med(d_full, d_true_dr):.2f})")
    ax.set_xlabel("burst time (s)"); ax.set_ylabel("density (Skymap moments)")
    ax.set_title(f"Density: methodology-corrected comparison  •  {chosen_name}\n"
                 f"the 'raw true' line is biased LOW because the cube has natural-zero bins; "
                 f"the dashed line uses only data-rich bins for the unbiased reference")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reconstruct_120s_compare.png"), dpi=110); plt.close(fig)
    print(f"[eval] wrote reconstruct_120s_compare.png", flush=True)

    # --- 4. Per-energy density (per energy shell, time-resolved) --- #
    print("[eval] per-energy density spectrogram ...", flush=True)
    def per_energy_density(cube_subset):
        # crude: integrate PSD over (theta, phi) per energy shell, per time
        phys = dp.to_physical_space(cube_subset)
        return phys.sum(axis=(2, 3))   # (n, 32)
    pe_true_raw = per_energy_density(y_cube[sel])
    pe_true_dr  = per_energy_density(np.where(always_data, y_cube[sel], 0.0))
    pe_imp = per_energy_density(imp[sel])
    pe_full = per_energy_density(full[sel])
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    vmax_se = max(pe_true_dr.max(), pe_full.max(), 1e-30)
    for ax, M, title in [(axes[0,0], pe_true_raw, "true (raw — biased)"),
                         (axes[0,1], pe_true_dr,  "true+missing (data-rich, unbiased)"),
                         (axes[1,0], pe_imp,      "imputed (synth-mask only)"),
                         (axes[1,1], pe_full,     "fully imputed")]:
        im = ax.imshow(np.log10(np.maximum(M.T, 1e-30)), aspect="auto", origin="lower",
                       cmap="viridis", extent=[t_mom[0], t_mom[-1], 0, 32])
        ax.set_title(title, fontsize=10); ax.set_xlabel("time (s)"); ax.set_ylabel("energy bin")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Per-energy density (log10) — energy-time spectrogram view", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spectrogram_compare.png"), dpi=110); plt.close(fig)
    print(f"[eval] wrote spectrogram_compare.png", flush=True)

    # --- 5. Temporal variance diagnostic --- #
    print("[eval] temporal variance diagnostic ...", flush=True)
    std_true = y_cube.std(axis=0); std_imp = imp.std(axis=0)
    sm = mask_default
    ratio = std_imp[sm].mean() / max(std_true[sm].mean(), 1e-10)
    metrics["temporal_var_ratio_seed13"] = float(ratio)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    vmax_s = float(std_true.max())
    axes[0].imshow(std_true[E_BIN], vmin=0, vmax=vmax_s, origin="lower", cmap="magma", aspect="auto")
    axes[0].set_title(f"TRUE per-bin std at E={E_BIN}", fontsize=10)
    axes[1].imshow(std_imp[E_BIN], vmin=0, vmax=vmax_s, origin="lower", cmap="magma", aspect="auto")
    axes[1].set_title(f"model imputed per-bin std at E={E_BIN}", fontsize=10)
    per_e_t = np.array([std_true[e][sm[e]].mean() for e in range(32)])
    per_e_p = np.array([std_imp[e][sm[e]].mean() for e in range(32)])
    axes[2].plot(per_e_t, "k-", lw=2, label="true")
    axes[2].plot(per_e_p, "C0-", lw=1.5, label="model")
    axes[2].set_xlabel("energy bin"); axes[2].set_title(f"per-E std in masked region\noverall ratio = {ratio:.2f}")
    axes[2].legend(); axes[2].set_ylabel("std")
    for ax in axes[:2]: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Temporal variance: does the model PREDICT VARYING values across the burst?\n"
                 f"(1.0 = matches truth, 0 = frozen). Current ratio in masked region = {ratio:.2f}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "temporal_variance.png"), dpi=110); plt.close(fig)
    print(f"[eval] wrote temporal_variance.png  (ratio = {ratio:.3f})", flush=True)

    # --- 6. Compact GIF of the 120-s reconstruction --- #
    print("[eval] writing GIF ...", flush=True)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), dpi=72)
    ims = []
    for j, lab in enumerate(("TRUE", "MASKED", "IMPUTED", "FULLY IMPUTED")):
        ims.append(axes[j].imshow(np.zeros((16, 32)), vmin=0, vmax=vmax, origin="lower", aspect="auto"))
        axes[j].set_title(lab, fontsize=9); axes[j].set_xticks([]); axes[j].set_yticks([])
    sup = fig.suptitle(f"t=0.0s   ({chosen_name})", fontsize=10)
    fig.tight_layout()
    def frame(i):
        masked_view = y_cube[i, E_BIN].copy(); masked_view[mask_default[E_BIN]] = 0.0
        ims[0].set_data(y_cube[i, E_BIN]); ims[1].set_data(masked_view)
        ims[2].set_data(imp[i, E_BIN]); ims[3].set_data(full[i, E_BIN])
        sup.set_text(f"t={t_axis[i]:.1f}s of {n_keep*real_t:.0f}s")
        return (*ims, sup)
    ani = FuncAnimation(fig, frame, frames=n_keep, blit=False, interval=int(SUB * 30))
    ani.save(os.path.join(out_dir, "reconstruct_120s.gif"),
             writer=PillowWriter(fps=int(round(1000/(SUB*30)))))
    plt.close(fig)
    print(f"[eval] wrote reconstruct_120s.gif", flush=True)

    # --- 7. Overall metrics: per-seed density err, store in metrics.json --- #
    metrics["per_seed_density_err"] = []
    for ev in seed_panels:
        ys = ev["y"]; sel_s = np.arange(0, ys.shape[0], MOM_STRIDE)
        d_dr  = density_data_rich(ys[sel_s], always_data, len(sel_s))
        d_imp_s = density(ev["imp"][sel_s], len(sel_s))
        d_fl_s  = density(ev["full"][sel_s], len(sel_s))
        metrics["per_seed_density_err"].append({
            "seed": ev["seed"], "file": os.path.basename(ev["chosen"]),
            "med_imp_vs_data_rich":  med(d_imp_s, d_dr),
            "med_full_vs_data_rich": med(d_fl_s, d_dr),
        })
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[eval] wrote metrics.json")

    print("\n[eval] HEADLINE NUMBERS:")
    for r in metrics["per_seed_density_err"]:
        print(f"  seed {r['seed']:4d}  imputed vs data-rich: {r['med_imp_vs_data_rich']:.3f}   "
              f"fully vs data-rich: {r['med_full_vs_data_rich']:.3f}")
    print(f"  temporal var ratio (seed 13, masked region): {metrics['temporal_var_ratio_seed13']:.3f}  "
          f"(closer to 1.0 = better)")


if __name__ == "__main__":
    main()
