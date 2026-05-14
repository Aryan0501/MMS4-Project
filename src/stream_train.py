"""Stream MMS burst files from the SDC, train on each, delete, repeat.

The end-to-end streaming pipeline the project was designed around:
    list all burst files from the SDC for a date range
    for each one:
        download to temp     (~0.4 GB, network bound)
        train on it          (incremental, with replay buffer, ~60-90 s on CPU)
        delete from disk     (no accumulation)
The trainer is the same `train_incremental.py` we've been using; it carries the
model from file to file and writes a checkpoint after each. Resume support is
inherited: ``--resume`` skips already-processed files in the stream.

Usage
-----
    python src/stream_train.py --start 2024-03-01 --end 2024-03-08 --sc 1 \
        --out outputs/stream_week_mar_2024 \
        --warm-start outputs/unet_v5/model_latest.h5 \
        --inner-epochs 1 --subsample 24 --base-filters 10

Or to resume an interrupted run:
    python src/stream_train.py ... --resume
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "MMS-FPI-Data-Gaps"))

import data_pipeline as dp
from model import build_unet3d
from losses import make_uniform_loss, make_masked_metrics
from train_incremental import random_wedge, _load_richness, build_mask
import tensorflow as tf

SDC = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--sc", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default="MMS-FPI-Data-Gaps")
    p.add_argument("--warm-start", default=None, help="path to model_latest.h5 to warm-start from")
    p.add_argument("--temp-dir", default=None, help="where to put the per-file CDF (deleted after; default: <out>/_tmp)")
    p.add_argument("--inner-epochs", type=int, default=1)
    p.add_argument("--subsample", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--base-filters", type=int, default=10)
    p.add_argument("--replay-per-file", type=int, default=24)
    p.add_argument("--replay-cap", type=int, default=700)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-files", action="store_true",
                   help="don't delete the CDF after training (debug only; will fill the disk)")
    p.add_argument("--resume", action="store_true",
                   help="skip files already in history.json")
    return p.parse_args()


def list_burst_files(sc, start, end):
    r = requests.get(f"{SDC}/file_names/science", params={
        "sc_id": f"mms{sc}", "instrument_id": "fpi", "data_rate_mode": "brst",
        "data_level": "l2", "descriptor": "des-dist",
        "start_date": start, "end_date": end,
    }, timeout=60)
    r.raise_for_status()
    return [f for f in r.text.strip().split(",") if f.endswith(".cdf")]


def download(name, dest, max_retries: int = 5, throttle_s: float = 1.0):
    """Download `name` from SDC to `dest`, with retries + throttling.

    The SDC /file_names/science endpoint returns path-prefixed names like
    'mms/data/mms1/fpi/.../file.cdf', but /download/science only works with the
    BARE filename — passing the path-prefixed one silently returns 204 No Content.
    Strip to basename. Adds:
      - Polite throttling between requests (default 1 s) so the SDC doesn't
        rate-limit us.
      - Exponential backoff on HTTP 429 (Too Many Requests) up to ~64 s.
      - Up to ``max_retries`` attempts before giving up.
    """
    import time as _time
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    bare = os.path.basename(name)

    backoff = throttle_s
    for attempt in range(max_retries):
        try:
            _time.sleep(throttle_s)        # always pause between requests
            with requests.get(f"{SDC}/download/science",
                              params={"file": bare}, stream=True, timeout=600) as r:
                if r.status_code == 429:
                    backoff = min(64.0, max(backoff * 2, 4.0))
                    print(f"  [throttle] 429 from SDC, backing off {backoff:.0f}s ...", flush=True)
                    _time.sleep(backoff)
                    continue
                r.raise_for_status()
                if r.status_code == 204 or r.headers.get("Content-Length") in ("0", None):
                    # fallback: about/browse direct URL
                    parts = bare.split("_")
                    sc, instr, _, level, desc = parts[0], parts[1], parts[2], parts[3], parts[4]
                    stamp = parts[5]; yyyy, mm, dd = stamp[:4], stamp[4:6], stamp[6:8]
                    url2 = (f"https://lasp.colorado.edu/mms/sdc/public/about/browse/"
                            f"{sc}/{instr}/brst/{level}/{desc}/{yyyy}/{mm}/{dd}/{bare}")
                    r2 = requests.get(url2, stream=True, timeout=600); r2.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for chunk in r2.iter_content(chunk_size=1 << 20): fh.write(chunk)
                else:
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20): fh.write(chunk)
            if os.path.getsize(tmp) == 0:
                os.remove(tmp); raise RuntimeError(f"got 0-byte download for {bare}")
            os.replace(tmp, dest)
            return
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            if attempt == max_retries - 1:
                raise
            print(f"  [retry] download {bare}: {e} (attempt {attempt+1}/{max_retries})", flush=True)
            _time.sleep(backoff)
            backoff = min(64.0, backoff * 2)


def local_path(name, base_dir):
    """Map a (possibly path-prefixed) SDC file name to a flat local path."""
    return os.path.join(base_dir, os.path.basename(name))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    tmp_dir = args.temp_dir or os.path.join(args.out, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"[stream] listing burst files mms{args.sc} {args.start}..{args.end} ...", flush=True)
    files = list_burst_files(args.sc, args.start, args.end)
    print(f"[stream] {len(files)} files queued (~{len(files) * 0.4:.0f} GB total before deletion)", flush=True)

    # held-out validation: take the LAST 4 files of the date range chronologically.
    # Capped low so we don't burn disk caching the validation set up front.
    n_val = min(4, max(1, int(round(len(files) * 0.05))))
    val_files_remote = files[-n_val:]
    train_files_remote = files[:-n_val]
    print(f"[stream] train={len(train_files_remote)}  val={len(val_files_remote)}", flush=True)

    # download + cache validation set ONCE up-front so each step has a stable val target
    print("[stream] downloading validation set (kept on disk) ...", flush=True)
    Xv_list, Yv_list = [], []
    use_pa, use_lb = True, True
    # build the data-aware mask -- use the ABSOLUTE repo root so this works no matter what
    # working directory the script is launched from
    fake_args = argparse.Namespace(mask="wedge", data_aware_mask=True, data_root=os.path.join(ROOT, "MMS-FPI-Data-Gaps"))
    mask = build_mask(fake_args)
    print(f"[stream] mask (data-aware wedge): {mask.mean()*100:.1f}% of bins", flush=True)
    rng = np.random.default_rng(args.seed)

    val_dir = os.path.join(args.out, "_val_cache")
    os.makedirs(val_dir, exist_ok=True)
    for vf in val_files_remote:
        local = local_path(vf, val_dir)
        download(vf, local)
        fb = dp.read_dist_file(local, subsample=args.subsample, with_phi=True)
        X, Y = dp.build_inputs(fb, mask, use_pitch_angle=use_pa, use_logb=use_lb, temporal_window=1)
        Xv_list.append(X); Yv_list.append(Y)
        del fb
    Xv = np.concatenate(Xv_list, 0); Yv = np.concatenate(Yv_list, 0)
    del Xv_list, Yv_list
    if Xv.shape[0] > 800:
        sel = np.random.default_rng(args.seed).choice(Xv.shape[0], 800, replace=False)
        Xv, Yv = Xv[sel], Yv[sel]
    print(f"[stream] validation: {Xv.shape[0]} samples cached", flush=True)

    # build model
    in_channels = Xv.shape[-1]
    model = build_unet3d(base_filters=args.base_filters, input_shape=(32, 16, 32, in_channels))
    if args.warm_start and os.path.exists(args.warm_start):
        model.load_weights(args.warm_start)
        print(f"[stream] warm-started from {args.warm_start}", flush=True)
    richness = _load_richness(fake_args)
    gt_only = ~richness["always_zero"]
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=make_uniform_loss(gt_only=gt_only),
                  metrics=make_masked_metrics(mask, gt_only=gt_only))
    model.summary(print_fn=lambda s: print("[model] " + s, flush=True))

    # resume state
    hist_path = os.path.join(args.out, "history.json")
    history = []
    completed = set()
    if args.resume and os.path.exists(hist_path):
        history = json.load(open(hist_path))["history"]
        completed = {r["file"] for r in history}
        ckpt = os.path.join(args.out, "model_latest.h5")
        if os.path.exists(ckpt):
            model.load_weights(ckpt)
            print(f"[stream] RESUMED: {len(completed)} files already done, model weights loaded",
                  flush=True)

    replay_X, replay_Y = [], []
    step = len(history); t0 = time.time()
    always_data = richness["always_data"]

    for i, fname in enumerate(train_files_remote):
        if fname in completed:
            continue
        local = local_path(fname, tmp_dir)
        try:
            t_dl = time.time()
            download(fname, local)
            dl_dt = time.time() - t_dl

            fb = dp.read_dist_file(local, subsample=args.subsample, with_phi=True)
            # data-aware random wedge for this file
            file_mask = random_wedge(rng, confine_to=always_data)
            Xf, Yf = dp.build_inputs(fb, file_mask, use_pitch_angle=use_pa, use_logb=use_lb,
                                      temporal_window=1)
            n_f = Xf.shape[0]
            if replay_X:
                Xtr = np.concatenate([Xf] + replay_X, 0)
                Ytr = np.concatenate([Yf] + replay_Y, 0)
            else:
                Xtr, Ytr = Xf, Yf

            h = model.fit(Xtr, Ytr, epochs=args.inner_epochs, batch_size=args.batch_size,
                          validation_data=(Xv, Yv), verbose=0)
            train_loss = float(h.history["loss"][-1])
            val_loss = float(h.history["val_loss"][-1])
            val_mmse = float(h.history.get("val_masked_mse", [float("nan")])[-1])
            step += 1; el = time.time() - t0
            print(f"[stream {step:4d}/{len(train_files_remote)}] {fname[:64]}  "
                  f"n={n_f:5d}  dl={dl_dt:4.0f}s  train={train_loss:.3e}  val={val_loss:.3e}  "
                  f"val_mmse={val_mmse:.3e}  ({el:.0f}s)", flush=True)
            history.append(dict(step=step, file=fname, n_samples=n_f,
                                train_loss=train_loss, val_loss=val_loss,
                                val_masked_mse=val_mmse, elapsed_s=el, dl_s=dl_dt))

            # replay buffer
            if args.replay_per_file > 0:
                m = min(args.replay_per_file, n_f)
                sel = rng.choice(n_f, m, replace=False)
                replay_X.append(Xf[sel].copy()); replay_Y.append(Yf[sel].copy())
                tot = sum(a.shape[0] for a in replay_X)
                while tot > args.replay_cap and len(replay_X) > 1:
                    tot -= replay_X.pop(0).shape[0]; replay_Y.pop(0)

            del fb, Xf, Yf, Xtr, Ytr, h; gc.collect()
            model.save(os.path.join(args.out, "model_latest.h5"))
            with open(hist_path, "w") as fh:
                json.dump({"args": vars(args), "history": history}, fh, indent=2)
        except Exception as e:
            print(f"[stream] file failed: {fname}  -> {e}", flush=True)
            history.append(dict(step=step + 1, file=fname, error=str(e)))
        finally:
            # ALWAYS delete the cached CDF, even on error
            if not args.keep_files and os.path.exists(local):
                os.remove(local)

    print(f"[stream] DONE  {len(history)} files processed", flush=True)


if __name__ == "__main__":
    main()
