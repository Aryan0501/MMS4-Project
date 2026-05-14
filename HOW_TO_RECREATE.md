# How to recreate this project from scratch (no AI assist required)

A self-contained walkthrough so anyone — including future-you with no
context — can rebuild what's here. Reads top to bottom; sections are short.

---

## 1. Why we are doing this

NASA's MMS spacecraft constellation measures the 3-D velocity-space
distribution of electrons (and ions) every 30 ms in burst mode. Each
measurement is a `32 (energy) × 16 (polar θ) × 32 (azimuth φ)` cube of phase
space density (PSD). Two distinct kinds of "missing" data make this hard to
work with:

- **Spacecraft body occlusion** — for any given look-direction (θ, φ), the
  spacecraft body sometimes blocks the detector; those bins are reported as
  zero PSD even though there is real plasma there.
- **Below-noise-floor** — the detector simply can't see the bin; reported as
  zero even though the underlying PSD is small but nonzero.

Both look the same in the raw data: `PSD = 0`. Plasma scientists then
integrate the cube to get moments (density, bulk velocity, pressure tensor,
heat flux). Missing bins systematically bias the moments **low**.

The goal of this project is a learned model that **fills in** the missing
bins from the visible ones, so the moments computed from the reconstructed
cubes are closer to physical truth.

---

## 2. The data

- **Source:** the MMS Science Data Center at LASP — `https://lasp.colorado.edu/mms/sdc/`.
- **Product:** `desc=des-dist`, `data_rate_mode=brst`, `data_level=l2`,
  `instrument=fpi`, `sc_id=mms1` (we used spacecraft 1; MMS2/3/4 are reserved for cross-validation).
- **Scale:** ~80 burst files per day per spacecraft (~0.4 GB each). Six months
  × four spacecraft is ~9,600 files (~3.9 TB). The full archive runs
  March 2015 → present (~70-80 TB just of this product).
- **What we actually trained on:** ~870 burst files spanning March + April
  2024 of MMS1, ~350 GB total — but **never more than ~0.4 GB on disk at a
  time** thanks to the streaming pipeline.
- **Auxiliary product used as features:** matched FGM burst files
  (`mms1_fgm_b_dmpa_brst_l2`) for the local magnetic-field vector.

API endpoints:
- list files in a date range:
  `GET /files/api/v1/file_names/science?sc_id=mms1&instrument_id=fpi&data_rate_mode=brst&data_level=l2&descriptor=des-dist&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD`
- download one file (use only the **basename** — passing the full path returns 204):
  `GET /files/api/v1/download/science?file=<filename>.cdf`

---

## 3. What goes IN to the model and what comes OUT

**Input** to the v8 model: a tensor of shape `(batch, 32, 16, 32, 8)` per cube — eight
channels stacked on the last axis:

| ch | content | what it represents |
|---|---|---|
| 0 | masked PSD at t-2  | temporal context, log-PSD in [0,1] |
| 1 | masked PSD at t-1  | "" |
| 2 | **masked PSD at t** (centre frame) | the frame we predict for |
| 3 | masked PSD at t+1  | "" |
| 4 | masked PSD at t+2  | "" |
| 5 | pitch-angle map    | angle between (θ,φ) look-direction and **B** |
| 6 | log\|B\|             | broadcast scalar from FGM, normalised to [0,1] |
| 7 | **validity flag**  | 1 = bin is reliable input (visible AND not always-zero), 0 = bin is unreliable |

The "masked PSD" channels have the synthetic occlusion mask **and** the
always-zero region replaced with the per-energy-shell visible-bin mean
(see §6) — never literal zero. The validity channel tells the model which
bins to trust.

**Output:** a single channel `(batch, 32, 16, 32, 1)` — the predicted full
distribution function for the centre frame, in the same normalised log space.
For the residual-temporal architecture, the network's actual output is
`sigmoid(centre_input + small_correction)`, so by construction it preserves
input variation (fixes the "predictions are too constant in time" problem
seen in earlier versions).

To convert back to physical PSD (s³/cm⁶) for moment calculation:
`f = 10**(x_norm * 10 - 33)`.

---

## 4. The problem (concretely)

Given a cube `f(E, θ, φ)`, ~25–50 % of the bins are unreliable (some always
zero, some occluded for this measurement only). We want to predict the true
PSD in those bins from the visible bins + temporal neighbours + B-field
context, accurately enough that the integrated plasma moments computed from
the reconstructed cube are within (target) ~10 % of the true moments.

What "true" means is itself nontrivial — see §7.

---

## 5. Architecture

A 3-D U-Net with skip connections, modified into a **residual-temporal**
form:

```
input  (32, 16, 32, 8)
  │
  ├── 5 temporal frames + 2 features + 1 validity flag
  │
  enc1: Conv3D × 2 + BN + ReLU                       ──┐ skip
  pool (energy and phi only, NOT theta — why: θ is already small)
  enc2: Conv3D × 2 + BN + ReLU                      ─┐│ skip
  pool                                                ││
  bottleneck: Conv3D × 2 + BN + ReLU                 ││
  upsample → concat enc2 → dec2                    ──┘│
  upsample → concat enc1 → dec1                     ──┘
  Conv3D 1×1×1 → "correction" via tanh
  output = sigmoid(centre_input_channel + correction)
```

About 200 K trainable parameters at `base_filters=12`. Small enough to
train on a CPU laptop, big enough to learn the structure.

**Why these choices:**

- **3-D** convolution because the input is a cube; energy / θ / φ all matter.
- **U-Net skip connections** because inpainting needs the decoder to copy
  high-resolution detail from the visible region; without skips the
  bottleneck destroys it (plain encoder→decoder fails completely; we tried).
- **Down-sample on energy and φ only** because θ is only 16 bins; pooling
  it would leave a 4×N feature map.
- **Residual head** (`output = sigmoid(centre_input + correction)`) because
  earlier non-residual models had predictions that were ~3-4× temporally
  flatter than truth — they collapsed to predicting the mean. With
  residual, the input frame's variation is preserved by construction; the
  network only learns the small correction.
- **Wider temporal window (5 frames)** so the model can lean on time-
  adjacent frames when the centre frame's information is incomplete.
- **Pitch-angle + log\|B\| feature channels** because magnetic-field geometry
  is what makes a plasma distribution anisotropic in the first place.

---

## 6. Three subtle data fixes that mattered more than architecture

These three fixes — none of them flashy — gave more improvement than any
architecture change.

### 6a. Data-aware random masking (`--data-aware-mask --random-mask`)

The synthetic occlusion mask used for training (the bins we artificially
hide so the model has something to learn from) is placed:

- **Only in the always-data region.** The model never has to "predict" a
  bin in the always-zero back-hemisphere because there's no ground truth
  there anyway.
- **At a random (θ, φ) origin per file.** The mask shape is the same; the
  position rotates. This stops the model from memorising one specific mask
  geometry and forces it to learn the *task* of inpainting.

### 6b. Ground-truth-only loss (`--gt-only-loss`)

The loss has zero weight on always-zero bins. Their "ground truth" of 0
isn't physical truth — it's "no measurement." Penalising the model for
putting non-zero predictions there teaches it the wrong thing.

### 6c. Shell-mean fill for unreliable input (`--missing-fill shell_mean`)

The inputs at masked + always-zero bins are NOT literal zero. They are
filled with the per-frame, per-energy-shell visible-bin mean. Why: with
literal zero in the input, the model can shortcut "predict 0 because
input is 0 and the loss agrees." With the shell-mean fill, the input is a
smooth interpolation; the model can only reduce loss by doing **better**
than the shell-mean.

Combined with the validity-flag channel (§3, ch 7), the model knows
exactly which bins are real-input vs. shell-mean-fill, and can attend
accordingly.

---

## 7. The "true density is also missing" problem

The truth itself is biased. When we compute `density(true_cube)` via the
Skymap moments code, the always-zero bins contribute literal zero to the
integral — so this "true" density is **biased low**. Our reconstruction can
look like it's doing badly when in fact the scoring reference is broken.

The methodology fix in evaluation: compute `density(true_cube)` using
**only the always-data bins** (zero everything else first). That's the
unbiased reference. It's slightly higher than the raw "true" line and gets
labeled `true+missing` or `true (data-rich)` in plots.

---

## 8. Streaming training: how to run on the laptop without filling the disk

`src/stream_train.py` does the entire end-to-end loop:

```
list all burst files in [start, end] for this spacecraft
hold out the chronologically-last 4 as validation
for each training file in stream:
    download   (~30-60s, 0.4 GB)
    train one inner epoch on it (~60-90s)
    save model + history + replay buffer
    delete the CDF
```

Replay buffer: keep up to 1500 random samples from files already seen,
mix into each new file's training data. This is the standard cheap defence
against catastrophic forgetting in streaming / continual learning.

Resume support: every per-file save writes `model_latest.h5`,
`history.json`, AND `replay.npz`. On `--resume`, we restore all three —
so a laptop sleep / kill / OOM costs at most one file's worth of work and
**doesn't lose the buffer** (an earlier bug fixed in commit `027a6d5`).

Throttling: 1 s polite pause + exponential backoff on HTTP 429 + up to 5
retries per file, so the SDC doesn't rate-limit us during a multi-day run.

---

## 9. The exact command to recreate the v8 (current best) training

```bash
# 0. Install
pip install tensorflow==2.9.1 numpy==1.23.2 scipy==1.9.0 cdflib==1.3.10 \
    matplotlib requests

# 1. Compute the data-richness map (one-time, takes ~3 min)
python _richness.py
# writes outputs/data_richness.npz

# 2. Stream-train v8 on a chunk of MMS1 burst data
python src/stream_train.py \
    --start 2024-03-01 --end 2024-04-30 --sc 1 \
    --out outputs/stream_v8_long_mms1 \
    --model unet_residual --temporal-window 2 \
    --base-filters 12 --batch-size 64 \
    --subsample 24 --inner-epochs 1 \
    --replay-cap 1500 \
    --validity-channel --missing-fill shell_mean
# CPU runtime: ~30 hours on 870 files (resume-safe; can stop+restart anytime)

# 3. Evaluate
python _evaluate_model.py \
    --ckpt outputs/stream_v8_long_mms1/model_latest.h5 \
    --model unet_residual --temporal-window 2 --base-filters 12 \
    --features pitch_angle,logb
# writes a comprehensive set of figures + metrics.json under the ckpt dir
```

---

## 10. What's still wrong, and what we'd try next

- **The prediction in the always-zero region is honest but cannot be
  validated.** We trust the model there because it was validated on the
  data-aware masked region. A reviewer can challenge this; the answer is
  "what else could we do — there's no ground truth."
- **Temporal-flatness gap** has improved with the residual-temporal
  architecture, but isn't fully closed. ConvLSTM3D might do better; or a
  transformer over the time axis with explicit temporal attention.
- **Data diversity** — even with 870 burst files, all are MMS1 magnetosheath /
  magnetopause from one season. A reviewer will ask about plasma sheet,
  solar wind, magnetotail. `TRAINING_PLAN.md` lays out the path; the
  rate-limiting step is GPU compute (a weekend on a Lambda Labs A100 ≈ $55
  would close 80 % of the gap).
- **No comparison to existing space-physics gap-filling baselines** beyond
  energy-shell-mean. Gaussian processes / Kriging / matrix completion are
  the standard things to compare against; that's a Phase-5 task in the
  training plan.

---

## 11. Where to find what

| Path | What |
|---|---|
| `src/data_pipeline.py` | CDF reading, log-transform + normalisation, masking, shell-mean fill |
| `src/features.py` | FGM B-field loading → pitch-angle + log\|B\| feature channels |
| `src/model.py` | All architectures: plain U-Net, residual-temporal, energy-attention, ConvLSTM, JEPA-lite, I-JEPA |
| `src/losses.py` | masked-region loss + uniform loss; metrics; gt-only support; JEPA loss |
| `src/baseline.py` | energy-shell-mean and constant-mean interpolation baselines |
| `src/train_incremental.py` | Single-machine training with all flags; no streaming |
| `src/stream_train.py` | Streaming training (download → train → delete) |
| `_evaluate_model.py` | "After training, run this" — generates 7 figures + metrics.json |
| `_richness.py` | Computes outputs/data_richness.npz (always-data / always-zero maps) |
| `notebooks/` | Jupyter wrappers (00_setup → 09_inpainting_movie) for offline use |
| `outputs/` | Per-run artifacts (model weights, history, figures, GIFs) |
| `MODEL_CHANGELOG.md` | v1 → v8 evolution and why each change |
| `RESULTS.md` | Methodology + headline numbers, paper-ready prose |
| `NEXT_STEPS.md` | Prioritised list of what's still open |
| `PUBLICATION.md` | Venues for a first paper + first-time-author guide |
| `TRAINING_PLAN.md` | 6-phase ~7-week schedule to a first submission |
| `HOW_TO_RECREATE.md` | this file |
