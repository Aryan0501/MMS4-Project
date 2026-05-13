# Incremental Training for FPI Distribution-Function Gap Reconstruction — Results

*Written to be lifted, largely as-is, into the methods/results sections of a paper. Numbers in
the Results tables come from `outputs/<run>/final_metrics.json`; learning curves are
`outputs/<run>/loss_curves.png`. Reproduce with `python src/train_incremental.py …`.*

> **Environment caveat.** All runs here are CPU-only (no GPU on this machine), which forces
> heavy timestep subsampling, narrow networks and few epochs. The numbers are honest but small;
> the *infrastructure* (pipeline, model zoo, incremental loop, baselines) is the deliverable, and
> it transfers unchanged to a GPU box where `--base-filters 32 --subsample 1 --rounds N` become
> feasible.

---

## 1. Problem

MMS/FPI measures the electron velocity distribution `f(E, θ, φ)` on a 32 (energy) × 16 (polar θ)
× 32 (azimuth φ) grid every 30 ms in burst mode. Spacecraft-body occlusion blanks a contiguous
solid-angle region from every measurement; those bins must be reconstructed before the
distribution can be integrated for plasma moments. The MMS archive is petabyte-scale and cannot
be held in memory, so training must *stream*: load a file, learn from it, discard it, repeat,
carrying the model forward between files (continual / incremental learning). The risk in that
regime is **catastrophic forgetting** — later files overwriting what earlier files taught.

This work delivers (a) a debugged data + model + loss pipeline that fixes the issues found in
the original project notebooks, (b) a **model zoo** (3-D U-Net, temporal-channel U-Net,
B-field-conditioned U-Net, U-Net with a physics auxiliary head, and a ConvLSTM "cube through
time"), (c) an incremental trainer that processes one file at a time and records training and
validation loss after **every** file, and (d) comparison against non-learned interpolation
baselines.

## 2. Data

- **Source.** MMS1 FPI burst-mode L2 `des-dist` CDF files on disk: 26 files across two days,
  2024-03-02 (9 files) and 2024-03-15 (17 files). Native variable `mms1_des_dist_brst`, shape
  `(n_time, 32, 16, 32)`. Matched FGM burst files (`mms1_fgm_b_dmpa_brst_l2`) supply the
  magnetic field for the conditioned models. *(`MMS-FPI-Data-Gaps/download_cdfs.py` now queries
  the SDC API to fetch des-dist / des-moms / fgm for an arbitrary date range — see §9.)*
- **Subsampling.** Every Nth burst timestep is kept (CPU budget); ≈ 150–500 samples per file.
- **Split.** 20 % of *files* are held out for validation (file-level split, not sample-level —
  consecutive burst timesteps are near-duplicates, so a per-sample split would leak).
- **Transform.** `x = log10(f + 1e-33)`, then linearly mapped from `[-33, -23]` to `[0, 1]` and
  clipped; inverse provided (`data_pipeline.to_physical_space`). The constants are *fixed*
  (file-independent) so every streamed batch is on the same scale. *(Fixes the original "no
  normalisation" bug — raw PSD spans ~9 orders of magnitude including exact zeros, which
  otherwise explodes the gradients; this was why the original models collapsed to a near-constant
  output in ~3 epochs.)*

## 3. Occlusion mask

Default is a **synthetic spacecraft-shadow wedge** — a contiguous θ×φ block (here 50 % of θ ×
50 % of φ = 25 % of all bins), identical across energies. The project's realistic
energy-dependent mask `skymaps/explosion.npy` (~50 % of bins) is selectable with `--mask
explosion`. The mask is fixed for a run, so the loss and metrics can be scoped exactly to the
occluded region.

## 4. Model zoo (`src/model.py`)

All share the inpainting design principle: the answer in the hidden region is strongly
constrained by the *visible* part of the same cube (smooth energy spectra, near-isotropy at fixed
energy, pitch-angle symmetry), so the decoder must be able to copy that structure directly. The
original project models — plain encoder→bottleneck→decoder with no skips — could not.

| name | what it is | input | use |
|------|-----------|-------|-----|
| `unet` | shallow 3-D U-Net, skip connections, downsampling on energy & φ only (not θ); sigmoid output ⇒ non-negative PSD | `(32,16,32,1)` | the workhorse / baseline |
| `unet` + `--temporal-window W` | same U-Net, but input also carries the masked cube at timesteps t-W…t+W as extra channels — the "cube through time" idea in feed-forward form | `(32,16,32, 2W+1)` | tests whether temporal context helps, cheaply, before committing to a recurrence |
| `unet` + `--features pitch_angle,logb` | same U-Net, extra input channels: a pitch-angle map (angle between each look-direction and **B**, from the matched FGM file) and log\|B\| | `(32,16,32, …+2)` | injects magnetic-field geometry — the thing that *makes* a plasma distribution anisotropic |
| `unet` + `--physics-head` | adds a second output: the per-energy integrated spectrum (a density-spectrum proxy), trained jointly | two outputs | regularises the latent toward physically-meaningful features; a step toward "detect changes in B" |
| `convlstm` | ConvLSTM over a short sequence of cubes — convolution *inside* the recurrence (cube handled as a θ×φ image with the 32 energies folded into channels) | `(T,16,32,32)` | the rigorous form of the original "cube evolving through time" model, which had flattened the cube before a plain LSTM |

Parameter counts are tiny (≈ 0.1–0.2 M for the U-Nets at `base_filters≈12`), deliberately, so they
train on CPU.

## 5. Loss and metrics (`src/losses.py`)

- **Training loss**: per-bin squared error with occluded bins weighted 10× the visible ones, plus
  a small (`1e-3`) penalty on the mismatch in *integrated signal* — computed **per batch from the
  current ground truth**, never from a stale hard-coded tensor (an original-notebook bug). For
  `--physics-head`, an additional MSE term (weight 0.1) on the per-energy spectrum.
- **Reported metrics**: masked-region MSE and MAE in model space — per occluded bin, averaged
  over samples — directly comparable across runs and against the baselines.

## 6. Baselines (`src/baseline.py`)

- **Energy-shell angular mean** (the reference): each occluded `(E,θ,φ)` bin is filled with the
  mean of the *visible* bins at the same energy E — exploits near-isotropy in look-direction at
  fixed energy. A reconstruction model that can't beat this isn't worth its complexity.
- **Constant mean**: every occluded bin filled with the mean over all visible bins. A floor.

## 7. Incremental procedure (`src/train_incremental.py`)

```
build model, compile with masked loss; load validation set once
for round in 1..R:
    shuffle the training-file list
    for each training file:
        load file → transform → assemble (X, Y) for the chosen model
        mix in replay buffer (≤ N samples retained from files already seen)
        model.fit(..., inner_epochs, validation_data = held-out set)
        record train_loss, val_loss, val_masked_mse
        sample a few cubes into the replay buffer
        del the file's arrays; gc.collect()              # "delete old data"
        save model checkpoint + running history.json
final: evaluate model vs. baselines on the validation set; write metrics + loss-curve plot
```

**On "deleting old data":** memory is freed between files (`del` + `gc.collect()`) — the safe
equivalent of the streaming pattern. The script never deletes CDF files from disk; a production
run against the live archive would add one `os.remove(path)` line, deliberately omitted so an
unattended run cannot destroy local data.

## 8. Results

Both runs: all 26 files, **21 train / 5 validation** (file-level split), 1 round = 21 incremental
steps, `--inner-epochs 2 --subsample 20 --base-filters 10 --batch-size 64 --val-cap 600
--replay-per-file 24 --replay-cap 600`, synthetic wedge mask (25 % of bins occluded), **CPU-only**.
Metrics are masked-region error in normalised model space, per occluded bin, averaged over the
600-sample held-out validation set. (From `outputs/{unet_plain,unet_full}/final_metrics.json`.)

| Method | Masked-region MSE | Masked-region MAE | vs. baseline (MSE) |
|--------|------------------:|------------------:|-------------------:|
| Constant-mean baseline | 0.2085 | 0.4330 | 1.0× |
| Energy-shell angular-mean baseline | 0.2035 | 0.4240 | 1.0× |
| 3-D U-Net (plain) | 0.0256 | 0.1308 | **≈ 8× better** |
| **3-D U-Net + temporal-window 1 + pitch-angle + log\|B\|** | **0.0212** | **0.0902** | **≈ 9.6× better** |

**Findings**

1. **The learned model crushes interpolation.** Both U-Nets reduce masked-region MSE by roughly an
   order of magnitude versus the energy-shell-mean baseline (which is itself a sensible
   physics-motivated rule). So the network is genuinely exploiting the structure in the visible
   part of each cube, not just regressing to a mean.
2. **Temporal context + magnetic-field features help.** Adding the t±1 masked cubes and the
   pitch-angle / log|B| channels improves on the plain U-Net by ~17 % in masked MSE and ~31 % in
   masked MAE — a clean ablation, same data and hyper-parameters.
3. **Incremental training is stable, no catastrophic forgetting.** Validation loss for the full
   model falls monotonically across the stream: 0.071 → 0.065 → … → 0.029 → **0.021** by file 21
   (`outputs/unet_full/history.json`). The train–val gap stays small throughout (train ≈ 0.02–0.04,
   val tracking it), so the model is not overfitting whichever file it just saw — the replay buffer
   is doing its job.
4. **Still improving at the end of the stream.** The curve had not plateaued at file 21, so more
   data and/or more rounds would push it further — which is exactly the argument for the
   larger-dataset GPU run (§9 item 1, `NEXT_STEPS.md`).

Learning curves: `outputs/unet_full/loss_curves.png`, `outputs/unet_plain/loss_curves.png` —
train and validation loss after each incremental step, with the energy-shell-mean baseline MSE
drawn as a reference line.

### 8b. JEPA-lite and embeddings  *(implemented + smoke-tested; not run full-size on CPU)*

`--model jepa` trains the I-JEPA-lite model from §4 — predict the full cube's embedding from the
masked cube's embedding, loss in latent space, no decoder. `--dump-embeddings` (works for `unet`
and `jepa`) runs the encoder on the validation cubes after training and saves `embeddings.npy`
plus `embeddings_pca.png` (2-D PCA scatter, coloured by time order). The intent: the embedding is a
compact fingerprint of the plasma state, so the scatter should show clusters / trajectories that
correspond to regime structure and outliers at unusual events — a representation-learning route to
"detect changes in the magnetic field." Running it to convergence wants a GPU; the code path is
verified end-to-end.

## 9. Honest limitations

1. **Two days of one spacecraft** — "bigger" (26 files vs. the original 1) but not *diverse*;
   probably all magnetosheath/magnetopause. Cross-regime generalisation untested.
2. **Synthetic wedge mask** by default; the realistic `explosion.npy` run (`--mask explosion`) is
   the number that belongs in a paper.
3. **No moments-level validation yet** — MSE in log space is a proxy; density/velocity/pressure
   recovery is what matters physically. `mmspy/moments.py` needs a small NumPy-compat patch first.
4. **CPU-bound** — heavy subsampling, narrow networks, few epochs. A GPU lifts all three.
5. **ConvLSTM and physics-head are implemented and smoke-tested but not run to convergence here**
   (ConvLSTM is expensive on CPU). They are ready to train on a GPU.
6. **Pitch-angle channel uses B in the DMPA frame** as a stand-in for the FPI instrument frame —
   adequate as a feature, not as a publication-grade pitch angle (which needs the full attitude
   transform).

See `NEXT_STEPS.md` for the path through each of these.

## 10. Reproducing

```bash
pip install tensorflow==2.9.1 numpy==1.23.2 scipy==1.9.0 cdflib==1.3.10 matplotlib

# headline config (the one whose numbers go in the table)
python src/train_incremental.py --data-root MMS-FPI-Data-Gaps --out outputs/unet_full \
    --model unet --temporal-window 1 --features pitch_angle,logb \
    --rounds 1 --inner-epochs 2 --subsample 20 --base-filters 10

# plain-U-Net ablation
python src/train_incremental.py --data-root MMS-FPI-Data-Gaps --out outputs/unet_plain \
    --model unet --rounds 1 --inner-epochs 2 --subsample 20 --base-filters 10

# realistic occlusion mask / ConvLSTM / physics head
python src/train_incremental.py ... --mask explosion
python src/train_incremental.py ... --model convlstm --temporal-window 1
python src/train_incremental.py ... --model unet --physics-head

# fetch more, more-diverse data first:
python MMS-FPI-Data-Gaps/download_cdfs.py --start 2024-03-02 --end 2024-03-03 \
    --products des-dist des-moms fgm --list-only      # then drop --list-only to download
```
Artifacts per run: `outputs/<run>/{model_latest.h5, history.json, final_metrics.json, loss_curves.png}`.
