# MMS FPI Data Gap Reconstruction — Project Summary

**Internship project @ Aurora Engineering**  
MMS spacecraft · Fast Plasma Investigation (FPI) · Deep learning for distribution function reconstruction

---

## Background

The [Magnetospheric Multiscale (MMS) mission](https://mms.gsfc.nasa.gov/) is a NASA constellation of four spacecraft studying magnetic reconnection in Earth's magnetosphere. Each spacecraft carries a **Fast Plasma Investigation (FPI)** instrument suite that measures 3D velocity distribution functions of electrons and ions at extremely high cadence:

| Mode | Electrons (DES) | Ions (DIS) |
|------|----------------|------------|
| Burst | **30 ms** | 150 ms |
| Fast Survey | 4.5 s | 4.5 s |

The instrument resolves distributions across **32 energy levels** (6.5 eV → 27,525 eV), **16 theta bins**, and **32 phi bins** — producing a 16,384-element snapshot every 30 ms in burst mode.

### The data gap problem

MMS collects roughly **100 Gbits of data per day** across all four spacecraft, but only **~4 Gbits can be downlinked to the ground** due to telemetry constraints. Even for transmitted data, portions of each distribution function are systematically missing due to spacecraft body occlusion — the spacecraft itself blocks certain look directions for the detectors. These gaps cannot be filled by interpolation alone because the distributions are highly non-isotropic and physically structured (pitch angle anisotropy, beam features, loss cones, etc.).

The full MMS archive (available publicly via the [MMS Science Data Center](https://lasp.colorado.edu/mms/sdc/public/) and [NASA CDAWeb](https://cdaweb.gsfc.nasa.gov/pub/data/mms/)) spans from March 2015 to present across four spacecraft, accumulating a data volume on the order of **petabytes** — far beyond what can be downloaded or processed locally. Any reconstruction method must therefore work efficiently on a per-file or per-interval basis.

---

## Reference codebase

This project was built in the context of the [Aurora-Engineering/MMS-FPI-Data-Gaps](https://github.com/Aurora-Engineering/MMS-FPI-Data-Gaps) repository, which provides:

- **`download_cdfs.py`** — download MMS L2 CDF files from the SDC by time interval and instrument
- **`mmspy/moments.py`** — physics-based computation of plasma moments (density, bulk velocity, pressure tensor, heat flux, entropy) from raw distribution functions via trapezoidal integration
- **`skymaps/skymap.py`** — `Skymap` class for managing 4D distribution data; `MomentsTimeSeries` for aggregated moments; `plotMomentsTimeSeries()` for standardised 6-panel visualisation
- **`skymaps/generator.py`** — PyTorch `Dataset` class with realistic spacecraft occlusion masking and despin-based augmentation
- **`example_read_cdf.py`**, **`example_moments.py`**, **`example_skymaps.py`** — worked examples for reading CDFs, computing moments, and visualising skymaps

---

## Work done

### 1. Data acquisition & preprocessing
**`data-preprocessing.ipynb`**

- Downloaded MMS1 FPI/DES burst-mode L2 CDF files for **1 March 2024** (5 matched FPI+FGM intervals)
- Extracted distribution functions `mms1_des_dist_brst` — shape `(4,333 timesteps × 32 energy × 16 θ × 32 φ)`
- Computed density moments from raw distributions using the `Moments` class and validated against official L2 moment products
- Confirmed pipeline correctness: computed density matched official products across the full interval

### 2. Pitch angle calculation
**`pitch_angle_calculation.ipynb`**

- Loaded FGM magnetic field data (`mms1_fgm_b_gse_brst_l2`) for all 5 matched intervals
- Interpolated B-field to FPI timestamps using linear interpolation
- Converted all (θ, φ) look directions to 3D unit vectors and computed the pitch angle between each detector look direction and the local B-field
- Generated 5 pitch angle heatmaps (φ vs. θ) with the magnetic field direction overlaid — these directly reveal which angular bins are field-aligned, anti-parallel, or perpendicular

### 3. Moments validation
**`moments_validation.ipynb`**

- Cross-validated custom `Moments` integration against official MMS L2 moments CDF products
- Extracted number density (~8.6–11.5 cm⁻³), bulk velocity components, and confirmed physical consistency across the time series

### 4. Model development — iterations

#### MODEL 3 · 3D Convolutional Autoencoder (baseline)
**`MODEL_3.ipynb`**

| Property | Value |
|----------|-------|
| Input shape | (16 φ, 32 θ, 32 E, 1 channel) |
| Architecture | Conv3D(32) → MaxPool → Conv3D(64) → MaxPool → UpSample × 2 |
| Parameters | 223,105 |
| Mask | Right half of θ dimension (16 bins) |
| Loss | MSE + physics-informed density loss |
| Test MSE | 1.18 × 10⁻⁴ |
| Density loss | 430.76 |

Established that a convolutional autoencoder can learn to reconstruct masked angular regions while preserving moments through a custom combined loss.

---

#### MODEL 4 · Pitch-Angle Conditioned Autoencoder
**`MODEL_4.ipynb`**

| Property | Value |
|----------|-------|
| Input shape | (16 φ, 32 θ, 32 E, **2 channels**: distribution + pitch angle) |
| Architecture | Same encoder-decoder backbone as Model 3 |
| Innovation | Pitch angle map (normalised to [0,1]) injected as second input channel |
| Masked region MAE | **1.36 × 10⁻⁹** |
| Masked region MSE | 1.56 × 10⁻¹⁴ |

Adding physical context (pitch angle) dramatically improved reconstruction accuracy in masked regions by giving the model geometric information about the particle's relationship to the local magnetic field.

---

#### Model 5 · Experimental conditioning refinement
**`Model_5.ipynb`**

Explored alternative concatenation strategies for the pitch angle channel. Encountered tensor dimension mismatches during development; architecture direction carried forward into Model 7.

---

#### Model 7 & 8 · Time-Distributed CNN + LSTM + Attention
**`Model_7.ipynb`, `Model_8.ipynb`**

| Property | Value |
|----------|-------|
| Architecture | Conv3D(64) ×4 → MaxPool3D → TimeDistributed → LSTM (512→256→128) → Attention ×2 → Dense(16384) → Reshape (32,16,32) |
| Total parameters | **13,118,848** (50 MB) |
| Trainable | 2,150,016 (8.2 MB) — convolutional backbone frozen |
| Design rationale | Capture temporal correlations in plasma dynamics across distribution snapshots |

This architecture treats a sequence of consecutive distribution function snapshots as a temporal input, using LSTM layers to model how plasma structures evolve in time and attention layers to focus reconstruction effort on the masked region.

---

### 5. Fine-tuning
**`fine_tuning.ipynb`**

Applied transfer learning: froze the convolutional encoder weights from the best-performing earlier model and retrained the LSTM and attention layers with a lower learning rate to adapt to the specific March 2024 dataset.

---

### 6. Final evaluation
**`final_evaluation.ipynb`**

- Loaded fine-tuned model and applied it to held-out test set (434 samples from the March 2024 burst interval)
- Reconstructed full distributions by combining model predictions (masked region) with original data (unmasked region)
- Computed plasma moments from reconstructed distributions and compared time-series against ground truth
- Generated side-by-side heatmaps (original vs. reconstructed) at multiple energy channels
- Confirmed that bulk density and velocity are preserved through reconstruction

---

## Results summary

| Model | Architecture | Masked region MSE | Masked region MAE | Physics-informed loss |
|-------|-------------|-------------------|-------------------|-----------------------|
| 3 — Baseline | Conv3D AE | 1.18 × 10⁻⁴ | — | ✓ (density) |
| 4 — Pitch angle | Conv3D AE + PA channel | 1.56 × 10⁻¹⁴ | 1.36 × 10⁻⁹ | ✓ (density) |
| 7/8 — Temporal | CNN + LSTM + Attention | evaluated via moments | — | ✓ (moments time-series) |

**Key findings:**
- Physics-informed loss functions are essential — MSE alone does not preserve plasma moments
- Pitch angle conditioning (injecting magnetic field geometry as an auxiliary input channel) is the single largest accuracy improvement
- Temporal modelling via LSTM captures plasma dynamics and improves coherence across consecutive snapshots
- All reconstructions maintain non-negativity via sigmoid output activation

---

## Repository structure

```
Aurora_Project/
├── src/                        # Debugged streaming pipeline + model zoo (new — see RESULTS.md)
│   ├── data_pipeline.py        # CDF reading, log-transform + normalisation, masking, temporal windows
│   ├── features.py             # FGM B-field loading → pitch-angle map + log|B| feature channels
│   ├── model.py                # 3-D U-Net (+ temporal channels / B-field conditioning / physics head); ConvLSTM
│   ├── losses.py               # masked-region loss + metrics (per-batch density proxy, not a stale tensor)
│   ├── baseline.py             # interpolation baselines the model must beat
│   └── train_incremental.py    # streaming train/free-memory loop; tracks train+val loss per file
├── outputs/                    # run artifacts: model_latest.h5, history.json, final_metrics.json, loss_curves.png
├── RESULTS.md                  # methodology + results + model zoo, written paper-ready
├── NEXT_STEPS.md               # prioritised path to a publishable result
├── MMS-FPI-Data-Gaps/          # Original internship notebooks + reference toolkit
│   ├── mmspy/moments.py        # Physics: density, velocity, pressure, heat flux
│   ├── skymaps/                # skymap.py, generator.py (occlusion masking), explosion.npy (real mask)
│   ├── data-preprocessing.ipynb, pitch_angle_calculation.ipynb, moments_validation.ipynb
│   ├── MODEL_3.ipynb … Model_8.ipynb, fine_tuning.ipynb, final_evaluation.ipynb
│   ├── download_cdfs.py        # rewritten: queries the SDC API for des-dist/des-moms/fgm by date range
│   ├── environment.yml, README.md
│   └── data_03_02/, data_03_15/   # downloaded burst CDFs (gitignored)
└── Untitled-1.ipynb            # Exploratory CDF/moments scratch notebook
```

> **Note:** Large data files (`.cdf`, `.npy`, `.h5`, `.pkl`, `data_*/`) are excluded from version control via `.gitignore`. Download MMS data on demand with `MMS-FPI-Data-Gaps/download_cdfs.py` (try `--list-only` first).

## Running the streaming trainer

```bash
pip install tensorflow==2.9.1 numpy==1.23.2 scipy==1.9.0 cdflib==1.3.10 matplotlib

# plain 3-D U-Net
python src/train_incremental.py --data-root MMS-FPI-Data-Gaps --out outputs/unet_plain --model unet

# headline: U-Net + temporal context + magnetic-field features
python src/train_incremental.py --data-root MMS-FPI-Data-Gaps --out outputs/unet_full \
    --model unet --temporal-window 1 --features pitch_angle,logb

# realistic occlusion mask / ConvLSTM "cube through time" / physics auxiliary head
python src/train_incremental.py ... --mask explosion
python src/train_incremental.py ... --model convlstm --temporal-window 1
python src/train_incremental.py ... --model unet --physics-head
```

It discovers every `*des-dist*.cdf` under `--data-root`, holds out 20 % of *files* for validation,
then processes the rest one at a time — fit a couple of epochs, record train+validation loss, free
the file from memory, checkpoint — for `--rounds` passes, mixing in a small replay buffer to resist
catastrophic forgetting. Results table and the full model zoo are in `RESULTS.md`; open work is in
`NEXT_STEPS.md`.

### Resuming a crashed / interrupted run

The trainer writes model weights, history, run state and replay buffer after **every file**, so an
interrupted run (machine sleep, OOM, kill -9) loses at most one file's worth of work. To pick up,
re-run the same command with `--resume` appended:

```bash
python src/train_incremental.py ... --out outputs/unet_full --resume
```

It reads `outputs/unet_full/{model_latest.h5, history.json, state.json, replay.npz}`, skips every
`(round, file)` pair already processed, and continues. Older runs that pre-date this feature still
work — the completed pairs are derived from `history.json` (the replay buffer just starts empty).

---

## Setup

```bash
conda env create -f MMS-FPI-Data-Gaps/environment.yml
conda activate mms
```

Data download (requires SDC access — data is publicly available):
```python
# Edit time interval and output path in download_cdfs.py, then:
python MMS-FPI-Data-Gaps/download_cdfs.py
```

---

## Data access

| Resource | URL |
|----------|-----|
| MMS Science Data Center | https://lasp.colorado.edu/mms/sdc/public/ |
| NASA CDAWeb MMS archive | https://cdaweb.gsfc.nasa.gov/pub/data/mms/ |
| Reference toolkit | https://github.com/Aurora-Engineering/MMS-FPI-Data-Gaps |

---

*Project conducted as part of an internship at Aurora Engineering.*
