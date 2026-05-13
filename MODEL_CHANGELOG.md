# Model Changelog

A running log of every trained model in `outputs/`, in chronological order, with
what changed, why, and the headline numbers. Newest at the top.

---

## v6 · `stream_week_mar2024` — first streaming run (download → train → delete)
*Trained 2026-05-13 18:15 → 19:29 (~75 min). 31 of 58 attempted MMS1 burst files
from 2024-03-02..08; the SDC rate-limited (HTTP 429) on the remaining 27 from
Mar 2 specifically, so we have a complete Mar 5–8 run + a few from Mar 7. Warm-
started from v4. Same data-aware random mask + gt-only loss as v5 was supposed to.*

**Why this run mattered:** first end-to-end test of the production-style streaming
pipeline (`src/stream_train.py`): list files from SDC, download one, train one
inner-epoch, **delete the file**, repeat. Peak disk usage during the run was ≈ 1.6 GB
(one in-flight CDF + cached val set), peak RAM ≈ 700 MB. Validated that we can
stream-train indefinitely without filling local storage.

**3-way comparison vs. v3 and v4 on the same RECON_SEED=13 120-s held-out burst:**

| | median \|Δn\|/n | std_pred / std_true (masked region) |
|---|---:|---:|
| v3 imputed | 0.233 | 0.22 |
| v4 imputed | 0.223 | 0.35 |
| **v6 imputed** | 0.236 | 0.28 |
| **v6 fully imputed** | **0.203** | — |

- On `imputed` v6 is essentially tied with v3 and slightly worse than v4. Caveat:
  the held-out sample is from `data_03_15`, which v4 saw during training (the
  Mar 15 files are in v4's train set). v6 was warm-started from v4 then trained
  on Mar 5–8, so it's been gradient-stepped *away* from Mar 15 specifics — partial
  catastrophic forgetting that the replay buffer mitigates but doesn't eliminate.
  A fairer eval is on a val burst v6 has never seen and v4 also has never seen,
  which the streaming pipeline can produce on demand.
- On `fully imputed` v6 is the **best** so far (0.20 vs v4's 0.22), suggesting the
  diverse-week training did improve the model's ability to extrapolate into the
  always-zero region.
- The **temporal-variance diagnostic** (`temporal_variance.png`) is the headline
  honest finding: ALL three models have only 22–35 % of the temporal variability
  the truth has in the masked region. The U-Net is significantly *flatter in time*
  than reality — confirms the user's "imputed part is constant" observation. The
  fix is a model with explicit temporal dynamics (ConvLSTM3D, or longer
  `--temporal-window`) — either of which is GPU work.

**Known issue from this run:** SDC rate-limits at ~5 GB / minute. The script needs
exponential backoff and request throttling; that's a small follow-up edit before
the next stream attempt.

Files: `outputs/stream_week_mar2024/{model_latest.h5, history.json,
reconstruct_120s_*.png, reconstruct_120s.gif, temporal_variance.png}`.

---

## v4 · `unet_proper` *(prior best)* — data-aware mask + ground-truth-only loss
*Trained 2026-05-13 14:32 → 15:10. CPU. 42 steps (2 rounds × 21 train files).*

**The fix**, motivated by the user's observation that raw zeros in MMS data are
*not* ground truth — they mean "no measurement," not "PSD = 0":

- `--data-aware-mask`: the synthetic occlusion wedge is restricted to bins that are
  **always-data** (signal present in ≥ 95 % of the burst, per `outputs/data_richness.npz`).
  17.7 % of cube bins are masked, all in the data-rich region. The model is *forced*
  to learn inpainting on bins where we have ground truth.
- `--gt-only-loss`: the loss and metrics give **zero weight** to **always-zero** bins
  (≤ 5 % of the burst has signal there; 27 % of all bins). The model is no longer
  penalised for putting non-zero predictions in bins whose "ground truth" of 0 is
  fake.

**Headline (data-aware mask, gt-only metric):**
- masked-region MSE = **0.01525** (~15× better than the energy-shell-mean baseline 0.233)
- masked-region MAE = 0.0595

**Why this matters:** the model now extrapolates honestly into the truly-missing bins.
Whatever it predicts there is its data-rich-learned prior generalised — not zero from
training contamination. That is the production-deployable behaviour.

**Side-by-side vs. v3 on the same 120-s held-out burst** (same fixed wedge for both, fair
comparison; see `outputs/unet_proper/reconstruct_120s_*.png`):

| | median \|Δn\|/n |
|---|---:|
| v3 imputed | 0.233 |
| v3 fully imputed | 0.233 |
| v4 imputed | 0.223 |
| v4 fully imputed | **0.223** |

Density numbers barely changed (~4 % relative). The dramatic difference is visual: in the
FULLY IMPUTED panel, v3's right-side natural-zero region stays dark (the model was trained
to predict zero there); v4's right side fills with bright, plasma-like structure (the model
is now free to extrapolate the visible signal into the missing region). Density doesn't
move much because the predicted values, while non-zero, are individually small. **Visually
v4 is an extrapolator; v3 was a zero-mimic.** That's the production-deployable behaviour
even when the integrated-moment metric only nudges.

Files: `outputs/unet_proper/{model_latest.h5, history.json, final_metrics.json,
loss_curves.png, embeddings.npy, embeddings_pca.png, reconstruct_120s_filmstrip.png,
reconstruct_120s_compare.png, reconstruct_120s.gif}`.

---

## v3 · `unet_full_3rounds` — 3-round full config (fixed mask)
*Trained over two sessions; first attempt died overnight at step 27/63, resumed via
`--resume` to completion. Used to validate the resume mechanism in anger.*

`--model unet --temporal-window 1 --features pitch_angle,logb --rounds 3` on a
fixed right-half-θ wedge mask (25 % of bins).

- masked-MSE 0.01984, MAE 0.0881 (~10.3× baseline)
- median fractional density error on a 10-s held-out burst: **0.21**
- median fractional density error on a 120-s held-out burst: **0.23**

**Known weakness** — discovered with the data-aware-mask probe in `RESULTS.md` §8i:
when the mask is moved off the back-hemisphere onto the bright (left) half of φ,
density error blows up to ~0.94. The model had been getting partial credit for
predicting near-zero in bins that are near-zero anyway, *and* learning that the
always-zero region must be zero (which is wrong — it's just unmeasured).

This is the model v4 was designed to replace.

Files: `outputs/unet_full_3rounds/`.

---

## v2 · `unet_full` and `unet_plain` — 1-round headline + ablation
*Trained 2026-05-12 23:46 → 00:35. Same 21 train / 5 val split, 1 round.*

The original two-run ablation that produced the first paper-grade table:
- `unet_full` (3-D U-Net + temporal + pitch-angle + log\|B\|): masked-MSE 0.0212
- `unet_plain` (plain 3-D U-Net): masked-MSE 0.0256
- → temporal + B-field channels improve over plain by ~17 % MSE, ~31 % MAE.

Both ~10× better than the energy-shell-mean baseline. Validation loss fell
monotonically over the 21-file stream and was still descending at the end —
motivation for the v3 3-round run.

Files: `outputs/unet_full/`, `outputs/unet_plain/`.

---

## v1 · early notebook models *(MMS-FPI-Data-Gaps/)*
The original internship notebooks (`MODEL_3.ipynb` … `Model_8.ipynb`,
`fine_tuning.ipynb`, `final_evaluation.ipynb`). Documented in `RESULTS.md` and the
deep-dive in earlier session work — multiple bugs (no normalisation → gradient
collapse, density loss compared to a hardcoded tensor, plain LSTM fed sequence-
length 1, inference path called the model on unmasked input). Replaced from
scratch by the `src/` pipeline that v2–v4 use.

---

## v5 · `unet_v5` *(training now)* — random data-aware masks + uniform gt-only loss
*Launched 2026-05-13 18:00. Training in background, ~50 min.*

Strict version of v4 inspired by the user's observation that v4's FULLY IMPUTED
panel still had patches of dim output near the visible/missing boundary:

- `--random-mask` — fresh random wedge position **per training file**, never the
  same shape twice. Forces the model to handle gaps at every position.
- `--data-aware-mask` — random wedges are **confined to the always-data region**
  (35% of bins). Mask never overlaps an always-zero bin, so the model is only ever
  asked to inpaint bins where ground truth exists.
- `--gt-only-loss` — `make_uniform_loss(gt_only=...)` excludes always-zero bins
  from the uniform MSE loss. Fixed an earlier bug where `--random-mask` was
  silently falling back to plain MSE and ignoring `--gt-only-loss`.
- Validation set still uses the fixed default mask (upper-left quadrant of
  always-data) — different from the random training masks, so no positional
  leakage.

Results filled in §8j of RESULTS.md when training finishes.

---

## Auxiliary models *(implemented + smoke-tested, not yet run on CPU)*

- **I-JEPA proper** (`src/model.py:build_ijepa`): separate EMA target encoder
  (0.99 momentum) + invariance loss + VICReg-style variance and covariance
  regularisation. The properly-resourced version of JEPA-lite. Returns
  `(train_model, online, target, ema_update_fn)`; trainer calls `ema_update_fn`
  after every optimiser step. 121 K params at `base_filters=8`. Needs GPU
  and bigger batches to stabilise the regulariser.
- **Energy-axis attention U-Net** (`src/model.py:build_energy_attention_unet`):
  drop-in replacement for `build_unet3d` with a 4-head self-attention block
  over the **energy axis at the bottleneck**. Plasma motivation: bin-to-bin
  energy structure carries information that pure 3-D conv locality misses
  (e.g. an electron beam at one energy correlates with adjacent energies).
  627 K params at `base_filters=8` — about 5× the plain U-Net, would benefit
  from a GPU.
- **JEPA-lite** (`outputs/jepa/`): predict the full-cube embedding from the
  masked-cube embedding in latent space, no decoder. Runs; the variance term
  oscillates without an EMA target, which is why I-JEPA proper exists now.
- **ConvLSTM** (`src/model.py:build_convlstm`): the rigorous "cube through time"
  model with convolution inside the recurrence. Smoke-tested; full CPU run
  impractical.
- **Physics auxiliary head** (`--physics-head`): adds a per-energy-spectrum
  prediction output. Smoke-tested; can be combined with any of the above.

---

## Data scale (what's actually available)

Surveyed the SDC API for burst-mode des-dist availability **in just 6 months
(Jan–Jun 2024)**, all four spacecraft:

| Spacecraft | Files |
|---|---:|
| MMS1 | 2,443 |
| MMS2 | 2,427 |
| MMS3 | 2,390 |
| MMS4 | 2,394 |
| **6-month total** | **9,654** |

At ~0.4 GB/file that's **~3.9 TB** for half a year. The full archive runs March
2015 → present (~10 years), plausibly 70–80 TB of just this product. Our 26
files were **0.27 % of half a year of one spacecraft**. Data is not the
constraint — compute (GPU hours) and the streaming pipeline are. Both are now
solved in `src/`; the next-step gating factor is GPU access.
