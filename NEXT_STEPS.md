# Next Steps

Ordered roughly by leverage. ✅ = built/wired; the rest is open.
Items 1–4 turn this from a working prototype into something paper-worthy.

## 0. Randomised-mask training  *(open — highest leverage, smallest effort)*
The θ-position failure in §8g (the model breaks when the mask is shifted into the originally-
visible half of θ) is entirely an artifact of training on **one fixed mask**. Fix: at each
file, sample a random (θ-offset, φ-offset) for the wedge before applying it; or just switch to
`--mask explosion` (the realistic energy-dependent mask, which already rotates per energy).
Either resolves the only embarrassing failure mode the current model has, with no new code
beyond a couple of lines in `data_pipeline.apply_mask`.

## 1. Validate at the moments level  *(open — highest priority)*
Reconstruction MSE is in normalised log space. What a physicist (and a reviewer) cares about is
whether the *plasma moments* — density, bulk velocity, pressure tensor, heat flux — computed from
the reconstructed cube match those from the true cube.
- Use `MMS-FPI-Data-Gaps/mmspy/moments.py` — **but first patch it**: it uses `numpy.float`, removed
  in NumPy ≥ 1.20, so it errors on this environment. Replace `num.float` → `float` / `np.float64`.
- Convert reconstructed cubes to physical units with `data_pipeline.to_physical_space`, then run the
  moments integration on (a) the true cube, (b) the reconstructed cube, (c) the interpolation
  baseline. Report median |Δn|/n, |Δv|, |ΔP|, … for each.
- Cross-check against the official `des-moms` product (now downloadable via `download_cdfs.py`).

## 2. Train on a genuinely diverse dataset  *(open)*
Current data: 26 burst files from 2 days (2024-03-02, 2024-03-15), all MMS1 — "bigger" but
probably all magnetosheath/magnetopause. For a real result, pull burst intervals spanning
different regions: pristine solar wind, magnetosheath, magnetopause crossings, magnetotail/plasma
sheet, and include MMS2–4.
- ✅ `MMS-FPI-Data-Gaps/download_cdfs.py` now takes `--start/--end/--sc/--products` and queries the
  SDC file-name API. `--list-only` shows what's there before committing the GB.
- ✅ The incremental loop already handles arbitrary file lists — point `--data-root` at the bigger
  archive. For a true streaming run, add `os.remove(path)` after each file in
  `src/train_incremental.py` (deliberately omitted now so an unattended run can't delete data).

## 3. Use the realistic occlusion mask  *(✅ wired — needs a full run)*
`--mask explosion` switches from the synthetic spacecraft-shadow wedge to `skymaps/explosion.npy`
(energy-dependent, ~50 % of bins). Run the headline config under it and confirm the model still
beats the baseline — that's the number that belongs in a paper. Also consider the *despun*
occlusion handling in `skymaps/generator.py` (the real mask rotates with the spacecraft).

## 4. Physics auxiliary head  *(✅ implemented — needs a full run)*
`--physics-head` adds a second output predicting the per-energy spectrum, trained jointly. Confirm
it (a) doesn't hurt reconstruction and (b) the spectrum prediction is accurate. Then extend the
head to predict a B-field-change indicator (e.g. |dB/dt| from the matched FGM data) — that is the
"detect changes in the magnetic field" capability you wanted, obtained as a by-product of training.

## 5. ConvLSTM run + temporal ablation  *(✅ implemented — needs runs)*
- `--model convlstm --temporal-window W` runs the rigorous "cube through time" model. Expensive on
  CPU; run it on a GPU.
- Compare `unet` vs. `unet --temporal-window {1,2}` vs. `convlstm` on identical data — this
  ablation ("does temporal context actually help, and is a recurrence worth it over just stacking
  neighbouring frames as channels?") is a natural figure.

## 6. Forgetting study  *(open — easy)*
The replay buffer (`--replay-per-file`, `--replay-cap`) is the only forgetting defence. Run with
`--replay-per-file 0` to see how badly the model forgets without it — a clean ablation figure
("incremental training is viable *if* you keep a small replay buffer"). Add multiple `--rounds` so
forgetting at round boundaries is visible.

## 7. More CDF variables  *(partly ✅)*
- ✅ Now used: the distribution itself, FGM B-field → pitch-angle map + log|B|, per-timestep
  azimuth table (`des_phi_brst`) for accurate look-directions.
- Still unused, worth trying as quality/feature channels: per-bin `des_errorflags_brst` and
  `des_compressionloss_brst`; the time-varying `des_energy_brst` step table; spacecraft position
  `fgm_r_gse`; and entirely separate products — `des-moms` (official moments, for validation),
  `dis-dist`/`dis-moms` (ions), EDP electric field, MEC ephemeris.

## Housekeeping
- ✅ `src/` now has: `data_pipeline.py`, `features.py`, `model.py`, `losses.py`, `baseline.py`,
  `train_incremental.py`. `download_cdfs.py` is rewritten.
- `MMS-FPI-Data-Gaps/mmspy/moments.py` still needs the NumPy-compat patch (item 1).
- Add a `requirements.txt`: TF 2.9.1 / Keras 2.9.0 / NumPy 1.23.2 / cdflib 1.3.10 / SciPy 1.9.0 /
  matplotlib, Python 3.10, CPU-only. The conda `environment.yml` is the old PyTorch 1.4 / Python 3.7
  stack and does not match the new `src/` code.
- No GPU on this machine — training is CPU-bound. A GPU box allows `--base-filters 32 --subsample 1
  --rounds N` and makes the ConvLSTM and physics-head runs practical.
