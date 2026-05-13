# Notebooks — Offline backup pipeline

Self-contained Jupyter notebooks that wrap `src/` so you can run the whole pipeline
without Claude. They mostly just call into `src/` rather than duplicating logic, so
the documentation in `RESULTS.md` and `NEXT_STEPS.md` still applies.

Run them in this order. Each notebook is short (a few cells) and the heavy
operations (download, training) are commented out by default — uncomment to run.

| # | Notebook | What it does | When to run |
|---|---|---|---|
| 0 | `00_setup.ipynb` | `pip install` deps; verify imports | Once after cloning |
| 1 | `01_download_data.ipynb` | Wraps `download_cdfs.py` to fetch des-dist + fgm by date range | Whenever you want more data |
| 2 | `02_explore_data.ipynb` | Read one CDF, view a sample distribution + the occlusion mask | Sanity check on any new data |
| 3 | `03_baseline.ipynb` | Compute the interpolation baseline you have to beat | Once per dataset (the number to beat) |
| 4 | `04_train_from_scratch.ipynb` | Full training run via `train_incremental.py` | First training of a new run |
| 5 | `05_resume_training.ipynb` | **Re-run after a crash / sleep** — appends `--resume` to the same command | Whenever a run is interrupted |
| 6 | `06_inspect_results.ipynb` | Plot the learning curve, view final metrics, view embeddings | After any run |
| 7 | `07_validate_with_skymap.ipynb` | Reconstructions side-by-side + per-energy spectrum + plasma-density moments via `Skymap` | After a trained run, for paper figures |

## Typical workflows

**First time / fresh dataset**: 0 → 1 → 2 → 3 → 4 → 6

**Adding more data, re-using existing model**: 1 → 5 (the trainer warm-starts from
the checkpointed weights and continues incrementally over the new files).

**Run died overnight**: just open 5 with the same `RUN_NAME` and re-run. The
trainer reads `outputs/<RUN>/{model_latest.h5, history.json, state.json, replay.npz}`
and skips every (round, file) pair already done. You lose at most one file's worth
of training.

**Trying a different architecture / config**: 4 with a fresh `RUN_NAME` and
different `MODEL` / `EXTRA` settings (`unet`, `convlstm`, `jepa`; `--temporal-window`,
`--features pitch_angle,logb`, `--physics-head`, `--mask explosion`).

## Notes

- The notebooks compute `ROOT` automatically as the parent of `notebooks/`, so they
  work whether you launch Jupyter from the repo root or from inside `notebooks/`.
- All real logic lives in `src/`. Editing the notebooks won't change the model — to
  modify the architecture, edit `src/model.py`; to change the loss, edit
  `src/losses.py`; etc. The notebooks are a thin, runnable façade.
- Long training runs are best launched in a terminal (so they survive the Jupyter
  kernel restarting):  
  `python src/train_incremental.py --data-root MMS-FPI-Data-Gaps --out outputs/<RUN> ...`  
  Notebook 04 prints the exact command for you to copy.
