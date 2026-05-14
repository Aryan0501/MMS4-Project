# Long-term training plan

A phased CPU-only campaign that produces a paper-ready model over ~6-10 weeks
of mostly-unattended training, using the streaming pipeline. **Architecture
complexity is allowed to grow as we go** — each phase upgrades the model and
the data scope. The streaming + resume infrastructure means crashes, sleeps,
and weekend interruptions cost at most one file's work.

## Hardware reality

Single Windows laptop, **CPU only** (no GPU). At current settings each file
is ~30-90 s download + ~60-120 s training = ~2 min/file. So:

- **1 day of CPU-time** ≈ 720 files trained → about **2 weeks of MMS1 burst data**
- **1 week of CPU-time** ≈ 5,000 files → about **3 months of MMS1**
- **1 month of CPU-time** ≈ 21,000 files → about **a year of MMS1, OR 3 months × 4 spacecraft**

A GPU box ($1-3/hr on RunPod or Lambda) would compress all of this 20-50×, but
the plan below assumes you stay on the laptop. If you ever rent a GPU for a
weekend, jump to the corresponding faster phase.

## Phase 0 — last-mile fixes  *(this week, ~1 hour each)*

- ✅ **Streaming pipeline** (`src/stream_train.py`) — done.
- ✅ **SDC throttling + retry** — done.
- ✅ **Residual-temporal architecture** (`--model unet_residual`) — coded; needs training.
- [ ] **Methodology fix to the comparison plot** — `true` density line should
  be computed from data-rich bins only (the raw cube has missing bins as
  literal zeros, which biases the "true" line low).
- [ ] **Moments-aware loss term** — penalty on `|density(prediction) - density(truth)|`
  computed per batch from the data-rich bins. Forces the model to preserve the
  integrated quantity that matters physically.
- [ ] **Gallery of multiple 120-s samples** with different RNG seeds (you don't
  trust one example, fair).

## Phase 1 — v7 residual-temporal trial run  *(~3 hours, this evening)*

Just convince ourselves the new architecture beats v6 before scaling.

```bash
python src/train_incremental.py --model unet_residual --temporal-window 2 \
    --features pitch_angle,logb --random-mask --data-aware-mask --gt-only-loss \
    --rounds 2 --inner-epochs 2 --subsample 20 --base-filters 10 \
    --out outputs/unet_v7_residual_temporal
```

- ~26 files (existing on disk) × 2 rounds × ~70 s = **~1 hour of training** on CPU.
- **Success criterion**: v7's std_pred / std_true ratio in the masked region
  improves from v6's 0.28 to **≥ 0.6** on the same 120-s sample. That's the
  diagnostic that motivates the architecture change.
- If v7 succeeds, lock the architecture and proceed to Phase 2.
- If it doesn't, try `--temporal-window 3` (7-frame stack) or wider
  `--base-filters 16`, or add the `--physics-head` for moments-aware
  regularisation.

## Phase 2 — broaden the data  *(~2 weeks of background runs, ~1 week of CPU-time)*

```bash
python src/stream_train.py --start 2024-03-01 --end 2024-03-31 --sc 1 \
    --warm-start outputs/unet_v7_residual_temporal/model_latest.h5 \
    --model unet_residual --temporal-window 2 \
    --features pitch_angle,logb --random-mask --data-aware-mask --gt-only-loss \
    --out outputs/stream_march_mms1
```

- 1 month of MMS1 burst ≈ 480 files × 2 min = **~16 hours of CPU**.
- Run overnight + weekends; resume each morning.
- Then repeat with `--start 2024-04-01 --end 2024-04-30`, `--start 2024-02-01`,
  etc. until you've covered ~3 months of MMS1.
- After each run: re-evaluate on the held-out 120-s sample (and others), commit
  the figures and a one-paragraph entry in `MODEL_CHANGELOG.md`.

## Phase 3 — MMS2/3/4 used for VALIDATION ONLY  *(no training)*

Per user direction: train only on MMS1 to keep the wall-clock budget bounded;
use the other three spacecraft strictly to **cross-validate** the model.

```bash
for sc in 2 3 4; do
  python _evaluate_model.py --ckpt outputs/<latest>/model_latest.h5 \
      --model unet_residual --temporal-window 2 --base-filters 12 \
      --features pitch_angle,logb \
      --data-root <download dir for MMSn> \
      --out outputs/<latest>/cross_validate_mms${sc}
done
```
- Download a small slice (~1 day) of burst data for each of MMS2-4.
- Run the standard 120-s reconstruction on each.
- Honest finding: does the model trained on MMS1 generalise to other
  spacecraft? Compare median \|Δn\|/n across the four. If MMS2-4 are much
  worse, that's a paper-worthy "trained-on-one-generalises-to-formation"
  finding either way.

## Phase 4 — architecture iterations  *(parallel, on the side)*

Once the streaming infrastructure is humming, try one architecture variant per
week on the same data:

| Week | Try | Expected benefit |
|------|-----|------------------|
| W1 | `--model unet_energy_attn` (already coded) | catches long-range energy-bin coupling |
| W2 | `--model unet_residual --temporal-window 3` | wider temporal context |
| W3 | `--model convlstm --temporal-window 1` | proper recurrence over time |
| W4 | `--model unet_residual --physics-head` | moments-aware joint loss |
| W5 | I-JEPA pre-training of the encoder, then fine-tune for inpainting | learned representation prior |

Each week: 1 architecture, ~1 day of CPU on the existing month-of-data, compare
on the standard 120-s validation sample. Keep what wins, drop what doesn't.

## Phase 5 — paper-grade evaluation  *(~2 weeks)*

Once Phase 3 is done and Phase 4 has produced a clear winner:

1. **Multi-event validation** — pick 5-10 known events from the MMS literature
   (magnetopause crossings, reconnection events, jet braking). Reconstruct each
   with the final model. Show the model preserves the known signature.
2. **Comparison vs. literature baselines** — pick one published gap-filling
   method from the heliophysics-ML literature, implement it, run on the same
   validation set. Numbers go in the paper's results table.
3. **Final ablation table** — plain U-Net / + temporal / + features / + residual /
   + physics-head. One row each on the same evaluation.
4. **Final figures**: filmstrip, density time-series with `true (data-rich)` line,
   per-energy-spectrum, embedding scatter, temporal-variance diagnostic, learning
   curves, ablation heatmap.

## Phase 6 — paper writing  *(~3 weeks)*

Follow the structure in `PUBLICATION.md`. Submit arXiv preprint first
(immediately, as soon as Phase 5 finishes), then AGU26 abstract (deadline
30 July), then ML4PS workshop paper (~Sept).

## Total time estimate (revised — 1-week training budget)

User constraint: max **1 week of wall-clock training time**, MMS1 only for
training, MMS2/3/4 for validation only.

| Phase | CPU-time | Wall-clock |
|---|---|---|
| 0 (last-mile fixes) | ~3 hours | this week |
| 1 (v7 trial) | (subsumed into Phase 2's stream which uses the v7 architecture) | — |
| 2 (stream MMS1 Mar+Apr 2024, ~900 files) | ~30 hours | **~1 week wall-clock** |
| 3 (MMS2/3/4 validation only) | ~3 hours | 1 day |
| 4 (architecture iterations, parallel) | ~10 hours | overlaps Phase 2 |
| 5 (multi-event paper-grade eval + literature baselines) | ~15 hours | 2 weeks |
| 6 (paper writing) | ~30 hours of writing | 3 weeks |
| **Total to first submission** | | **~6-7 weeks** |

Phase 2 (the heavy stream) was just launched (`outputs/stream_long_mms1/`).
Architecture is residual-temporal U-Net (v7 design); throttling + retry are in
place; resume support handles laptop sleep. Run it for 1 week, then evaluate
whatever was completed.

## Practical workflow each session

1. Open the laptop, check `outputs/<latest_run>.log` and `git log --oneline -5`.
2. If a run died: `python src/stream_train.py ... --resume` (one line; rest is identical).
3. If a run finished: re-run the 120-s comparison, commit figures, update MODEL_CHANGELOG.
4. Pick a date range / spacecraft for the next run. `nohup python src/stream_train.py ... &`.
5. Close laptop, come back later. Resume support handles sleep.
