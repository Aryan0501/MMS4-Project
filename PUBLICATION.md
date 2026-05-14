# Publication shortlist

Candidate venues for this work, given current maturity (working pipeline, ~10×
pixel-MSE / ~2.5× moments improvement over interpolation baseline, ~20% median
fractional density error on held-out burst data, model is mask-aware but still
~3× temporally flatter than truth in masked regions). Ordered by fit + timing.
Sources are the official CFPs/journal pages as of 2026-05-13.

## Tier 1 — best fit (workshop / poster, near-term)

| # | Venue | Next deadline | Format | Notes |
|---|---|---|---|---|
| 1 | **ML4PS @ NeurIPS 2026** ([ml4physicalsciences.github.io](https://ml4physicalsciences.github.io/)) | ~Sept 2026 | 4 pp NeurIPS template, OpenReview | Perfect — physical-sciences ML, accepts in-progress work, lightly reviewed. NeurIPS in San Diego, Dec 2026. **This is the recommended target.** |
| 2 | **AGU26 Fall Meeting** ([agu.org/annual-meeting](https://www.agu.org/annual-meeting)) | ~30 Jul 2026 | 1-paragraph abstract → poster | SM (space physics/aeronomy) section runs many MMS sessions. Almost-guaranteed acceptance as poster. Best way to get community feedback before committing to a journal. |
| 3 | **EGU 2027 General Assembly** ([egu.eu/meetings/general-assembly](https://www.egu.eu/meetings/general-assembly/)) | ~15 Jan 2027 | 1-page abstract → poster/oral | ST (Solar-Terrestrial) division, Vienna, Apr 4–9 2027. Low bar, European audience. |
| 4 | **SPAICE 2027** ([spaice.esa.int](https://spaice.esa.int/)) | ~Mar 2027 (CFP for 2026 cycle closed 1 Mar 2026) | 4 pp excl refs, OpenReview | ESA's AI-in-space conference; bookmark for next cycle. |

## Tier 2 — broader ML-for-science workshops

| # | Venue | Next deadline | Format |
|---|---|---|---|
| 5 | **AI4Science @ ICML 2026** ([ai4sciencecommunity.github.io/icml26](https://ai4sciencecommunity.github.io/icml26.html)) | ~late May 2026 | 4–8 pp ICML template — 2026 theme leans agentic, less ideal fit |
| 6 | **AI4Science @ NeurIPS 2026** | ~Sept 2026 | 4 pp — sister to ML4PS, broader scope |

## Tier 3 — journals (when the headline error drops below ~10 % and we have a science demonstration)

| # | Journal | Page limit | Notes |
|---|---|---|---|
| 7 | **JGR: Machine Learning and Computation** ([jgr-machinelearning-submit.agu.org](https://jgr-machinelearning-submit.agu.org/)) | 25 PUs | New AGU journal explicitly for ML methods in geophysics — strongest journal fit. |
| 8 | **Earth and Space Science** (AGU, gold OA, [agupubs](https://agupubs.onlinelibrary.wiley.com/journal/23335084)) | 25 PUs | Has "Models and Machine Learning: Applications" track, rolling submission, lower bar than JGR:SP. |
| 9 | **Frontiers Astronomy & Space Sciences** ([frontiersin.org](https://www.frontiersin.org/journals/astronomy-and-space-sciences)) | flexible | Has prior heliophysics-ML Research Topics; rolling. |
| 10 | **JGR: Space Physics** | no hard cap | Higher bar; better once we have a science result rather than just a method. |
| 11 | **GRL** | 12 PUs (~4500 words) | Wants high-impact short papers; method-only at 20 % error likely below bar. |

## Tier 4 — applications journals (the right place for a first paper)

This is an **ML-applications** paper, not a core space-physics result. For a
first-time author, applications venues are friendlier: editors expect technical
methodology + a real dataset, not a deep new physics insight. Reviewers tend to
be other ML/applied-ML people who'll engage with the method on its own terms.

| # | Venue | Page limit | Fee | Notes |
|---|---|---|---|---|
| 12 | **arXiv preprint** ([arxiv.org](https://arxiv.org/), `cs.LG` + `astro-ph.SR` cross-list) | none | free | **Do this first regardless.** Gets a citable DOI in ~24 h, no review barrier, lets you get feedback / share the link before formal submission. Costs nothing, blocks nothing. |
| 13 | **IEEE Access** ([ieeeaccess.ieee.org](https://ieeeaccess.ieee.org/)) | ~10 pp typical | $1750 (OA) | Broad-scope ML+applications, fast review (~5 weeks), open access, friendly to first-time authors, well-indexed. **Top recommendation for the first journal target.** |
| 14 | **Scientific Reports** (Nature, [nature.com/srep](https://www.nature.com/srep/)) | flexible (~10 pp) | $2790 (OA) | Very broad — accepts ML applications across all sciences, helpful editors, prestigious umbrella, ~50 % accept. |
| 15 | **Remote Sensing** (MDPI, [mdpi.com/journal/remotesensing](https://www.mdpi.com/journal/remotesensing)) | flexible | $2700 (OA) | Many published papers on satellite-data ML inpainting/gap-filling — directly your topic. Fast turnaround (~25 days first decision). Some reputational concerns about MDPI but the venue is well-cited and friendly. |
| 16 | **Frontiers in Big Data** ([frontiersin.org/journals/big-data](https://www.frontiersin.org/journals/big-data)) | flexible | $2950 (OA) | Editor-driven, helpful pre-review feedback. Good for "petabyte-scale streaming pipeline" framing. |
| 17 | **PLOS ONE** ([journals.plos.org/plosone](https://journals.plos.org/plosone/)) | flexible | $1805 (OA) | Method judged on rigor, not novelty/impact. Forgiving for first-time authors. |

## Recommended path for a first-time author

**Now / this week (free, low-stakes):**
1. **Write up an arXiv preprint** (4-6 pages). Gets a citable URL, lets you put
   "preprint at arXiv:XXXX.XXXXX" on your CV immediately. **No review, no risk.**
   Use the JMLR / NeurIPS LaTeX template. Cross-list to `astro-ph.SR`.

**Within 1-2 months (free, low-effort):**
2. **Submit AGU26 abstract** (30 July 2026 deadline) — costs nothing, takes
   1-2 hours to write the paragraph, almost-guaranteed poster acceptance, gets
   you in front of the MMS community. The Fall Meeting itself (Dec 2026) is
   where you'd present.

**Sept 2026 (workshop paper, low bar):**
3. **ML4PS @ NeurIPS 2026** — 4-page short paper. Even if rejected, the writing
   exercise produces the foundation of the eventual journal version. Reviewers
   are constructive at workshops.

**Oct-Dec 2026 (first journal, after some iteration):**
4. **IEEE Access** — convert the arXiv preprint + workshop paper to a 10-page
   journal article. Apply lessons from the workshop reviews. Open-access fee
   is ~$1750; check your university for an OA-fee waiver (most universities
   have institutional agreements that cover IEEE Access).

## First-time-author cheat sheet

- **Paper structure** that works for ML applications: Abstract / 1. Introduction
  (motivate the problem & gap in literature) / 2. Data (the MMS/FPI archive,
  what you have, the inpainting problem) / 3. Method (your U-Net architecture,
  data-aware mask, gt-only loss, streaming pipeline) / 4. Experiments
  (ablation table, the moments-validation finding, the temporal-variance
  diagnostic) / 5. Limitations & next steps / 6. Conclusion / References.
- **Length-by-section, ~10 pp paper:** intro 1.5 pp, data 1 pp, method 2-2.5 pp,
  experiments 3 pp, limitations 0.5 pp, conclusion 0.5 pp, refs ~1 pp.
- **Figures**: aim for 4-6. The reconstruction filmstrip + the temporal-variance
  diagnostic + a results table + a learning-curve plot already gets you most of
  the way.
- **Reviewers will ask**: "Why this architecture vs. X?" "Why these baselines
  and not Gaussian processes / Kriging / matrix completion?" "How does it
  generalise across plasma regimes?" "What's the failure mode?" The honesty
  in `RESULTS.md` and `MODEL_CHANGELOG.md` already addresses most of this.
- **Reviewer time**: typical applications journal turnaround is 2-3 months for
  the first decision, then 1-2 rounds of revision over another 2-4 months.
  Plan for 4-9 months from submission to acceptance for a journal; 1-3 months
  for a workshop.
- **Don't be afraid of rejection.** First papers often get rejected; the reviews
  are the most valuable feedback you'll ever get. "Reject and resubmit" is a
  normal cycle.

## What still needs to happen technically before submission

- Train on ≥ a few hundred files spanning multiple plasma regimes
  (`TRAINING_PLAN.md` lays out the phased schedule)
- Land the residual-temporal architecture (already coded, see MODEL_CHANGELOG v7)
- Reach < 10 % median fractional density error (currently 20 %)
- One science demonstration: recover a known event feature through a synthetic gap
