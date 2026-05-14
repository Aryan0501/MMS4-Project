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

## Recommendation

**Submit to two venues in parallel:**

1. **ML4PS @ NeurIPS 2026** — 4-page short paper, ~Sept 2026 deadline. Get the
   method written up in front of an ML audience.
2. **AGU26 Fall Meeting abstract** — 30 July 2026 deadline, poster format. Get
   the method in front of the space-physics community for feedback.

Save journal submission (JGR: ML&C or ESS) until after iterating on:
- Train on ≥ a few hundred files spanning multiple plasma regimes
- Close the residual-temporal "model is too flat in time" gap (already coded —
  needs GPU)
- Reach < 10 % median fractional density error
- Add one concrete science demonstration (recover a known feature through a gap)
