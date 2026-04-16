# Brief for agents: methodology overview — Rank Surprise burst detection (three stages)

Use this document when you are given **`BurstDetection/`** inside **`burstDetection_sangerlab`** and asked to draft or review a **methods** section. **In scope:** (1) **signal / unit filtering** applied before any burst detection, and (2) the **three-stage Rank Surprise (RS)** hierarchy: **unit → region → network**. **Out of scope for this paper:** Max-ISI or alpha–mean-ISI methods, pooling, co-activity analyses, region-exclusion or other ablation studies, proxy analyses, channel reduction / ROC / validation side projects — do not describe those as part of the core methodology even if they appear elsewhere in the repo.

---

## What this project is (publication focus)

The pipeline implements **hierarchical burst detection using Rank Surprise only** (Gourevitch & Eggermont, 2007; see citation in `pipeline/detection.py`). The intended scientific emphasis is **synchronous bursting across anatomy**: single units are screened and then bursts are detected at **unit** level, then **region** level (onsets of unit bursts within each anatomical region), then **network** level (onsets pooled across included units on a hemisphere), each stage using **RS** with its own parameters.

---

## Repository layout

- **Workspace root:** `burstDetection_sangerlab/`
- **Code package:** `burstDetection_sangerlab/BurstDetection/`

Assume the working directory is **`BurstDetection/`** for imports and data paths.

---

## Read order (code trace for the methodology paper)

1. **`config.py`** — **Filtering:** `SNR_MIN`, `SNR_MAX` (optional upper bound via `snr_fails_filter` in `pipeline/stats.py`), `FR_MIN_HZ`, and related burst gates (`MIN_SPIKES_IN_BURST`, `MIN_BURST_DURATION`, `APPLY_MIN_BURST_DURATION_ALL_STAGES`, etc.). **RS only:** ensure `rankSurprise_toggle` is the active method for the runs you describe (`maxisi_toggle` / `alpha_meanisi_toggle` are **not** part of this methodology document). **RS triple:** parameters for **cluster (unit)**, **region**, and **network** stages (`RS_alpha_*`, `RS_Limit_*`, `RS_Percentile_Limit_*`, `RS_NETWORK_ONSETS_TOGGLE`, region-stage toggle if present). **Anatomy:** `infer_region` / `COMBINE_NUMBERED_REGIONS` / `REGION_COLORS` affect how electrodes map to regions for stage 2.
2. **`pipeline/stats.py`** — **`build_cache`**: builds **`STATS`** from the recording; only clusters passing **FR** and **SNR** rules enter the index used for all later RS stages. **`snr_fails_filter`** is the single SNR rule (NaN, below `SNR_MIN`, above `SNR_MAX` if set).
3. **`pipeline/detection.py`** — **`RS_detect_burst`**: core RS on spike trains or on sorted **onset** times. **`rs_burst_detection`**: orchestrates **stage 1 (unit)** → **stage 2 (region)** → **stage 3 (network)** for combined / left / right, writes timing CSVs under **`burst_timings/`** via **`pipeline/burst_paths.py`**. **`compute_spans_and_bars`**: maps RS windows to reported spans (e.g. median onset/end vs onset + median length, per config).
4. **`main.py`** — Entry: load data → **`build_cache`** → **`rs_burst_detection`** (and any figure export). For **methodology text**, trace only the path that uses **Rank Surprise** and **`STATS`** from `build_cache`. Ignore branches devoted to other studies unless verifying file locations.
5. **`pipeline/utils.py`** — **`iter_units_from_stats`**: only **`STATS.index`** units participate in RS. **`infer_region`**, **`split_spike_struct_by_side`**, recording length helpers.
6. **`pipeline/burst_paths.py`** — Exact **output filenames** for unit / region / network CSVs (e.g. under `run_dir/burst_timings/`).

**Optional context:** `CONTEXT.md` / `README.md` for day-to-day running of the repo; your **paper** should still be anchored on the sections above, not on auxiliary analyses.

---

## Methodological story (what the paper should reconstruct)

### A. Pre-detection filtering (before any Rank Surprise)

Clusters must satisfy **minimum firing rate** (`FR_MIN_HZ`) and **SNR bounds** (`SNR_MIN`, and `SNR_MAX` if not `None`), with NaN SNR excluded. Optionally require at least two spikes in the window used for stats. Only passing clusters are rows in **`STATS`**; **`iter_units_from_stats`** restricts all RS stages to those units. This block is part of the **published methodology** — it is not optional narrative.

### B. Stage 1 — Unit-level bursts (Rank Surprise on spike trains)

For each included unit, **RS** is run on the spike train (millisecond times). Output: burst intervals, spike counts, and RS statistics. Thresholding uses the **cluster / stage-1** RS parameters in `config.py`.

### C. Stage 2 — Region-level bursts (Rank Surprise on unit-burst onsets)

Unit bursts are assigned an anatomical **region** from the electrode label. Within each region, **onset times** of unit-level bursts are collected and **RS** is applied again (**region** RS parameters). That yields **region-level** burst windows. Spans for tables/figures follow **`compute_spans_and_bars`** (and related config, e.g. minimum unique channels at region level if enabled).

### D. Stage 3 — Network-level bursts (Rank Surprise on pooled onsets)

**Onset times** from unit bursts (scope as implemented in `rs_burst_detection`, e.g. per hemisphere / combined) are passed to **RS** a third time with **network** parameters. This stage targets **cross-channel / cross-region temporal alignment** of bursts. Post-RS gates may include **minimum duration**, **minimum unique channels**, etc., as in `config.py`. Reported event times/spans again use **`compute_spans_and_bars`** where applicable.

---

## Units and naming

- Internal times for detection and **`STATS`** are in **milliseconds**; figures may use seconds — state this explicitly in the paper.
- Electrode strings encode region and side; parsing is in **`pipeline/utils.py`** (`parse_electrode`, `infer_region`).

---

## Framing “novelty” (for the author, not a claim about RS itself)

- **Established:** Rank Surprise for burst detection (cite Gourevitch & Eggermont, 2007).
- **Emphasize as contribution:** A **clear three-stage RS cascade** — spikes → unit bursts → regional synchrony on unit-burst onsets → network-scale synchrony on pooled onsets — with **separate, reportable RS hyperparameters** at each stage, preceded by **explicit unit-quality filtering**. Do **not** anchor the novelty claim on Max-ISI, alpha–mean-ISI, proxy, or ancillary studies.

---

## Parameters to report in the paper

At minimum: **SNR** bounds (and that NaN is excluded), **FR** cutoff, **MIN_SPIKES_IN_BURST**, **MIN_BURST_DURATION** and whether it applies only at network or at all stages, the **three RS parameter sets** (unit / region / network: alpha or alpha-percentile equivalents, limits, percentile caps), any **minimum unique channel** requirements and **span definition** flags (e.g. onset + median length vs median onset/end). Mirror symbols in **`build_run_params`** / **`run_tag_from_params`** only insofar as they correspond to this RS pipeline.

---

## Checklist before asserting “what the code does”

- [ ] Confirm **`rankSurprise_toggle`** is on and the other method toggles are off for the runs you describe.
- [ ] Trace **`build_cache`** → **`STATS`** and confirm filtering matches the text (grep **`snr_fails_filter`**, **`FR_MIN_HZ`**).
- [ ] Trace **`rs_burst_detection`**: unit loop, region loop, network loop; note combined vs L/R outputs.
- [ ] Confirm CSV names and folders from **`burst_paths.py`** (not legacy filenames from old docs).
- [ ] Do **not** describe pooling, co-activity, region exclusion, proxy, or channel reduction as core methodology unless the user explicitly expands scope later.

---

## One-line instruction for a new agent session

> Work in `burstDetection_sangerlab/BurstDetection/`. Read `PAPER_METHODOLOGY_AGENT_BRIEF.md`, then trace `config.py` (RS + filtering only) → `pipeline/stats.py` (`build_cache`, `snr_fails_filter`) → `pipeline/detection.py` (`RS_detect_burst`, `rs_burst_detection`, `compute_spans_and_bars`) → `main.py` and `pipeline/burst_paths.py`. Draft a **methods overview** of **pre-filtering plus three-stage Rank Surprise** (unit, region, network). Ignore Max-ISI, alpha–mean-ISI, pooling, co-activity, region-exclusion studies, proxy, and channel-reduction code.

---

*Orientation only; not a substitute for the code or for final statistical wording in a manuscript.*
