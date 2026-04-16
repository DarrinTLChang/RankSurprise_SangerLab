# Paper methodology context (BurstDetection)

This file is intended to capture paper-ready methodological details and rationale behind key analysis choices in `BurstDetection/`.

## RankSurprise at multiple scales (unit → region → network)

### Baseline issue with compiled onset trains

At higher scales we compile lower-level burst **onset times** into a single event train:

- **Region level**: compile unit/channel burst onsets within a region into one regional onset train.
- **Network level**: compile regional burst onsets into one network onset train.

If we then apply RankSurprise (RS) directly to the compiled train using only its own inter-event interval (ISI) structure as an implicit baseline, repeated cross-channel synchrony can be under-emphasized. For example, if many channels burst near the same repeated times (e.g. clustered around 1s, 3s, and 10s), the compiled train itself will contain dense clusters at those times. When RS is computed on that compiled train alone, those clusters can become “typical” for the compiled train and appear less surprising, even though the repeated multi-channel synchrony is precisely the phenomenon of interest.

### Offset-based independence-preserving reference (planned change)

To quantify how surprising the observed clustering is **relative to an independence model**, we define an offset-based reference construction for both region- and network-level RS:

1. **Observed compiled train** (unchanged):
   - Region level: merge unit/channel onset trains in the region into one event train.
   - Network level: merge region onset trains into one event train.

2. **Reference (null) compiled train** (new):
   - For each contributing source train (unit/channel for region; region for network), apply a **random constant time offset** \(\Delta_i\) that is unique to that source.
   - Merge the offset trains into a single reference event train.
   - The constant offset breaks cross-source synchrony while preserving within-source timing statistics.

3. **Reproducibility**:
   - Offsets are sampled from a pseudo-random number generator with a **fixed, documented seed**, so results are exactly repeatable.
   - The seed should be reported (or deterministically derived) and kept constant across re-runs of the same analysis.

4. **Units and range**:
   - Event times are represented in **milliseconds** in the pipeline.
   - The paper must specify the offset sampling range (e.g. uniform over \([0, T)\) ms or another stated interval) and any boundary handling (wrap vs clip), because these affect the null distribution.

5. **RS comparison**:
   - The RS statistic at region/network level should be evaluated with respect to the offset-based reference so that repeated multi-source synchronous clustering yields a stronger “surprise” signal than would be obtained from the compiled train alone.

### Notes for reporting

- Clearly distinguish **what is detected** (bursts/onsets in the observed data) from **what is used as the null** (offset-based merged trains).
- Report how many sources contribute to each region/network reference and how offsets are drawn.
- Report the RNG seed strategy and the offset range.

