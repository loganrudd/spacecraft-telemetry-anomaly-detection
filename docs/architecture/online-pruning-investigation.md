# Online Anomaly Pruning — Investigation Note (Option A)

**Status:** deferred. Captured 2026-05-30 so it can be picked up cold.
**Owner:** unassigned.

## Why this note exists

We added Hundman §3.3 false-positive *pruning* to the offline scoring pipeline
(`model/scoring.py::prune_anomalies`). On the local single channel it is highly
effective: a peak-error gate collapses ~10,000 false-positive segments to a
handful while keeping the strongly-separated true anomalies.

But pruning is **retrospective and batch**: it ranks the peak smoothed error of
*every* predicted sequence across the whole series, then drops the low-peak tail
that blends into the noise floor. The online serving engine
(`api/inference.py::ChannelInferenceEngine`) is **streaming and O(1)/tick** — at
time *t* it has no access to future sequences and therefore no global ranking to
prune against.

To keep train/serve parity we chose **Option B** (see decision below): HPO and
the *headline* scoring metric use the **un-pruned** pipeline (exactly what
serving produces); pruning is reported only as an offline "ceiling." This note
records what Option A (real online pruning) would take, if we ever decide the
ceiling is worth chasing in the live path.

## Decision in force (Option B)

- `ray_training/tune.py::SEARCH_SPACE` tunes 4 params; **`prune_min_decrease`
  is not tuned.** Objective = segment-overlap F0.5 on the un-pruned pipeline.
- `_TUNABLE_SCORING_FIELDS` (`ray_training/runner.py`) excludes
  `prune_min_decrease`, so a tuned config can never carry it into serving.
- `model/scoring.py::score_channel` reports un-pruned headline metrics +
  `pruned_seg_*` ceiling metrics (fixed `settings.model.prune_min_decrease`).
- Serving ignores pruning entirely (`ScoringParams` in `model/io.py` has no
  prune field).

Revisit Option A only if the gap between the served (un-pruned) segF0.5 and the
offline pruned ceiling is large enough to matter for the demo/dashboard.

## Why naive online pruning fails

The batch algorithm needs, per channel: the set of all predicted sequences,
their peak smoothed errors, and the max non-flagged error (the noise baseline).
Online we have none of the future ones. Two specific breakages:

1. **No global peak ranking.** The §3.3 reset logic walks peaks top-down and
   prunes the tail below the last significant relative drop. That ordering is
   undefined until the series ends.
2. **Retroactive un-flagging.** Pruning can *reclassify an already-emitted
   anomaly* as nominal once later sequences reveal it was part of the noise
   tail. SSE events are already on the wire — we cannot un-send them.

## Candidate approaches to investigate

### A1. Static peak-error gate (simplest)
Replace the dynamic rank with an absolute floor: emit an anomaly only if the
sequence's running peak smoothed error exceeds `g`. Derive `g` offline from the
pruning behaviour (e.g. the noise-segment peak p99, or the smallest kept-peak
across channels in a subsystem).
- **Pros:** trivial to add to the streaming engine; one new scalar; no
  retroactive edits.
- **Cons:** static threshold ≠ dynamic per-series rank; needs per-channel or
  per-subsystem calibration; drifts if the noise floor is non-stationary
  (channel_22's noise *is* bimodal — see below).

### A2. Rolling-window pruning
Run §3.3 over a trailing window of the last *N* sequences instead of the whole
series. Emit with a deliberate lag of one sequence so a just-closed sequence can
be pruned before it is reported.
- **Pros:** closer to the batch semantics; adapts to non-stationarity.
- **Cons:** introduces detection latency (one sequence); window size is a new
  hyperparameter; still approximate near window edges.

### A3. Two-tier emit (provisional → confirmed)
Emit a *provisional* anomaly immediately, then a *confirmed/retracted* event
once enough trailing context exists to run pruning on that sequence.
- **Pros:** no information lost; dashboard can show "pending" vs "confirmed."
- **Cons:** protocol + UI work (event types, retraction handling); most complex.

## Evidence to start from (channel_22, ESA-Mission1)

Full diagnostic lives in the conversation that produced this note; key facts:
- 8 labelled anomaly segments; 5 are **forecast-invisible** (peak smoothed error
  0.28–0.52, below the noise median in the high-error regime) — *no* pruning
  scheme recovers these; they need a different detector and are out of scope per
  project rules.
- 3 segments are detectable (peak 2.5–7.9); 2 separate cleanly above the
  **noise-segment peak ceiling of 2.72**, which is exactly what makes a static
  gate (A1) viable for the strong ones.
- The noise floor is **non-stationary / bimodal** (quiet regime ~0.15, loud
  regime ~2–5). A single global `g` (A1) will mis-serve one regime — favours
  A2/A3 if we need both regimes.

## Open questions before implementing
1. Is the served-vs-ceiling segF0.5 gap actually large on a *multi-channel*
   subsystem, or is channel_22 an outlier? (Need the subsystem sweep first.)
2. Does the dashboard requirement tolerate detection latency (A2) or event
   retraction (A3)? If neither, A1 is the only option.
3. Can `g` be calibrated per subsystem offline and shipped in `ScoringParams`
   without violating the "tuned config = what serving uses" parity rule?

## Touch points if/when implemented
- `api/inference.py::ChannelInferenceEngine` — emit logic, any buffering/lag.
- `model/io.py::ScoringParams` + `load_scoring_params` — carry the gate param.
- `model/scoring.py` — keep batch `prune_anomalies` as the source of truth the
  online approximation is validated against.
- Tests: an integration test asserting the online approximation matches batch
  pruning within tolerance on a fixture series.
