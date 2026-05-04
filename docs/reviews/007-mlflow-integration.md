# Big Review — Phase 7: MLflow Integration

**Branch:** `phase07` vs `main`
**Date:** 2026-05-04
**Scope:** 32 files changed, 2089 insertions, 147 deletions

Sections: Architecture · Code Quality · Tests · Performance. Up to 4 issues per section. Each issue: file/line ref, 2–3 options (including "do nothing"), effort/risk/impact, recommendation.

> **Note:** Pre-review fix — `mlflow*.db` (the two committed SQLite blobs) was
> added to `.gitignore`. `mlruns/` and `mlartifacts/` were already gitignored.
> No further action.

---

## 1. Architecture

### A1. Filesystem-primary + MLflow-as-observability is the wrong source-of-truth pattern

The current architecture writes most artifacts to the filesystem via `model/io.py`
(`save_model`, `save_errors`, `save_threshold`, `save_metrics`, `save_train_log`,
`save_norm_params`) and *additionally* registers a copy via MLflow:
- [training.py:223-232](src/spacecraft_telemetry/model/training.py#L223-L232) calls `save_model`, `save_norm_params`, `save_train_log` (filesystem)
- [training.py:240-245](src/spacecraft_telemetry/model/training.py#L240-L245) calls `register_pytorch_model` → `mlflow.pytorch.log_model` (MLflow artifact store, separate copy)
- [scoring.py:272-278](src/spacecraft_telemetry/model/scoring.py#L272-L278) calls `save_errors`, `save_threshold`, `save_metrics` (filesystem)
- [scoring.py:311-314](src/spacecraft_telemetry/model/scoring.py#L311-L314) `log_artifact_bytes(metrics.json)` (MLflow, redundant copy)

This is a deviation from MLflow's intended pattern. MLflow expects the
configured artifact store (local FS, GCS, S3) to be the **canonical** source-of-truth
for run-scoped artifacts. Filesystem files outside the MLflow artifact store are
not tracked, not versioned, not visible in the UI, and not loadable via
`models:/{name}/{stage}` URIs without custom plumbing. Phase 9 (FastAPI) was
going to need that custom plumbing; Phase 11 (cloud) was going to need it
twice (filesystem `gs://` *and* MLflow `gs://`).

The right pattern: write **once** through MLflow's logging APIs
(`mlflow.pytorch.log_model`, `mlflow.log_artifact`, `mlflow.log_dict`).
Configure the artifact store via the tracking URI / `MLFLOW_ARTIFACTS_DESTINATION`.
Reads go through `mlflow.pytorch.load_model` (registry URIs) or
`mlflow.tracking.MlflowClient().download_artifacts` (run-scoped non-model files).

**User decision: full pivot.** `model/io.py` becomes a thin MLflow client
wrapper or is deleted; downstream readers (`_prepare_channel_data` in tune.py,
FastAPI in Phase 9) load artifacts via MLflow APIs.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Full pivot — drop `save_*` filesystem path; use MLflow logging APIs exclusively; migrate read sites in tune.py + tests | L (~1 day) | Medium — touches every artifact read site; needs test-fixture migration | Single source of truth; unblocks Phase 9 registry-based loading without custom path plumbing; eliminates `gs://` indirection in `model/io.py`; simplifies mental model |
| Hybrid — model weights & run-scoped artifacts via MLflow; Spark-produced intermediates (Parquet, normalization params) stay on filesystem | M (~half day) | Low | Smaller refactor but `model/io.py` keeps a partial role |
| Document the deviation, defer | S | Low | Tech debt; Phase 9 / 11 inherit it |

**Recommendation: Option 1 (full pivot).** Per user decision.

**Scope of change:**
- `model/io.py` — replace `save_*` and `_write_bytes`/`_read_bytes` with MLflow log calls or delete
- `model/training.py` — drop the `save_*` calls; rely on `mlflow.pytorch.log_model` + `mlflow.log_dict` for `train_log.json`, `model_config.json`, `normalization_params.json`
- `model/scoring.py` — drop `save_errors`/`save_threshold`/`save_metrics`; use `mlflow.log_artifact` for the bytes that need persisting; metrics already go to `mlflow.log_metrics`
- `ray_training/tune.py` `_prepare_channel_data` — load `errors.npy` from MLflow (download artifact from the most recent scoring run for the channel) instead of from `paths.errors`
- `ray_training/runner.py` — `score_all_channels` artifact discovery shifts from filesystem `errors.npy` checks to MLflow run lookups
- `tests/model/conftest.py` and downstream tests — fixtures that build a fake `models/{mission}/{channel}/` directory tree need to be replaced with MLflow-stored equivalents (or use the real MLflow client against a tmp_path SQLite store)
- `.gitignore` — `models/` directory becomes obsolete (already gitignored); `mlruns/` / `mlartifacts/` is now the sole local artifact root

**Cascade effects on later findings.** This pivot reframes several other items in this review. Cross-references:
- **A2** — partially dissolves: subsystem becomes a run tag, not a lookup that `model/scoring.py` performs.
- **A4** — *more* important: `tracking_uri` is now load-bearing for *all* artifact I/O.
- **C2 + T2** — more important: registry semantics matter more when the registry is the only source.
- **P1** — partially dissolves with A2.
- **P2** — recommendation flips to "drop the dual-write" (was "document the dual-write").
- **P3** — MLflow lookup is now the *correct* paradigm; fix is just to bound the search to the current sweep.
- **T1, T3, T4** — unchanged but proportionally more important.

### A2. Circular import dance between `runner.py` ↔ `model/training.py` and `model/scoring.py`

[training.py:122-126](src/spacecraft_telemetry/model/training.py#L122-L126) and [scoring.py:282-286](src/spacecraft_telemetry/model/scoring.py#L282-L286) lazy-import `load_channel_subsystem_map` from `ray_training.runner` inside the function body, wrapped in `try/except: pass`. This means: (1) `model/` now structurally depends on `ray_training/` (layering inversion — per `.claude/rules/pytorch.md` and `fastapi.md`, `model/scoring.py` should be importable without pulling in Ray for Phase 9 FastAPI); (2) errors in subsystem lookup are silently swallowed; (3) the lazy import hides a real circular-dep design smell.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Move `load_channel_subsystem_map` to a neutral location (`model/metadata.py` or `core/metadata.py`) | S (~30 min) | Low | Removes the circle, restores clean layering, removes 2 swallowed exceptions |
| Pass `subsystem` in as an argument from the caller (CLI / Ray task) | M (~1 hour) | Low | Subsystem becomes an explicit dependency; function easier to test |
| Do nothing | 0 | Future refactors will re-trip the circle; FastAPI Phase 9 may transitively pull in Ray | None |

**Recommendation: Option 1.** Cheap, decoupling, and the data is metadata not behaviour — belongs in a non-runner module anyway.

**Note:** This finding partially dissolves under A1's MLflow-primary pivot if subsystem becomes a run tag at write-time rather than a lookup at score-time. Re-evaluate after A1 lands.

### A3. `configure_mlflow()` mutates process-global state

[training.py:95](src/spacecraft_telemetry/model/training.py#L95) and [scoring.py:236](src/spacecraft_telemetry/model/scoring.py#L236) call `configure_mlflow(settings)` which sets a process-global tracking URI. When a Ray worker process is reused for multiple tasks (Ray's default — `num_cpus=1` tasks share workers), the second task overwrites the first's tracking URI. Today nothing exercises this; concurrency-correctness bug waiting in ambush if multi-mission processes become a thing.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Document the invariant + comment | S | Invariant true today, easy to violate later | Cheap insurance |
| Refactor to per-call URI scoping (`mlflow.MlflowClient(tracking_uri=…)`) | M-L (~2-3 hours) | Medium — MLflow's fluent API leaks globals | Actually concurrency-safe |
| Cache + assert in `configure_mlflow` (store URI on first call, raise if a later call differs) | S (~15 min) | Low | Fails loudly on the bug |

**Recommendation: Option 1 + Option 3.** 15-min insurance policy. Option 2 is worth it at Phase 11 (multi-mission cloud deploy); not now.

### A4. `_with_abs_paths` doesn't resolve `mlflow.tracking_uri`

[runner.py:148-171](src/spacecraft_telemetry/ray_training/runner.py#L148-L171) resolves `processed_data_dir`, `artifacts_dir`, and `raw_data_dir` to absolute paths before `ray.put(settings)`. But `mlflow.tracking_uri = "sqlite:///mlflow.db"` is a *relative* path inside a URI — Ray workers running from a session temp dir will silently create a fresh `mlflow.db` in the wrong place. The driver process logs to repo-root `mlflow.db`; Ray-task training runs may be writing to a session-dir DB nobody sees.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Detect `sqlite:///` in URI inside `_with_abs_paths`, resolve, rewrite | S (~30 min) | Low | Prevents silent-loss bug |
| Document absolute-tracking_uri requirement + assertion in `configure_mlflow` | S (~10 min) | Low | Pushes burden to user, clear error |
| Resolve relative SQLite URLs in `core/config.py` at load time using `_REPO_ROOT` | S (~10 min) | Low | Silent fix; matches "MLflow file lands at repo root" expectation |

**Recommendation: Option 3.** Pair with Option 2 as belt-and-suspenders if env-var expansion isn't trusted.

**Note:** *More* important under A1's MLflow-primary pivot, since `tracking_uri` becomes load-bearing for *all* artifact I/O, not just metric metadata.

---

## 2. Code Quality

### C1. CLI `mlflow_promote` reimplements `registry.promote()` — DRY violation

[cli.py:1028-1050](src/spacecraft_telemetry/cli.py#L1028-L1050) duplicates the version-resolution + `transition_model_version_stage` logic from [registry.py:63-98](src/spacecraft_telemetry/mlflow_tracking/registry.py#L63-L98). Two implementations now drift independently — `registry.promote()` raises `ValueError` with a helpful message, the CLI raises `ClickException` with a different message. `registry.promote()` is in `__all__` but nothing calls it.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| CLI calls `registry.promote()`, catches `ValueError` → `ClickException` | S (~10 min) | Low | One source of truth; deletes ~15 lines and an `MlflowClient` import from `cli.py` |
| Delete `registry.promote()`, keep CLI logic | S | Medium — loses programmatic API for Phase 9; would need to delete its tests too | Smaller `mlflow_tracking` surface, narrower API |
| Do nothing | 0 | Two implementations will drift | None |

**Recommendation: Option 1.** Textbook fix; CLI should be a thin shell over the API.

### C2. `register_pytorch_model` fallback can return wrong run's version

[registry.py:58-60](src/spacecraft_telemetry/mlflow_tracking/registry.py#L58-L60): when the `run_id` filter returns empty, returns the first version of any run for that model name. Concurrent training (e.g. parallel Ray task on the same channel during a re-train) could silently return the wrong `ModelVersion`. The training caller (training.py:240-245) doesn't use the return value, so the fallback exists for nobody.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Remove fallback; return `None` when run_id search is empty (+ update test_registry.py:80-95) | S (~5 min) | Low | Removes a footgun |
| Sort `all_versions` by creation timestamp DESC, document the race | S (~15 min) | Medium — still wrong under concurrency, just biased toward newest | Minimal real improvement |
| Do nothing | 0 | Race remains; relies on caller not using the return value | None |

**Recommendation: Option 1.** The fallback exists for no real caller — cut it. *More* important under A1's pivot — registry semantics matter more when MLflow is the only source.

### C3. Inconsistent exception suppression: `try/except: pass` vs `contextlib.suppress`

Mix of patterns:
- `with suppress(Exception):` at [runs.py:85-86](src/spacecraft_telemetry/mlflow_tracking/runs.py#L85-L86), [training.py:129-130](src/spacecraft_telemetry/model/training.py#L129-L130) — clean
- `try: … except Exception: pass` at [training.py:122-126](src/spacecraft_telemetry/model/training.py#L122-L126), [scoring.py:282-286](src/spacecraft_telemetry/model/scoring.py#L282-L286), [tune.py:319-333](src/spacecraft_telemetry/ray_training/tune.py#L319-L333)

The `tune.py:319-333` case is the worst — a 14-line `try` block wrapping client setup + experiment lookup + run search. Any failure leaves `best_run_id = None` silently; downstream scoring loses lineage. The Phase 7 plan said "tracking failures demoted to warnings" but `tune.py` skipped that step.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Convert all to `with suppress(Exception):` with smallest-possible regions; add `log.warning` on tune.py block | S (~30 min) | Low | Consistent style; failures become observable |
| Replace bare `Exception` with specific classes (`MlflowException`, `OSError`) | M (~1 hour) | Medium — easy to miss an exception class and let a real bug surface | Catches genuine errors, more maintenance |
| Do nothing | 0 | Cosmetic inconsistency; behaves correctly today | None |

**Recommendation: Option 1.** Keeps "MLflow never breaks training" promise but makes failures observable.

### C4. Duplicate float-coercion of metrics across training + scoring

[training.py:234-238](src/spacecraft_telemetry/model/training.py#L234-L238) casts ints to floats; [scoring.py:310](src/spacecraft_telemetry/model/scoring.py#L310) does the same dict-comprehension cast. The `evaluate()` return dict has int values (`n_true_positive_labels`, `n_predicted_positive_labels`) that must be cast every time; if a third caller (Phase 8 monitoring?) calls `log_metrics_final`, they'll need to remember the cast or get a cryptic MLflow type error.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Widen helper signature: `log_metrics_final(metrics: dict[str, float \| int])`, cast inside | S (~10 min) | Low | Removes cast at every callsite; future callers get it free |
| Document at callsites, leave casting where it is | S (~5 min) | Low | No behaviour change |
| Do nothing | 0 | Two callsites today; bug surface small | None |

**Recommendation: Option 1.** Coercion belongs in the helper; callers shouldn't know about MLflow plumbing.

---

## 3. Tests

### T1. Entire `tuned_from_run` lineage feature has zero tests

The HPO → scoring run lineage was a key Phase 7 deliverable: `score_channel(parent_hpo_run_id=...)` writes `tuned_from_run` tag at [scoring.py:289-290](src/spacecraft_telemetry/model/scoring.py#L289-L290), `runner.py` plumbs it through [runner.py:317-340](src/spacecraft_telemetry/ray_training/runner.py#L317-L340), `tune.py` looks up `best_run_id` at [tune.py:319-333](src/spacecraft_telemetry/ray_training/tune.py#L319-L333). Nothing asserts any of this. The temporal-split + lineage feature shipped without a regression net.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Unit test calling `score_channel(parent_hpo_run_id="abc123")` asserting `tags["tuned_from_run"] == "abc123"` | S (~15 min) | Low | Locks in the lineage feature with one cheap test |
| Above + slow integration test running `run_hpo_sweep` then `score_channel(parent_hpo_run_id=best["run_id"])` | M (~30 min) | Low | End-to-end lineage proof |
| Do nothing | 0 | Lineage feature shipped on faith | None |

**Recommendation: Option 1 for sure; Option 2 only if Phase 8 (drift) plans to consume the lineage tag.**

### T2. `test_falls_back_to_all_versions_when_run_id_filter_empty` pins the C2 bug as a contract

[test_registry.py:89-105](tests/mlflow_tracking/test_registry.py#L89-L105) asserts that when the run_id filter returns empty, `register_pytorch_model` returns "the first version of any run." That's the silent-bug behavior from C2 — the test treats the bug as a contract. Inverse coverage — actively prevents the right fix.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Delete this test as part of C2; replace with `test_returns_none_when_run_id_filter_empty` | S (~5 min) | Low (changes test alongside code) | Test behavior matches corrected contract |
| Leave alone if C2 stays | 0 | Locks in buggy behavior | None |
| Convert to xfail/skip with comment pointing at C2 | S (~5 min) | Low | Same as Option 1 but signals "we know this is wrong" |

**Recommendation: Option 1, bundled with C2 fix.**

### T3. `subsystem` and `training_data_hash` tags required by `.claude/rules/mlflow.md` but not asserted

CLAUDE.md mlflow rules say: *"Set tags for: mission_id, channel_id, subsystem_category, training_data_hash."* The code dutifully sets all four (training.py:133-140, scoring.py:293-300), but [test_train_channel_creates_mlflow_run](tests/model/test_training.py#L188-L222) only asserts `model_type`, `mission_id`, `channel_id`. If the lazy subsystem lookup at training.py:122-126 silently fails (it can — bare `try/except: pass`), the run is created without those two tags and no test notices.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Add `tags["subsystem"]` + `tags["training_data_hash"]` assertions to existing test | S (~10 min) | Low | Closes the rule → test gap |
| Above + negative test: when `load_channel_subsystem_map` returns empty, run still succeeds but tag absent | M (~20 min) | Low | Documents graceful-degradation contract for missing metadata |
| Do nothing | 0 | Tags ship without verification; rule drift undetected | None |

**Recommendation: Option 1 at minimum. Add Option 2 when fixing A2** — A2 reorganizes the metadata lookup; that's the natural moment to add the negative test.

### T4. "MLflow failures don't break training" promise has no end-to-end test

The Phase 7 plan explicitly promised: *"Tracking failures are caught and demoted to warnings — the training run completes with filesystem artifacts intact."* Each piece tested in isolation, but no test calls `train_channel(settings_with_broken_tracking_uri)` and asserts the user-facing contract. `configure_mlflow` itself is unguarded — if `mlflow.set_tracking_uri()` raises (it can on bad URIs), `configure_mlflow` raises *before* `open_run` is called, and `train_channel` aborts. The current test surface wouldn't catch this regression.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| `@pytest.mark.slow` test pointing `tracking_uri` at unreachable backend; assert `train_channel` still produces filesystem artifacts | M (~20 min) | Low — fails today | Verifies the promise |
| Above + wrap `configure_mlflow` itself in try/except inside `train_channel` | M (~30 min) | Low | Actually delivers the promise |
| Do nothing | 0 | Promise ships untested | None |

**Recommendation: Option 2.** The test from Option 1 would fail today because `configure_mlflow` is unguarded. Either guard the call or weaken the promise — both fine, neither is not.

---

## 4. Performance

### P1. `load_channel_subsystem_map` reads `channels.csv` once per channel under Ray fan-out

[training.py:122-126](src/spacecraft_telemetry/model/training.py#L122-L126) and [scoring.py:282-286](src/spacecraft_telemetry/model/scoring.py#L282-L286) lazy-call `load_channel_subsystem_map` inside every per-channel function. Under Ray fan-out across 100 channels: 100 process-boundary lazy imports + 100 disk reads + 100 CSV parses. Mapping is invariant within a sweep. Plus the `try/except: pass` swallows any read error.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Compute mapping once in driver, attach to `settings` (or pass as arg) | M (~30 min) | Low | 100× fewer disk reads on full sweep; resolves A2 cleanly |
| `functools.lru_cache(maxsize=1)` on `load_channel_subsystem_map` | S (~5 min) | Medium — Ray workers are separate processes; each worker's cache independent | Cuts only repeats within the same worker |
| Do nothing | 0 | Few seconds of disk over a multi-minute sweep | None |

**Recommendation: Option 1.** Bundles cleanly with the A2 fix — two issues, one change.

**Note:** Likely dissolves under A1's MLflow-primary pivot if subsystem flows as a run tag rather than a per-call lookup.

### P2. `mlflow.pytorch.log_model` doubles model artifact write, bypasses `model/io.py` indirection

[registry.py:49-53](src/spacecraft_telemetry/mlflow_tracking/registry.py#L49-L53) writes to MLflow's artifact store; [training.py:224](src/spacecraft_telemetry/model/training.py#L224) `save_model()` writes a second copy to `models/{mission}/{channel}/model.pt`. Same data, two locations, two upload paths under cloud `gs://`. `.claude/rules/pytorch.md` says **all** model I/O must go through `model.io._write_bytes` so Phase 5's `gs://` support works — `mlflow.pytorch.log_model` knows nothing about that indirection.

**Reframed under A1's MLflow-primary pivot:** the dual-write is the symptom; the root cause is `model/io.py` existing as a parallel artifact store. Recommendation flips.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Drop `save_model` and the entire `model/io.py` filesystem-write path; use only `mlflow.pytorch.log_model`. Read path becomes `mlflow.pytorch.load_model("models:/.../<stage>")` or per-run artifact downloads. | M-L (~3-5 hours) bundled with A1; includes test-fixture migration | Medium — touches every artifact read site | Single source of truth; ~50% less disk; gs:// indirection works for registry too |
| ~~Document as deliberate dual-write~~ | — | — | Rejected per A1 |
| ~~Keep both writes~~ | — | — | Rejected per A1 |

**Recommendation: Option 1.** This is mechanically the bulk of A1's implementation.

### P3. `run_hpo_sweep` best_run_id lookup is unbounded scan AND can return a prior sweep's run

[tune.py:322-331](src/spacecraft_telemetry/ray_training/tune.py#L322-L331):
```python
_runs = _client.search_runs(
    [_exp.experiment_id],
    filter_string=f"tags.subsystem = '{subsystem}'",
    order_by=["metrics.f0_5 DESC"],
    max_results=1,
)
```

Two problems:
1. **Performance:** filter is by experiment + subsystem only, no time bound. After 10 re-runs of HPO on `subsystem_1`, scans all 10 sweeps × num_samples trials.
2. **Correctness (more important):** `order_by metrics.f0_5 DESC` returns the all-time best across history. If the *current* sweep produces a worse best than a previous sweep, `best_run_id` points at the old run — `tuned_from_run` and `_meta.run_id` silently lie.

**Reframed under A1's MLflow-primary pivot:** querying MLflow for the best run is now the *correct* paradigm (MLflow is the system of record). The fix is just to scope the query to the current sweep.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Use `best_result.run_id` straight from Ray Tune's `Result` object | M (~30 min) | Low-Medium — verify Ray Tune exposes the MLflow run_id mapping | O(1) lookup, always correct |
| Add `attributes.start_time > <sweep_start_ts>` to the filter | S (~15 min) | Low | Bounded scan, correct lineage |
| Do nothing | 0 | Wrong lineage on re-runs; scan grows | None |

**Recommendation: Option 2.** Same correctness guarantee, smaller surface change. Option 1 bypasses MLflow which under A1 is the wrong direction.

### P4. Local SQLite tracking backend serializes all MLflow writes — caps Ray fan-out throughput

`mlflow.db` (SQLite) supports one writer at a time. Concurrent Ray training tasks all write metrics/params/artifacts to the same SQLite file: with 4 concurrent Ray workers each running 35 epochs, ~140 logging ops fighting for a single SQLite write lock per second. Expect `database is locked` errors or, more insidiously, throughput collapse where workers serialize on the lock and "parallel" training runs at sequential speed. The Phase 7 plan acknowledged the cloud fix (PostgreSQL) but local dev will hit it first.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Document in README + configs/local.yaml; suggest `max_concurrent_trials=1` for local sweeps | S (~10 min) | Low | Sets expectations; doesn't fix |
| Retry-with-backoff wrapper around MLflow writes | M (~30 min) | Medium — masks contention; throughput still bad | Stops crashes, slow runs continue |
| Switch local default to `mlflow server --backend-store-uri sqlite:///mlflow.db` and have configs point at `http://localhost:5000` | L (~1-2 hours) | Medium — extra process to manage | Solves contention by giving SQLite a single writer; unblocks parallel training locally |

**Recommendation: Option 1 now, Option 3 at Phase 11 if/when local full-mission sweeps become routine.** Don't spend the budget yet on a portfolio project where local sweeps are usually 5 channels, not 100.

---

## Action plan

After model switch, the user will indicate which findings to act on.
Recommended order (dependencies first; **A1 reshapes everything below it**,
so most other items either dissolve or get reframed once A1 lands):

**Pre-A1 (cheap, independent of the pivot):**
1. **C1** — CLI `mlflow_promote` calls `registry.promote()`. ~10 min.
2. **C4** — widen `log_metrics_final` signature, drop callsite casts. ~10 min.
3. **C3** — unify exception suppression; add `log.warning` on tune.py block. ~30 min.
4. **C2 + T2** — bundled: delete the buggy registry fallback and its inverse test. ~10 min. *(More important after A1 — registry semantics matter more.)*

**A1 — the big pivot:**
5. **A1 + P2** — bundled: drop `save_model`/filesystem write path; use MLflow logging APIs only; migrate read sites in tune.py + tests. ~1 day.

**Post-A1 (some items dissolve, others get easier):**
6. **A4** — resolve `sqlite:///` relative URI. *More* critical after A1 since all artifact I/O now flows through tracking_uri. ~30 min.
7. **A2 + P1** — likely dissolves: subsystem becomes a run tag, not a runtime lookup. Re-evaluate after A1. If anything remains, ~15 min.
8. **T1** — add `tuned_from_run` lineage test. ~15 min.
9. **T3** — assert `subsystem` and `training_data_hash` tags. ~10 min.
10. **T4** — graceful-degradation test + guard `configure_mlflow`. ~30 min.
11. **P3** — bound HPO best_run_id lookup to current sweep window. ~15 min.

**Defer:**
12. **P4** — README note about SQLite write contention; revisit at Phase 11.
13. **A3** — theoretical concurrency concern; revisit at Phase 11.

Independent items in the "Pre-A1" group can interleave. The A1 + P2 bundle should land as one logical change so tests track the pivot. Post-A1 items should be re-evaluated against the actual post-pivot code (some may be no-ops by then).
