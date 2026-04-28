# Big Review — Phase 3: Feast Feature Store Integration

**Branch:** `phase03` vs `main`
**Date:** 2026-04-28
**Scope:** 27 files changed, 1467 insertions, 37 deletions

Sections: Architecture · Code Quality · Tests · Performance. Up to 4 issues per section. Each issue: file/line ref, 2–3 options (including "do nothing"), effort/risk/impact, recommendation.

---

## 1. Architecture

### A1. Module-level `load_settings()` in `feature_repo/registry.py`
[feature_repo/registry.py:24](feature_repo/registry.py#L24) reads YAML and builds Pydantic objects every time the module is imported. The plan explicitly accepted this trade-off, but it has knock-on effects: tests can't import the module (so they go through `feast_client.repo.build_feature_view` instead — which is why that builder exists), and Phase 9 boot-time imports will pay the cost.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | Tests already work around it | Phase 9 FastAPI Cloud Run cold start adds ~ms; not catastrophic |
| Wrap in a `_get_settings()` function called only inside `apply_definitions` | S | Low — `feast apply` CLI already needs settings via the wrapper | Eliminates module-level side effect, makes registry.py importable in any context |
| Move the registry construction into `feast_client/repo.py` and have `feature_repo/registry.py` just call a builder | M | Low | Cleanest — single source of FV construction, registry.py becomes a 5-line shim |

**Recommendation: Option 3** before Phase 9. The current shape duplicates FeatureView construction logic between `registry.py` (production) and `repo.build_feature_view` (tests), and they've already drifted (one uses `_settings.feast.source_path`, the other takes an arg).

### A2. `sys.path` mutation inside `apply_definitions`
[store.py:60-64](src/spacecraft_telemetry/feast_client/store.py#L60-L64) inserts `repo_path.parent` into `sys.path` to import `feature_repo.registry`. Global mutation, never cleaned up.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | Low for single-process CLI; fragile under multi-tenant servers (Phase 9) | Pollutes import system globally |
| Use `importlib.util.spec_from_file_location` to load the module by path | S | Low | No global mutation, explicit dependency |
| Refactor per A1 Option 3 (registry built programmatically, no file import needed) | M | Low | Eliminates the import entirely |

**Recommendation: Option 3** — solves A1 and A2 in one shot. If you want a quick fix today, Option 2.

### A3. `feast materialize` and `feast retrieve` always re-apply the registry
[cli.py:408](src/spacecraft_telemetry/cli.py#L408), [cli.py:475](src/spacecraft_telemetry/cli.py#L475) call `apply_definitions(store)` unconditionally. This conflates the lifecycle: apply is a *registration* operation, materialize is a *data movement* operation, retrieve is a *read* operation. They shouldn't all do registration.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | Phase 4 will keep paying the apply cost on every retrieval | Confusing to users who run `feast apply` then wonder why `materialize` reports apply logs too |
| Drop the apply call from `materialize` and `retrieve`; let users run `feast apply` first | S | Medium — first-time users will hit "no feature view registered" errors | Cleaner separation, requires a docs note |
| Make apply lazy: if registry exists, skip; else apply automatically | S | Low | Best UX, hides the gotcha |

**Recommendation: Option 3.** Check `Path(repo_path / "data" / "registry.db").exists()` and only apply if missing. Log which path was taken.

### A4. CLI historical retrieve contradicts the plan's pass-through decision
[cli.py:489](src/spacecraft_telemetry/cli.py#L489) synthesizes a `pd.date_range(..., freq="90s")` for `feast retrieve --mode historical`. The approved plan explicitly rejected the regular-grid helper because of NaN inflation across the irregular ESA sampling — yet here it is, in the CLI.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | Phase 4 won't use this code path (it builds its own entity_df), so contained to CLI debugging | Still surprising if anyone copies this pattern |
| Replace with: read the offline Parquet's actual `telemetry_timestamp` values in the window and use those | M | Low | Honors the plan, but adds a Parquet read inside the CLI |
| Keep the grid but document it as "debug-only, synthesizes a regular grid; not representative of real timestamps" | S | Low | Cheapest, matches what it actually is |

**Recommendation: Option 3.** This is a CLI inspection tool, not a training path. Document the limitation and move on. (The 90s constant is also hardcoded — pull from a config or comment.)

---

## 2. Code Quality

### Q1. Online vs historical key naming asymmetry
[client.py:88](src/spacecraft_telemetry/feast_client/client.py#L88) returns `{key: values[0]}` where `key` is whatever Feast emits (often `"telemetry_features__rolling_mean_10"` with the view-name prefix). Meanwhile `get_historical_features` returns columns named `"rolling_mean_10"`. The test [test_client.py:126](tests/feast_client/test_client.py#L126) papers over this with substring matching. Phase 4/9 will hit this mismatch and need a key-cleaner.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | Phase 4 writes its own normalization; Phase 9 too | Two normalizations to maintain |
| Strip the `{view_name}__` prefix in `get_online_features_for_channel` | XS | Low — but check that Feast's online dict format is stable | Symmetric API — one mental model |
| Change historical to return prefixed keys (the other direction) | S | Higher — point-in-time joins use bare names by convention | Worse |

**Recommendation: Option 2.** Three lines: `prefix = f"{view_name}__"` then `key.removeprefix(prefix)` per entry. Add a unit test for both formats.

### Q2. Silent `float64` → `Float32` aliasing
[repo.py:29](src/spacecraft_telemetry/feast_client/repo.py#L29) maps `"float64"` to `Float32` because Feast 0.47 has no Float64. The map entry is speculative — every current `FeatureDefinition` declares `dtype="float32"`, and Phase 2 already casts `value_normalized` to float32 before writing Parquet. The alias only fires for hypothetical future features, and when it does, it's a silent precision downcast.

Note: storage dtype and compute dtype are already independent. [definitions.py:221](src/spacecraft_telemetry/features/definitions.py#L221) computes in float64 (`np.asarray(values, dtype=np.float64)`) for numerical accuracy and stores in float32. That's correct and should stay. The fix is to make the storage contract explicit — float32 only — so anyone adding a float64 storage dtype later has to make a deliberate decision instead of getting a quiet downcast.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | Future float64 feature → silent precision loss | Bug magnet |
| Remove the `"float64": Float32` map entry | XS | None — no current feature uses it | `KeyError` on future misuse, points to the right answer |
| Option 2 + tighten `FeatureDefinition.dtype` to `Literal["float32"]` (or a wider literal of supported dtypes) | XS | None | Mypy catches the mistake before runtime; contract becomes explicit at the type level |
| Option 3 + add a CI test that scans `FEATURE_DEFINITIONS` and asserts every dtype is in `_DTYPE_MAP` | S | None | Belt-and-suspenders |

**Recommendation: Option 3.** Two-line diff: drop the map entry, narrow the type annotation. Update `test_float64_maps_to_float32` to `test_float64_raises_key_error`. Any future float64 storage need then surfaces as a deliberate API change rather than a quiet downcast. Option 4 is overkill at the current registry size.

### Q3. Cross-layer config coupling in `_resolve_feast_settings`
[cli.py:315](src/spacecraft_telemetry/cli.py#L315) derives `source_path = settings.spark.processed_data_dir / mission / "features"`. The Feast layer now reaches into `SparkConfig` to find its own source. If a future phase produces features by a non-Spark route (e.g. a Ray batch job or a streaming ingest), this breaks.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | Spark is the only feature producer through Phase 12 | Couples two configs; rename of `processed_data_dir` would break Feast CLI |
| Add `feast.source_root: Path = Path("data/processed")` and derive `source_root / mission / "features"` | XS | Low | Decouples, mirrors the rest of `FeastConfig` |
| Move the per-mission derivation into a helper method on `FeastConfig` | S | Low | Encapsulates the convention |

**Recommendation: Option 2.** Three-line config addition, eliminates the cross-layer reach.

### Q4. Removed runtime validation in `transforms.py`
[transforms.py](src/spacecraft_telemetry/spark/transforms.py) drops the `if strategy != "forward_fill" / method != "z-score"` guards. Defensible — `Literal` types at the Pydantic layer make these unreachable for typed callers — but `# type: ignore[comparison-overlap]` was the only thing suppressing the typecheck warning, and you've now removed both. Anything calling these from untyped code (pickled config, dynamic dispatch) loses the safety net.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing (current state) | 0 | Untyped callers can pass garbage | Loud failure later instead of a clean error here |
| Restore the guards with `# type: ignore` | XS | None | Belt-and-suspenders |
| Add a one-line `assert` instead of `raise` | XS | None | Same intent, nicer in dev |

**Recommendation: Do nothing** — the project is fully typed and Pydantic-validated at the boundary. The guards were dead code. (Just calling out the change for awareness — none of the existing tests covered them.)

---

## 3. Tests

### T1. `test_handles_query_before_data_window` doesn't test what the plan specified
[test_client.py:78-94](tests/feast_client/test_client.py#L78-L94) queries `1999-12-31` (before all synthetic data starts at `2000-01-01`) and asserts the result is empty. The plan called for "query before window warmup → NaN, no exception" — that's testing rows 0-9 where `rolling_mean_10` should be NaN because the rolling buffer hasn't filled. The current test exercises a different code path (Feast returning 0 rows because no source row has `telemetry_timestamp ≤ entity_ts`), not the NaN-warmup case Phase 4 will actually hit.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | Phase 4 will discover the warmup-NaN behavior at training time | Coverage gap |
| Add a second test: query at row 5 (within warmup), assert feature is NaN. Keep this one too. | S | Low | Both edge cases covered |
| Replace this one with the warmup-NaN test | S | Low | Fewer tests but tighter to the plan |

**Recommendation: Option 2.** The current test does cover the "no rows" path which is also a real edge case (Phase 4 might retrieve a timestamp before the channel's first reading). Just add the warmup test alongside it. Note: this requires the synthetic data generator to actually produce nulls for the first `window_size - 1` rows, which it currently doesn't (every row has `i * 0.01`). So a second fixture or a parametrized variant is needed.

### T2. Weak assertion in `test_returns_dict_for_known_channel`
[test_client.py:115-119](tests/feast_client/test_client.py#L115-L119) only asserts `isinstance(result, dict)` and `len(result) > 0`. That's "didn't crash", not a behavior check. If `get_online_features_for_channel` started returning `{"error": "no data"}` this would pass.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | The other two tests in the class catch real bugs | Vestigial test |
| Strengthen: assert all expected feature names are present | XS | None | Replaces `test_all_features_present` partially — could merge |
| Delete this test, keep the other two | XS | None | Removes redundancy |

**Recommendation: Option 3.** `test_returns_latest_row_values` and `test_all_features_present` already cover the real contract.

### T3. No coverage of `feast retrieve --mode historical` output path
[test_cli.py:405-432](tests/test_cli.py#L405-L432) tests that historical mode requires `--start`, but no test actually invokes it with `--start` set. The 90s grid synthesis ([cli.py:489](src/spacecraft_telemetry/cli.py#L489)) has zero coverage.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | Buggy grid generation could ship | Regression risk on a code path that's already drifted from the plan |
| Add a test that mocks `get_historical_features` and asserts the entity_df shape | S | None | Catches future drift |
| Don't bother since A4 recommends documenting it as debug-only | 0 | If you redesign the path per A4 Option 2, write the test then | Defer until the design is settled |

**Recommendation: Option 2.** Even as a debug command, untested output shaping is a future-bug magnet.

### T4. `test_apply_idempotent` tests Feast, not our code
[test_store.py:61-69](tests/feast_client/test_store.py#L61-L69) calls `store.apply([…])` twice. We don't define `apply_definitions` to be idempotent — Feast's `store.apply` is. This test asserts Feast's behavior, which Feast's own test suite already covers.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | Costs ~1s of test time, doesn't hurt | Misleading — looks like it covers our code |
| Replace with `apply_definitions(store)` calls (the actual function we wrote), so we test our wrapper | XS | None | Tests our code path, including the lazy import + sys.path mutation |
| Delete the test | XS | None | Reduces noise |

**Recommendation: Option 2.** If A2 is fixed (sys.path mutation removed), this test becomes more meaningful. Either way, route through `apply_definitions` not raw `store.apply`.

---

## 4. Performance

### P1. `apply_definitions` runs on every materialize/retrieve invocation
Same cause as A3. Each `feast materialize` or `feast retrieve` re-walks the registry, re-imports `feature_repo.registry` (which calls `load_settings()` → YAML parse → Pydantic build), re-mutates `sys.path`, and re-writes the registry proto. Order of magnitude: 50–200ms per CLI call.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | One-shot CLI calls; not a hot path | Slow but not painful |
| Implement A3 Option 3 (skip apply if registry exists) | S | Low | Cuts ~100ms off most CLI runs |
| Cache the loaded settings module-level in `registry.py` | XS | Low | Marginal — apply itself is the cost, not settings |

**Recommendation: Option 2** (combined with A3).

### P2. Function-scoped `materialized_store` fixture re-builds for every test
[conftest.py:108-133](tests/feast_client/conftest.py#L108-L133) is `@pytest.fixture()` (function scope). Five tests in `test_client.py` each pay the full apply + materialize cost — apply takes ~200ms, materialize ~500ms on the synthetic 200-row data. ~3.5s of repeated work in `test_client.py` alone.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | Tests are already fast (~5s total) | Wasted CI minutes accumulate |
| Make `materialized_store` session-scoped | S | Medium — tests must not mutate the store, otherwise cross-contamination | Cuts test_client.py runtime ~70% |
| Make it module-scoped | XS | Low — module isolation preserved | ~50% speedup |

**Recommendation: Option 3.** Module-scoped is the sweet spot — fast enough, isolation per test file, no cross-file leakage. The current tests are all read-only against the materialized store, so this is safe.

### P3. CLI historical retrieve does O(N) point-in-time joins for a small window
[cli.py:489](src/spacecraft_telemetry/cli.py#L489) — `pd.date_range(..., freq="90s")` over a 1-month window is ~29,000 entity rows. Feast's point-in-time join does a sort + merge per row. At 90s × 30 days = 28,800 rows, this is `O(N log N)` over the full Parquet — for what's likely a debug command someone runs once.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | Slow for large windows; user can interrupt | Bad UX |
| Default `freq="1h"` or `"10m"` for the CLI; users can pass `--freq` | S | None | Cap output to ~30 rows for a 30-day window |
| Replace synthesis with reading actual timestamps from the Parquet (per A4 Option 2) | M | Low | Honors plan, faster |

**Recommendation: Option 2** as a quick fix; Option 3 as the longer-term answer.

### P4. `_make_synthetic_df` builds 200 × 14 lists in pure Python
[conftest.py:44-63](tests/feast_client/conftest.py#L44-L63) — 14 list comprehensions of 200 elements each, then a DataFrame from dict. ~10ms per call, called once per `tmp_feast_repo` fixture instantiation. Combined with P2, this happens many times per test session.

| Option | Effort | Risk | Impact |
|---|---|---|---|
| Do nothing | 0 | Fast enough | Trivial cost |
| `np.arange(200) * 0.01` once, broadcast across 14 columns | XS | None | ~1ms |
| Combine with P2 (session-scope) so it runs once total | XS | Low | Solves both |

**Recommendation: Bundle with P2** — fix once, both go away.

---

## Summary

The biggest issues, in priority order:

1. **A3 + P1** (re-apply on every command) — concrete UX & speed win, ~½ day
2. **A1 + A2** (module-level config, sys.path mutation) — refactor `feature_repo/registry.py` into a thin shim that calls `feast_client.repo.build_feature_view`. Solves both. ~½ day
3. **A4 + P3 + T3** (CLI historical retrieve drift from plan) — document or rebuild the path, add a test
4. **Q1 + Q2** (online/historical key asymmetry, float32-only contract) — small, high Phase 4 value
5. **T1 + T4** (test coverage gaps) — small, do alongside Q1/Q2

Items 1–2 are the right thing to do *before* Phase 4 starts consuming this code. Items 3–5 can wait but should land before the phase closes out.
