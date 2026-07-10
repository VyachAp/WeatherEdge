# Graveyard

A permanent record of WeatherEdge decisions about hypotheses that are **dead**
(removed) or **suspended** (gated off, code kept). The single place to answer
"why is this gone / why is this off / what would it take to bring it back" —
so we don't re-attempt a failed idea or rebuild a deliberately-paused one.

This is the **decision-mark** index. Every disabled or removed hypothesis links
here from a one-line code sentinel:

```python
# GRAVEYARD: <name> — docs/graveyard.md#<anchor>   (status: removed | suspended)
```

## Statuses

- **removed** — code deleted; lives only in git history. Re-introducing means
  re-writing it. Use the *removed-entry format* below.
- **suspended** — code + `.env`/Settings flag kept (an operational lever), but
  the feature is OFF and must NOT be turned on until its **re-enable criterion**
  is met. Use the *suspended-entry format*. Listed in [Suspended hypotheses](#suspended-hypotheses).

## Entry formats

```markdown
## YYYY-MM-DD — <feature / symbol name> (<files removed>)   [removed]
**Removed in:** <commit SHA>
**Why originally added:** <one paragraph or "unknown — predates audit">
**Why removed:** <evidence: no callers / lost $X / unpopulated for N days / superseded by Y>
**What we learned:** <so we don't re-attempt; may be "TBD" for older entries>
```

```markdown
### <FLAG_NAME> — <one-line what>   [suspended]
**Suspended:** <date> · **Flag:** <env/Settings state> · **Code:** <where it lives>
**Why off:** <evidence: lost $X / overconfident Npp / fails gate>
**Re-enable when:** <the concrete prerequisite + how to verify> (or "permanent — do not re-enable")
```

---

## Suspended hypotheses

Gated off, code retained as an operational lever. **Do not flip on** without the
stated prerequisite. Deeper history for each lives in CLAUDE.md + `~/.claude` memory;
this is the consolidated decision index.

### BRACKET_LIKE_NO_DISABLED — kill switch for bracket-like (`exactly`/`range`/`bracket`) BUY_NO in the probability path
**Suspended:** 2026-05-30 · **Flag:** live `.env`=`True` (Settings default `False`) · **Code:** `src/signals/edge_calculator.py::binary_market_edge`
**Why off:** all-time −$260 / −7.4% ROI on 383 trades; surviving NO evals still −0.023 EV/$1 after the three single-bucket guards. Each single-°C bucket evaluated as an independent binary → tight Gaussian on an over-forecast peak makes every neighbour look near-impossible.
**Re-enable when:** `SIGMA_FLOOR_LEAD_TIME_ENABLED` ships AND `evals-report --operator bracket-like` baseline (all-passing) row shows +EV. Lock-path range YES / threshold ops / `exactly` YES are unaffected and keep firing.

### RANGE_UNDERSHOOT_LOCK_ENABLED — `range_undershoot` NO lock branch
**Suspended:** 2026-06-03 · **Flag:** live `.env`=`False` (Settings default `True`, so the `.env` line is REQUIRED to gate it) · **Code:** `src/signals/lock_rules.py::_evaluate_range_lock`
**Why off:** ~9pp model-overconfident (won 71.4% vs 80.8% price-implied break-even) → −$26.52 / 35 trades / 30d. Measurable resolver divergence is unbiased (±0.1°F), so this is a σ-collapse symptom (σ too tight on narrow windows), NOT a °C correction target.
**Re-enable when:** `SIGMA_FLOOR_LEAD_TIME_ENABLED` lands and the class turns +EV. `range_in_window` (YES) is unaffected.

### NEAR_LOCK_CONVICTION_SIZING_ENABLED — prob-cap bypass for near-lock threshold bets
**Suspended:** 2026-06-21 · **Flag:** `.env` + DO env=`False` (Settings default `False`) · **Code:** `src/execution/binary_market.py::near_lock_conviction_eligible`, `src/risk/kelly.py::size_position(prob_cap=)`
**Why off:** the v1 gate (`prob≥0.99 + hours_until_peak≤0`) fired 96% on degenerate Open-Meteo-failed states, amplifying the NO/forecast bets the HARD-lock path correctly refuses. Gate REBUILT 2026-06-21 (requires `has_forecast` + target-day-anchored observed max ≥ threshold + `NEAR_LOCK_CONVICTION_MARGIN_F`, betting-hot direction only).
**Re-enable when:** the rebuilt gate is validated firing on real observational locks (`decision_logs.metadata.conviction=true` → `trade_filled` on genuine YES-`at_least` locks, settling +EV).

### SIGMA_FLOOR_LEAD_TIME_ENABLED — lead-time-aware σ floor (probability-engine Gaussian)
**Suspended:** 2026-05-30 · **Re-enabled (standalone):** 2026-06-27 (prod env `true`, slope at default 0.3) · **Flag:** Settings default `False` · **Code:** `src/signals/probability_engine.py::_compute_sigma`
**Why off:** addressed the bracket-like-NO / range-undershoot σ-collapse root cause but also touches the working threshold path; `backtest-v2` A/B is neutral (Brier 0.0537→0.0525, no clear win). `SHADOW_SIGMA_LEADTIME_ENABLED` (default True, pure telemetry) measured it dark.
**Re-enabled because (2026-06-27, see playbook §5):** gates 1/3/4 green — `forecast-error-report` shows RMSE 1.49× the effective engine σ at 12h lead (calibrated at peak); `shadow-report --key sigma` positive far-from-peak (+1.81°F at 12h+, dormant near peak); `backtest-v2` neutral (no regression). Flipped **standalone** — this *corrects* the original "pair with `BRACKET_LIKE_NO_DISABLED=False`" criterion below: the σ-floor is the low-risk **prerequisite** that must land first; it only widens σ far from peak. The killed classes stay killed (their gate 5 — `evals-report --operator bracket-like` baseline +EV — is unsamplable until the forward-forecast-archive code task lands; see playbook Iteration 2). Kill criterion: revert if it suppresses the working at/past-peak +EV book (should be dormant there).

### PER_OPERATOR_CALIBRATION_ENABLED — split calibration fit by operator class
**Suspended:** 2026-05-31 · **Flag:** Settings default `False` · **Code:** `src/signals/calibration.py`
**Why off:** the n=55 threshold-class fit was degenerate (slope +3.64 → mapped raw 0.78 to 0.04, would have destroyed threshold trading). A degenerate-fit guardrail (slope ∈ [0.2, 2.0]) now makes a bad class-fit fall back to pooled, so the flag is safe to flip — but the data still says NO.
**Re-enable when:** `shadow-report` shows the per-class threshold fit un-squashes the 0.78–0.85 band (positive `cal.delta` in-band) on a non-degenerate sample.

### VALLEY_BLOCK_ENABLED / VALLEY_MIN_EDGE — price-band "overconfidence valley" policy
**Suspended:** 2026-06-06 · **Flag:** Settings default `False` / `None` · **Code:** `src/signals/edge_calculator.py::_in_price_valley`
**Why off:** per-trade EV is U-shaped in the side-buy price; the [0.60,0.85) mid valley is −EV. P1 (block) and P2 (raised edge floor) were tuned in-sample on n=8 removed trades. `SHADOW_VALLEY_POLICY_ENABLED` (default True) stamps the counterfactual.
**Re-enable when:** `valley-report` (joins `shadow_json.valley` to `market_resolutions.yes_won`) confirms the P1/P2 cohort is +EV out-of-sample on a meaningful n.

### BRACKET_MARKETS_ENABLED — evaluate `bracket`/`range` multi-outcome markets
**Suspended:** 2026-05-08 · **Flag:** Settings default `False` · **Code:** gate in `src/scheduler` `job_unified_pipeline` + `job_fast_lock_poll`
**Why off:** live data showed −21pp model overconfidence on bracket markets. The *dead* multi-bucket evaluator (`compute_edges`) was removed 2026-05-30 (see [removed](#removed-hypotheses)); what remains is the operational kill switch that skips `bracket`/`range` markets. `exactly` markets are unaffected (handled by `binary_market_edge`'s single-bucket bracket-like branch). NOT removed because flipping/deleting the gate changes which markets trade.
**Re-enable when:** 4 consecutive weeks of threshold-market Brier < 0.10, then a 2-week bracket measurement window with `CLUSTER_STAKE_CAP_USD` capping cluster risk (see `docs/improvements.md`).

### CLIMATE_PRIOR_ENABLED — Bayesian climate-prior blend (probability engine)
**Suspended:** planned-not-built · **Flag:** Settings default `False` · **Code:** `src/signals/probability_engine.py::_apply_climate_prior`, `src/ingestion/station_normals.py`
**Why off:** the entire blend is wired and tested, but `station_normals` is unpopulated — `get_normal()` returns `None` and the blend no-ops. NOT cruft (the wiring is the asset); the one missing piece is the backfill loader.
**Re-enable when:** `scripts/backfill_station_normals.py` (the ERA5/Open-Meteo archive pull, ~50–100 LOC; see `docs/improvements.md [climate-prior backlog]`) populates `station_normals`, then flip the flag.

---

## Removed hypotheses

## 2026-06-24 — Residual-slope projection "v2" (`PROJECTION_RESIDUAL_SLOPE_ENABLED`)   [removed]

**Removed in:** _(pending commit — Phase 1 of the 2026-06-24 refactor)_
**Files:** `src/signals/projection.py` (slope branch in `project_with_residual` + the
`prefer_slope` param — `project_with_residual` merged into `project_daily_max`),
`src/signals/forecast_exceedance.py` (dual `legacy_projected=` A/B logging + the
`_project_with_residual` import + the `RESIDUAL_SLOPE_*` module aliases), `src/config.py`
(`PROJECTION_RESIDUAL_SLOPE_ENABLED` + `EXCEEDANCE_RESIDUAL_SLOPE_MIN_POINTS`/`_HOURS_CAP`/`_MAX_F_PER_HR`),
`.env` (flag line), `tests/test_forecast_exceedance.py` (the `TestProjectDailyMaxResidualSlope`
class → replaced by a 2-test `TestProjectDailyMaxPrePeakHalflife`).

**Why originally added:** "lever A" of the projection-latency redesign — project the
forecast residual forward at its observed hourly slope (instead of halflife-decaying the level
residual) to catch "forecast falling further behind every hour" 1-2 hours earlier.

**Why removed:** live data showed v2 **overshoots the warming concavity** at high-volume
stations (KLAX/KJFK/RKSI/LFPG strictly worse than the raw forecast). It was disabled in the live
`.env` (`PROJECTION_RESIDUAL_SLOPE_ENABLED=False`) "pending a v3 redesign" that never came — the
slope branch had been dead in production for weeks while `project_with_residual` carried a
`prefer_slope` param whose only remaining purpose was a now-redundant A/B log (both legs degenerate
to the halflife branch when the flag is off, so `projected==legacy_projected` on every row).

**What we learned:** the `forecast_residual_slope_f_per_hr` / `forecast_residual_count` fields on
`WeatherState` (and `state_aggregator._compute_residual_slope`) are **kept** — they still feed the
diagnostic `slope=…°F/hr n=…` reasoning trail and have independent tests. Removing a projector
≠ removing its inputs; check who else reads the inputs before deleting them.

## 2026-06-24 — `range_overshoot` NO lock branch (`RANGE_OVERSHOOT_LOCK_ENABLED`)   [removed]

**Removed in:** _(pending commit — Phase 1 of the 2026-06-24 refactor)_
**Files:** `src/signals/lock_rules.py` (`_evaluate_range_lock` overshoot branch +
`LockDecision` docstring entry), `src/config.py` (`RANGE_OVERSHOOT_LOCK_ENABLED`),
`src/signals/config_epoch.py` (tracked-flag entry), `tests/test_lock_rules.py`
(`test_overshoot_*` methods + the `TestRangeOvershootFlag` class), comment refs in
`cli.py` / `market_resolution.py` / `lock_rule_executor.py`.

**Why originally added:** a deterministic NO lock for range/`exactly` markets — fire NO when the
observed routine-METAR daily max already overshoots the range high by 2× margin (mathematically
"can't land in the window"), gated behind a 2×-margin + rc≥4 filter.

**Why removed:** won only 56% across 18 trades (−$57.61), concentrated in °C cities — our
routine-METAR daily max reads systematically hotter than Polymarket's resolver, so the "obviously
locked" overshoot fired on a max the resolver never recorded. The Phase-3 divergence audit
(2026-05-31, 535 settled rows) found measurable divergence ~0.00°C/abs-mean 0.28°F — i.e. the
hot-bias theory is **unsupported at current volume**, so there's no °C-correction path that revives
this branch. It sat `RANGE_OVERSHOOT_LOCK_ENABLED=False` (off) with no path forward.

**What we learned:** `RANGE_LOCK_MARGIN_MULTIPLIER` is **kept** — it's shared by the
`range_undershoot` branch (suspended) and the super-margin EASY gate. `range_undershoot` stays
(suspended, has a σ-floor re-enable path); only the overshoot twin, which has no path, was removed.
The `lock_branch` enum comment in `models.py` keeps `'range_overshoot'` because historical `Signal`
rows still carry it.

> ### ⚠ 2026-07-10 — THIS REMOVAL WAS A MISTAKE. Superseded by `bucket_overshoot`.
>
> The "56% / 18 trades / −$57.61" figure is **not reproducible from the phantom-safe book**. The
> branch's real realized book was **9 filled trades, 8 W / 1 L (88.9%), −$3.70** — the rest were
> NULL-fill phantom rows (see `project_phantom_recovery_pnl_shift_2026-05-30`). The single real loss
> (Buenos Aires, −$10.22) was caused by the **pre-2026-05-26 wrong-day bug**: `resolve_target_local_day`
> returned *title-day − 1* for −UTC cities, so the bot read **May 19's** max (17°C) and fired NO on a
> "14°C" bucket for **May 20**, whose true max was 14°C. Not °C resolver divergence. Every real trade
> of this branch fired 2026-05-17..21, i.e. entirely before that fix. The branch also bought NO at an
> **average price of 0.901** (break-even 90.1%) — it paid the edge away.
>
> The underlying rule is sound and is the bot's only verified edge. Re-validated 2026-07-10 on 3,580
> rule-triggered candidates priced from real CLOB 1-min history at the bot's true METAR arrival time,
> scored against on-chain resolution: on trusted stations the "dead" bucket wins **3 / 3580 = 0.08%**
> of the time. Restored as **`bucket_overshoot`** with the three fixes the old branch lacked: a °C-aware
> certainty boundary, an ex-ante station-divergence exclusion, and a **max-cost gate** (0.93) so it
> never again buys certainty at 0.90+. See `src.config.Settings.BUCKET_OVERSHOOT_*`.
>
> **Generalized lesson:** before killing a branch, (a) filter `fill_price NOT NULL`, and (b) check
> whether its losses predate a known input bug. A rule can be correct while its *inputs* are wrong.

## 2026-05-30 — `Signal.gfs_prob` / `ecmwf_prob` / `aviation_prob` / `wx_prob` columns

**Removed in:** _(pending commit — first entry of the audit pass)_
**Files:** `src/db/models.py` (4 column declarations), Alembic migration
`q7r8s9t0u1v2_drop_signal_legacy_probability_columns.py` (new), CLAUDE.md
(2 mentions removed), `docs/improvements.md` (backlog entry removed).

**Why originally added:** the v1 architecture (pre-unified-pipeline) ran four
independent probability models in parallel — a GFS-driven Gaussian, an
ECMWF-driven Gaussian, an aviation-METAR observation-projection model, and
the Weather.com v3 observation model — each contributing one `Signal.<X>_prob`
column. The chosen-trade `model_prob` was a weighted blend of the four. The
per-model breakdown lived on `Signal` so the alerter "Detail" view and the
calibration dashboard could explain which model dragged the blend.

**Why removed:** the unified pipeline (`src/scheduler/__init__.py::job_unified_pipeline`)
merges all signals into one `WeatherState` upstream of `compute_distribution`,
which produces a single bucket distribution. There has been no writer for
these four columns since the unified refactor — they are written by no code
in `src/`, read by no code in `src/`. The alerter detail view and calibration
dashboard that consumed them were cleaned up at the same time (per CLAUDE.md
Gotcha). The columns have sat NULL on every new `Signal` row since.

Additionally: `gfs_prob` and `ecmwf_prob` were never created by any Alembic
migration — they were declared in `models.py` and would only appear in
databases where `Base.metadata.create_all()` was run with sufficient DDL
privileges. Production DBs created via `alembic upgrade head` may not have
these columns at all (same failure class as the 2026-05-16 `bot_state` table
drift). The drop migration therefore uses `DROP COLUMN IF EXISTS`.

**What we learned:**

1. **When retiring an architecture, retire its telemetry columns in the same
   migration.** Keeping NULL-only columns "for schema compat" doesn't preserve
   anything (no readers means no compat surface) and creates ambient confusion
   for anyone reading the model file.
2. **Schema changes via `models.py` alone (without a matching migration) are
   silently environment-dependent.** Audit any column whose history shows no
   `op.add_column` against suspicion that prod and dev schemas have diverged.
   Drop migrations for such columns must be `IF EXISTS`-defensive.
3. **Don't re-add per-model probability telemetry to `Signal`.** If we ever
   want to split telemetry by model source again, do it via a side table
   keyed to `signal_id` — that way deprecation is a single `DROP TABLE`, not
   a schema-bloat decision per column.

---

## 2026-05-30 — `Signal.confidence` column overload + `alerter._confidence_label`

**Removed in:** _(pending commit — Module 2 of the audit pass)_
**Files:** `src/db/models.py` (column dropped), `src/persistence/dedup.py`
(`upsert_signal` signature: `confidence` removed, `lock_margin_f` added),
`src/scheduler/__init__.py` (probability-path duplicate `confidence=`
kwarg removed), `src/execution/lock_rule_executor.py` (`confidence=margin_f`
→ `lock_margin_f=margin_f`), `src/execution/alerter.py` (dead
`_confidence_label` helper deleted; Detail view became path-aware),
`src/cli.py` (`bet info`/`status` display became path-aware),
`src/monitoring/dashboard.py` (3 SELECT queries + Signals table column
+ Calibration scatter Y-axis switched to `model_prob`), tests updated,
CLAUDE.md Database section updated, Alembic migration
`r8s9t0u1v2w3_signal_confidence_to_lock_margin.py`.

**Why originally added:** `Signal.confidence` predates the audit pass.
The probability path wrote `confidence = our_probability` (a duplicate
of the value also stored as `model_prob` on the same row); the lock
path repurposed the same column to carry `decision.margin_f` (a °F
margin from threshold), so the Detail view could show "how locked was
this" via the same Telegram formatter. The shared formatter never
branched on `signal_kind`, and the dashboard treated the column
throughout as a probability (`{:.1%}` table format, calibration scatter
`y="confidence"`).

**Why removed:** the column had two incompatible meanings (probability
0–1 vs °F margin 0–10+) written by two different paths through the same
helper, and three readers (Telegram detail view, `bet info`, Streamlit
dashboard) that all treated it as one dimension. Lock-path alerts
shipped `Confidence: 5.00` to the operator (looks like 500% probability,
not a 5°F margin) and dashboard lock rows displayed `450.0%`. Cleaner
to split: `lock_margin_f` slots into the existing
`lock_branch`/`lock_routine_count`/`lock_observed_max_f` cluster and is
NULL for the probability path; probability-path callers stop passing the
duplicate `confidence=` kwarg entirely (the dimension they cared about
is already `model_prob`). The migration backfills
`lock_margin_f = confidence WHERE signal_kind = 'lock'` before dropping,
so no lock-path data is lost. `alerter._confidence_label` had zero
callers anywhere in `src/` or `tests/` — its 0.75/0.55 thresholds
assumed probability semantics and would have produced nonsense for
lock-path values regardless.

**What we learned:**

1. **Don't overload columns across paths.** A column written with two
   different units by two paths through the same upsert helper is a
   semantic bug that will surface as a UX bug at the first read site
   that doesn't `signal_kind`-branch — and most read sites don't.
2. **Keyword-arg writes are easy to miss in audits.** The Explore agent
   pass that produced the original plan reported `Signal.confidence` as
   "orphaned (never written)" because it only grepped for
   `signal.confidence = ...` attribute assignments. The actual writes
   were `upsert_signal(..., confidence=X)` (keyword arg through a helper
   into `pg_insert(...).values(...)`) — invisible to the grep pattern.
   Future audits: when checking "is column X written?", grep for
   `X=<value>` as a kwarg, not just `.X = <value>` attribute assignment.
3. **The plan's claim that this was a "latent crash" was wrong.** The
   column was always populated, so the f-string `:.2f` formatter never
   hit None. The real bug was unit confusion in the read path, not a
   null-pointer crash. Worth flagging audit findings as "verify writers
   exist by reading actual call sites, not by grep-pattern alone".
4. **When splitting an overloaded column, preserve only the meaningful
   side via backfill.** The probability-path value was a duplicate of
   `model_prob`; the lock-path value was the only one carrying
   information not stored elsewhere. The migration discards the duplicate
   and preserves the unique data.

---

## 2026-05-30 — `edge_calculator.compute_edges` (multi-bucket bracket path)

**Removed in:** _(pending commit — Module 3 of the audit pass)_
**Files:** `src/signals/edge_calculator.py` (function deleted, ~60 LOC),
`src/scheduler/__init__.py` (dead import at the top of
`job_unified_pipeline` removed, 1 line), `tests/test_edge_calculator.py`
(7 test classes deleted: `TestEdgeComputation`, `TestMinProbabilityFilter`,
`TestPriceFilter`, `TestRoutineCountFilter`, `TestMarketCloseFilter`,
`TestDepthFilter`, `TestMultipleBuckets` — ~140 LOC), CLAUDE.md (4
references corrected: pipeline ASCII art, Edge-calculator section
description, threshold-floor note, File Index row including the
nonexistent `MIN_EDGE` re-export that the Gotcha section had already
flagged as removed).

**Why originally added:** the original bracket-evaluation design. A
bracket market (e.g. "Temp 80-84 / 85-89 / 90-94 °F") has multiple
outcome buckets, each its own YES/NO binary. `compute_edges` produced
one `BucketEdge` per bucket so the scheduler could pick the highest-edge
one. Tested in isolation; the filter set (`MIN_EDGE` / `MIN_PROBABILITY`
/ price band / depth / close-buffer) lived inside it.

**Why removed:** superseded by `binary_market_edge`'s bracket-like
branch (lines ~205-218: `op in ("exactly", "range", "bracket")` →
`market_range_f(market)` returns a single `(low, high)` window; the
function collapses the multi-bucket evaluation into one side-pick with
the new single-bucket NO guards). When `BRACKET_MARKETS_ENABLED=False`
went live on 2026-05-08, the scheduler's import of `compute_edges`
became orphaned (`scheduler/__init__.py:222` imported it but never
called it — the only call sites were in the test file). For 3+ weeks
the function has been a dead `def` reachable only from its own tests,
while CLAUDE.md's File Index still listed it as part of the live
exports. Dropping it closes a real cruft surface; if bracket markets
ever reactivate the existing `binary_market_edge` path serves them via
the single-bucket window. The multi-bucket-per-outcome variant lives in
git history if ever needed.

**What we learned:**

1. **An orphaned import is a signal that the live caller drifted.**
   When a module-level `from X import Y` is the only reference and Y is
   never invoked, the audit verdict is usually "Y is also dead, not
   just the import" — the import was the last thread tying Y to the
   call graph. Don't just drop the `import` line; drop Y too. Module 3
   would have left ~60 LOC of `compute_edges` standing if we'd only
   fixed the cosmetic import.
2. **Kill-switched features fork into "operational kill switch with
   reactivation criterion" vs "superseded design with no reactivation
   path".** `BRACKET_MARKETS_ENABLED=False` is the former (criterion
   documented in `docs/improvements.md`). `compute_edges` was the
   latter — even if brackets get re-enabled, the replacement
   (`binary_market_edge`'s bracket-like branch) is what will run. Audit
   should distinguish these and drop the latter; the operational
   switches stay.
3. **CLAUDE.md drift compounds when one section is updated without
   another.** The Gotcha section already documented "the
   `edge_calculator.MIN_EDGE` re-export was removed; `_check_filters`
   reads `settings.MIN_EDGE` directly", but the File Index row for the
   same file still listed `MIN_EDGE` as an export. Two contradictory
   claims in the same file. The lesson: when removing a symbol, grep
   CLAUDE.md for the symbol name and fix every mention in one pass.

---

## 2026-05-30 — WX (Weather Company v3) pipeline

**Removed in:** _(pending commit — Module 6 of the audit pass)_
**Files:** `src/ingestion/wx.py` (620 LOC, full module deleted),
`tests/test_wx.py` (8 test classes deleted), `src/db/models.py`
(`WxObservation` ORM class deleted), `src/config.py` (5 `WX_*` Settings
deleted), `src/scheduler/__init__.py` (WX retention-cleanup block in
`job_daily_settlement` deleted), `src/cli.py` (`bet find --station`
observation-enriched branch removed; falls straight through to
`find_markets_for_station`), CLAUDE.md (3 references: Data-sources row,
Auxiliary models line, File Index row), Alembic migration
`s9t0u1v2w3x4_drop_wx_observations.py` (defensive `DROP TABLE IF EXISTS
wx_observations`).

**Why originally added:** Weather Company v3 (`api.weather.com`)
publishes per-station current observations on a ~5-minute cadence —
roughly 10× the routine-METAR rate — with a richer field set (temp,
dewpoint, humidity, wind, wind gust, wind direction, pressure +
tendency, 1/6/24h precip + snow, `temperatureMaxSince7Am`,
`temperatureMax24Hour`, cloud cover, visibility, UV index). The intent
was to supplement the once-an-hour METAR feed with fresh observations
for last-mile peak detection. The module shipped a full ingestion +
trend-analysis + threshold-event-detection pipeline (HTTP fetch with
retry, daily-budget rate limiter, 60s response cache, in-process
rolling buffer for ~10h of obs, linear-regression rate fit, multi-
indicator peak-likely-passed detection, threshold-crossed / peak-
likely-done event types). Tests passed in isolation.

**Why removed:** the writer was never (or no longer) scheduled in
production. No code in `src/` ever calls `poll_stations` /
`poll_and_store` / `fetch_wx_current`. The `wx_observations` table
therefore accumulated zero rows; the nightly cleanup at
`job_daily_settlement` (gated on `WX_API_KEY`) deleted zero rows
every night; the `bet find --station` CLI hook called
`get_buffer_history` against an always-empty in-process buffer and
silently fell through behind `if buf:`. Two clear tombstones — the
cleanup job and the CLI hook — both presupposed a writer that no
longer exists.

When asked "was this shipped intentionally for future use, like
`forecast_archive` and `station_normals`?" the answer was: no, an
early experiment that didn't survive. Unlike the climate-prior
infra (a known bootstrap step away from working) or the forecast
archive (writing every tick and waiting for a reader), the WX
pipeline needs a new scheduler job + ICAO-selection logic + error-
budget plumbing + monitoring to ever run again — heavier than the
"just add the missing piece" pattern of Modules 4-5, and not
justified by a concrete signal-quality theory beyond "more frequent
obs sounds useful".

**What we learned:**

1. **A cleanup job for a table no one writes is a tombstone.** When
   `job_daily_settlement` carries an `if settings.X_API_KEY: DELETE
   FROM x WHERE created_at < cutoff` block but no scheduler job
   anywhere writes to `x`, the cleanup is documentation that the
   pipeline once ran and got de-scheduled. Audits should treat both
   the cleanup AND the data path as suspect.
2. **A silently-degrading CLI hook is the other tombstone shape.**
   The `bet find --station` flow's `if buf:` guard against an always-
   empty buffer never alerted the operator that the underlying
   pipeline was dead — it just rendered a less-rich output. The
   lesson: when a feature degrades silently because its dependency
   isn't wired, prefer a clear log line ("WX poller not running;
   falling back to plain station lookup") so future-me debugs the
   dependency, not the consumer. Or — as in this commit — remove
   the dependent path entirely.
3. **Distinguish "forward-looking infra waiting for one missing
   piece" from "abandoned mid-design".** Modules 4 and 5 fit the
   former pattern (the missing piece is small, the value is concrete,
   the writer or reader is already running). Module 6 fit the latter
   (re-wiring needs a new scheduler job + a non-trivial set of
   policy decisions, and no current signal-quality investigation
   names 5-min WX obs as the lever). The two patterns have opposite
   audit verdicts. Ask the user; their answer determines which is
   which.
