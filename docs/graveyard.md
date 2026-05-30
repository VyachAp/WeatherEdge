# Graveyard

A permanent record of features, modules, and code paths removed from
WeatherEdge — so we don't re-attempt the same failed idea later. Append
only; one entry per deletion.

## Entry format

```markdown
## YYYY-MM-DD — <feature / symbol name> (<files removed>)

**Removed in:** <commit SHA>
**Why originally added:** <one paragraph or "unknown — predates audit">
**Why removed:** <evidence: no callers / lost $X / unpopulated for N days / superseded by Y>
**What we learned:** <so we don't re-attempt; may be "TBD" for older entries>
```

---

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
