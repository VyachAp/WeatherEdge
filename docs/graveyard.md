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
