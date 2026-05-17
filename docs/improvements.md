# Improvements log

A running list of hypotheses, experiments, and follow-ups for WeatherEdge.
Captures the *forward-looking* stuff: things considered but not shipped,
ideas to revisit when data accumulates, optimisations that didn't make
the cut today. Things that *have* shipped live in git history + CLAUDE.md
gotchas, not here.

**Status tags:**
- `[backlog]` — proposed, not started
- `[in-flight]` — actively testing or partially implemented
- `[done]` — kept here only when the outcome is worth remembering (e.g.
  "tried X, didn't work because Y"). Otherwise prune to git history.
- `[rejected]` — decided against; include the *why* so it isn't
  re-proposed.

**Entry template:**

```
## [status] Short action-oriented title
**Why:** what triggers this idea / what hypothesis we're testing
**Success criteria:** how we'll know it worked
**Effort:** rough size (hours / days / week)
**Leverage:** volume / signal-quality / risk / observability
**Files:** likely touchpoints (omit if speculative)
**Notes:** anything else (data, links, open questions)
```

---

## [in-flight] Monitor stuck-exposure fix + post-Phase-E observation window

**Why:** On 2026-05-17 the bot was discovered silenced for ~24h —
`decision_logs` (Phase C dividend) immediately revealed 100% of 419
decisions over 24h hit `stake_below_min` with `dd_multiplier=1.0`.
Root cause: 16 OPEN+delayed+null-fill trades from May 13–14 pinned
the exposure cap at $226/$110. Phase E shipped 5 changes:
`admin reconcile-stuck` (E1), `resolve_trades` catch-22 fallback (E2),
widened `job_reconcile_orders` lookback (E3), exposure-cap Telegram
warning (E4), `MAX_EXPOSURE_USD_FLOOR=$300` (E5). Need clean
attribution window before more changes.
**Success criteria:** by 2026-05-24, confirm: (a) operator ran
`admin reconcile-stuck` and exposure dropped from $226 → ~$0;
(b) `decision_logs.outcome` distribution shifts from 100%
`stake_below_min` to a healthy mix (`trade_filled`/`trade_pending`/
`dup_blocked_db` dominant); (c) the Telegram exposure-cap warning
hasn't fired (or has fired and was actionable); (d) `job_reconcile_orders`
catches any new delayed-no-fill trades within hours, not days.
**Effort:** ~5 min/day morning checks.
**Leverage:** measurement.
**Notes:** if exposure starts pinning again, query `decision_logs`
filtered to `outcome='stake_below_min'` and look at `dd_level`
(drawdown-induced) vs `requested_stake_usd > 0` (Kelly returned
non-zero but exposure cap ate it).

## [done] 2026-05-16 monitoring window — superseded

Earlier monitoring entry for the 2026-05-16 deploy (Phase A-D) is
folded into the entry above; same observation pattern, same queries.
Outcome: scheduler ran cleanly overnight, surfaced the exposure-cap
silencer in `decision_logs` in 30 seconds (vs the 4-hour debug that
caused Phase C to exist). Phase C's first dividend.

---

## [backlog] Investigate OPEN-promotion bug for delayed orders

**Why:** CLAUDE.md gotcha says `_update_fill_details` keeps
`exchange_status='delayed'` trades at `status=PENDING` when the order
isn't yet retrievable. But the 16 stuck May-13/14 trades had
`status=OPEN AND exchange_status='delayed' AND fill_price IS NULL` —
something promoted them despite no confirmed fill. Once E1-E5
shipped, the operational symptom (exposure pinning) is fixed, but
this is the deeper bug that allowed the bad state.
**Success criteria:** identify the code path that flips `status` to
OPEN before `_update_fill_details` has confirmed a fill. Likely
candidate: `src/execution/lock_rule_executor.py` line ~339 where
`actual_stake = trade.stake_usd or 0.0 > 0` is checked. If `stake_usd`
retains the requested value (instead of being zeroed on no-fill),
the OPEN branch fires. Add a `logger.warning` at the OPEN-promotion
site that triggers when `fill_price is None`. Watch one day of logs.
**Effort:** 2-3h (trace + targeted fix + test).
**Leverage:** prevention (eliminates one source of stuck-exposure
backlog so reconcile-stuck rarely needs to be run).
**Files:** `src/execution/lock_rule_executor.py`, `src/execution/polymarket_client.py::_update_fill_details`.

## [backlog] Per-station lock-rule auto-disable

**Why:** Calibration data showed bimodal lock-rule performance — RJTT /
WIII / NZWN / RKSI hit ~95% win rate, while RPLL / SAEZ / ZBAA sit
near 50-60%. Current global `LOCK_RULE_LOSS_DISABLE_COUNT=3` over 72h
is too crude: one bad streak on the weak stations would mute the path
entirely, including the high-confidence stations.
**Success criteria:** per-station loss tracker prunes the marginal
stations during a streak without halting the strong ones. Measure: 14d
lock-rule cohort win rate post-deploy ≥ pre-deploy.
**Effort:** 4h code + tests.
**Leverage:** volume + risk (preserves throughput on strong stations).
**Files:** `src/risk/circuit_breakers.py` (new `_lock_losses_by_icao`
dict + helper), `src/execution/lock_rule_executor.py::try_lock_rule_trade`
gate before sizing.

## [backlog] Re-enable BRACKET_MARKETS_ENABLED

**Why:** Disabled 2026-05-08 after live data showed −21pp
overconfidence on bracket markets (memory:
`project_bracket_disabled_2026-05-08`). Calibration is now on +
per-station σ floor lands variance correctly per Phase D. Worth
revisiting once we have 4 weeks of clean threshold-market Brier
scores.
**Success criteria:** 4 consecutive weeks of threshold-market Brier
< 0.10 → re-enable for a 2-week measurement window with
`CLUSTER_STAKE_CAP_USD=100` already capping cluster risk. If bracket
trades' Brier stays < 0.15 over those 2 weeks, keep enabled.
**Effort:** 5 min config + 4 weeks of patience.
**Leverage:** universe expansion (~30% of weather markets are brackets
per the 2026-05-16 scan: 297 bracket / 1617 total).
**Files:** `.env` (`BRACKET_MARKETS_ENABLED=true`).

## [backlog] Weekly station_calibration_report cadence

**Why:** Calibration drifts, station performance varies, bias-runaway
stations need pruning. Currently ad-hoc — should be a routine.
**Success criteria:** Friday afternoon habit + reading the markdown,
adjusting `_EXCLUDED_ICAOS` if any station hits |bias|>2.5°C or
proj_RMSE > 6°F.
**Effort:** 30 min/week of attention.
**Leverage:** signal-quality (catches stations decaying out of model
range before they bleed PnL).
**Files:** `scripts/station_calibration_report.py` (already exists).

## [backlog] Aggregation CLI for decision_logs funnel

**Why:** Phase C wrote 12 instrumentation points but querying the
funnel breakdown still requires hand-rolled SQL. A `python -m src.cli
decisions-report --days 1` that prints outcome counts grouped by
`signal_kind` would make the post-filter funnel as cheap to inspect as
`evals-report`.
**Success criteria:** one command outputs total / passing-by-signal-kind
/ outcome breakdown / top reject reasons. Run weekly to spot drift in
the funnel shape (e.g. sudden spike in `cluster_cap_hit` = bracket
markets re-enabled and clustering aggressively).
**Effort:** 1-2h.
**Leverage:** observability (closes the loop on the Phase C investment).
**Files:** new subcommand in `src/cli.py`, mirroring `evals_report`.

## [backlog] Bound the keyword-scan fallback offset configurably

**Why:** `fetch_weather_markets` hard-codes `MAX_OFFSET=10000` for the
legacy keyword scan after the Phase A6 rewrite. Should be a setting
so future Gamma API changes can be tuned without code.
**Success criteria:** `POLYMARKET_LEGACY_SCAN_MAX_OFFSET` in
`src/config.py`; default 10000; can be set to 0 to skip the fallback
entirely (saves ~100 API calls per scan).
**Effort:** 15 min.
**Leverage:** maintenance.
**Files:** `src/ingestion/polymarket.py`, `src/config.py`.

## [backlog] Drop dead Signal columns

**Why:** `Signal.gfs_prob` / `ecmwf_prob` / `aviation_prob` / `wx_prob`
are always NULL per CLAUDE.md gotcha — no live readers. Schema bloat
hurts nothing today but Adds Confusion to anyone reading the model.
**Success criteria:** alembic migration drops the four columns; tests
still pass.
**Effort:** 30 min.
**Leverage:** code clarity (low priority).
**Files:** new alembic migration, `src/db/models.py::Signal`.

## [backlog] Reconsider MIN_PROBABILITY for threshold markets only

**Why:** `MIN_PROBABILITY=0.85` was raised from 0.50 in the 2026-05-08
bracket-overconfidence fix. Brackets are now disabled, so the 0.85
floor is gating *threshold* markets that may have a different
calibration curve. Threshold-only band 0.75-0.85 might be profitable
post-calibration.
**Success criteria:** after `decisions-report` CLI lands, look at win
rate by predicted-probability bucket for threshold-only trades over
30 days. If 0.75-0.85 band wins >75%, lower `MIN_PROBABILITY` for
threshold markets only (or globally, since brackets are disabled).
**Effort:** measurement first; if data supports, 5 min config change.
**Leverage:** volume (would re-open a probability band currently
silently filtered).
**Files:** `.env`, `src/signals/edge_calculator.py` (if we want per-kind
floors, more invasive).

## [backlog] Investigate Polymarket Gamma listing recovery

**Why:** Phase A6 rewrote `fetch_weather_markets` to slug-enumerate
because Gamma's `/markets?tag=weather` silently returned generic
markets starting 2026-05-15. If Polymarket fixes the listing
behaviour, the keyword + tag scans (kept as defensive fallbacks)
would be more complete than slug enumeration (which depends on
hardcoded city + day patterns and only catches `highest-temperature-in-*`
events).
**Success criteria:** weekly probe of
`/markets?active=true&closed=false&tag=weather&limit=100` — if it
starts returning weather markets again, demote slug enumeration to
fallback.
**Effort:** 5 min/week to probe; if needed, 30 min to flip the
strategy ordering.
**Leverage:** discovery breadth (catches `lowest-temperature-*` or
`avg-temperature-*` if Polymarket ever ships them).
**Files:** `src/ingestion/polymarket.py`.

## [backlog] Persist station_rmse for backtest reproducibility

**Why:** Phase D's `get_station_rmse` is an in-process TTL cache
computed on-the-fly. Replay-capable backtests in `src/risk/simulate.py`
won't see the same RMSE values the live bot used at that historical
tick. For now this doesn't matter (the global floor was the prior
behaviour) but if we want to A/B the floor strategies on replay, the
historical RMSE per (station, date) would need to be persisted.
**Success criteria:** new `StationRmseSnapshot` table written nightly
by `job_daily_settlement`; backtest reads from there instead of
recomputing.
**Effort:** 4-6h (migration + nightly write + backtest plumbing).
**Leverage:** measurement (only matters if we run rigorous A/B
backtests on the σ floor).
**Files:** `src/db/models.py`, new alembic migration,
`src/scheduler/__init__.py::job_daily_settlement`, `src/risk/simulate.py`.

## [backlog] POLYMARKET_USE_NEW_EXCHANGES verification

**Why:** Config notes that Polymarket migrated to new exchange
contracts (pUSD collateral). The flag monkey-patches py-clob-client
to sign for the new exchange; without it orders are rejected with
`order_version_mismatch`. Worth verifying the current value is
correct for the live wallet before any larger trading-volume push.
**Success criteria:** one successful FAK on the new exchange this
week confirms the flag is right.
**Effort:** 5 min.
**Leverage:** risk (avoid `order_version_mismatch` halting trades
after a Polymarket-side change).
**Files:** `.env` (`POLYMARKET_USE_NEW_EXCHANGES`).