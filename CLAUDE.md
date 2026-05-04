# CLAUDE.md

## Project Overview

WeatherEdge is a weather-driven trading bot for Polymarket daily "highest temperature in [city]" markets. Two trading paths run concurrently:

1. **Probability path** — every 5 minutes, aggregate live METAR observations + a deterministic Open-Meteo forecast (with multi-model ensemble σ) per city, build a full probability distribution over integer temperature buckets, compare to live Polymarket CLOB prices, and size positions via fractional Kelly.
2. **Lock-rule path** — every 30 seconds, re-check active binary markets for new routine METARs and fire deterministic trades when the daily max is mathematically locked above (or below) threshold.

The resolution source for these markets is the Wunderground history page, which mirrors **routine** (non-SPECI) METARs. The bot reads METARs directly and typically sees the same data 1–2 hours before the UI updates — that's the edge both paths exploit.

## Quick Reference

```bash
# Run tests
pytest tests/                         # full suite

# Type check
mypy src/ --ignore-missing-imports
# (Ruff lint not used in this project — skip it.)

# One-shot market scan
python -m src.cli scan

# Start full daemon (health server on :8080)
python -m src.cli run

# Backfill historical markets
python -m src.cli backfill --days 30

# Replay the probability pipeline against resolved markets
python -m src.cli backtest-v2 --days 30

# Dry simulation against historical snapshots (no orders)
python -m src.cli paper-trade --days 30

# Look up a market in the local DB (faster than Gamma search)
python -m src.cli bet find --city Phoenix --date 2026-04-30

# Backtest the lock-rule trader specifically
python scripts/backtest_lock_rule.py --days 30
```

## Architecture

### Pipeline (critical path)

The unified pipeline runs in `src/scheduler.py::job_unified_pipeline` every 5 min. A second loop, `job_fast_lock_poll`, runs every 30 s and re-fires the same lock-rule logic between unified ticks.

```
circuit_breakers ─► get_active_weather_markets ─► group by ICAO (skip _EXCLUDED_ICAOS)
        │                                             │
        ▼                                             ▼
  Phase 1 (concurrent, sem=8):                 Phase 2 (sequential, per market):
    aggregate_state(session, icao, lat, lon)     ├─ skip future-day markets (_should_skip_future_day)
      ├─ fetch_metar_history(icao, 24h)          ├─ get_token_ids (Gamma)
      ├─ fetch_deterministic_forecast \\         ├─ get_best_bid_ask → live yes_price
      │  +  fetch_ensemble_forecast    } gather  ├─ get_orderbook_depth
      ├─ _blend_forecasts (det. central +        ├─ _try_lock_rule_trade ── if locked: size_locked_position,
      │   ensemble σ at peak hour)               │   filter, place_order, dedup, continue
      ├─ get_bias(session, icao)                 ├─ skip if price ≥ 0.99 or ≤ 0.01
      ├─ check_and_record_daily_max_alert        ├─ compute_distribution(state, buckets)
      └─ returns WeatherState                    ├─ _binary_market_edge / compute_edges
                                                 ├─ size_position (fractional Kelly × dd_mult)
                                                 └─ place_order

  job_fast_lock_poll (30s):
    fetch_latest_metars(active_icaos) ─► for each new routine (icao, market):
        evaluate_lock(_minimal_state, market) ─► _try_lock_rule_trade
        _fast_poll_projection_check(icao, station_metars) ─► reuses cached
          forecast/bias/normals from last unified tick → build_state_from_metars
          → check_and_record_daily_max_alert (no extra HTTP)
```

`_locked_markets_fired_today`, `_last_routine_seen`, and the `_state_cache` (in `state_aggregator.py`) are in-process dicts cleared **per-station at the station's local-day rollover** by `_maybe_clear_per_station_caches` (called at the top of every `job_unified_pipeline` tick). A sibling `_market_to_icao` map populated at lock-fire time lets the rollover find which `market_id` entries belong to which station; `_local_day_seen` cookies the last-observed `today_local(tz)` per ICAO. A bias-runaway station (`|bias| > 3°C`) is skipped at Phase 1 grouping.

### Scheduler Jobs (`src/scheduler.py`)

| Job | Schedule | Purpose |
|-----|----------|---------|
| `job_scan_markets` | Every 15 min | Fetch Polymarket weather markets via Gamma API, upsert `Market` + `MarketSnapshot` |
| `job_resolve_trades` | Every 5 min (30s offset from unified) | Settle expired markets within minutes of their `end_date`. Calls `resolve_trades` (idempotent: gates on `Market.end_date < now` and `Trade.status=OPEN`); refreshes `current_yes_price` from CLOB before applying the 0.95/0.05 thresholds so a 30-min-stale price doesn't delay resolution. Telegram-pushes each new resolution. |
| `job_unified_pipeline` | Every `UNIFIED_PIPELINE_INTERVAL_MINUTES` (5m default) | Aggregate per-city weather state, evaluate lock + probability paths, place trades. Calls `_maybe_clear_per_station_caches` at the top of each tick. |
| `job_fast_lock_poll` | Every `FAST_LOCK_POLL_INTERVAL_SECONDS` (30s default) | Bulk `fetch_latest_metars` for active binary-market ICAOs; fire EASY-direction lock trades on new routine METARs without waiting for the 5-min tick, AND re-run the forecast-exceedance projection check using cached forecast/bias from the last unified tick. Dedup via `_locked_markets_fired_today`, cleared per-station on local-day rollover. Disabled by `FAST_LOCK_POLL_ENABLED=false` or `LOCK_RULE_ENABLED=false`. |
| `job_daily_settlement` | 22:00 UTC | Bankroll/drawdown bookkeeping (records current equity to `BankrollLog`), Telegram daily summary, WX retention cleanup, station-bias recording, weekly calibration check. **Does not** resolve trades or clear caches — both moved to higher-cadence per-station jobs. |

`job_startup` runs once on boot: starts Telegram alerter, loads drawdown state, runs an initial market scan.

`job_fast_lock_poll` only fires the **EASY** lock direction (observed max already clears threshold + margin) — it builds a `_minimal_state_for_easy_lock` with no forecast, no solar/cloud, just `routine_history`. The **HARD** direction (no-more-heating: market-day max below threshold AND past-peak signal AND forecast peak < threshold) needs full forecast context and stays in `job_unified_pipeline`.

For the **forecast-exceedance projection** check, fast-poll instead reuses the previous unified tick's forecast / station-bias / climate-normals via the `_state_cache` (in `state_aggregator.py`, 30-min TTL keyed by ICAO). `_fast_poll_projection_check` merges the new METAR into cached history, calls `build_state_from_metars`, then `check_and_record_daily_max_alert`. Closes the 5-min cadence gap on projection alerts (~30s instead of up to 5 min). No-op when no cached state exists yet (unified hasn't run for this station, or cache aged out).

### Data sources

| Source | Module | Purpose |
|--------|--------|---------|
| Polymarket Gamma API | `src/ingestion/polymarket.py` | Active market list, parsed threshold/operator/location/date |
| Polymarket CLOB | `src/execution/polymarket_client.py` | Live best bid/ask, orderbook depth, order placement |
| Aviation METAR (6 providers) | `src/ingestion/aviation/` | Routine METAR history, daily max, trend, cycle detection |
| Open-Meteo (deterministic + ensemble) | `src/ingestion/openmeteo.py` | Hourly temp/cloud/solar/dewpoint forecast (no API key). Two endpoints fetched concurrently per city: `fetch_deterministic_forecast` for the central peak/hourly arrays (used as the **bias reference frame**), `fetch_ensemble_forecast` (`ENSEMBLE_MODELS=ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless`) for inter-model spread at peak hour. `_blend_forecasts` combines them. Either alone is acceptable as fallback; ensemble σ falls back to the hours-based schedule when fewer than `ENSEMBLE_MIN_MODELS=3` models return data. |
| Weather.com v3 | `src/ingestion/wx.py` | Optional auxiliary station observations (`WX_API_KEY` gated) |

Aviation providers fail over in order: AWC → IEM → OGIMET → NOAA → CheckWX/AVWX (API-key gated). See `src/ingestion/aviation/_aggregator.py`.

### Database (`src/db/models.py`)

Core: `Market`, `MarketSnapshot`, `Signal`, `Trade`, `BankrollLog`, `StationBias`, `StationNormal`
Aviation: `MetarObservation`, `TafForecast`, `Pirep`, `SynopObservation`, `AviationAlert`
Alerting: `ForecastExceedanceAlert` (every same-hour delta > 0.5°F, with `alerted` flag for Telegram-pushed rows)
Backtest archive: `ForecastArchive` — one row per (station, target-local-day, fetched_at) capturing the blended Open-Meteo forecast. Written best-effort by `aggregate_state` after `_blend_forecasts`; consumed by the replay-capable backtest in `simulate_distribution_pipeline`. Hourly arrays stored as JSONB.
Telemetry: `EvaluationLog` — append-only, one row per per-side edge evaluation (PASSING and REJECTED). Captures `model_prob/market_prob/edge/passes/reject_reason/depth_usd/minutes_to_close/routine_count/signal_kind`. Volume ≈ 11.5K rows/day. Source of truth for `MIN_EDGE`/`MIN_PROBABILITY`/`MIN_DEPTH_USD`/etc filter-tuning backtests, since `signals` is now de-duplicated to one row per (market, side) and only carries passing edges.
Auxiliary: `WxObservation`

`Signal` carries `signal_kind` ('probability' | 'lock') so post-mortems can split realised P&L by which path produced the row. When `signal_kind='lock'`, `lock_branch`/`lock_routine_count`/`lock_observed_max_f` carry the structured `LockDecision` context; NULL otherwise.

`Signal` also has legacy `gfs_prob`/`ecmwf_prob`/`aviation_prob`/`wx_prob` columns — always NULL in the unified pipeline; kept for schema compatibility (still read by `alerter._build_detail_message` for the inline-button "detail" view).

`Trade` carries `submit_yes_bid`/`submit_yes_ask`/`submit_depth_usd`/`submit_at` — snapshot of the live YES bid/ask and buy-side orderbook depth at the moment `place_order` was called. Lets slippage post-mortems decompose `fill_price - entry_price` into "spread we accepted" vs "depth we walked". NULL on rows from before migration `j0k1l2m3n4o5` and on backtest paths.

## Key Patterns

### Probability engine (`src/signals/probability_engine.py`)

`compute_distribution(state, buckets)` produces a `BucketDistribution` with `probabilities: {bucket_f: prob}` summing to 1.0.

Signal combination:

1. **Baseline Gaussian** centered on `state.forecast_peak_f` (deterministic Open-Meteo peak + 30-day station bias).
2. **σ from `_compute_sigma`**: ensemble spread when available (`state.forecast_sigma_f × ENSEMBLE_SPREAD_MULTIPLIER (1.3)`), clipped to `[ENSEMBLE_MIN_SIGMA_F=1.0, ENSEMBLE_MAX_SIGMA_F=5.0]` °F, with a soft floor at half the hours-based schedule. Hours-based fallback (`_hours_based_sigma`): 1.0 past peak, 1.5 (≤2h), 2.5 (≤4h), 3.5 (>4h) — used when fewer than `ENSEMBLE_MIN_MODELS=3` models returned data.
3. **METAR trend shift** (three branches):
   - *Pre-peak* (`hours_until_peak > 0`): shift by **residual rate** (`metar_trend_rate − forecast_slope_to_peak_f_per_hr`) when ≥0.5°F/hr — only the part of the obs slope the forecast didn't already account for. Falls back to the legacy raw-rate shift when forecast slope is unavailable. Capped at +2°F.
   - *Past peak + declining* (`hours_until_peak ≤ 0` AND `rate ≤ 0`): lock center to observed max.
   - *Past peak + still rising* (`hours_until_peak ≤ 0` AND `rate > POST_PEAK_MIN_TREND_F_PER_HR=0.5`): extrapolate `rate × hours × POST_PEAK_TREND_CARRY_K (0.75)`, capped at `POST_PEAK_MAX_SHIFT_F=3°F`, damped by `solar_decline_magnitude` and `cloud_rise_magnitude`. Reason: Open-Meteo's nominal peak is systematically too early in hot arid stations (OPKC/OMDB/Phoenix).
4. **Solar/cloud cap** (`_apply_solar_cloud_cap`): combined `solar_declining` AND `cloud_rising` → hard cap at observed max. Strong solar decline alone (`solar_decline_magnitude > 0.7`) → soft cap at `observed_max + 1°F`.
5. **Dewpoint adjustment**: rising Td >1°F/hr tightens sigma (max 0.5°F); falling Td <-1°F/hr widens it (max 0.3°F).
6. **Monotonicity constraint** (hard): `P(bucket < current_max_f) = 0`. **Degenerate fallback**: when all mass is zeroed, collapse to the bucket closest to `current_max_f` with prob 1.0 (logged as "degenerate: all mass on Xº°F (no upside)").

Every applied signal is appended to `distribution.reasoning` — the log tail you see in `[LTFM] binary ...` lines is this list.

#### State aggregator (`src/signals/state_aggregator.py`)

`aggregate_state` returns a `WeatherState` snapshot per city. Key fields beyond the obvious ones:

- `metar_trend_rate` / `metar_trend_rate_short` — 6h and 2.5h linear regressions on routine METAR temps. Probability engine uses the 6h slope; `forecast_exceedance` uses the 2.5h slope (the 6h regression stays positive past the real peak on sharp-cooling afternoons).
- `routine_history` — sorted `(observed_at_utc, temp_f)` tuples for last 24 h, used by `lock_rules._market_daily_max` to compute the per-market daily max anchored to `market.end_date`'s **station-local** day (matters for markets that close at e.g. 07:00 local time).
- `latest_obs_temp_f`, `forecast_temp_now_f`, `forecast_slope_to_peak_f_per_hr`, `forecast_residual_f` — bias-adjusted residuals; trajectory comparison drives the residual-rate METAR trend signal.
- `forecast_residual_slope_f_per_hr`, `forecast_residual_count` — `d(residual)/dt` linear regression over routines in the 6h window (computed by `_compute_residual_slope`). Captures "forecast falling further behind hour-over-hour" — invisible to the point-residual field above. Drives the v2 path of `_project_daily_max` when `forecast_residual_count >= 3`. None / 0 when forecast missing or fewer than 2 routines.
- `forecast_sigma_f`, `ensemble_model_count` — ensemble σ at peak hour (None when ensemble unavailable).
- `has_forecast` — gates HARD-direction lock evaluation.
- `_routine_daily_max` boundaries are **station-local day**, not UTC.
- When Open-Meteo fails entirely, `WeatherState` is still returned: `forecast_peak_f = current_max_f`, `hours_until_peak = 0`, solar/cloud signals False, `forecast_sigma_f = None`. The probability engine degenerates to a narrow band around current max; lock rule's HARD direction is disabled.

After every successful aggregation, `aggregate_state` also stashes its inputs (`forecast`, `bias_c`, `climate_prior_*`, `history`, timestamp) into the module-level `_state_cache` keyed by ICAO. The fast-poll loop reads via `get_cached_aggregation_inputs(icao)` (30-min TTL) so it can rebuild state with a fresh METAR appended without re-paying the forecast / bias / normals HTTP cost. Cleared in `job_daily_settlement` via `clear_state_cache()`.

### Lock-rule trader (`src/signals/lock_rules.py`, `scheduler._try_lock_rule_trade`)

Deterministic complement to the probability engine. Returns a `LockDecision(side, reasons, margin_f)` rather than a probability.

Operators in scope:
- **Threshold** (`above`, `at_least`, `below`, `at_most`) — EASY/HARD branches below.
- **Range / bracket / `exactly`** — routed through `_evaluate_range_lock`. `market_range_f` (in `scheduler.py`) builds the `[low_f, high_f]` window: parsed `(low, high)` for explicit bracket questions, or a synthetic single-bucket range for `exactly` (e.g. `=10°C` → `[50, 51]°F`). Fires NO on overshoot (`current_max_f > high_f + margin`), NO on undershoot (`current_max_f < low_f - margin` AND `_no_more_heating`), YES on in-range (`low_f <= current_max_f <= high_f` AND past-peak with no upward signal).

**Lowest/minimum temperature markets are filtered upstream** in `polymarket.parse_question` (`_LOWEST_TEMP_RE`) — the entire pipeline assumes daily-max physics, so comparing `current_max_f` against a daily-min threshold would be a category error. A defensive guard in `evaluate_lock` also drops them by question-text match in case a stale Market row from before the filter still has `parsed_operator` set.

Two threshold-market directions:

- **EASY** — market-day max ≥ threshold + `LOCK_MARGIN_F` (default 2.0°F). Mathematically locked because daily max is monotonic. Decides YES for `above`/`at_least`, NO for `below`/`at_most`. No forecast required.
- **HARD** — market-day max < threshold − margin AND `_no_more_heating(state, threshold)`:
  - `state.has_forecast` must be True;
  - `forecast_peak_f < threshold`;
  - past-peak signal: `solar_declining` OR `metar_trend_rate ≤ 0`.
  Decides NO for `above`/`at_least`, YES for `below`/`at_most`.

Per-market daily max comes from `_market_daily_max(state, market.end_date, now_utc)` — anchored to **`market.end_date`'s local day**, not the snapshot's local day. Source: `state.routine_history`.

Routine-count guard is **direction-dependent**:
- Hard floor of **2** routines for any lock (single-METAR fluke prevention).
- **Super-margin EASY** (overshoot ≥ 2× `LOCK_MARGIN_F` = 4°F by default): allowed at routine_count = 2. Daily max is monotonic — two confirming obs already 4°F+ over threshold cannot be undone by a third. Cuts ~30-60 min of morning lag on hot days.
- **Standard EASY** (margin between `LOCK_MARGIN_F` and 2× `LOCK_MARGIN_F`): requires `MIN_ROUTINE_COUNT` (3 by default).
- **HARD** + **range/exactly**: requires `MIN_ROUTINE_COUNT` (forecast-dependent, no fluke shortcut).

`_try_lock_rule_trade` (in `scheduler.py`) is the executor used by both `job_unified_pipeline` (Phase 2, before the near-resolved skip) and `job_fast_lock_poll`. Flow:

1. `evaluate_lock(state, market)` → if `side is None`, return `None` (caller falls through to probability).
2. Compute `effective_price` = `yes_price` (YES side) or `1 - yes_price` (NO side); reject if outside `[LOCK_RULE_MIN_PRICE=0.05, LOCK_RULE_MAX_PRICE=0.95]`.
3. Pull buy-side depth, run shared `_check_filters` with `min_routine_count=2` (lock-rule already gated routine count per its own super-margin / standard rules), close buffer, depth — stub edge/prob, no edge gate.
4. Size via `size_locked_position` (`LOCK_POSITION_PCT=2%` of bankroll, capped at `MAX_POSITION_USD/2` and 15% of orderbook depth), apply drawdown multiplier, place FOK order.
5. On execution, write `Signal(model_prob=1.0, market_prob=effective_price, edge=1-effective_price)` and `Trade(direction, stake_usd, entry_price=effective_price)`. Telegram 🔒 alert.

Settings: `LOCK_RULE_ENABLED`, `LOCK_RULE_MIN_PRICE`, `LOCK_RULE_MAX_PRICE`, `LOCK_MARGIN_F`, `LOCK_POSITION_PCT`, `LOCK_RULE_LOSS_WINDOW_HOURS=72`, `LOCK_RULE_LOSS_DISABLE_COUNT=3`, plus the fast-poll knobs `FAST_LOCK_POLL_ENABLED`, `FAST_LOCK_POLL_INTERVAL_SECONDS`.

### Forecast-exceedance alerts (`src/signals/forecast_exceedance.py`)

Diagnostic + Polymarket-discovery layer that runs from `aggregate_state` every unified tick AND from `_fast_poll_projection_check` (in `scheduler.py`) on every fresh fast-poll METAR. Two outputs:

1. **DB row** in `forecast_exceedance_alerts` for every routine METAR whose same-hour delta vs forecast exceeds `EXCEEDANCE_THRESHOLD_F=0.5°F`. This is calibration history — bucket by `alerted` to compare push rate vs reality. The `(station_icao, observed_at)` uniqueness constraint dedupes between fast-poll and unified for the same observation.
2. **Telegram 🌡 push** when *all* of:
   - `_peak_passed(state)` is False (heuristics on `metar_trend_rate_short`, solar, hours_until_peak);
   - routine count gate: `routine_count_today >= 3` normally, or `>= 2` when `same_hour_delta_f > STRONG_RESIDUAL_DELTA_F=1.0°F` (the obs is already running clearly hot, so 2 confirming routines suffice);
   - `_project_daily_max(state) - state.forecast_peak_f > DELTA_THRESHOLD_F=1.0°F`;
   - station hasn't already pushed within `ALERT_COOLDOWN=30 min`.

`_project_daily_max` has two pre-peak branches; both share the post-peak trend carry, dewpoint nudge, overshoot cap (`forecast_peak + MAX_OVERSHOOT_F=5°F`), and observed-max floor:

- **v2 (slope, default when ≥ `RESIDUAL_SLOPE_MIN_POINTS=3` routines and `PROJECTION_RESIDUAL_SLOPE_ENABLED=True`)** — projects the residual forward at its observed slope: `projected = forecast_peak + (current_residual + slope × min(hours_until_peak, RESIDUAL_SLOPE_HOURS_CAP=3))`. Slope clipped to `±RESIDUAL_SLOPE_MAX_F_PER_HR=1.5°F/hr`; the slope contribution is damped by `solar_decline_magnitude` and `cloud_rise_magnitude`. Captures "forecast falling further behind every hour" 1-2 hours earlier than v1.
- **v1 (legacy halflife decay, fallback)** — `α·forecast_residual_f` (`α=exp(-h/2)`) plus a separate damped trend-residual carry (`obs_slope − forecast_forward_slope`). Used when fewer than 3 routines, slope unavailable, or the v2 setting is off.

For A/B comparison while v2 stabilizes, every exceedance log line carries both projections side-by-side: `projected=… (live v2), legacy_projected=… (v1)`. Grep `legacy_projected=` over a few days to compare lead time and RMS error vs the realized daily max — promote/revert via `PROJECTION_RESIDUAL_SLOPE_ENABLED`.

Optional Polymarket-discovery line via `projected_market_lookup.lookup_projected_binary` — finds the active binary closest to the projected max and prints its operator/threshold/YES price. Tunables live as module-level constants in `forecast_exceedance.py` (no config plumbing) except for the v2 master switch above; edit + restart.

### Edge calculator (`src/signals/edge_calculator.py`)

`compute_edges()` (for bracket markets) and `_binary_market_edge()` in `scheduler.py` (for binary threshold markets) both delegate filter checks to `_check_filters()`.

All filters must pass:

| Filter | Default | Source |
|--------|---------|--------|
| `edge >= MIN_EDGE` | 0.05 | `edge_calculator.MIN_EDGE` (hardcoded) |
| `our_probability >= MIN_PROBABILITY` | 0.60 | `settings.MIN_PROBABILITY` |
| `MIN_ENTRY_PRICE <= price <= MAX_ENTRY_PRICE` | 0.40 / 0.97 | `settings` |
| `routine_count >= min_routine_count` | 3 | `settings.MIN_ROUTINE_COUNT`, overridable via the `min_routine_count` kwarg (lock-rule path passes 2 since it gates routine count per its own super-margin rules) |
| `minutes_to_close >= MARKET_CLOSE_BUFFER_MINUTES` | 30 | `settings` |
| `depth >= MIN_DEPTH_USD` | 10 | `settings` |

Near-resolved markets (`price >= 0.95 or 0 < price <= 0.05`) are skipped in `job_unified_pipeline` **before** edge computation.

### Binary vs bracket markets

`_is_binary_market(market)` in `scheduler.py` returns True when `parsed_threshold` and `parsed_operator` are set and operator != "bracket".

- **Binary** (`above`/`at_least`/`below`/`at_most`/`exactly`): buckets span `[current_max-1, max(threshold, forecast_peak)+10]`, prob collapsed by operator, single `BucketEdge`.
- **Bracket**: buckets extracted from `market.outcomes` via regex, prices pulled from the market row, full per-bucket edge list. (Only `current_yes_price` is populated right now, so most brackets only get one priced bucket.)

### Station bias (`src/ingestion/station_bias.py`)

Per-ICAO rolling mean of `observed_daily_max_c - forecast_peak_c` over `STATION_BIAS_WINDOW_DAYS` (30). Added to Open-Meteo peak to produce `forecast_peak_f`.

- `get_bias(session, icao)` — 30-day mean, falls back to `DEFAULT_STATION_BIAS_C` (1.0°C) when no history.
- `record_daily_outcome(...)` — called in `job_daily_settlement` for every station with ≥3 routine METARs.
- `is_bias_runaway(session, icao)` — returns True when `|bias| > STATION_BIAS_MAX_C` (3°C). Such cities are skipped in Phase 1 for that tick.

### Kelly sizing (`src/risk/kelly.py`)

`size_position()` applies cascading caps:
1. Fractional Kelly with `KELLY_FRACTION=0.25`
2. Per-trade cap: `MAX_POSITION_PCT=5%` of bankroll
3. Total exposure cap: `MAX_EXPOSURE_PCT=25%` of bankroll minus current open exposure
4. Hard USD cap: `MAX_POSITION_USD=200`
5. Orderbook depth cap: `DEPTH_POSITION_CAP_PCT=20%` of visible depth
6. Minimum viable trade: `MIN_TRADE_USD=5` (below returns 0)

The drawdown monitor multiplier (`DrawdownMonitor.check`) is applied on top of the Kelly stake in the scheduler loop before the `MIN_STAKE_USD` check. The lock-rule path (`size_locked_position`) uses fixed sizing instead of Kelly: `LOCK_POSITION_PCT=2%` of bankroll, capped at `MAX_POSITION_USD/2=$100` and 15% of depth — but the same `MAX_EXPOSURE_PCT`, `MIN_TRADE_USD`, and drawdown multiplier still apply.

#### Drawdown state machine (`src/risk/drawdown.py`)

Four states, not three. Multipliers in parentheses:

| State | When | Multiplier |
|---|---|---|
| `NORMAL` (1.0) | `current_bankroll ≥ peak`, OR `dd_pct < CAUTION_THRESHOLD (10%)` while previously NORMAL |
| `CAUTION` (0.5) | `dd_pct ≤ PAUSE_THRESHOLD (20%)` |
| `PAUSED` (0.0) | `dd_pct > PAUSE_THRESHOLD` |
| `RECOVERY` (0.5) | `dd_pct < CAUTION_THRESHOLD` AND previous level was CAUTION/PAUSED/RECOVERY — provides hysteresis so the bot doesn't snap back to full size on a single up-tick |

Exit `RECOVERY` → `NORMAL` only when `current_bankroll >= peak`.

#### Bankroll = equity, not wallet liquidity (`src/resolution.py`)

`get_current_bankroll(session)` returns **equity**, not the live USDC balance:

```
bankroll = wallet_usdc + Σ (stake_usd + pnl) for trades WHERE status=WON AND redeemed_at IS NULL
```

Polymarket wins do not auto-settle into wallet USDC — the user must run `bet redeem` (which calls `redeemPositions()` on-chain) to convert the conditional tokens into wallet balance. Until then the future payout sits as conditional tokens. Without the unredeemed-WON adjustment, the drawdown monitor sees `peak = wallet + pnl` (from the prior `BankrollLog` write) but `current = wallet`, triggers a phantom `CAUTION`/`PAUSED`, and shrinks every Kelly stake by 0.5×–0× until the user redeems. The `(stake + pnl)` term equals `stake/entry` — exactly the dollar value the conditional tokens will return on redemption — so once redeemed, wallet rises by the same amount the unredeemed sum drops by, with no double-count. `get_unredeemed_won_payout(session)` exposes the adjustment for diagnostics.

`bet redeem` stamps `Trade.redeemed_at = now()` on every WON row whose `market_id` matches a successful `redeemPositions()` receipt (`Market.id` is the `condition_id` in our DB). `BankrollLog` is written at 22:00 UTC by `job_daily_settlement` from the same `get_current_bankroll` value, so the persisted history reflects equity too.

### Circuit breakers (`src/risk/circuit_breakers.py`)

Checked once per unified pipeline tick *and* every fast-poll tick, before any city is evaluated:
- Daily loss stop: halt when today's P&L < `-DAILY_LOSS_STOP_USD` (-$200).
- Consecutive loss stop: `CONSECUTIVE_LOSS_PAUSE_COUNT` LOST trades in a row (default 3) → pause `CONSECUTIVE_LOSS_PAUSE_HOURS` (2h). `_paused_until` lives in process memory; consecutive losses are re-queried from the DB each check, so the pause re-engages on restart if the streak still holds.

Both counts are configurable via `settings`. Per-city routine-count and bias-runaway checks live in the aggregator and edge filter, not here. The fast-poll dedup state (`_locked_markets_fired_today`, `_last_routine_seen`) is also in-process and reset per-station at the station's local-day rollover (`_maybe_clear_per_station_caches`).

## How To: Common Tasks

### Add a new city

1. Add `"city name": (lat, lon)` to `CITIES` in `src/signals/mapper.py`.
2. Add `"city name": "ICAO"` to `CITY_ICAO`.
3. Confirm the ICAO is **not** in `_EXCLUDED_ICAOS` in `src/scheduler.py` (currently `{"VHHH", "LLBG"}` for Hong Kong / Tel Aviv where Polymarket's resolution source diverges from the routine METAR feed). Removing an entry from that set requires verifying the actual resolver station first.
4. If the city uses °C in its market title, make sure the regex in `src/ingestion/polymarket.py` captures it (°C/CELSIUS handled by `_market_unit` in `scheduler.py`).
5. Seed `StationBias` if you have historical data — otherwise the default +1.0°C bias applies until enough settlements accumulate.

### Tune trade filters

Everything is in `src/config.py` (Pydantic `Settings`, overridable via `.env`):
- `MIN_EDGE` (default 0.05; harmonised — `edge_calculator._check_filters` reads `settings.MIN_EDGE` directly)
- `MIN_PROBABILITY`, `MIN_ENTRY_PRICE`, `MAX_ENTRY_PRICE`
- `MIN_DEPTH_USD`, `MIN_ROUTINE_COUNT`, `MARKET_CLOSE_BUFFER_MINUTES`
- Kelly caps: `KELLY_FRACTION`, `MAX_POSITION_PCT`, `MAX_POSITION_USD`, `DEPTH_POSITION_CAP_PCT`
- `APPLY_CALIBRATION` (default False) — when True, the unified pipeline refreshes the consensus calibration cache each tick and applies the linear correction to the chosen side's probability before edge filtering. See `src/signals/consensus.py`.

The lock-rule path has its **own** knob set (`LOCK_RULE_*`, `LOCK_MARGIN_F`, `LOCK_POSITION_PCT`, `FAST_LOCK_POLL_*`) — changing the directional `MIN_EDGE` etc. does not affect it. The ensemble σ knobs (`ENSEMBLE_*`) only affect Phase 1 σ in the probability engine.

Forecast-exceedance / projection knobs:
- `PROJECTION_RESIDUAL_SLOPE_ENABLED` (default `True`) — flip off to revert `_project_daily_max` to the legacy halflife decay path without code change.
- Tunables live as module-level constants in `src/signals/forecast_exceedance.py`: `RESIDUAL_SLOPE_MIN_POINTS=3`, `RESIDUAL_SLOPE_HOURS_CAP=3.0`, `RESIDUAL_SLOPE_MAX_F_PER_HR=1.5`, `STRONG_RESIDUAL_DELTA_F=1.0`, `STRONG_RESIDUAL_MIN_ROUTINES=2`, `MIN_ROUTINE_COUNT_FOR_PUSH=3`, `DELTA_THRESHOLD_F=1.0`, `MAX_OVERSHOOT_F=5.0`, `ALERT_COOLDOWN=30 min`. Edit + restart.

### Change pipeline cadence

`UNIFIED_PIPELINE_INTERVAL_MINUTES` in `.env` (default 5). `job_resolve_trades` is hardcoded to a 5-min interval (with a 30s offset from unified to spread CLOB load). The cron-style daily settlement job is fixed at 22:00 UTC in `setup_scheduler()`.

### Add / modify probability signals

Signals live in `src/signals/probability_engine.py` as `_apply_*` helpers. Each should:
- Take `(center_or_sigma, state, reasoning)` and return the updated value.
- Append a one-line explanation to `reasoning`.
- Respect the monotonicity constraint (applied last, after raw distribution).

### Test the edge calculator without running the daemon

`pytest tests/test_edge_calculator.py` — uses synthetic `BucketDistribution` + price dicts. See `compute_edges(dist, prices, routine_count, market_end_time, orderbook_depths=...)` for the call shape.

## Testing Conventions

- Tests in `tests/test_<module>.py`.
- Mock external APIs at the module boundary (e.g. `@patch("src.signals.state_aggregator.fetch_forecast")`).
- `AsyncMock` for async functions; session mocking via `AsyncMock()` with explicit `session.flush = AsyncMock()`.
- Do **not** run `ruff` as part of the verification flow — it's not used on this project.

## File Index

| File | Purpose | Key exports |
|------|---------|-------------|
| `src/scheduler.py` | APScheduler jobs + health server + binary/bracket edge helpers + lock-rule executor + fast-poll projection + per-station cache rollover + DB-backed dedup helpers + telemetry log writer | `job_scan_markets`, `job_resolve_trades`, `job_unified_pipeline`, `job_fast_lock_poll`, `job_daily_settlement`, `_maybe_clear_per_station_caches`, `_try_lock_rule_trade`, `_fast_poll_projection_check`, `_has_active_trade`, `_upsert_signal`, `_log_evaluation`, `_is_binary_market`, `_binary_market_edge`, `_should_skip_future_day`, `_EXCLUDED_ICAOS`, `run_scheduler` |
| `src/signals/state_aggregator.py` | Per-ICAO weather state snapshot, deterministic + ensemble forecast blend, residual-slope fit, fast-poll input cache | `WeatherState`, `CachedAggregationInputs`, `aggregate_state`, `build_state_from_metars`, `_blend_forecasts`, `_compute_residual_slope`, `get_cached_aggregation_inputs`, `clear_state_cache`, `clear_state_cache_for_icao` |
| `src/signals/probability_engine.py` | Signal-based bucket distribution | `BucketDistribution`, `compute_distribution` |
| `src/signals/edge_calculator.py` | Per-bucket edge + filter checks (`min_routine_count` overridable) | `BucketEdge`, `compute_edges`, `_check_filters`, `MIN_EDGE` |
| `src/signals/lock_rules.py` | Deterministic physical-lock decisions (super-margin EASY at routine #2) | `LockDecision`, `evaluate_lock` |
| `src/signals/forecast_exceedance.py` | "Daily max set to beat forecast" alerts + DB calibration history; v2 slope-projection (`PROJECTION_RESIDUAL_SLOPE_ENABLED`) with v1 parallel-logged | `check_and_record_daily_max_alert`, `_project_daily_max`, `_project_with_residual` |
| `src/signals/projected_market_lookup.py` | Find the active binary closest to a projected daily max | `lookup_projected_binary` |
| `src/signals/mapper.py` | Geocoding, ICAO lookup, operator/date/threshold normalisation, station-local timezones | `CITIES`, `CITY_ICAO`, `icao_for_location`, `cities_for_icao`, `geocode`, `icao_timezone`, `unit_for_station`, `normalize_operator`, `convert_threshold`, `f_to_c` |
| `src/signals/consensus.py` | Linear recalibration `actual ≈ slope*predicted + intercept` from resolved signals, plus an in-process TTL cache (`refresh_calibration` / `get_cached_calibration`) and a sync `apply_calibration(prob) → (corrected, applied_bool)`. Wired into `_binary_market_edge` post-side-selection when `settings.APPLY_CALIBRATION=True`. | `get_calibration_coefficients`, `refresh_calibration`, `apply_calibration`, `MIN_CALIBRATION_SAMPLES` |
| `src/signals/reverse_lookup.py` | Find markets by city/station/observation | `find_markets_for_city`, `find_markets_for_station`, `find_markets_for_observation`, `find_markets_for_event` |
| `src/ingestion/polymarket.py` | Gamma API scanner + question parser | `scan_and_ingest`, `ingest_markets`, `get_active_weather_markets`, `parse_question`, `is_weather_market`, `parse_temperature_brackets` |
| `src/ingestion/openmeteo.py` | Deterministic + ensemble hourly forecast, solar/cloud/dewpoint helpers | `OpenMeteoForecast`, `fetch_deterministic_forecast`, `fetch_ensemble_forecast`, `solar_declining`, `cloud_rising`, `dewpoint_trend` |
| `src/ingestion/aviation/` | Multi-provider METAR/TAF/PIREP/SIGMET | `fetch_metar_history`, `fetch_latest_metars`, `get_routine_daily_max`, `detect_metar_cycle`, `get_temp_trend`, `taf_amendment_count`, `has_severe_weather_reports`, `alerts_affecting_location`, `get_aviation_weather_picture` |
| `src/ingestion/station_bias.py` | Per-station forecast bias tracking (anchored to deterministic peak) | `get_bias`, `record_daily_outcome`, `is_bias_runaway` |
| `src/ingestion/wx.py` | Weather.com v3 observations (optional) | gated by `WX_API_KEY` |
| `src/execution/polymarket_client.py` | CLOB client, orderbook, orders | `is_live`, `get_token_ids`, `place_order`, `check_order_status`, `cancel_order`, `get_best_bid_ask`, `get_orderbook_depth`, `get_daily_spend` |
| `src/execution/alerter.py` | Telegram notification queue + inline buttons (exec/skip/detail) + market-discovery alerts | `Alerter`, `AlertType`, `get_alerter`, `_escape_md2` |
| `src/risk/kelly.py` | Fractional Kelly + lock-rule fixed sizing | `PositionSize`, `size_position`, `size_locked_position`, `MIN_TRADE_USD`, `MAX_EXPOSURE_PCT` |
| `src/risk/drawdown.py` | Four-state drawdown machine with hysteresis | `DrawdownMonitor`, `DrawdownLevel` (`NORMAL`/`CAUTION`/`PAUSED`/`RECOVERY`) |
| `src/risk/circuit_breakers.py` | Daily loss + consecutive loss halts | `check_circuit_breakers`, `CircuitBreakerState` |
| `src/risk/simulate.py` | Paper-trade simulator | used by `cli paper-trade` |
| `src/resolution.py` | Trade settlement, bankroll-as-equity, exposure, CLOB-refreshed resolution price | `resolve_trades`, `_refresh_market_price`, `get_current_bankroll`, `get_unredeemed_won_payout`, `get_current_exposure`, `calculate_daily_pnl` |
| `src/db/models.py` | SQLAlchemy ORM. `Trade.redeemed_at` is the redemption-tracking timestamp set by `bet redeem` (NULL means winnings still locked in conditional tokens). `Trade.submit_*` are submit-time market context for slippage analysis. `Signal.signal_kind`/`lock_*` split rows by which path produced them. | `Market`, `Signal`, `Trade`, `EvaluationLog`, `StationBias`, `StationNormal`, `MetarObservation`, `TafForecast`, `Pirep`, `WxObservation`, `AviationAlert`, `ForecastExceedanceAlert`, `ForecastArchive` |
| `src/ingestion/forecast_archive.py` | Best-effort writer that snapshots every `OpenMeteoForecast` blend into the `ForecastArchive` table for replay-capable backtests | `archive_forecast_snapshot` |
| `src/config.py` | Pydantic settings | `settings` |
| `src/cli.py` | CLI entry points | `run`, `scan`, `backfill`, `status`, `paper-trade`, `backtest-v2`, `migrate`, `approve`, `test-trade`, `bet {place,info,search,find,cancel,orders,portfolio,redeem}` |
| `scripts/backtest_lock_rule.py` | Replay the lock-rule trader against resolved markets + DB METARs | standalone CLI |
| `scripts/debug_pipeline.py` | Trace the unified pipeline for one market/station, no orders | standalone CLI |
| `scripts/inspect_loss.py` | Post-mortem drilldown for a single losing trade | standalone CLI |
| `scripts/backfill_station_bias_tz.py` | Idempotent recompute of station bias from DB METARs (timezone-correct windows) | one-shot |
| `scripts/compare_projection_versions.py` | Phase 1.3 — joins `forecast_exceedance_alerts` to realised daily max (single batch query, not N+1); reports v2 projection RMSE / mean-bias by station + lead bucket + alerted flag. v1/v2 head-to-head deferred until `forecast_archive` has ≥1 week of data. | `PYTHONPATH=. python scripts/compare_projection_versions.py --days 30` |
| `scripts/station_calibration_report.py` | Phase 1.4 — per-ICAO markdown dashboard combining `StationBias` rolling mean, projection error, push hit rate, lock-rule strike rate, and probability-path realised edge / Brier. Output committed to `reports/calibration/stations_baseline.md`. | `PYTHONPATH=. python scripts/station_calibration_report.py --days 30` |

## Live Execution

Orders are placed via `py-clob-client` in `src/execution/polymarket_client.py::place_order`.

**Flow per trade:**
1. Scheduler creates `Signal` + `Trade(status=PENDING)` row.
2. `place_order` checks `DAILY_SPEND_CAP_USD`, resolves YES/NO token IDs (Gamma API), places a FOK (fill-or-kill) market order.
3. On fill: trade → `OPEN`, `order_id`/`fill_price`/`filled_size` populated, Telegram alert.
4. On failure: trade stays `PENDING` (no retry logic), warning logged.

**Safety:**
- `AUTO_EXECUTE=false` (default) — no orders sent; signals still logged and Telegram-notified.
- `POLYMARKET_PRIVATE_KEY` empty → dry-run mode; `py_clob_client` never imported.
- `DAILY_SPEND_CAP_USD=200` — hard 24h rolling cap, checked before every order.
- `MIN_STAKE_USD=5` — orders below this skipped after drawdown multiplier.
- `DrawdownMonitor` multiplies stake by a level-dependent factor (`NORMAL=1.0`, `CAUTION=0.5`, `PAUSED=0.0`, `RECOVERY=0.5`).
- Lock-rule auto-disable: `LOCK_RULE_LOSS_DISABLE_COUNT=3` lock losses within `LOCK_RULE_LOSS_WINDOW_HOURS=72` flip the path off until manually re-enabled.

**Going live:**
1. Fund a Polygon wallet with USDC.
2. Approve contracts (once) — see `docs/` for the allowance script. Contracts:
   - `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` (main exchange)
   - `0xC5d563A36AE78145C45a50134d48A1215220f80a` (neg-risk exchange)
   - `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296` (neg-risk adapter)
3. `.env`: `POLYMARKET_PRIVATE_KEY=0x...`, `AUTO_EXECUTE=true`, `DAILY_SPEND_CAP_USD=50` (start small).
4. There is no testnet — Polygon mainnet only, real USDC.

## Telemetry (calibration data sources)

Three append-/refresh-as-you-go data streams unblock strategy iteration once the daemon is producing live trades. Migration `j0k1l2m3n4o5_strategy_telemetry` introduced them; `signals` continues to carry one row per `(market_id, direction)` and is the chosen-trade rationale, while these capture the surrounding evaluation context.

- **`evaluation_logs`** — append-only row per per-side edge evaluation, written by `_log_evaluation` from BOTH the probability path (after `_binary_market_edge`) and `_try_lock_rule_trade` (after `evaluate_lock`, plus at each rejection point). Captures `model_prob/market_prob/edge/passes/reject_reason/depth_usd/minutes_to_close/routine_count/signal_kind`. Use this — not `signals` — for filter-tuning backtests (lower `MIN_EDGE`? raise `MIN_DEPTH_USD`?), since `signals` only carries passing edges and is de-duplicated. Volume ≈ 11.5K rows/day; index `ix_eval_logs_market_created` supports per-market time-series queries.
- **`trades.submit_yes_bid` / `submit_yes_ask` / `submit_depth_usd` / `submit_at`** — populated by `place_order` before the FAK call (live AND dry-run, since the diagnostic value of "what did the book look like at submission" is the same). Lets slippage analyses decompose `fill_price - entry_price` into spread vs depth-walked. NULL on rows from before the migration.
- **`signals.signal_kind`** ('probability' | 'lock') + **`signals.lock_branch`** ('easy_super' / 'easy_standard' / 'hard' / 'range_overshoot' / 'range_undershoot' / 'range_in_window') + `lock_routine_count` + `lock_observed_max_f` — set by `_upsert_signal` on every refresh; lets you split realised P&L by which deterministic lock branch fired (e.g. "are EASY-super-margin locks at routine_count=2 over- or under-firing?").

## Gotchas

- **Near-resolved skip is 0.99 / 0.01**, not 0.95. Threshold was raised so the lock-rule path can evaluate prices in the 0.90–0.99 zone (where Wunderground UI hasn't yet caught up to a freshly-published METAR).
- **Lock-rule fires before the near-resolved skip.** Phase 2 runs `_try_lock_rule_trade` first; only if it returns `None` does the probability path see the market. A lock that's "fired but rejected" (price out of `[0.05, 0.95]`, depth too thin, close-buffer breach) returns `0.0` and the market is skipped for the rest of this tick.
- **Dedup is DB-backed first, in-process second.** `_has_active_trade(session, market_id, direction)` is the durable safety net used in BOTH the probability path and `_try_lock_rule_trade`: any PENDING/OPEN Trade row for this `(market_id, direction)` short-circuits a second attempt. This is what stops the live daemon from placing 5+ real bets on the same market+side as Kelly exposure caps slowly catch up. The in-process `_unified_fired_today` / `_locked_markets_fired_today` sets are kept as same-tick speed-ups (skip the DB roundtrip when we know we already fired this tick) but no longer load-bearing. Combined with `uq_signals_market_direction` on the Signal table (migration `i9j0k1l2m3n4`) and the `_upsert_signal` helper that does INSERT-or-REFRESH, repeat ticks now refresh the existing Signal row instead of inserting duplicates.
- **Fast-poll lock-rule dedup is also aggressive on filter rejects.** `_locked_markets_fired_today.add(market.id)` fires inside `job_fast_lock_poll` even when the lock was *filtered out* (depth, close buffer) — by design, re-trying every 30s wouldn't change the rejection cause. Cleared per-station when the station's `today_local(tz)` rolls over (`_maybe_clear_per_station_caches`).
- **Per-station rollover, not global UTC clear.** `_maybe_clear_per_station_caches` runs at the top of every unified tick. On the first sighting of a station it just seeds `_local_day_seen` (no clear) — this avoids clobbering dedup entries that fast-poll may have written between process boot and the first unified tick. The dedup→ICAO link is stored in `_market_to_icao` at lock-fire time; if a market is added to `_locked_markets_fired_today` outside `job_fast_lock_poll`, the rollover won't find it.
- **Fast-poll projection check needs a warm cache.** `_fast_poll_projection_check` is a no-op for a station until `aggregate_state` has run for it once (cache populated) and within the 30-min TTL. After a restart, the first 5 minutes of fast-poll alerts will be silent until the next unified tick warms the cache — expected, not a bug.
- **`_state_cache` is in-process, never persisted.** Resets on restart and per-station on local-day rollover. Forecast and bias drift slowly within 30 minutes, so the staleness ceiling is fine for fast-poll's projection use case but not for anything that needs day-spanning data.
- **`job_resolve_trades` ≠ `job_daily_settlement`.** Resolution moved to a 5-min interval job; settlement at 22:00 UTC only does bankroll/drawdown bookkeeping, daily summary, station bias, and weekly calibration. A trade's `closed_at` is now within ≤5 min of the market's `end_date`, not the next 22:00 UTC tick.
- **Resolution price is refreshed at settlement time.** `resolve_trades` calls `_refresh_market_price` on each candidate before applying the 0.95/0.05 thresholds. The 15-min `job_scan_markets` and 5-min `job_unified_pipeline` writes can leave `current_yes_price` 30+ min stale by the time the market expires; the live CLOB mid is the source of truth for resolution. Failures fall back to the stored price and retry next tick.
- **Bankroll is equity, not wallet.** `get_current_bankroll` adds `Σ(stake + pnl) for WON trades with redeemed_at IS NULL` to the wallet balance. Without this, won trades whose conditional tokens haven't been on-chain redeemed yet show as a phantom drawdown. `bet redeem` stamps `Trade.redeemed_at` on success; `Market.id` doubles as the on-chain `condition_id` for the matching update.
- **`Trade.redeemed_at` migration backfill assumes pre-existing wins were already redeemed** (sets `redeemed_at = closed_at`). If you have actual unredeemed wins at deploy time, NULL their `redeemed_at` after running `alembic upgrade head`, or run `bet redeem` first. Migration `g7h8i9j0k1l2_add_trade_redeemed_at`. The Postgres `tradestatus` enum stores **uppercase names** (`'WON'`/`'LOST'`/`'OPEN'`/`'PENDING'`) — SQLAlchemy's default `Enum(TradeStatus)` uses enum names, not `.value` strings; raw SQL touching this column must use the uppercase form.
- **`projected_max_f` in `forecast_exceedance_alerts` is whichever path is live (v2 by default).** The legacy v1 value is **only** in the JSON logs (`legacy_projected=`), not the DB. If you need to compare v1 vs v2 historically, parse logs. Promoting v2 permanently means deleting the parallel-logging block in `check_and_record_daily_max_alert`.
- **Bias is recorded against the deterministic single-source peak**, not the ensemble blend (`job_daily_settlement` step 4). Reason: keep the bias reference frame stable when the ensemble model list changes. The pipeline still trades on the bias-adjusted ensemble peak.
- **Dry-run trades stay `PENDING` in BOTH paths and keep their requested `stake_usd`.** When `place_order` returns from the dry-run branch (`polymarket_client.py:308-318`) it sets `trade.exchange_status="dry_run"` without touching `stake_usd`. Both the lock-rule executor and the probability-path post-place block check that flag, leave `status=PENDING`, and emit a Telegram alert prefixed "(dry-run)" with an "Indicative" stake/price line. The DB row carries the requested stake (useful for paper-trade analysis) — exposure / P&L math stays clean because every consumer (`get_current_exposure`, `resolve_trades`, `calculate_daily_pnl`, `get_unredeemed_won_payout`, `cli status`) filters on `status IN (OPEN, WON, LOST)`, never PENDING. **Invariant for new code:** any future query that sums `Trade.stake_usd` must filter on `status` (or exclude `exchange_status='dry_run'`); otherwise dry-run rows will double-count.
- **Hong Kong (VHHH) and Tel Aviv (LLBG) are silently skipped** even though they're in `CITIES`/`CITY_ICAO`. `_EXCLUDED_ICAOS` in `scheduler.py` filters them at the grouping step because their Polymarket resolution station diverges from the routine METAR feed we consume.
- **`Signal.aviation_prob` is still read by the alerter** (inline-button "detail" view) — it's NULL for unified-pipeline signals, so the detail view shows a stale dash. Surprising but not a bug.
- **Orderbook fetching is not disabled.** `job_unified_pipeline` calls `get_best_bid_ask` and `get_orderbook_depth` for every market with token IDs; 404s on resolved tokens produce the `Could not fetch orderbook for token X after 3 attempts` warning twice per market (once per call, since the failed fetch isn't cached). The subsequent skip comes from the `price >= 0.99 or price <= 0.01` check using the stale DB price.
- `py_clob_client` and `eth_account` are imported **inside** functions in `polymarket_client.py`, not at module level, so the system runs in dry-run mode without those deps installed.
- `_fetch_orderbook` caches only **successful** fetches for 30s. Failures are not negative-cached, so two back-to-back calls to a dead token retry 3×3×throttle delay.
- `WeatherState` is produced **even if Open-Meteo fails** — in that case `forecast_peak_f = current_max_f`, `hours_until_peak = 0`, solar/cloud signals are False, `forecast_sigma_f = None`, `has_forecast = False`. Probability distribution degenerates to a narrow band around current max; lock rule's HARD direction is disabled.
- Dewpoint trend in `state_aggregator.py` uses recent METARs (6h window), not Open-Meteo. `openmeteo.dewpoint_trend` exists but isn't wired into the pipeline.
- `consensus.py` is a vestigial filename — it no longer blends multi-model forecasts. Today it (a) fits linear calibration coefficients from resolved signals, (b) caches them in-process for 30 min via `refresh_calibration`, and (c) exposes `apply_calibration` which the unified pipeline calls inside `_binary_market_edge` (post-side-selection) **only when `settings.APPLY_CALIBRATION=True`** (default False). Default-off because the bake-off in `reports/calibration/` hasn't been run yet — flip on, run a backtest, compare Brier, then promote.
- `MIN_EDGE` is now harmonised on `settings.MIN_EDGE` (default 0.05). The module-level `edge_calculator.MIN_EDGE` is kept as a re-export for back-compat but `_check_filters` reads `settings.MIN_EDGE` directly. If you have an older `.env` with `MIN_EDGE=0.10`, that value still wins via env override — review it before assuming the default applies.
- `gfs.py`/`ecmwf.py`/`detector.py`/old `mapper.py` signal-generation paths are gone. The `Signal.gfs_prob`/`ecmwf_prob` columns remain in the schema but are always NULL.
- **Backtest harnesses exist as standalone scripts**, not as a module in the pipeline: `python -m src.cli backtest-v2` (probability path) and `python scripts/backtest_lock_rule.py` (lock path). Earlier docs claimed there was no backtest — that's stale.
- Token IDs are refetched from Gamma per order and per price call. High-frequency trading would benefit from caching on the `Market` row, but current 5-min cadence makes it a non-issue.
- FOK orders fully fill or cancel — no partial fills; thin books leave trades in `PENDING`.
- **`place_order` posts FAK buys via `MarketOrderArgsV2` + `create_market_order`, not `OrderArgsV2` + `create_order`.** The CLOB enforces a stricter precision rule on FAK/FOK buys (`market buy orders maker amount supports a max accuracy of 2 decimals`) than on resting limit orders. The limit-order builder produces `makerAmount = round(size, 2) × price` with up to `round_config.amount` decimals (4 for tick=0.01) and is rejected; the market-order builder takes USDC `amount` directly (rounded to `round_config.size=2`) and derives shares as `amount/price`, which fits the rule by construction. We pass our tick-snapped `limit_price` so the SDK skips its network `calculate_market_price` call. If you ever switch back to the limit builder you'll see `400 invalid amounts` on every live order.
- Aviation has its own rate-limit semaphores per provider (`_RATE_LIMIT_RPS` constants) independent of the pipeline semaphore (`_UNIFIED_CONCURRENCY=8`).
- `_paused_until` circuit-breaker state resets on process restart. Consecutive losses are re-queried from DB each check, so the pause re-engages if the streak still holds.
- `Market.current_yes_price` is **read-only** in the unified pipeline, fast-lock poll, and resolve_trades paths. They take the live mid from `get_best_bid_ask` into a local `live_price` and never assign back to the ORM row — `expire_on_commit=False` plus default autoflush meant any later `session.add(Signal/Trade)` was flushing dirty Market mutations to disk and causing cross-transaction deadlocks (multiple jobs UPDATEing overlapping markets in different orders). Only `job_scan_markets` (`ingest_markets`) persists this column, and it iterates raw markets sorted by id so the UPDATEs are emitted in deterministic primary-key order.
