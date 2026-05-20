# CLAUDE.md

## Project Overview

WeatherEdge is a weather-driven trading bot for Polymarket daily "highest temperature in [city]" markets. Two trading paths run concurrently:

1. **Probability path** — every 5 minutes, aggregate live METAR observations + a deterministic Open-Meteo forecast (with multi-model ensemble σ) per city, build a probability distribution over integer temperature buckets, compare to live Polymarket CLOB prices, and size positions via fractional Kelly.
2. **Lock-rule path** — every 30 seconds, re-check active binary markets for new routine METARs and fire deterministic trades when the daily max is mathematically locked above (or below) threshold.

Resolution source is the Wunderground history page, which mirrors **routine** (non-SPECI) METARs. The bot reads METARs directly and typically sees the same data 1–2 hours before the UI updates — that's the edge both paths exploit.

## Quick Reference

```bash
pytest tests/                          # full test suite
mypy src/ --ignore-missing-imports     # type check (no ruff)

python -m src.cli scan                 # one-shot market scan
python -m src.cli run                  # full daemon (health on :8080)
python -m src.cli backfill --days 30   # historical markets
python -m src.cli backtest-v2 --days 30   # replay probability pipeline
python -m src.cli paper-trade --days 30   # dry simulation
python -m src.cli evals-report --days 7   # filter-tuning markdown report
python -m src.cli bet find --city Phoenix --date 2026-04-30  # market lookup
python scripts/backtest_lock_rule.py --days 30  # lock-rule backtest
```

## Architecture

### Pipeline (critical path)

The unified pipeline runs in `src/scheduler/__init__.py::job_unified_pipeline` every 5 min. `job_fast_lock_poll` runs every 30 s.

```
circuit_breakers ─► get_active_weather_markets ─► group by ICAO (skip _EXCLUDED_ICAOS)
        │                                             │
        ▼                                             ▼
  Phase 1 (concurrent, sem=8):                 Phase 2 (sequential, per market):
    aggregate_state(session, icao, lat, lon)     ├─ skip future-day markets
      ├─ fetch_metar_history(icao, 24h)          ├─ get_token_ids (Gamma)
      ├─ fetch_deterministic_forecast \\         ├─ get_best_bid_ask → live yes_price
      │  +  fetch_ensemble_forecast    } gather  ├─ get_orderbook_depth
      ├─ _blend_forecasts (det. central +        ├─ _try_lock_rule_trade — if locked: size,
      │   ensemble σ at peak hour)               │   filter, place_order, continue
      ├─ get_bias(session, icao)                 ├─ skip if price ≥ 0.99 or ≤ 0.01
      ├─ check_and_record_daily_max_alert        ├─ compute_distribution(state, buckets)
      └─ returns WeatherState                    ├─ _binary_market_edge / compute_edges
                                                 ├─ size_position (fractional Kelly × dd_mult)
                                                 └─ place_order

  job_fast_lock_poll (30s):
    fetch_latest_metars(active_icaos) ─► for each new routine (icao, market):
        evaluate_lock(_minimal_state, market) ─► _try_lock_rule_trade
        _fast_poll_projection_check(icao, station_metars) — reuses _state_cache
```

`_locked_markets_fired_today`, `_last_routine_seen`, and the `_state_cache` (in `state_aggregator.py`) are in-process dicts cleared **per-station at the station's local-day rollover** by `_maybe_clear_per_station_caches` (called at the top of every `job_unified_pipeline` tick). A sibling `_market_to_icao` map populated at lock-fire time lets the rollover find which `market_id` entries belong to which station; `_local_day_seen` cookies the last-observed `today_local(tz)` per ICAO. A bias-runaway station (`|bias| > 3°C`) is skipped at Phase 1 grouping.

`job_fast_lock_poll` only fires the **EASY** lock direction (observed max already clears threshold + margin). The **HARD** direction (no-more-heating: market-day max below threshold AND past-peak signal AND forecast peak < threshold) needs full forecast context and stays in `job_unified_pipeline`. For the **forecast-exceedance projection** check, fast-poll reuses the previous unified tick's forecast / bias / climate-normals via `_state_cache` (30-min TTL keyed by ICAO).

### Scheduler Jobs (`src/scheduler/__init__.py`)

| Job | Schedule | Purpose |
|-----|----------|---------|
| `job_scan_markets` | Every 15 min | Fetch Polymarket weather markets via Gamma API, upsert `Market` + `MarketSnapshot` |
| `job_resolve_trades` | Every 5 min (30s offset) | Settle expired markets. Refreshes `current_yes_price` from CLOB before applying 0.95/0.05 thresholds so stale prices don't delay resolution. Telegram-pushes each new resolution |
| `job_unified_pipeline` | Every `UNIFIED_PIPELINE_INTERVAL_MINUTES` (5m) | Aggregate per-city weather state, evaluate lock + probability paths, place trades |
| `job_fast_lock_poll` | Every `FAST_LOCK_POLL_INTERVAL_SECONDS` (30s) | Bulk `fetch_latest_metars` for active binary-market ICAOs; fire EASY-direction lock trades + re-run projection check on new routines. Disabled by `FAST_LOCK_POLL_ENABLED=false` or `LOCK_RULE_ENABLED=false` |
| `job_daily_settlement` | 22:00 UTC | Bankroll/drawdown bookkeeping, Telegram daily summary, redeem nudge (💸 push when any WON trade has `redeemed_at IS NULL`), WX retention cleanup, station-bias recording, weekly calibration. **Does not** resolve trades or clear caches |
| `job_reconcile_orders` | Every `ORDER_RECONCILE_INTERVAL_MINUTES` (5m) | Polls trades where `order_id IS NOT NULL`, `fill_price IS NULL`, `exchange_status IN ('delayed','matched','matching')`, `status IN (PENDING, OPEN, WON)` (WON included since 2026-05-19 to backfill pre-fix CLOB-mid-flipped terminals; recomputes `pnl` against the corrected `entry_price` after backfill). Scope = trades opened in the last `ORDER_RECONCILE_LOOKBACK_HOURS` (24h) OR whose `market.end_date` is within the last 7 days (long-tail catch). Calls `check_order_status` per row. After the per-row pass: log-only **stale-OPEN sweep** (OPEN+null-fill, market ended past `STALE_OPEN_RECONCILE_GRACE_HOURS=4h`) + 🚨 Telegram heartbeat when the stuck set crosses `STUCK_ALERT_MIN_COUNT=5` OR `STUCK_ALERT_EXPOSURE_FRACTION=50%` of the effective cap (cooldown `STUCK_ALERT_COOLDOWN_HOURS=4h` in `bot_state`). The sweep NEVER marks LOST — that's the operator's `admin reconcile-stuck` flow. No-op in dry-run |

`job_startup` runs once on boot.

### Data sources

| Source | Module | Purpose |
|--------|--------|---------|
| Polymarket Gamma | `src/ingestion/polymarket.py` | Active market list (via event-slug enumeration), parsed threshold/operator/location/date |
| Polymarket CLOB | `src/execution/polymarket_client.py` | Live best bid/ask, orderbook depth, order placement |
| Aviation METAR (6 providers) | `src/ingestion/aviation/` | Routine METAR history, daily max, trend, cycle detection. Failover: AWC → IEM → OGIMET → NOAA → CheckWX/AVWX |
| Open-Meteo | `src/ingestion/openmeteo.py` | Hourly temp/cloud/solar/dewpoint. Two endpoints fetched concurrently: `fetch_deterministic_forecast` (central peak — bias reference frame) + `fetch_ensemble_forecast` (`ENSEMBLE_MODELS=ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless`, inter-model spread). `_blend_forecasts` combines them. Either alone is acceptable; ensemble σ falls back to hours-based schedule when fewer than `ENSEMBLE_MIN_MODELS=3` models return data |
| Weather.com v3 | `src/ingestion/wx.py` | Optional auxiliary station observations (`WX_API_KEY` gated) |

### Database (`src/db/models.py`)

Core: `Market`, `MarketSnapshot`, `Signal`, `Trade`, `BankrollLog`, `StationBias`, `StationNormal`
Aviation: `MetarObservation`, `TafForecast`, `Pirep`, `SynopObservation`, `AviationAlert`
Alerting: `ForecastExceedanceAlert` (every same-hour delta > 0.5°F, with `alerted` flag)
Backtest: `ForecastArchive` — one row per (station, target-local-day, fetched_at) capturing the blended forecast. Written best-effort by `aggregate_state`; consumed by `simulate_distribution_pipeline`
Telemetry: `EvaluationLog` — append-only, one row per per-side edge evaluation (PASSING and REJECTED). Source of truth for filter-tuning backtests, since `signals` is de-duplicated per (market, side) and only carries passing edges
Telemetry: `DecisionLog` — append-only, one row per per-side **post-filter** decision (added 2026-05-16). Captures the funnel between "passed `_check_filters`" and "Trade row persisted": outcomes are `signal_written`/`trade_pending`/`trade_filled`/`dup_blocked_inproc`/`dup_blocked_db`/`stake_below_min`/`drawdown_paused`/`cluster_cap_hit`/`cap_exceeded`/`no_token_ids`/`no_client`/`no_fill`/`order_failed`. Use this when investigating "evals passed but no trades fired" — the 2026-05-16 drawdown-stuck-at-PAUSED debug took 4h because this gap was invisible
Runtime state: `BotState(key, value, updated_at)` — generic key/value table (JSONB value) for runtime values that must survive a process restart. Consumers: `circuit_breakers.paused_until` (consecutive-loss pause window), `reconcile.stuck_alert_last_pushed_at` (stuck-OPEN heartbeat cooldown — added 2026-05-18). Add new keys without a migration per key; dotted-string namespace convention.
Auxiliary: `WxObservation`

`Signal.signal_kind` ('probability' | 'lock') splits realised P&L by path. When `signal_kind='lock'`, `lock_branch`/`lock_routine_count`/`lock_observed_max_f` carry the structured `LockDecision` context. Legacy `gfs_prob`/`ecmwf_prob`/`aviation_prob`/`wx_prob` columns are always NULL in the unified pipeline; kept for schema compatibility (no live readers — drop via migration when convenient).

`Trade.submit_yes_bid`/`submit_yes_ask`/`submit_depth_usd`/`submit_at` — snapshot at `place_order` time. Lets slippage post-mortems decompose `fill_price - entry_price` into spread vs depth-walked. NULL on pre-migration rows and backtest paths.

## Key Patterns

### Probability engine (`src/signals/probability_engine.py`)

`compute_distribution(state, buckets)` produces a `BucketDistribution` with `probabilities: {bucket_f: prob}` summing to 1.0.

Signal combination:

1. **Baseline Gaussian** centered on `state.forecast_peak_f` (deterministic Open-Meteo peak + 30-day station bias).
2. **σ from `_compute_sigma`**: ensemble spread when available (`forecast_sigma_f × ENSEMBLE_SPREAD_MULTIPLIER=1.3`), clipped to `[floor, ENSEMBLE_MAX_SIGMA_F=5.0]`°F where `floor` comes from `_effective_sigma_floor(state)`: per-station rolling forecast-error RMSE (clamped to `[1.5, 5.0]`°F) when the station has ≥`_PER_STATION_SIGMA_MIN_DAYS=14` days of `station_biases` rows, else the global `ENSEMBLE_MIN_SIGMA_F=2.0`. Per-station floor was added 2026-05-16 because the global 2.0°F simultaneously over-pads low-variance stations (RJTT RMSE ≈ 1.7°F) and under-pads high-variance ones (KAUS/KLAX RMSE ≈ 5-8°F). Soft floor at half the hours-based schedule prevents runaway overconfidence. Hours-based fallback: 1.0 past peak, 1.5 (≤2h), 2.5 (≤4h), 3.5 (>4h).
3. **METAR trend shift** (three branches): *pre-peak* — shift by residual rate (`metar_trend_rate − forecast_slope_to_peak_f_per_hr`) when ≥0.5°F/hr, capped at +2°F (falls back to raw rate when forecast slope missing); *past peak + declining* (`rate ≤ 0`) — lock center to observed max; *past peak + still rising* (`rate > POST_PEAK_MIN_TREND_F_PER_HR=0.5`) — extrapolate `rate × hours × POST_PEAK_TREND_CARRY_K (0.75)`, capped at `POST_PEAK_MAX_SHIFT_F=3°F`, damped by solar/cloud magnitudes.
4. **Solar/cloud cap**: combined `solar_declining` AND `cloud_rising` → hard cap at observed max. Strong solar decline alone (`solar_decline_magnitude > 0.7`) → soft cap at `observed_max + 1°F`.
5. **Dewpoint adjustment**: rising Td >1°F/hr tightens sigma (max 0.5°F); falling Td <-1°F/hr widens it (max 0.3°F).
6. **Monotonicity constraint** (hard): `P(bucket < current_max_f) = 0`. Degenerate fallback collapses all mass to bucket closest to `current_max_f` when zeroed.

Every applied signal is appended to `distribution.reasoning`.

### State aggregator (`src/signals/state_aggregator.py`)

`aggregate_state` returns a `WeatherState` snapshot per city. Key fields:

- `metar_trend_rate` / `metar_trend_rate_short` — 6h and 2.5h linear regressions on routine METAR temps. Probability engine uses the 6h slope; `forecast_exceedance` uses the 2.5h slope.
- `routine_history` — sorted `(observed_at_utc, temp_f)` tuples for last 24 h. `lock_rules._market_daily_max` uses this to compute per-market daily max anchored to `market.end_date`'s **station-local** day.
- `latest_obs_temp_f`, `forecast_temp_now_f`, `forecast_slope_to_peak_f_per_hr`, `forecast_residual_f` — bias-adjusted residuals.
- `forecast_residual_slope_f_per_hr`, `forecast_residual_count` — `d(residual)/dt` linear regression over routines (computed by `_compute_residual_slope`). Drives v2 path of `_project_daily_max` when count ≥3.
- `forecast_sigma_f`, `ensemble_model_count` — ensemble σ at peak hour (None when ensemble unavailable).
- `has_forecast` — gates HARD-direction lock evaluation.

When Open-Meteo fails entirely, `WeatherState` is still returned: `forecast_peak_f = current_max_f`, `hours_until_peak = 0`, solar/cloud signals False, `forecast_sigma_f = None`, `has_forecast = False`. Probability distribution degenerates to a narrow band; lock rule's HARD direction is disabled. The METAR / deterministic / ensemble fetches now run with `asyncio.gather(..., return_exceptions=True)` — a 5xx from the ensemble endpoint no longer voids the deterministic-only fallback, and a deterministic failure no longer voids ensemble-alone (lossy bias frame but at least usable for distribution width).

After every successful aggregation, `aggregate_state` stashes its inputs (`forecast`, `bias_c`, `climate_prior_*`, `history`, timestamp) into the module-level `_state_cache` keyed by ICAO (30-min TTL). Fast-poll reads via `get_cached_aggregation_inputs(icao)`.

### Lock-rule trader (`src/signals/lock_rules.py` decision + `src/execution/lock_rule_executor.py` side effects)

Deterministic complement to the probability engine. Returns a `LockDecision(side, reasons, margin_f)`.

Operators in scope:
- **Threshold** (`above`, `at_least`, `below`, `at_most`) — EASY/HARD branches below.
- **Range / bracket / `exactly`** — routed through `_evaluate_range_lock`. `market_range_f` (in `execution/binary_market.py`) builds the `[low_f, high_f]` window: parsed `(low, high)` for explicit brackets, synthetic single-bucket range for `exactly` (e.g. `=10°C` → `[50, 51]°F`). Fires NO on overshoot (`current_max_f >= high_f + RANGE_LOCK_MARGIN_MULTIPLIER × LOCK_MARGIN_F`, 2× = 4°F), NO on undershoot (`current_max_f <= low_f - 2 × LOCK_MARGIN_F` AND `_no_more_heating`), YES on in-range (past-peak with no upward signal). Both overshoot and undershoot require `routine_count >= RANGE_LOCK_MIN_ROUTINES=4`.

**Lowest/minimum temperature markets are filtered upstream** in `polymarket.parse_question` (`_LOWEST_TEMP_RE`) — the pipeline assumes daily-max physics. A defensive guard in `evaluate_lock` also drops them by question-text match.

Two threshold-market directions:

- **EASY** — market-day max ≥ threshold + `LOCK_MARGIN_F` (2.0°F). Mathematically locked. Decides YES for `above`/`at_least`, NO for `below`/`at_most`. No forecast required.
- **HARD** — market-day max < threshold − margin AND `_no_more_heating(state, threshold)` (requires `has_forecast`, `forecast_peak_f < threshold`, `hours_until_peak <= 0` pre-peak guard, AND past-peak signal: `solar_declining` OR `metar_trend_rate ≤ 0`). Decides NO for `above`/`at_least`, YES for `below`/`at_most`. The same `hours_until_peak <= 0` gate applies to `range_undershoot`.

Per-market daily max comes from `_market_daily_max(state, market.end_date, now_utc)` — anchored to **`market.end_date`'s local day**, source `state.routine_history`.

Routine-count gates:
- **Super-margin EASY** (overshoot ≥ 2× `LOCK_MARGIN_F`): allowed at routine_count = 2 (monotonicity makes two confirming obs already 4°F+ over threshold bulletproof).
- **Standard EASY / HARD**: require `MIN_ROUTINE_COUNT` (3).
- **Range overshoot/undershoot**: require `max(MIN_ROUTINE_COUNT, RANGE_LOCK_MIN_ROUTINES=4)` + 2× margin bound. The `range_in_window` (YES) branch is *not* tightened — its filter is already stricter (forecast caps + past peak + no upward signals).

`_try_lock_rule_trade` flow: (1) `evaluate_lock` → return None if no side; (2) `effective_price` = `yes_price` (YES) or `1 - yes_price` (NO), reject outside `[LOCK_RULE_MIN_PRICE=0.05, LOCK_RULE_MAX_PRICE=0.95]`; (3) `_check_filters` with `min_routine_count=2` (already gated), close buffer, depth — stub edge/prob, no edge gate; (4) size via `size_locked_position` (`LOCK_POSITION_PCT=2%`, capped at `MAX_POSITION_USD/2` and 15% of depth), apply drawdown multiplier, place FOK; (5) write `Signal(model_prob=1.0, market_prob=effective_price, edge=1-effective_price)` and `Trade(direction, stake_usd, entry_price=effective_price)`. Telegram 🔒 alert.

Settings: `LOCK_RULE_ENABLED`, `LOCK_RULE_{MIN,MAX}_PRICE`, `LOCK_MARGIN_F`, `LOCK_POSITION_PCT`, `LOCK_RULE_LOSS_WINDOW_HOURS=72`, `LOCK_RULE_LOSS_DISABLE_COUNT=3`, `FAST_LOCK_POLL_{ENABLED,INTERVAL_SECONDS}`. Range constants are module-level in `lock_rules.py`: `RANGE_LOCK_MIN_ROUTINES=4`, `RANGE_LOCK_MARGIN_MULTIPLIER=2.0`.

### Forecast-exceedance alerts (`src/signals/forecast_exceedance.py`)

Diagnostic + Polymarket-discovery layer. Runs from `aggregate_state` every unified tick AND from `_fast_poll_projection_check` on every fresh fast-poll METAR. Two outputs:

1. **DB row** in `forecast_exceedance_alerts` for every routine METAR whose same-hour delta vs forecast exceeds `EXCEEDANCE_THRESHOLD_F=0.5°F`. `(station_icao, observed_at)` uniqueness dedupes fast-poll vs unified.
2. **Telegram 🌡 push** when *all* of: `_peak_passed(state)` False; `routine_count_today >= 3` (or `>= 2` when `same_hour_delta_f > STRONG_RESIDUAL_DELTA_F=1.0°F`); `_project_daily_max(state) - forecast_peak_f > DELTA_THRESHOLD_F=1.0°F`; station hasn't pushed within `ALERT_COOLDOWN=30 min`.

`_project_daily_max` has two pre-peak branches; both share the post-peak trend carry, dewpoint nudge, overshoot cap (`MAX_OVERSHOOT_F=5°F`), and observed-max floor:

- **v2 (residual slope, default when `PROJECTION_RESIDUAL_SLOPE_ENABLED=True` and ≥3 routines)**: `projected = forecast_peak + (current_residual + slope × min(hours_until_peak, RESIDUAL_SLOPE_HOURS_CAP=3))`. Slope clipped to ±`RESIDUAL_SLOPE_MAX_F_PER_HR=1.5°F/hr`; damped by `solar_decline_magnitude` and `cloud_rise_magnitude`.
- **v1 (legacy halflife decay, fallback)**: `α·forecast_residual_f` (`α=exp(-h/2)`) plus damped trend-residual carry.

For A/B logging, every exceedance line carries `projected=… (live v2), legacy_projected=… (v1)`. Promote/revert via `PROJECTION_RESIDUAL_SLOPE_ENABLED`.

Optional Polymarket-discovery line via `projected_market_lookup.lookup_projected_binary`. Tunables live as module-level constants in `forecast_exceedance.py`; edit + restart.

### Edge calculator (`src/signals/edge_calculator.py`)

`compute_edges()` (brackets) and `binary_market_edge()` (binary thresholds) — both in `signals/edge_calculator.py` — delegate filter checks to `_check_filters()`.

All filters must pass:

| Filter | Default | Source |
|--------|---------|--------|
| `edge >= MIN_EDGE` | 0.05 | `settings.MIN_EDGE` |
| `our_probability >= MIN_PROBABILITY` | 0.85 | `settings.MIN_PROBABILITY` |
| `MIN_ENTRY_PRICE <= price <= MAX_ENTRY_PRICE` | 0.40 / 0.97 | `settings` |
| `routine_count >= min_routine_count` | 3 | `settings.MIN_ROUTINE_COUNT` (overridable kwarg; lock-rule passes 2) |
| `minutes_to_close >= MARKET_CLOSE_BUFFER_MINUTES` | 30 | `settings` |
| `depth >= MIN_DEPTH_USD` | 10 | `settings` |

### Binary vs bracket markets

`_is_binary_market(market)` returns True when `parsed_threshold` and `parsed_operator` are set and operator != "bracket".

- **Binary** (`above`/`at_least`/`below`/`at_most`/`exactly`): buckets span `[current_max-1, max(threshold, forecast_peak)+10]`, prob collapsed by operator, single `BucketEdge`.
- **Bracket**: buckets extracted from `market.outcomes` via regex, prices pulled from market row, full per-bucket edge list.

### Station bias (`src/ingestion/station_bias.py`)

Per-ICAO rolling mean of `observed_daily_max_c - forecast_peak_c` over `STATION_BIAS_WINDOW_DAYS` (30). Added to Open-Meteo peak to produce `forecast_peak_f`.

- `get_bias(session, icao)` — 30-day mean, falls back to `DEFAULT_STATION_BIAS_C` (1.0°C) when no history.
- `record_daily_outcome(...)` — called in `job_daily_settlement` for every station with ≥3 routine METARs.
- `is_bias_runaway(session, icao)` — returns True when `|bias| > STATION_BIAS_MAX_C` (3°C). Such cities are skipped in Phase 1 for that tick.
- `get_station_rmse(session, icao)` — 30-day rolling `sqrt(avg(bias_c²))` over the same window, converted to °F. Returns `(rmse_f, sample_days)`. In-process TTL cache (`_RMSE_TTL=1h`). Drives the per-station σ floor in `probability_engine._compute_sigma`; falls back to global `ENSEMBLE_MIN_SIGMA_F` when `sample_days < 14`. Computed on-the-fly from the existing `station_biases` rows — no new schema, no nightly job.

Bias is recorded against the deterministic single-source peak (not the ensemble blend) so the reference frame stays stable when the ensemble model list changes.

### Kelly sizing (`src/risk/kelly.py`)

`size_position()` applies cascading caps:
1. Fractional Kelly with `KELLY_FRACTION=0.25`
2. Per-trade cap: `MAX_POSITION_PCT=5%` of bankroll
3. Total exposure cap: `max(MAX_EXPOSURE_PCT=25% × bankroll, MAX_EXPOSURE_USD_FLOOR=$300)` minus current open exposure. The USD floor binds at small bankroll (< ~$1200) so a few stuck OPEN trades can't pin the cap and silence the bot — added 2026-05-17 after that exact failure mode silenced trading for ~24h
4. Hard USD cap: `MAX_POSITION_USD=200`
5. Orderbook depth cap: `DEPTH_POSITION_CAP_PCT=20%` of visible depth
6. Minimum viable trade: `MIN_TRADE_USD=5` (below returns 0)

Drawdown monitor multiplier (`DrawdownMonitor.check`) is applied on top of Kelly stake before the `MIN_STAKE_USD` check. The lock-rule path (`size_locked_position`) uses fixed sizing instead of Kelly: `LOCK_POSITION_PCT=2%`, capped at `MAX_POSITION_USD/2=$100` and 15% of depth — but `MAX_EXPOSURE_PCT`, `MIN_TRADE_USD`, and drawdown multiplier still apply.

#### Drawdown state machine (`src/risk/drawdown.py`)

Four states. Multipliers in parentheses:

| State | When | Multiplier |
|---|---|---|
| `NORMAL` (1.0) | `current_bankroll ≥ peak`, OR `dd_pct < CAUTION_THRESHOLD (10%)` while previously NORMAL |
| `CAUTION` (0.5) | `dd_pct ≤ PAUSE_THRESHOLD (20%)` |
| `PAUSED` (0.0) | `dd_pct > PAUSE_THRESHOLD` |
| `RECOVERY` (0.5) | `dd_pct < CAUTION_THRESHOLD` AND previous was CAUTION/PAUSED/RECOVERY — hysteresis so the bot doesn't snap back to full size on a single up-tick |

Exit `RECOVERY` → `NORMAL` only when `current_bankroll >= peak`.

#### Bankroll = equity, not wallet liquidity (`src/resolution.py`)

`get_current_bankroll(session)` returns **equity**:

```
bankroll = wallet_usdc
         + Σ (stake_usd / entry_price × per_share_value) for OPEN trades
         + Σ (stake_usd + pnl) for trades WHERE status=WON AND redeemed_at IS NULL
```

where `per_share_value` is `Market.current_yes_price` for `BUY_YES`, `1 - current_yes_price` for `BUY_NO`, falling back to cost basis when missing. Both unsettled lifecycle stages (OPEN on-book and WON-unredeemed) must be included; otherwise the drawdown monitor trips PAUSED whenever the bot has unresolved trades.

Wins don't auto-settle — run `bet redeem --all` to convert WON conditional tokens to USDC. The default path is **DB-driven**: it queries `Trade(status=WON AND redeemed_at IS NULL)` directly, groups by `(market_id, direction)`, and reads the redemption metadata (`condition_id`, `neg_risk`, `clob_token_ids`) from the local `markets` row for that bounded set — scaling with WON-unredeemed count, not lifetime trade count. These three columns are populated on every `job_scan_markets` tick from the Gamma payload (see `_market_to_row` in `src/ingestion/polymarket.py`), so any market scanned at least once while live has metadata cached before it resolves. For legacy rows where the columns are NULL (market resolved before this caching existed), redeem falls back to a single bulk `data-api.polymarket.com/positions?user=<funder>` call: every on-chain position the wallet holds — including resolved-but-unredeemed — comes back keyed by `asset` (= `trades.token_id`) with `conditionId` + `negativeRisk`. CLOB `/markets/<conditionId>` is then called once per missing condition to fill the sibling token_id for `clob_token_ids` (skipped when the user owns both YES and NO — both `asset`s are already in the data-api response). Recovered metadata is persisted back to `markets`. If data-api returns no position for the token, the market is logged and skipped (operator runs `--full-scan` as the last-resort recovery). The previously-used Gamma `?clob_token_ids=<token_id>` fallback was removed: empirically Gamma drops resolved markets from **both** the `id` and `clob_token_ids` indexes (returns `[]`), and the camelCase `?clobTokenIds=` form is silently ignored by the API and returns an unrelated 20-row default listing — unsafe to treat as authoritative. `resolve_trades` setting `status=WON` at resolution time IS the "redeemable" mark; no separate flag exists. `--full-scan` falls back to the legacy flow that paginates the entire CLOB trade history via `get_trades_history` + `compute_positions` — slower, kept as a safety net when local DB may have drifted from on-chain state. Either path stamps `Trade.redeemed_at = now()` on every WON row whose `market_id` matches a successful `redeemPositions()` receipt (`Market.id` is the **Gamma integer id**, not the on-chain condition_id — the condition_id is read from Gamma's `conditionId` field at redeem time and passed to `redeemPositions()`). At the end of every run (and as the sole action when invoked with `--reconcile`), an idempotent reconciliation pass queries on-chain `balanceOf` for every WON-unredeemed Trade and stamps any whose tokens are already gone — this self-heals positions redeemed via the Polymarket UI, redeemed before stamping existed, or that have fallen out of the CLOB trade-history window. Trades whose `token_id` is NULL get resolved via Gamma's `clobTokenIds` (cached per-market) and persisted back. `BankrollLog` is written at 22:00 UTC by `job_daily_settlement`. `get_unredeemed_won_payout(session)` and `get_open_trade_value(session)` expose each adjustment for diagnostics.

`bet portfolio` follows the same DB-first pattern as `bet redeem`: by default it queries `Trade(status=OPEN OR (status=WON AND redeemed_at IS NULL))`, eager-loads `Market` via `selectinload`, and fetches live best-bid/ask for every position in parallel via `asyncio.gather(asyncio.to_thread(get_best_bid_ask, …))` — replacing the prior per-position sequential `get_last_trade_price` round-trips. `--all` extends the set to WON-redeemed + LOST (rendered with realised PnL = payout − cost). `--full-scan` falls back to the legacy walk that pulls the entire CLOB trade history via `get_trades_history` + `compute_positions` and then probes on-chain `balanceOf` to filter redeemed-but-resolved tokens; use it as a safety net when positions may have been taken outside the bot. Helpers: `_load_active_positions_from_db()` (DB path) and `_load_active_positions()` (legacy CLOB path) — both return the same `(ordered, token_to_market)` shape; `bet sell` still uses the CLOB-walk helper.

### Circuit breakers (`src/risk/circuit_breakers.py`)

Checked once per unified pipeline tick *and* every fast-poll tick, before any city is evaluated:
- Daily loss stop: halt when today's P&L < `-DAILY_LOSS_STOP_USD` (-$200).
- Consecutive loss stop: `CONSECUTIVE_LOSS_PAUSE_COUNT` LOST trades in a row (3) → pause `CONSECUTIVE_LOSS_PAUSE_HOURS` (2h). `_paused_until` lives in process memory; consecutive losses are re-queried from DB each check.
- **Submission-failure stop** (2026-05-20): `SUBMIT_FAIL_PAUSE_COUNT` (5) `exception:*` orders in the last `SUBMIT_FAIL_PAUSE_WINDOW_MINUTES` (10m) → pause `SUBMIT_FAIL_PAUSE_MINUTES` (30m) and 🚨 Telegram-alert. Counts rows with `trades.exchange_status LIKE 'exception:%'`. Catches a Polymarket protocol migration / wallet-allowance break before it buries the decision logs (the May 2026 PolyApiException class — 115 rows in one week, never alerted).

Per-city routine-count and bias-runaway checks live in the aggregator and edge filter, not here.

## How To: Common Tasks

### Add a new city

1. Add `"city name": (lat, lon)` to `CITIES` in `src/signals/mapper.py`.
2. Add `"city name": "ICAO"` to `CITY_ICAO`.
3. Confirm the ICAO is **not** in `_EXCLUDED_ICAOS` in `src/scheduler/__init__.py` (currently `{"VHHH", "LLBG"}` where Polymarket's resolution source diverges from the routine METAR feed). Removing requires verifying the actual resolver station first.
4. If the city uses °C in its market title, confirm `market_unit` in `src/execution/binary_market.py` handles it.
5. Seed `StationBias` if you have historical data — otherwise the default +1.0°C bias applies until enough settlements accumulate.

### Tune trade filters

All knobs in `src/config.py` (Pydantic `Settings`, overridable via `.env`):

- `MIN_EDGE` (0.05), `MIN_PROBABILITY` (0.85), `MIN_ENTRY_PRICE`, `MAX_ENTRY_PRICE`
- `MIN_DEPTH_USD`, `MIN_ROUTINE_COUNT`, `MARKET_CLOSE_BUFFER_MINUTES`
- Kelly caps: `KELLY_FRACTION`, `MAX_POSITION_PCT`, `MAX_POSITION_USD`, `DEPTH_POSITION_CAP_PCT`
- `CLUSTER_STAKE_CAP_USD` (100) — total stake cap across same-day same-city bracket/exactly cluster. Outcomes are anti-correlated (only one bucket wins) so independent Kelly sizing per bucket over-allocates. Applied in probability path AND `_try_lock_rule_trade`. Set 0.0 to disable
- `BRACKET_MARKETS_ENABLED` (False) — gates whether unified pipeline + fast-poll evaluate `parsed_operator IN ('bracket','range')` at all. Threshold/`exactly` unaffected
- `APPLY_CALIBRATION` (True) — refreshes the calibration cache each tick and applies the linear correction to the chosen side's probability before edge filtering. Requires ≥`MIN_CALIBRATION_SAMPLES=50` resolved trades; below that returns raw probability unchanged. See `src/signals/calibration.py`
- `ORDER_RECONCILE_INTERVAL_MINUTES` (5), `ORDER_RECONCILE_LOOKBACK_HOURS` (24) — supplemented by a 7-day long-tail filter that catches null-fill delayed trades whose market is still recent (fixes the 2026-05-17 drift bug)
- `RESOLVE_NO_PRICE_GRACE_HOURS` (4) — `resolve_trades` waits this long before marking null-fill trades LOST when the CLOB returns no price (the catch-22 gotcha)
- `MAX_EXPOSURE_USD_FLOOR` ($300) — absolute USD floor on the exposure cap; binds at small bankroll. Tune higher to increase concurrent-position capacity at the cost of larger drawdown risk per losing streak
- `STALE_OPEN_RECONCILE_GRACE_HOURS` (4), `STUCK_ALERT_MIN_COUNT` (5), `STUCK_ALERT_EXPOSURE_FRACTION` (0.50), `STUCK_ALERT_COOLDOWN_HOURS` (4) — stuck-OPEN sweep + heartbeat in `job_reconcile_orders`. Sweep is log-only; heartbeat fires when count OR exposure-fraction threshold trips (cooldown stored in `bot_state` so a restart doesn't drop the timer). Catches the same silent-silencer failure mode as `job_daily_settlement`'s 80%-of-cap alert, but every 5 min instead of once a day

The lock-rule path has its **own** knob set (`LOCK_RULE_*`, `LOCK_MARGIN_F`, `LOCK_POSITION_PCT`, `FAST_LOCK_POLL_*`). The ensemble σ knobs (`ENSEMBLE_*`) only affect Phase 1 σ in the probability engine.

Forecast-exceedance projection: master switch `PROJECTION_RESIDUAL_SLOPE_ENABLED` in `.env`. Per-knob tunables are now `Settings` fields too (no more "edit + restart" source-code edits): `EXCEEDANCE_THRESHOLD_F`, `EXCEEDANCE_DELTA_THRESHOLD_F`, `EXCEEDANCE_MIN_ROUTINE_COUNT`, `EXCEEDANCE_STRONG_RESIDUAL_*`, `EXCEEDANCE_MAX_OVERSHOOT_F`, `EXCEEDANCE_RESIDUAL_SLOPE_*`, `EXCEEDANCE_ALERT_COOLDOWN_MINUTES`, etc. The module-level names in `forecast_exceedance.py` still work — they bind to `settings.*` at import time. Restart required for `.env` changes (Pydantic loads once at boot).

Post-peak trend carry (`POST_PEAK_HOURS_CAP`, `POST_PEAK_TREND_CARRY_K`, `POST_PEAK_MIN_TREND_F_PER_HR`, `POST_PEAK_MAX_SHIFT_F`) is a single Settings source shared by `forecast_exceedance` (alert) and `probability_engine` (Gaussian center shift) so both stay in lockstep. Range-lock gates (`RANGE_LOCK_MIN_ROUTINES`, `RANGE_LOCK_MARGIN_MULTIPLIER`) are also Settings fields.

### Change pipeline cadence

`UNIFIED_PIPELINE_INTERVAL_MINUTES` in `.env` (default 5). `job_resolve_trades` is hardcoded to 5 min (30s offset from unified). Daily settlement is fixed at 22:00 UTC.

### Add / modify probability signals

Signals live in `src/signals/probability_engine.py` as `_apply_*` helpers. Each takes `(center_or_sigma, state, reasoning)`, returns the updated value, and appends a one-line explanation to `reasoning`. The monotonicity constraint is applied last.

## Testing Conventions

Tests in `tests/test_<module>.py`. Mock external APIs at the module boundary (e.g. `@patch("src.signals.state_aggregator.fetch_forecast")`). `AsyncMock` for async functions; session mocking via `AsyncMock()` with explicit `session.flush = AsyncMock()`. Do **not** run `ruff` — not used on this project.

## File Index

| File | Purpose | Key exports |
|------|---------|-------------|
| `src/scheduler/__init__.py` | APScheduler jobs + setup. Re-exports the extracted helpers so existing `from src.scheduler import _has_active_trade` callers keep working. Package shape — the body is currently a single `__init__.py`; future internal split into `jobs.py` + `setup.py` is deferred. | `job_*` (see Scheduler Jobs table), `_try_lock_rule_trade`, `_fast_poll_projection_check`, `_has_active_trade`, `_cluster_stake_used`, `_upsert_signal`, `_log_evaluation`, `_binary_market_edge`, `_load_stuck_alert_cooldown`, `_persist_stuck_alert_cooldown`, `_EXCLUDED_ICAOS`, `run_scheduler`, `configure_logging`, `backfill_markets` |
| `src/execution/binary_market.py` | Pure per-market shape helpers — binary-vs-bracket detection, °F bucket grid, unit display, future-day skip. No I/O / no DB / no scheduler state. | `is_binary_market`, `market_range_f`, `should_skip_future_day`, `market_unit`, `display_bucket`, `make_binary_buckets`, `is_bracket_like` |
| `src/execution/lock_rule_executor.py` | Side-effect wrapper around `signals.lock_rules.evaluate_lock`. Owns: filter gating, sizing, Signal/Trade persistence, FAK order placement, Telegram alert dispatch. Mirrors `probability_engine` (pure) vs `polymarket_client` (side effects). | `try_lock_rule_trade`, `extract_bracket_buckets`, `extract_market_prices`, `minimal_state_for_easy_lock` |
| `src/persistence/dedup.py` | DB-backed dedup helpers — race-safe `INSERT … ON CONFLICT DO UPDATE … RETURNING` upsert + PENDING/OPEN trade-existence check. | `has_active_trade`, `upsert_signal` |
| `src/persistence/cache_rollover.py` | Per-station local-day in-process cache rollover + the dedup dicts that are reset on station-local midnight. | `locked_markets_fired_today`, `unified_fired_today`, `last_routine_seen`, `market_to_icao`, `local_day_seen`, `maybe_clear_per_station_caches`, `record_lock_fire` |
| `src/risk/cluster_cap.py` | Anti-correlation guard summing currently-staked $ across same parsed_location + same end_date.date() bracket/exactly cluster. Returns 0 for non-bracket markets. | `cluster_stake_used` |
| `src/monitoring/logging.py` | Structured JSON logging for the production scheduler. ``JSONFormatter`` merges ``extra={...}`` fields into the JSON payload so callers can ``logger.info("msg", extra={"icao": ..., "market_id": ...})`` without baking identifiers into the message string. | `JSONFormatter`, `configure_logging` |
| `src/monitoring/health.py` | stdlib asyncio health-check server. Closes over a scheduler-status callable to avoid circular import. | `start_health_server` |
| `src/signals/state_aggregator.py` | Per-ICAO weather state + det/ensemble blend + residual-slope fit + fast-poll input cache | `WeatherState`, `aggregate_state`, `build_state_from_metars`, `_blend_forecasts`, `_compute_residual_slope`, `get_cached_aggregation_inputs`, `clear_state_cache` |
| `src/signals/probability_engine.py` | Signal-based bucket distribution. `_effective_sigma_floor` picks per-station (rolling RMSE, ≥14 sample days) vs global `ENSEMBLE_MIN_SIGMA_F` floor for the Gaussian σ. | `BucketDistribution`, `compute_distribution`, `_effective_sigma_floor` |
| `src/signals/edge_calculator.py` | Per-bucket edge + filter checks + binary-market YES/NO side selection (was `scheduler._binary_market_edge`). | `BucketEdge`, `compute_edges`, `binary_market_edge`, `_check_filters`, `MIN_EDGE` |
| `src/signals/lock_rules.py` | Deterministic physical-lock decisions | `LockDecision`, `evaluate_lock`, `RANGE_LOCK_MIN_ROUTINES`, `RANGE_LOCK_MARGIN_MULTIPLIER` |
| `src/signals/projection.py` | Pure daily-max projection math (v1 halflife + v2 residual-slope). Split out of `forecast_exceedance` so backtest / replay code can import without DB or Telegram coupling. | `project_daily_max`, `project_with_residual`, `legacy_project_daily_max`, `peak_passed`, `effective_trend`, `pick_latest_routine`, `closest_hour_index`, `c_to_f` |
| `src/signals/forecast_exceedance.py` | Side-effect wrapper: DB row write + Telegram push trigger. Imports pure math from `signals.projection`. | `check_and_record_daily_max_alert` |
| `src/signals/evaluation_log.py` | Append-only telemetry writer for `evaluation_logs` (one row per per-side edge evaluation, passing OR rejected). | `log_evaluation` |
| `src/signals/decision_log.py` | Append-only telemetry writer for `decision_logs` (one row per per-side **post-filter** decision). Outcome enum is the source of truth for the funnel: `OUTCOME_SIGNAL_WRITTEN`, `OUTCOME_TRADE_PENDING`, `OUTCOME_TRADE_FILLED`, `OUTCOME_DUP_*`, `OUTCOME_STAKE_BELOW_MIN`, `OUTCOME_DRAWDOWN_PAUSED`, `OUTCOME_CLUSTER_CAP`, `OUTCOME_CAP_EXCEEDED`, `OUTCOME_NO_FILL`, `OUTCOME_ORDER_FAILED`. Add new outcome constants here, never inline them. | `log_decision`, `OUTCOME_*` |
| `src/signals/projected_market_lookup.py` | Find the active binary closest to a projected daily max | `lookup_projected_binary` |
| `src/signals/mapper.py` | Geocoding, ICAO lookup, operator/date/threshold normalisation, station-local timezones | `CITIES`, `CITY_ICAO`, `icao_for_location`, `cities_for_icao`, `geocode`, `icao_timezone`, `unit_for_station`, `normalize_operator`, `convert_threshold`, `f_to_c` |
| `src/signals/calibration.py` | Linear recalibration `actual ≈ slope*predicted + intercept` from resolved signals + 30-min TTL cache. Wired into `binary_market_edge` post-side-selection when `settings.APPLY_CALIBRATION=True` | `get_calibration_coefficients`, `refresh_calibration`, `apply_calibration`, `MIN_CALIBRATION_SAMPLES` |
| `src/signals/reverse_lookup.py` | Find markets by city/station/observation | `find_markets_for_city`, `find_markets_for_station`, `find_markets_for_observation`, `find_markets_for_event` |
| `src/ingestion/polymarket.py` | Gamma API scanner + question parser. Primary discovery is **event-slug enumeration** (every city × upcoming-day → `GET /events?slug=highest-temperature-in-<city>-on-<month>-<day>-<year>`) because Gamma's `/markets?tag=weather` listing endpoint silently ignores the filter and returns generic markets. Keyword-paginated scan + tag scan remain as defensive fallbacks. | `scan_and_ingest`, `ingest_markets`, `get_active_weather_markets`, `parse_question`, `is_weather_market`, `parse_temperature_brackets`, `_event_slug_for`, `_city_to_polymarket_slug` |
| `src/ingestion/openmeteo.py` | Deterministic + ensemble hourly forecast, solar/cloud/dewpoint helpers | `OpenMeteoForecast`, `fetch_deterministic_forecast`, `fetch_ensemble_forecast`, `solar_declining`, `cloud_rising`, `dewpoint_trend` |
| `src/ingestion/aviation/` | Multi-provider METAR/TAF/PIREP/SIGMET | `fetch_metar_history`, `fetch_latest_metars`, `get_routine_daily_max`, `detect_metar_cycle`, `get_temp_trend` |
| `src/ingestion/station_bias.py` | Per-station forecast bias tracking (anchored to deterministic peak) + rolling forecast-error RMSE feeding the probability-engine σ floor | `get_bias`, `record_daily_outcome`, `is_bias_runaway`, `get_station_rmse`, `clear_station_rmse_cache` |
| `src/ingestion/wx.py` | Weather.com v3 observations (optional, `WX_API_KEY` gated) | — |
| `src/execution/polymarket_client.py` | CLOB client, orderbook, orders. `place_order` is FAK BUY; `sell_position(token_id, size)` is FAK SELL `MarketOrderArgsV2` for `bet sell` | `is_live`, `get_token_ids`, `place_order`, `sell_position`, `check_order_status`, `cancel_order`, `get_best_bid_ask`, `get_orderbook_depth`, `get_daily_spend` |
| `src/execution/alerter.py` | Telegram queue + inline buttons (exec/skip/detail) + discovery alerts + daily redeem nudge | `Alerter`, `AlertType`, `get_alerter`, `_escape_md2` |
| `src/risk/kelly.py` | Fractional Kelly + lock-rule fixed sizing | `PositionSize`, `size_position`, `size_locked_position`, `MIN_TRADE_USD`, `MAX_EXPOSURE_PCT` |
| `src/risk/drawdown.py` | Four-state drawdown machine with hysteresis | `DrawdownMonitor`, `DrawdownLevel` (`NORMAL`/`CAUTION`/`PAUSED`/`RECOVERY`) |
| `src/risk/circuit_breakers.py` | Daily loss + consecutive loss + submission-failure halts | `check_circuit_breakers`, `CircuitBreakerState`, `_get_recent_submit_failures` |
| `src/risk/simulate.py` | Paper-trade simulator | used by `cli paper-trade` |
| `src/resolution.py` | Trade settlement (chain-gated via on-chain `payoutDenominator`, with legacy CLOB-mid fallback for markets without `condition_id`), bankroll-as-equity, exposure. Null-fill grace fallback now gates on `balanceOf` to prevent phantom losses (2026-05-20) | `resolve_trades`, `_apply_settlement`, `_build_ctf_readonly`, `_query_payout_outcome`, `_query_token_balance`, `_backfill_null_fill`, `_refresh_market_price` (legacy fallback only), `get_current_bankroll`, `get_unredeemed_won_payout`, `get_open_trade_value`, `get_current_exposure`, `calculate_daily_pnl` |
| `src/db/models.py` | SQLAlchemy ORM | `Market`, `Signal`, `Trade`, `EvaluationLog`, `DecisionLog`, `StationBias`, `MetarObservation`, `ForecastExceedanceAlert`, `ForecastArchive`, etc. |
| `src/ingestion/forecast_archive.py` | Snapshots every `OpenMeteoForecast` blend into `ForecastArchive` for replay-capable backtests | `archive_forecast_snapshot` |
| `src/config.py` | Pydantic settings | `settings` |
| `src/cli.py` | CLI entry points | `run`, `scan`, `backfill`, `status`, `paper-trade`, `evals-report`, `backtest-v2`, `migrate`, `approve`, `test-trade`, `bet {place,info,search,find,cancel,orders,portfolio,redeem}`, `admin {reset-drawdown-peak,reconcile-stuck [--include-lost]}` |
| `scripts/backtest_lock_rule.py` | Replay lock-rule trader against resolved markets + DB METARs | standalone |
| `scripts/debug_pipeline.py` | Trace the unified pipeline for one market/station, no orders | standalone |
| `scripts/inspect_loss.py` | Post-mortem drilldown for a single losing trade | standalone |
| `scripts/backfill_station_bias_tz.py` | Idempotent recompute of station bias from DB METARs | one-shot |
| `scripts/compare_projection_versions.py` | RMSE / mean-bias of v2 projection by station + lead bucket + alerted flag | standalone |
| `scripts/station_calibration_report.py` | Per-ICAO markdown dashboard (bias, projection error, lock strike rate, Brier) | standalone |
| `scripts/audit_lock_observed_max.py` | Recompute daily max from `metar_observations` for lock-rule LOST trades; flag snapshot contradictions | standalone |

## Live Execution

Orders placed via `py-clob-client` in `src/execution/polymarket_client.py::place_order`.

**Flow per trade:**
1. Scheduler creates `Signal` + `Trade(status=PENDING)`.
2. `place_order` checks `DAILY_SPEND_CAP_USD`, resolves YES/NO token IDs (Gamma), places FOK market order.
3. On fill: trade → `OPEN`, `order_id`/`fill_price`/`filled_size` populated, Telegram alert.
4. On failure: trade stays `PENDING` (no retry logic), warning logged.

**Safety:**
- `AUTO_EXECUTE=false` (default) — no orders sent; signals still logged + Telegram-notified.
- `POLYMARKET_PRIVATE_KEY` empty → dry-run; `py_clob_client` never imported.
- `DAILY_SPEND_CAP_USD=400` — hard 24h rolling cap (set in `.env`).
- `MIN_STAKE_USD=5` — orders below this skipped after drawdown multiplier.
- `DrawdownMonitor` multiplies stake by level (`NORMAL=1.0`, `CAUTION=0.5`, `PAUSED=0.0`, `RECOVERY=0.5`).
- Lock-rule auto-disable: `LOCK_RULE_LOSS_DISABLE_COUNT=3` losses within `LOCK_RULE_LOSS_WINDOW_HOURS=72` flips path off until manually re-enabled.

## Telemetry (calibration data sources)

Migration `j0k1l2m3n4o5_strategy_telemetry`. `signals` carries one row per `(market_id, direction)` — the chosen-trade rationale; the streams below capture surrounding evaluation context.

- **`evaluation_logs`** — append-only row per per-side edge evaluation. Written by `_log_evaluation` from BOTH paths (after `_binary_market_edge`, after `evaluate_lock` + at rejection points). Captures `model_prob/market_prob/edge/passes/reject_reason/depth_usd/minutes_to_close/routine_count/signal_kind`. Use this — not `signals` — for filter-tuning backtests, since `signals` only carries passing edges and is deduplicated.
- **`decision_logs`** — append-only row per per-side **post-filter** decision (added 2026-05-16). Closes the funnel between `evaluation_logs.passes=true` and `Trade` row written: outcomes cover both success branches (`trade_pending`/`trade_filled`) and skip/failure branches (`dup_blocked_*`, `stake_below_min`, `drawdown_paused`, `cluster_cap_hit`, `no_fill`, `order_failed`). Wired into 12 call sites across `src/scheduler/__init__.py::job_unified_pipeline` (probability path) and `src/execution/lock_rule_executor.py::try_lock_rule_trade` (lock path). Use this when investigating "evals passed but no Signal/Trade rows" — the failure mode that cost ~4h of debugging on 2026-05-16. The `stake_below_min` / `drawdown_paused` metadata also carries `size_reason` (which cap binds — per-trade, exposure, USD, depth, MIN_TRADE_USD floor — taken from `PositionSize.reason`), `kelly_pct` (raw fractional Kelly), and `depth_usd` (orderbook depth at decision time) so subsequent debug can identify the binding constraint in one query (added 2026-05-17).
- **`trades.submit_yes_bid` / `submit_yes_ask` / `submit_depth_usd` / `submit_at`** — populated by `place_order` before FAK (live AND dry-run). Lets slippage analyses decompose `fill_price - entry_price` into spread vs depth-walked.
- **`signals.signal_kind`** ('probability' | 'lock') + **`signals.lock_branch`** ('easy_super' / 'easy_standard' / 'hard' / 'range_overshoot' / 'range_undershoot' / 'range_in_window') + `lock_routine_count` + `lock_observed_max_f` — set by `_upsert_signal` on every refresh.

## Gotchas

- **Near-resolved skip is 0.99 / 0.01**, not 0.95 — so the lock-rule path can evaluate prices in the 0.90–0.99 zone (Wunderground UI catches up later than the METAR feed).
- **Lock-rule fires before the near-resolved skip.** Phase 2 runs `_try_lock_rule_trade` first; only if it returns `None` does the probability path see the market. A fires-but-rejects (price/depth/close-buffer) returns `0.0` and the market is skipped for the rest of this tick.
- **HARD / range-undershoot requires `hours_until_peak <= 0` pre-peak guard.** Without it, overnight radiative cooling makes the 6h trend regression negative and the past-peak OR check fires NO locks hours before daytime heating begins.
- **Dedup is DB-backed first, in-process second.** `_has_active_trade(session, market_id, direction)` is the durable safety net used in BOTH paths: any PENDING/OPEN Trade row short-circuits a second attempt. In-process `_unified_fired_today` / `_locked_markets_fired_today` are same-tick speed-ups. `uq_signals_market_direction` + `_upsert_signal` (single atomic ``INSERT ... ON CONFLICT DO UPDATE ... RETURNING``) prevents duplicate Signal rows on repeat ticks — race-safe at the DB level, no longer relies on APScheduler `max_instances=1` to serialize a SELECT/INSERT window.
- **Fast-poll lock-rule dedup is aggressive on filter rejects** — `_locked_markets_fired_today.add(market.id)` fires even when filtered out (depth, close buffer). Cleared per-station at local-day rollover.
- **Per-station rollover, not global UTC clear.** `_maybe_clear_per_station_caches` runs every unified tick. The dedup→ICAO link is `_market_to_icao` at lock-fire time; entries added outside `job_fast_lock_poll` won't be found by rollover.
- **Fast-poll projection check needs a warm `_state_cache` (30-min TTL).** No-op for a station until `aggregate_state` has run for it once. After a restart, the first ~5 min of fast-poll alerts are silent. `_state_cache` is in-process and never persisted.
- **`job_resolve_trades` ≠ `job_daily_settlement`.** Resolution is a 5-min job; settlement (22:00 UTC) only does bankroll/drawdown bookkeeping, daily summary, redeem nudge, station bias, and weekly calibration.
- **Resolution is chain-gated, not CLOB-gated (changed 2026-05-19).** `resolve_trades` calls `bet_helpers.get_payout_outcome(condition_id)` → `payoutDenominator` on the Polygon ConditionalTokens contract as the **primary** signal. If UMA has reported (denom > 0) → map `payoutNumerators` to YES/NO and settle. If UMA hasn't reported and we're past `RESOLVE_NO_PRICE_GRACE_HOURS` (4h): null-fill rows → mark LOST (release exposure, order never landed); populated-fill rows → log WARNING telling the operator to run `admin reconcile-stuck`. The legacy CLOB-mid 0.95/0.05 heuristic is only used as a fallback when `market.condition_id IS NULL` (pre-caching legacy rows) or the Polygon RPC connection fails. **Why:** pre-fix, the bot flipped trades to WON whenever the CLOB mid crossed 0.95/0.05 — but on-chain UMA reporting can lag hours-to-days behind that signal. That window produced false `💸 redeem` Telegram nudges that, when the operator acted on them, reverted with `result for condition not received yet` from the ConditionalTokens contract. `_refresh_market_price` is now legacy-only.
- **`job_reconcile_orders` includes WON+null-fill (since 2026-05-19).** Status filter is `PENDING/OPEN/WON` (LOST stays excluded — null-fill there is permanent). Backfills fill data on terminal rows that `resolve_trades` flipped before fill details landed; recomputes `pnl` against the corrected `entry_price` when the trade is already WON. Originally added because the on-chain-gating fix doesn't help the existing pre-fix stuck rows.
- **Postgres `tradestatus` enum stores uppercase names** (`'WON'`/`'LOST'`/`'OPEN'`/`'PENDING'`). Raw SQL must use the uppercase form (SQLAlchemy uses enum names, not `.value`).
- **`projected_max_f` in `forecast_exceedance_alerts` is whichever path is live (v2 by default).** The legacy v1 value is only in JSON logs (`legacy_projected=`), not the DB.
- **Dry-run trades stay `PENDING` in BOTH paths and keep their requested `stake_usd`.** `place_order`'s dry-run branch sets `trade.exchange_status="dry_run"` without touching `stake_usd`. Both executors check that flag, leave `status=PENDING`, and emit a "(dry-run)" Telegram alert. **Invariant for new code:** any future query summing `Trade.stake_usd` must filter on `status IN (OPEN, WON, LOST)` (or exclude `exchange_status='dry_run'`); otherwise dry-run rows double-count.
- **Hong Kong (VHHH) and Tel Aviv (LLBG) are silently skipped** even though they're in `CITIES`/`CITY_ICAO`. `_EXCLUDED_ICAOS` filters them at grouping because their Polymarket resolution station diverges from the routine METAR feed.
- **Bracket markets (`parsed_operator IN ('bracket','range')`) are silently skipped by default** — `BRACKET_MARKETS_ENABLED=False` gates BOTH unified pipeline AND `job_fast_lock_poll`. Threshold/`exactly` unaffected.
- **Cluster-cap helper anti-correlation guard.** `_cluster_stake_used` sums currently-staked $ across same `parsed_location` + same `end_date.date()` (UTC) bracket/exactly cluster, excluding dry-run rows. Both paths reject if `cluster_used + new_stake > settings.CLUSTER_STAKE_CAP_USD`. Threshold markets unaffected (helper returns 0.0 for non-bracket-like operators).
- **`apply_calibration` is on by default but degrades gracefully.** Needs ≥`MIN_CALIBRATION_SAMPLES=50` resolved trades. Below that, `get_calibration_coefficients` returns None and `apply_calibration` returns the raw probability unchanged with `applied=False`. Cache TTL 30 min in `_CACHE_TTL_SEC`.
- **`MIN_EDGE` is harmonised on `settings.MIN_EDGE` (default 0.05).** Module-level `edge_calculator.MIN_EDGE` is kept as a re-export but `_check_filters` reads `settings.MIN_EDGE` directly. **The current `.env` has `MIN_EDGE=0.10`** — that env value wins via override; review before assuming default applies.
- **`_fetch_orderbook` caches only successful fetches (30s).** Failures are not negative-cached, so two back-to-back calls to a dead token retry 3×3×throttle delay — except 404 ("No orderbook exists for the requested token id"), which short-circuits to `None` on attempt 1 and is silenced by `_CloudflareNoiseFilter`. Resolved markets surface as 404 here.
- **`_get_client()` is thread-safe via `_client_build_lock`.** `bet portfolio` fans out `get_best_bid_ask` through `asyncio.to_thread`; without the lock every worker raced past the `is None` check and rebuilt the client, each posting to `/auth/api-key`. `bet_helpers.get_clob_client` now delegates to the same singleton instead of building its own. The 400 `Could not create api key` line (transparently retried as a GET-derive by the SDK) is also filtered out.
- **FOK orders fully fill or cancel** — no partial fills; thin books leave trades in `PENDING`.
- **`place_order` posts FAK buys via `MarketOrderArgsV2` + `create_market_order`, not `OrderArgsV2` + `create_order`.** CLOB enforces stricter precision on FAK/FOK buys; the limit-order builder triggers `400 invalid amounts`. We pass our tick-snapped `limit_price` so the SDK skips its network `calculate_market_price` call.
- **`delayed` orders post successfully but aren't yet retrievable.** `post_order` returns `status="delayed"` with a real `orderID`, but `client.get_order(order_id)` returns None until the matching engine picks it up. `_update_fill_details` guards against None so the trade keeps `status=PENDING`, `exchange_status="delayed"`. The fill-details call in `place_order`'s success branch is wrapped in its own try/except so a lookup failure can't orphan a live order. `job_reconcile_orders` re-runs `check_order_status` on a 5-min cadence to populate `fill_price`/`filled_size` once the engine retrieves the order.
- **`WeatherState` is produced even if Open-Meteo fails** — `forecast_peak_f = current_max_f`, `hours_until_peak = 0`, `has_forecast = False`. HARD-direction lock disabled; probability degenerates.
- **`_paused_until` circuit-breaker state survives restart via `bot_state`.** The in-process variable is hydrated from `bot_state.value['until_iso']` on the first `check_circuit_breakers` call after boot; subsequent calls read in-process. Writes (engage / clear-on-expiry) write-through to DB so a crash mid-pause doesn't drop the protective window. Consecutive-loss counting is still re-queried each tick — the persisted pause is the *length of the window*, not the streak count. **Fail-open on missing table:** if the `m3n4o5p6q7r8_bot_state_table` migration hasn't been applied, `_load_paused_until` / `_persist_paused_until` catch `ProgrammingError`, log a warning, and return None rather than killing the pipeline. Verified failure mode: pre-2026-05-16 deploy of Tier 1 silently halted every unified/fast-lock tick for ~6 hours because the prod DB role's `Base.metadata.create_all` didn't auto-create the table and the migration wasn't run.
- **`Market.current_yes_price` is read-only in the unified pipeline, fast-lock poll, and `resolve_trades`** — they take the live mid from `get_best_bid_ask` into a local `live_price` and never assign back to the ORM row (the dirty Market would otherwise autoflush with `session.add(Signal/Trade)` and cause cross-transaction deadlocks). Only `job_scan_markets` (`ingest_markets`) persists this column, iterating sorted by id so UPDATEs are emitted in deterministic primary-key order.
- **`Signal.gfs_prob`/`ecmwf_prob`/`aviation_prob`/`wx_prob` columns remain in the schema but are always NULL.** No live readers; the alerter detail view and the calibration dashboard were cleaned up. Drop via migration when convenient.
- **Polymarket Gamma `/markets` listing no longer surfaces weather markets.** Verified 2026-05-16: `?tag=weather`, `?feeType=weather_fees`, `?negRiskMarketID=…`, `?category=Weather` are all silently ignored — Gamma returns the default 100-market trending listing (same failure mode as the documented `?clobTokenIds=` issue). Paginating `?active=true&closed=false` returns 0 weather markets at any offset up to ~10000 (Gamma's 422 cap). Direct `/events?slug=highest-temperature-in-<city>-on-<month>-<day>-<year>` still works and returns the event with all bucket markets embedded. `fetch_weather_markets` now uses event-slug enumeration as the primary path (every city in `CITY_ICAO` × next `_EVENT_SLUG_LOOKAHEAD_DAYS` days = ~600 slug probes per scan; concurrent with `Semaphore(8)`, well under the 10 rps rate limit). City-name → Polymarket-slug overrides live in `POLYMARKET_CITY_SLUG_OVERRIDES` (e.g. NYC's slug is `nyc`, not `new-york`); add a row when a city's first scan returns 0 events at the expected slug. Keyword + tag scans are kept as defensive fallbacks in case Polymarket restores the listing behavior.
- **`resolve_trades` catch-22 + exposure-cap silencer.** (Note: superseded as the *primary* signal by the on-chain payout-gating change of 2026-05-19; the catch-22 path now only fires in the legacy CLOB-fallback branch when `condition_id IS NULL`.) `_refresh_market_price` returns None for markets the CLOB dropped after resolution (gotcha above on `_fetch_orderbook` 404s). Pre-2026-05-17 the loop's `continue` left OPEN trades unresolved forever, pinning `MAX_EXPOSURE_PCT × bankroll` and silently zeroing every Kelly stake via `size_position`'s exposure-cap check. **Fallback (added 2026-05-17):** when `_refresh_market_price` returns None AND `market.end_date < now - RESOLVE_NO_PRICE_GRACE_HOURS (4h)`, (a) `fill_price IS NULL` → mark LOST (order never landed on-chain), (b) `fill_price IS NOT NULL` → log a warning telling the operator to run `admin reconcile-stuck` for on-chain payout settlement. **`admin reconcile-stuck`** does the heavier lift: queries `balanceOf` for each stuck trade's token; `bal=0` → LOST, `bal>0` → backfills `fill_price` and looks up on-chain `payoutNumerators` (via `bet_helpers.get_payout_outcome`) to settle WON/LOST inline. Backfills missing `Market.condition_id` from `data-api.polymarket.com/positions` (Gamma drops resolved markets so direct lookup returns []). **`MAX_EXPOSURE_USD_FLOOR=$300`** keeps the bot trading at small bankroll — the exposure cap is now `max(MAX_EXPOSURE_PCT × bankroll, MAX_EXPOSURE_USD_FLOOR)`; floor binds only when bankroll < ~$1200. **`job_daily_settlement`** now emits a Telegram warning at 80% of the effective cap so the operator catches this class of bug before throughput dies. **`job_reconcile_orders`** lookback was widened: it now also catches `fill_price IS NULL AND exchange_status IN ('delayed','matched','matching')` trades regardless of age, as long as their market ended within the last 7 days — prevents the long-tail drift that caused 16 May-13/14 trades to escape reconcile entirely.
- **Stuck-OPEN silencer recurrence 2026-05-18.** Same failure class, two new gaps closed: **(1)** `check_order_status` was calling `order.get("status", ...)` without a None check; `client.get_order` returns None for both queued AND post-resolution dropped orders, raising `AttributeError` into a bare `except` that logged a contextless one-liner. 24 `matched`-but-no-fill rows spammed the same warning for 12+ hours while staying stuck OPEN. Fix: explicit None branch + `exc_info=True` on the lookup-raised branch (mirrors `_update_fill_details`'s existing pattern). **(2)** `cli admin reconcile-stuck` filtered `exchange_status == "delayed"` only, silently skipping the 24 `matched` rows. Fix: widened to `IN ('delayed','matched','matching')` to match `job_reconcile_orders`. **(3)** observability gap — the 80%-of-cap alert in `job_daily_settlement` is once-a-day; can be up to ~24h before the operator notices. Added 5-min cadence **stuck-OPEN heartbeat** in `job_reconcile_orders` (log-only sweep + Telegram alert when stuck-row count ≥ `STUCK_ALERT_MIN_COUNT=5` OR stuck stake ≥ `STUCK_ALERT_EXPOSURE_FRACTION=50%` of effective cap). Cooldown timer in `bot_state` (`reconcile.stuck_alert_last_pushed_at`) so the queue isn't spammed every tick during a multi-hour silence. Pre-existing bug also fixed in this change: `job_daily_settlement`'s 80%-of-cap alert was reading `settings.MAX_EXPOSURE_PCT` which doesn't exist on Settings — the real constant lives at module scope in `src/risk/kelly.py`. The `try/except` swallowed the `AttributeError` so the alert had never actually fired. Now imports `MAX_EXPOSURE_PCT` from `src.risk.kelly` at the top of `scheduler/__init__.py`.
- **Phantom-LOST in `resolve_trades` null-fill grace 2026-05-20.** Pre-fix `src/resolution.py:163` treated `fill_price IS NULL` past the 4h grace as "order never landed → mark LOST with `pnl=-stake`". That conflated `exchange_status='exception:*'` (truly never landed) with `exchange_status='matched'` + valid `order_id` (DID land — `fill_price=None` is just a `job_reconcile_orders` backfill gap). 24 of 25 null-fill LOST trades in a 7d window were UNRESOLVED on-chain (phantom $228 of "loss" on 2026-05-20). **Fix:** both the chain-reachable branch and the legacy CLOB-mid branch now `balanceOf(token_id, funder_address)` via the new `_query_token_balance` helper before deciding. `bal>0` → `_backfill_null_fill(trade)` (fill_price ← entry_price, filled_size ← stake/entry) and preserve OPEN; `bal==0` → LOST with **`pnl=0`** (true release-exposure, no invented loss); `bal=None` (chain unreachable) → conservative pre-fix `pnl=-stake` so an RPC outage can't silently hide losses. `_build_ctf_readonly` now returns `(ctf, funder_address)` tuple. **Recovery:** `cli admin reconcile-stuck --include-lost` widens the row-set to `status IN (OPEN, LOST)` and reverts phantom-LOST + `bal>0` rows back to OPEN (clears `pnl`/`exit_price`/`closed_at`, backfills `fill_price`). Verified dry-run: 24 backfill-fill + 46 mark-LOST (bal=0, now `pnl=0`) + 2 settle-LOST. Future bankroll math should now treat `LOST + fill_price IS NULL` rows as accounting cleanup, not realized losses.
- **Submission-failure circuit breaker 2026-05-20.** `src/risk/circuit_breakers.py` now counts `trades.exchange_status LIKE 'exception:%'` within the recent `SUBMIT_FAIL_PAUSE_WINDOW_MINUTES=10` window; once ≥`SUBMIT_FAIL_PAUSE_COUNT=5` it pauses new submissions for `SUBMIT_FAIL_PAUSE_MINUTES=30` and 🚨 Telegram-alerts (one-shot per pause-window). Catches CLOB protocol migrations / wallet allowance drops — the May 2026 PolyApiException class (115 rows in 7d, no alert, no auto-pause). Persists via the existing `_paused_until` machinery so a restart can't drop the pause. Pair with `Trade.exchange_error` (Text column, captures full `str(exc)` truncated to 2000 chars; replaces the pre-fix `f"exception:{type(exc).__name__}"[:50]` that dropped the SDK / CLOB response body). Migration: `o5p6q7r8s9t0_trade_exchange_error`.
