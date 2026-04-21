# CLAUDE.md

## Project Overview

WeatherEdge is a weather-driven trading bot for Polymarket daily "highest temperature in [city]" markets. Every 5 minutes it aggregates live METAR observations + Open-Meteo forecasts per city, builds a full probability distribution over integer temperature buckets, compares it to live Polymarket CLOB prices, and sizes positions via fractional Kelly.

The resolution source for these markets is the Wunderground history page, which mirrors **routine** (non-SPECI) METARs. The bot reads METARs directly and typically sees the same data 1–2 hours before the UI updates — that's the edge.

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
```

## Architecture

### Pipeline (critical path)

The unified pipeline runs in `src/scheduler.py::job_unified_pipeline` every 5 min:

```
circuit_breakers ─► get_active_weather_markets ─► group by ICAO
        │                                             │
        ▼                                             ▼
  Phase 1 (concurrent, sem=8):                 Phase 2 (sequential):
    aggregate_state(session, icao, lat, lon)     for each market in city:
      ├─ fetch_metar_history(icao, 24h)            ├─ get_token_ids (Gamma)
      ├─ fetch_forecast(lat, lon) (Open-Meteo)     ├─ get_best_bid_ask → live price
      ├─ get_bias(session, icao)                   ├─ get_orderbook_depth
      └─ returns WeatherState                      ├─ skip if price <= 0.05 or >= 0.95
                                                   ├─ compute_distribution(state, buckets)
                                                   ├─ compute_edges (or _binary_market_edge)
                                                   ├─ size_position (fractional Kelly)
                                                   └─ place_order
```

### Scheduler Jobs (`src/scheduler.py`)

| Job | Schedule | Purpose |
|-----|----------|---------|
| `job_scan_markets` | Every 15 min | Fetch Polymarket weather markets via Gamma API, upsert `Market` + `MarketSnapshot` |
| `job_unified_pipeline` | Every `UNIFIED_PIPELINE_INTERVAL_MINUTES` (5m default) | Aggregate per-city weather state, compute edges, place trades |
| `job_daily_settlement` | 22:00 UTC | Resolve expired trades, record station bias, send daily summary, weekly calibration check |

`job_startup` runs once on boot: starts Telegram alerter, loads drawdown state, runs an initial market scan.

### Data sources

| Source | Module | Purpose |
|--------|--------|---------|
| Polymarket Gamma API | `src/ingestion/polymarket.py` | Active market list, parsed threshold/operator/location/date |
| Polymarket CLOB | `src/execution/polymarket_client.py` | Live best bid/ask, orderbook depth, order placement |
| Aviation METAR (6 providers) | `src/ingestion/aviation/` | Routine METAR history, daily max, trend, cycle detection |
| Open-Meteo | `src/ingestion/openmeteo.py` | Hourly temp/cloud/solar/dewpoint forecast (no API key) |
| Weather.com v3 | `src/ingestion/wx.py` | Optional auxiliary station observations (`WX_API_KEY` gated) |

Aviation providers fail over in order: AWC → IEM → OGIMET → NOAA → CheckWX/AVWX (API-key gated). See `src/ingestion/aviation/_aggregator.py`.

### Database (`src/db/models.py`)

Core: `Market`, `MarketSnapshot`, `Signal`, `Trade`, `BankrollLog`, `StationBias`
Aviation: `MetarObservation`, `TafForecast`, `Pirep`, `SynopObservation`, `AviationAlert`
Auxiliary: `WxObservation`

`Signal` has legacy `gfs_prob`/`ecmwf_prob`/`aviation_prob`/`wx_prob` columns — always NULL in the unified pipeline; kept for schema compatibility.

## Key Patterns

### Probability engine (`src/signals/probability_engine.py`)

`compute_distribution(state, buckets)` produces a `BucketDistribution` with `probabilities: {bucket_f: prob}` summing to 1.0.

Signal combination:
1. **Baseline Gaussian** centered on `state.forecast_peak_f` (Open-Meteo peak + station bias).
2. **Time-dependent sigma** from `hours_until_peak`: 1.0 past peak, 1.5 (≤2h), 2.5 (≤4h), 3.5 (>4h).
3. **METAR trend shift**: if rising >0.5°F/hr before peak, center shifts +min(rate*0.5, 2°F). If declining past peak, center locks to observed max.
4. **Solar/cloud cap**: if solar dropping >50% AND clouds rising >10% to >70%, hard cap upside at observed max. Strong solar decline alone → soft cap at max+1°F.
5. **Dewpoint adjustment**: rising Td >1°F/hr tightens sigma; falling Td widens it.
6. **Monotonicity constraint** (hard): `P(bucket < current_max_f) = 0`. If all mass gets zeroed, collapse to observed max bucket.

Every applied signal is appended to `distribution.reasoning` — the log tail you see in `[LTFM] binary ...` lines is this list.

### Edge calculator (`src/signals/edge_calculator.py`)

`compute_edges()` (for bracket markets) and `_binary_market_edge()` in `scheduler.py` (for binary threshold markets) both delegate filter checks to `_check_filters()`.

All filters must pass:

| Filter | Default | Source |
|--------|---------|--------|
| `edge >= MIN_EDGE` | 0.05 | `edge_calculator.MIN_EDGE` (hardcoded) |
| `our_probability >= MIN_PROBABILITY` | 0.60 | `settings.MIN_PROBABILITY` |
| `MIN_ENTRY_PRICE <= price <= MAX_ENTRY_PRICE` | 0.40 / 0.97 | `settings` |
| `routine_count >= MIN_ROUTINE_COUNT` | 3 | `settings` |
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

The drawdown monitor multiplier (`DrawdownMonitor.check`) is applied on top of the Kelly stake in the scheduler loop before the `MIN_STAKE_USD` check.

### Circuit breakers (`src/risk/circuit_breakers.py`)

Checked once per unified pipeline tick, before any city is evaluated:
- Daily loss stop: halt when today's P&L < `-DAILY_LOSS_STOP_USD` (-$200)
- Consecutive loss stop: 3 LOST trades in a row → pause `CONSECUTIVE_LOSS_PAUSE_HOURS` (2h), state held in process memory (`_paused_until`)

Per-city routine-count and bias-runaway checks live in the aggregator and edge filter, not here.

## How To: Common Tasks

### Add a new city

1. Add `"city name": (lat, lon)` to `CITIES` in `src/signals/mapper.py`.
2. Add `"city name": "ICAO"` to `CITY_ICAO`.
3. If the city uses °C in its market title, make sure the regex in `src/ingestion/polymarket.py` captures it (°C/CELSIUS handled by `_market_unit` in `scheduler.py`).
4. Seed `StationBias` if you have historical data — otherwise the default +1.0°C bias applies until enough settlements accumulate.

### Tune trade filters

Everything is in `src/config.py` (Pydantic `Settings`, overridable via `.env`):
- `MIN_EDGE` (hardcoded to 0.05 in `edge_calculator.py`, separate from `settings.MIN_EDGE`)
- `MIN_PROBABILITY`, `MIN_ENTRY_PRICE`, `MAX_ENTRY_PRICE`
- `MIN_DEPTH_USD`, `MIN_ROUTINE_COUNT`, `MARKET_CLOSE_BUFFER_MINUTES`
- Kelly caps: `KELLY_FRACTION`, `MAX_POSITION_PCT`, `MAX_POSITION_USD`, `DEPTH_POSITION_CAP_PCT`

### Change pipeline cadence

`UNIFIED_PIPELINE_INTERVAL_MINUTES` in `.env` (default 5). The cron-style settlement job is fixed at 22:00 UTC in `setup_scheduler()`.

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
| `src/scheduler.py` | APScheduler jobs + health server + binary/bracket edge helpers | `job_scan_markets`, `job_unified_pipeline`, `job_daily_settlement`, `run_scheduler` |
| `src/signals/state_aggregator.py` | Per-ICAO weather state snapshot | `WeatherState`, `aggregate_state` |
| `src/signals/probability_engine.py` | Signal-based bucket distribution | `BucketDistribution`, `compute_distribution` |
| `src/signals/edge_calculator.py` | Per-bucket edge + filter checks | `BucketEdge`, `compute_edges`, `_check_filters`, `MIN_EDGE` |
| `src/signals/mapper.py` | Geocoding, ICAO lookup, operator/date/threshold normalisation | `CITIES`, `CITY_ICAO`, `icao_for_location`, `cities_for_icao`, `geocode`, `normalize_operator`, `convert_threshold` |
| `src/signals/consensus.py` | Linear recalibration from resolved signals (weekly) | `get_calibration_coefficients`, `MIN_CALIBRATION_SAMPLES` |
| `src/signals/reverse_lookup.py` | Find markets by city/station/observation | `find_markets_for_city`, `find_markets_for_station`, `find_markets_for_observation`, `find_markets_for_event` |
| `src/ingestion/polymarket.py` | Gamma API scanner + question parser | `scan_and_ingest`, `ingest_markets`, `get_active_weather_markets`, `parse_question`, `is_weather_market` |
| `src/ingestion/openmeteo.py` | Hourly forecast + solar/cloud/dewpoint helpers | `OpenMeteoForecast`, `fetch_forecast`, `solar_declining`, `cloud_rising`, `dewpoint_trend` |
| `src/ingestion/aviation/` | Multi-provider METAR/TAF/PIREP/SIGMET | `fetch_metar_history`, `fetch_latest_metars`, `get_routine_daily_max`, `detect_metar_cycle`, `get_temp_trend`, `taf_amendment_count`, `has_severe_weather_reports`, `alerts_affecting_location` |
| `src/ingestion/station_bias.py` | Per-station forecast bias tracking | `get_bias`, `record_daily_outcome`, `is_bias_runaway` |
| `src/ingestion/wx.py` | Weather.com v3 observations (optional) | gated by `WX_API_KEY` |
| `src/execution/polymarket_client.py` | CLOB client, orderbook, orders | `is_live`, `get_token_ids`, `place_order`, `check_order_status`, `cancel_order`, `get_best_bid_ask`, `get_orderbook_depth`, `get_daily_spend` |
| `src/execution/alerter.py` | Telegram notification queue + inline buttons | `Alerter`, `get_alerter` |
| `src/risk/kelly.py` | Fractional Kelly with cascading caps | `PositionSize`, `size_position` |
| `src/risk/drawdown.py` | Drawdown state machine (NORMAL/CAUTION/PAUSED) | `DrawdownMonitor`, `DrawdownLevel` |
| `src/risk/circuit_breakers.py` | Daily loss + consecutive loss halts | `check_circuit_breakers`, `CircuitBreakerState` |
| `src/resolution.py` | Trade settlement, bankroll, exposure | `resolve_trades`, `get_current_bankroll`, `get_current_exposure`, `calculate_daily_pnl` |
| `src/db/models.py` | SQLAlchemy ORM | `Market`, `Signal`, `Trade`, `StationBias`, `MetarObservation`, `TafForecast`, `Pirep`, `WxObservation`, `AviationAlert` |
| `src/config.py` | Pydantic settings | `settings` |
| `src/cli.py` | CLI entry points | `scan`, `run`, `backfill`, bet helpers |

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
- `DrawdownMonitor` multiplies stake by a level-dependent factor (NORMAL=1.0, CAUTION<1.0, PAUSED=0).

**Going live:**
1. Fund a Polygon wallet with USDC.
2. Approve contracts (once) — see `docs/` for the allowance script. Contracts:
   - `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` (main exchange)
   - `0xC5d563A36AE78145C45a50134d48A1215220f80a` (neg-risk exchange)
   - `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296` (neg-risk adapter)
3. `.env`: `POLYMARKET_PRIVATE_KEY=0x...`, `AUTO_EXECUTE=true`, `DAILY_SPEND_CAP_USD=50` (start small).
4. There is no testnet — Polygon mainnet only, real USDC.

## Gotchas

- **Orderbook fetching is not disabled.** `job_unified_pipeline` calls `get_best_bid_ask` and `get_orderbook_depth` for every market with token IDs; 404s on resolved tokens produce the `Could not fetch orderbook for token X after 3 attempts` warning twice per market (once per call, since the failed fetch isn't cached). The subsequent skip comes from the `price <= 0.05 or price >= 0.95` check using the stale DB price.
- `py_clob_client` and `eth_account` are imported **inside** functions in `polymarket_client.py`, not at module level, so the system runs in dry-run mode without those deps installed.
- `_fetch_orderbook` caches only **successful** fetches for 30s. Failures are not negative-cached, so two back-to-back calls to a dead token retry 3×3×throttle delay.
- `WeatherState` is produced **even if Open-Meteo fails** — in that case `forecast_peak_f = current_max_f`, `hours_until_peak = 0`, and solar/cloud signals are False. Distribution will degenerate to a narrow band around current max.
- Dewpoint trend in `state_aggregator.py` uses recent METARs (6h window), not Open-Meteo. `openmeteo.dewpoint_trend` exists but isn't wired into the pipeline.
- `consensus.py` is a vestigial filename — it no longer blends multi-model forecasts; it only fits linear calibration coefficients (`slope*predicted + intercept`) from resolved signals, logged on Sundays. Not applied to live probabilities.
- `MIN_EDGE` is defined twice: `edge_calculator.MIN_EDGE = 0.05` (used by the unified pipeline) and `settings.MIN_EDGE = 0.10` (unused today; legacy).
- `gfs.py`/`ecmwf.py`/`detector.py`/old `mapper.py` signal-generation paths are gone. The `Signal.gfs_prob`/`ecmwf_prob` columns remain in the schema but are always NULL.
- Token IDs are refetched from Gamma per order and per price call. High-frequency trading would benefit from caching on the `Market` row, but current 5-min cadence makes it a non-issue.
- FOK orders fully fill or cancel — no partial fills; thin books leave trades in `PENDING`.
- Aviation has its own rate-limit semaphores per provider (`_RATE_LIMIT_RPS` constants) independent of the pipeline semaphore (`_UNIFIED_CONCURRENCY=8`).
- `_paused_until` circuit-breaker state resets on process restart. Consecutive losses are re-queried from DB each check, so the pause re-engages if the streak still holds.
- `Market.current_yes_price` is **overwritten** in-memory during Phase 2 with the live CLOB mid — that write is never committed to the DB row unless the session flushes; it only affects the current tick's probability/price comparison.
