# CLAUDE.md

## Project Overview

WeatherEdge is a weather-driven trading bot for Polymarket. It ingests NWP ensemble forecasts (GFS 31-member, ECMWF 51-member) and aviation weather data (METAR/TAF/PIREP/SIGMET), maps them to weather contract outcomes, detects probabilistic edges, and sizes positions via fractional Kelly criterion.

## Quick Reference

```bash
# Run tests
pytest tests/                         # full suite (~245 tests, ~17s)
pytest tests/test_signals.py -v       # signal pipeline only (~81 tests)

# Lint & type check
ruff check src/ tests/
mypy src/ --ignore-missing-imports

# One-shot scan (no scheduler)
python -m src.cli scan

# Start full daemon
python -m src.cli run
```

## Architecture

### Signal Pipeline (critical path)

```
polymarket.py (scan) -> mapper.py (geocode + fetch) -> consensus.py (blend) -> detector.py (filter + persist)
```

1. **Ingestion** (`src/ingestion/`): Polymarket markets, GFS/ECMWF ensembles, aviation METAR/TAF
2. **Mapping** (`src/signals/mapper.py`): Geocodes locations, resolves variable aliases, fetches all three probability sources concurrently, gathers aviation context
3. **Consensus** (`src/signals/consensus.py`): Dynamic horizon-based weighting with aviation intelligence adjustments
4. **Detection** (`src/signals/detector.py`): Edge computation, tiered filtering, persistence, deduplication

### Scheduler Jobs (`src/scheduler.py`)

| Job | Schedule | Purpose |
|-----|----------|---------|
| `job_scan_markets` | Every 15 min | Fetch and parse Polymarket markets |
| `job_forecast_pipeline` | 04:30, 10:30, 16:30, 22:30 UTC | Full NWP ingest + signal detection |
| `job_short_range_pipeline` | Every 60 min (configurable) | Aviation-focused, markets <= 30h only, skips NWP fetch |
| `job_daily_settlement` | 22:00 UTC | Resolve trades, P&L, calibration |

### Database (10 ORM tables in `src/db/models.py`)

Core: `Market`, `MarketSnapshot`, `Forecast`, `Signal`, `Trade`, `BankrollLog`
Aviation: `MetarObservation`, `TafForecast`, `Pirep`, `AviationAlert`

## Key Patterns

### Variable Alias System

Markets parsed as `snowfall`, `freeze`, or `heat_wave` are translated to NWP-compatible variables via `VARIABLE_ALIASES` in `mapper.py`. The `resolve_variable()` function handles this before any forecast fetch.

```python
# mapper.py
VARIABLE_ALIASES = {
    "snowfall":  {"nwp_variable": "precipitation"},
    "freeze":    {"nwp_variable": "temperature", "threshold_override": 32.0, "operator_override": "below"},
    "heat_wave": {"nwp_variable": "temperature", "operator_override": "above"},
}
```

To add a new alias: add an entry to `VARIABLE_ALIASES`. No changes needed to NWP or aviation code if the underlying variable is already supported.

### Consensus Weighting (horizon-based)

Aviation weight tapers by hours to resolution:
- 0-6h: 50% aviation, 30% ECMWF, 20% GFS
- 6-12h: 30% aviation
- 12-24h: 15% aviation
- 24-30h: 8% aviation
- >30h: 0% aviation (NWP only)

Weights are computed in `_compute_weights()` in `consensus.py`. The ECMWF/GFS 60/40 split is preserved within the NWP share.

### Tiered Filters

Three filter tiers in `passes_filters()` (`detector.py`):

| Tier | Condition | MIN_LIQ | MIN_VOL | MIN_EDGE | MIN_DAYS |
|------|-----------|---------|---------|----------|----------|
| Ultra-short | aviation + <=12h | 100 | 50 | 0.065 | 0 |
| Short-range | aviation + <=30h | 200 | 75 | 0.075 | 0 |
| Standard | no aviation | 300 | 100 | 0.10 | 1 |

### AviationContext

`AviationContext` (dataclass in `mapper.py`) carries intelligence from TAF amendments, SPECI events, PIREPs, and SIGMETs. It flows through the pipeline and adjusts confidence in `compute_consensus()`:
- TAF amendments > 2: -0.05 per extra (forecast instability)
- SPECI events in 2h: +0.05 (rapid change = edge)
- Severe PIREPs: +0.08
- Active SIGMETs: +0.10

### Short-Range Pipeline

`detect_signals_short_range()` in `detector.py` and `map_short_range_markets()` in `mapper.py` form the fast path for the 60-min scheduler job. Key differences from the full pipeline:
- Only processes markets with `hours_to_resolution <= 30`
- Deduplicates: skips markets with signals created in last 60 min
- Uses `_size_and_create_trades()` shared helper (extracted from `job_forecast_pipeline`)

## How To: Common Tasks

### Add a new weather variable to NWP models

1. Add GRIB variable definition to `VARIABLE_MAP` in both `src/ingestion/gfs.py` and `src/ingestion/ecmwf.py`
2. Add variable name to `SUPPORTED_VARIABLES` set in `src/signals/mapper.py`
3. If the variable needs special unit conversion, add a case to `convert_threshold()` in `mapper.py`
4. Add regex pattern to `src/ingestion/polymarket.py` if not already parsed

### Add a variable alias (remap market variable to existing NWP variable)

1. Add entry to `VARIABLE_ALIASES` in `src/signals/mapper.py`
2. If the market uses an unsupported operator (like `"occurs"`), add it to `OPERATOR_MAP`
3. Aviation `get_realtime_probability()` auto-resolves aliases via `resolve_variable()`

### Add a new aviation probability function

1. Implement `_prob_<variable>()` in `src/ingestion/aviation.py`
2. Add the variable case to `compute_taf_based_probability()` (line ~913)
3. The variable must be in `SUPPORTED_VARIABLES` or have a `VARIABLE_ALIASES` entry

### Tune short-range trading parameters

All short-range tunables are in `src/config.py` and `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SR_MIN_LIQUIDITY` | 100 | Min liquidity for aviation-confirmed <=12h signals |
| `SR_MIN_VOLUME` | 50 | Min volume for aviation-confirmed <=12h signals |
| `SR_MIN_EDGE_DISCOUNT` | 0.65 | Edge multiplier for ultra-short signals (0.65 = 35% reduction) |
| `SR_PIPELINE_INTERVAL_MINUTES` | 60 | Short-range pipeline frequency |

### Adjust consensus weights

Edit `_compute_weights()` in `src/signals/consensus.py`. The aviation weight schedule is a series of `if/elif` checks on `hours_to_resolution`. The confidence adjustments from `AviationContext` are applied after the base confidence calculation in `compute_consensus()`.

## Testing Conventions

- Tests live in `tests/` with `test_<module>.py` naming
- Use `_make_market(**overrides)` helper for mock Market objects
- Use `_future_date_str(days)` for date strings relative to now
- Mock external APIs at the module boundary (e.g., `@patch("src.signals.mapper.gfs.get_probability")`)
- Mock aviation with `@patch("src.signals.mapper.get_realtime_probability")`
- E2E signal tests mock: `get_active_weather_markets`, both NWP `get_probability`, `get_realtime_probability`, and `get_calibration_coefficients`
- AsyncMock for all async functions; session mocking via `AsyncMock()` with `session.flush = AsyncMock()`

## File Index (most-edited files)

| File | Purpose | Key exports |
|------|---------|-------------|
| `src/signals/mapper.py` | Market-to-forecast mapping, geocoding, aliases | `MarketSignal`, `AviationContext`, `map_market`, `map_all_markets`, `map_short_range_markets`, `resolve_variable` |
| `src/signals/consensus.py` | Multi-model blending + calibration | `ConsensusResult`, `compute_consensus`, `compute_calibrated_consensus`, `_compute_weights` |
| `src/signals/detector.py` | Edge detection, filtering, persistence | `ActionableSignal`, `detect_signals`, `detect_signals_short_range`, `passes_filters` |
| `src/ingestion/aviation.py` | METAR/TAF/PIREP/SIGMET parsing + probabilities | `get_realtime_probability`, `compute_taf_based_probability`, `taf_amendment_count`, `detect_speci_events`, `has_severe_weather_reports`, `alerts_affecting_location` |
| `src/ingestion/polymarket.py` | Market scanner + 15 regex parsers | `scan_and_ingest`, `get_active_weather_markets`, `parse_question` |
| `src/ingestion/gfs.py` | GEFS 31-member ensemble via Herbie | `get_probability`, `get_ensemble_stats` |
| `src/ingestion/ecmwf.py` | ECMWF 51-member ensemble | `get_probability`, `get_ensemble_stats` |
| `src/scheduler.py` | APScheduler jobs + health server | `job_forecast_pipeline`, `job_short_range_pipeline`, `job_scan_markets` |
| `src/execution/alerter.py` | Telegram notification queue | `Alerter`, `get_alerter` |
| `src/execution/polymarket_client.py` | CLOB order execution + spend tracking | `place_order`, `is_live`, `get_token_ids`, `check_order_status`, `cancel_order` |
| `src/config.py` | Pydantic settings from `.env` | `settings` |
| `src/risk/kelly.py` | Fractional Kelly position sizing | `size_position` |
| `src/risk/drawdown.py` | Drawdown state machine | `DrawdownMonitor` |

## Live Execution

### How It Works

Orders are placed via `py-clob-client` (Polymarket CLOB SDK) in `src/execution/polymarket_client.py`.

**Execution flow:**
1. `_size_and_create_trades()` creates a Trade row with `status=PENDING`
2. Calls `place_order(trade, session)` which:
   - Checks daily spend cap ($200 default)
   - Resolves YES/NO token IDs from Gamma API
   - Places a FOK (Fill-or-Kill) market order
   - Updates trade with `order_id`, `exchange_status`, `fill_price`
3. On success: trade status → OPEN, Telegram alert sent with "AUTO-EXECUTED" tag
4. On failure: trade stays PENDING, logged with error details

**Safety mechanisms:**
- `AUTO_EXECUTE=false` (default): no orders placed, signals logged only. Telegram buttons allow manual execution.
- `DAILY_SPEND_CAP_USD=200`: hard cap on 24h rolling spend. Checked before every order.
- `MIN_STAKE_USD=5`: orders below $5 are skipped.
- Drawdown state machine: reduces position size or pauses trading entirely at >20% drawdown.
- Dry-run mode: if `POLYMARKET_PRIVATE_KEY` is empty, all orders are logged but not sent.

**To go live:**
1. Fund a Polygon wallet with USDC
2. Approve USDC + Conditional Token contracts (see Polymarket docs)
3. Set in `.env`:
   ```
   POLYMARKET_PRIVATE_KEY=0x...
   AUTO_EXECUTE=true
   DAILY_SPEND_CAP_USD=200
   ```
4. Start with a small `DAILY_SPEND_CAP_USD` and monitor Telegram alerts

### Token Allowances (Required for EOA Wallets)

Before first trade, approve these contracts for USDC and Conditional Tokens:
- `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` (Main exchange)
- `0xC5d563A36AE78145C45a50134d48A1215220f80a` (Neg risk exchange)
- `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296` (Neg risk adapter)

## Gotchas

- `get_realtime_probability()` imports `resolve_variable` from `mapper.py` at call time (lazy import to avoid circular dependency). Don't move it to module level.
- The `asyncio.gather` in `map_market()` uses `return_exceptions=True` everywhere. Always check `isinstance(result, BaseException)` before using return values.
- `_FETCH_SEMAPHORE` (8 concurrent) in mapper.py limits parallel market mapping. Aviation has its own semaphore (1 RPS) in `aviation.py`. These are independent.
- The `Signal.aviation_prob` column already exists in the DB schema. No migration needed for it.
- `compute_consensus()` accepts `aviation_context` as `object | None` (not the concrete type) to avoid circular imports between consensus.py and mapper.py. It uses `getattr()` for field access.
- `MIN_DAYS=0` is only allowed when aviation data confirms the signal. Without aviation, `MIN_DAYS=1` still applies (same-day NWP-only markets are too uncertain).
- GRIB cache TTL is 24h by default. Stale cache is cleaned by `cleanup_stale_cache()` in gfs.py.
- `polymarket_client.py` lazily imports `py_clob_client` inside functions, not at module level. This keeps the import optional — the system runs without the SDK in dry-run mode.
- Token IDs are fetched from the Gamma API per-order. For high-frequency trading, consider caching them on the Market model.
- FOK orders either fill completely or cancel. No partial fill handling needed. If the order book is too thin, the order fails and the trade stays PENDING.
- There is no Polymarket testnet. All orders execute on Polygon mainnet with real USDC. Start with `DAILY_SPEND_CAP_USD=50` when first testing.
- The Trade model has new nullable columns (`order_id`, `token_id`, `fill_price`, `filled_size`, `exchange_status`). Run an Alembic migration to add them: `alembic revision --autogenerate -m "add execution fields to trades"` then `alembic upgrade head`.
