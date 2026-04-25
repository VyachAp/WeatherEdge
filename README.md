# weather-edge

Weather-driven trading bot for Polymarket "highest temperature in [city]" markets. Two trading paths run concurrently:

1. **Probability path** (every 5 min) — aggregates routine METARs and an Open-Meteo deterministic + ensemble forecast per city, builds a full probability distribution over integer temperature buckets, compares to live Polymarket CLOB prices, sizes positions via fractional Kelly.
2. **Lock-rule path** (every 30 s) — re-checks active binary markets and fires deterministic trades the moment a new METAR makes the daily max mathematically locked above (or below) the market threshold.

Both paths exploit the same structural edge: routine METARs publish 1–2 hours before the Wunderground UI (Polymarket's resolution source) updates.

## Architecture

```
                  Polymarket Gamma + CLOB             Aviation METAR (6 providers)        Open-Meteo (det. + ensemble)
                          │                                       │                                   │
                          ▼                                       ▼                                   ▼
                ┌───────────────────┐                  ┌───────────────────────┐         ┌─────────────────────────┐
                │ Market scanner    │                  │ fetch_metar_history   │         │ fetch_deterministic_*  +│
                │ + question parser │                  │ + cycle detection     │         │ fetch_ensemble_*       │
                └─────────┬─────────┘                  └──────────┬────────────┘         └─────────┬───────────────┘
                          │                                       │                                   │
                          │                                       └───────────┬───────────────────────┘
                          ▼                                                   ▼
                ┌──────────────────┐                              ┌────────────────────────┐
                │ Market + Snapshot│                              │ aggregate_state →      │
                │ + Signal + Trade │                              │ WeatherState (per ICAO)│
                │ tables           │                              └──────────┬─────────────┘
                └─────────┬────────┘                                         │
                          │                                  ┌───────────────┴────────────────┐
                          │                                  ▼                                ▼
                          │                       ┌────────────────────┐         ┌──────────────────────┐
                          │                       │ Lock-rule trader   │         │ Probability engine   │
                          │                       │ (deterministic,    │         │ (signal-based bucket │
                          │                       │  EASY/HARD locks)  │         │  distribution)       │
                          │                       └─────────┬──────────┘         └──────────┬───────────┘
                          │                                 │                               │
                          │                                 ▼                               ▼
                          │                       ┌────────────────────┐         ┌──────────────────────┐
                          │                       │ size_locked_position│         │ Edge calculator +    │
                          │                       │ (fixed % of bankroll│         │ size_position (Kelly │
                          │                       │  + depth cap)       │         │  + exposure caps)    │
                          │                       └─────────┬──────────┘         └──────────┬───────────┘
                          │                                 └──────────────┬────────────────┘
                          ▼                                                ▼
                ┌─────────────────────┐                       ┌────────────────────────────────────────┐
                │ Drawdown monitor    │ ────── multiplier ──► │ Polymarket CLOB place_order            │
                │ (NORMAL/CAUTION/    │                       │ + Telegram alerts (🔒 / 💰 / 🌡 / ⚠️ / 📊)│
                │  PAUSED/RECOVERY)   │                       └────────────────────────────────────────┘
                └─────────────────────┘
```

**Scheduler** (APScheduler) runs four recurring jobs:
- `job_scan_markets` — every 15 minutes
- `job_unified_pipeline` — every `UNIFIED_PIPELINE_INTERVAL_MINUTES` (default 5)
- `job_fast_lock_poll` — every `FAST_LOCK_POLL_INTERVAL_SECONDS` (default 30)
- `job_daily_settlement` — 22:00 UTC

See `CLAUDE.md` for the full pipeline contract and `docs/operator-guide.md` for day-to-day ops.

## Prerequisites

- Docker & Docker Compose
- Python 3.11+ and [Poetry](https://python-poetry.org/) (for local development)

## Quick Start

```bash
# 1. Clone and enter
git clone <repo-url> && cd weather-edge

# 2. Setup (starts DB, runs migrations, creates .env)
make setup

# 3. Edit configuration
vim .env

# 4. Start all services
make run
```

The scheduler runs on port 8080 (health check) and the dashboard on port 8501.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://weather:weather@localhost:5432/weatheredge` | Database connection string |
| `TELEGRAM_BOT_TOKEN` | _(empty)_ | Telegram bot token; dry-run if empty |
| `TELEGRAM_CHAT_ID` | _(empty)_ | Telegram chat ID for alerts |
| `MIN_EDGE` | `0.10` | Legacy setting; live pipeline uses hardcoded `edge_calculator.MIN_EDGE=0.05` |
| `KELLY_FRACTION` | `0.25` | Fractional Kelly multiplier (0–1) for the probability path |
| `MAX_POSITION_PCT` | `0.05` | Max single directional trade as % of bankroll |
| `INITIAL_BANKROLL` | `750` | Starting capital in USD |
| `UNIFIED_PIPELINE_INTERVAL_MINUTES` | `5` | Probability-path cadence |
| `FAST_LOCK_POLL_INTERVAL_SECONDS` | `30` | Lock-path cadence |
| `LOCK_RULE_ENABLED` | `true` | Master switch for the lock-rule trader |
| `LOCK_MARGIN_F` | `2.0` | Safety margin (°F) above/below threshold before declaring a lock |
| `LOCK_POSITION_PCT` | `0.02` | Lock trade size as % of bankroll (no Kelly) |
| `ENSEMBLE_MODELS` | `ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless` | Open-Meteo models for σ |
| `AUTO_EXECUTE` | `false` | When true, place real CLOB orders |
| `DAILY_SPEND_CAP_USD` | `200` | Hard 24h rolling spend cap |

## CLI Commands

```bash
python -m src.cli run             # Start scheduler daemon (probability + lock paths)
python -m src.cli scan            # One-shot: fetch markets & ingest snapshots
python -m src.cli backfill -d 7   # Backfill historical market snapshots
python -m src.cli status          # Bankroll, positions, recent signals
python -m src.cli paper-trade -d 30   # Dry-run simulation against history
python -m src.cli backtest-v2 -d 30   # Replay probability pipeline
python -m src.cli approve         # On-chain USDC/CTF contract approvals (one-time)
python -m src.cli test-trade      # Place a tiny FOK to validate live execution

# Manual betting helpers
python -m src.cli bet search "Phoenix"    # Gamma API search
python -m src.cli bet find --city Phoenix # Local-DB lookup (faster)
python -m src.cli bet info <market-id>
python -m src.cli bet place <market-id> --side YES --usd 10
python -m src.cli bet portfolio
python -m src.cli bet redeem --all        # CTF redemption after resolution

# Standalone backtest of the lock-rule trader
python scripts/backtest_lock_rule.py --days 30
```

## Dashboard

The Streamlit dashboard provides four pages:

- **Overview** -- KPIs, bankroll curve, today's signals, open positions
- **Signal Explorer** -- Historical signals with filters and scatter plots
- **Model Calibration** -- Reliability diagrams and Brier scores
- **Market Scanner** -- Live active markets with edge highlighting

Access at `http://localhost:8501` or run locally:

```bash
make dashboard
```

## Makefile Targets

| Target | Description |
|---|---|
| `make setup` | Start DB, run migrations, create `.env` |
| `make run` | Start all services (db + app + dashboard) |
| `make scan` | One-shot signal scan (local Python) |
| `make dashboard` | Start Streamlit locally |
| `make test` | Run pytest |
| `make logs` | Tail scheduler logs |
| `make down` | Stop containers |
| `make clean` | Stop containers and delete volumes |

## Deployment

### Docker Compose (local / VPS)

```bash
make run
# All three services start: db, app, dashboard
# Health check: curl http://localhost:8080/
```

### Digital Ocean App Platform

```bash
# Install doctl
brew install doctl   # or snap install doctl
doctl auth init

# Deploy from app spec (creates app + managed Postgres)
doctl apps create --spec .do/app.yaml

# Set secrets via dashboard or CLI
doctl apps update <app-id> --spec .do/app.yaml
```

The app spec (`.do/app.yaml`) provisions a `basic-xxs` instance and a managed PostgreSQL 16 database.

### Digital Ocean Droplet

```bash
# 1. Create a Droplet (Ubuntu 24.04, 1GB RAM)
# 2. SSH in and install Docker
ssh root@<droplet-ip>
curl -fsSL https://get.docker.com | sh

# 3. Clone and configure
git clone <repo-url> /opt/weather-edge && cd /opt/weather-edge
cp .env.example .env && vim .env

# 4. Start services
docker compose up -d

# 5. Run migrations
docker compose exec app python -m alembic upgrade head
```

For the Droplet approach, enable DO Managed Database for PostgreSQL instead of the containerized TimescaleDB for production reliability. Update `DATABASE_URL` in `.env` accordingly.

## Development

```bash
# Install dependencies
poetry install

# Run tests
make test

# Type check
mypy src/ --ignore-missing-imports

# (Ruff is not used on this project — skip lint.)
```

## Project Structure

```
src/
  cli.py                       CLI entry point (click) + manual bet helpers
  config.py                    Pydantic settings from .env
  scheduler.py                 APScheduler jobs + health server + lock-rule executor
  resolution.py                Trade settlement, bankroll, exposure
  db/
    engine.py                  SQLAlchemy async engine
    models.py                  ORM models (markets, trades, METARs, exceedance alerts, …)
  ingestion/
    polymarket.py              Gamma scanner + question parser
    openmeteo.py               Deterministic + ensemble forecast
    aviation/                  6-provider METAR/TAF/SIGMET stack (AWC, IEM, OGIMET, NOAA, CheckWX, AVWX)
    station_bias.py            Per-ICAO rolling bias correction
    wx.py                      Optional Weather.com v3 observations
  signals/
    state_aggregator.py        WeatherState builder + forecast blender
    probability_engine.py      Signal-based bucket distribution
    edge_calculator.py         Per-bucket edge + filter checks
    lock_rules.py              Deterministic lock decisions (EASY/HARD)
    forecast_exceedance.py     "Daily max set to beat forecast" alerts
    projected_market_lookup.py Market lookup by projected daily max
    mapper.py                  Geocoding, ICAO map, timezone helpers
    consensus.py               Weekly calibration coefficients (vestigial; not applied live)
    reverse_lookup.py          Find markets by city/station/observation
  risk/
    kelly.py                   Fractional Kelly + lock-rule fixed sizing
    drawdown.py                4-state drawdown machine with hysteresis
    circuit_breakers.py        Daily loss + consecutive loss halts
    simulate.py                Paper-trade simulator
  execution/
    polymarket_client.py       CLOB client (FOK orders, orderbook depth)
    alerter.py                 Telegram queue + inline buttons (exec/skip/detail)

scripts/
  backtest_lock_rule.py        Replay lock-rule against resolved markets
  debug_pipeline.py            Trace pipeline for one market/station
  inspect_loss.py              Drilldown on a losing trade
  backfill_station_bias_tz.py  Idempotent bias recompute (tz-correct windows)
```

## FAQ

**Q: Do I need a Telegram bot?**
A: No. If `TELEGRAM_BOT_TOKEN` is empty, alerts run in dry-run mode (logged but not sent).

**Q: Disk space?**
A: No GRIB cache anymore — forecasts come from Open-Meteo's HTTP API. The Postgres footprint is dominated by `metar_observations` and `market_snapshots`; expect <1 GB after a few months.

**Q: Can I run just the scanner without the full scheduler?**
A: Yes. `make scan` or `python -m src.cli scan` runs a one-shot scan and exits.

**Q: How do I add a new city?**
A: Add it to `CITIES` and `CITY_ICAO` in `src/signals/mapper.py`. Verify the ICAO is *not* in `_EXCLUDED_ICAOS` in `src/scheduler.py` (currently `{"VHHH", "LLBG"}` for Hong Kong / Tel Aviv, where Polymarket's resolution station diverges from the routine METAR feed). See `CLAUDE.md` for the full checklist.

**Q: What happens during a drawdown?**
A: The drawdown state machine has four states with hysteresis. >10% drawdown = `CAUTION` (half size), >20% = `PAUSED` (no new trades). On recovery, the bot enters `RECOVERY` (half size) until bankroll exceeds the prior peak — preventing snap-back to full size on a single up-tick.

**Q: How do I disable the lock-rule path in an emergency?**
A: Set `LOCK_RULE_ENABLED=false` in `.env` and restart. The probability path is unaffected. To kill *only* the 30-second fast-poll loop while keeping main-pipeline locks active, use `FAST_LOCK_POLL_ENABLED=false`.
