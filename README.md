# weather-edge

Weather-driven Polymarket edge detection and trading signals. Ingests NWP ensemble forecasts (GFS, ECMWF), maps them to Polymarket weather contracts, detects probabilistic edges, and sizes positions via fractional Kelly criterion.

## Architecture

```
                    Polymarket API          NCEP GFS/GEFS          ECMWF Open Data
                         |                      |                       |
                         v                      v                       v
                  +--------------+      +---------------+      +-----------------+
                  |  polymarket  |      |    gfs.py     |      |   ecmwf.py      |
                  |  scanner     |      |  31 ensemble  |      |  51 ensemble    |
                  +--------------+      |  members      |      |  members        |
                         |              +---------------+      +-----------------+
                         |                      |                       |
                         v                      v                       v
               +-------------------+    +--------------------------------------+
               | Market + Snapshot |    |           Forecast table             |
               |    tables         |    |    (ensemble mean, std, members)     |
               +-------------------+    +--------------------------------------+
                         |                              |
                         +------------+  +--------------+
                                      |  |
                                      v  v
                              +----------------+
                              |  Signal mapper |
                              |  + detector    |
                              +----------------+
                                      |
                            +---------+---------+
                            |                   |
                            v                   v
                    +---------------+   +----------------+
                    | Kelly sizing  |   |   Drawdown     |
                    | + exposure    |   |   state        |
                    | caps          |   |   machine      |
                    +---------------+   +----------------+
                            |                   |
                            +--------+----------+
                                     |
                                     v
                            +------------------+
                            | Trade records +  |
                            | Telegram alerts  |
                            +------------------+
                                     |
                                     v
                          +---------------------+
                          | Streamlit dashboard |
                          | :8501               |
                          +---------------------+
```

**Scheduler** (APScheduler) runs three recurring jobs:
- Market scan every 15 minutes
- Forecast pipeline at 04:30, 10:30, 16:30, 22:30 UTC
- Daily settlement at 22:00 UTC

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
| `MIN_EDGE` | `0.10` | Minimum edge to generate a signal |
| `KELLY_FRACTION` | `0.25` | Fractional Kelly multiplier (0-1) |
| `MAX_POSITION_PCT` | `0.05` | Max single trade as % of bankroll |
| `INITIAL_BANKROLL` | `750` | Starting capital in USD |
| `GEFS_CACHE_DIR` | `/data/grib_cache/gfs` | GFS GRIB file cache directory |
| `GEFS_CACHE_TTL_HOURS` | `24` | GFS cache time-to-live |
| `ECMWF_CACHE_DIR` | `/data/grib_cache/ecmwf` | ECMWF GRIB file cache directory |
| `ECMWF_CACHE_TTL_HOURS` | `24` | ECMWF cache time-to-live |

## CLI Commands

```bash
weather-edge run          # Start scheduler daemon
weather-edge scan         # One-shot: fetch markets & generate signals
weather-edge backfill -d7 # Backfill 7 days of market snapshots
weather-edge status       # Show bankroll, positions, recent signals
weather-edge paper-trade -d30  # Simulate 30 days of trading
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

# Lint
ruff check src/ tests/

# Type check
mypy src/ --ignore-missing-imports
```

## Project Structure

```
src/
  cli.py              CLI entry point (click)
  config.py           Pydantic settings from .env
  scheduler.py        APScheduler orchestration + health check
  resolution.py       Trade resolution & P&L
  db/
    engine.py         SQLAlchemy async engine
    models.py         ORM models (6 tables)
  ingestion/
    gfs.py            GEFS ensemble ingestion (31 members)
    ecmwf.py          ECMWF ensemble ingestion (51 members)
    polymarket.py     Market scanner & parser
  signals/
    mapper.py         Market-to-forecast mapping
    detector.py       Edge detection & filtering
    consensus.py      Multi-model Bayesian consensus
  risk/
    kelly.py          Fractional Kelly sizing
    drawdown.py       Drawdown protection state machine
    simulate.py       Backtesting simulator
  execution/
    alerter.py        Telegram alert queue
  monitoring/
    dashboard.py      Streamlit dashboard (4 pages)
```

## FAQ

**Q: Do I need a Telegram bot?**
A: No. If `TELEGRAM_BOT_TOKEN` is empty, alerts run in dry-run mode (logged but not sent).

**Q: How much disk space does the GRIB cache need?**
A: Each forecast cycle downloads ~200MB. With a 24-hour TTL, expect ~1GB peak usage.

**Q: Can I run just the scanner without the full scheduler?**
A: Yes. `make scan` or `weather-edge scan` runs a one-shot scan and exits.

**Q: How do I add a new weather variable?**
A: Add the GRIB variable mapping in `src/ingestion/gfs.py` and `src/ingestion/ecmwf.py`, then update the mapper in `src/signals/mapper.py`.

**Q: What happens during a drawdown?**
A: The drawdown state machine reduces position sizes: >10% drawdown = half size, >20% = trading paused. Resumes at half size when recovering.
