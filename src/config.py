from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    DATABASE_URL: str = ""

    @field_validator("DATABASE_URL")
    @classmethod
    def normalize_db_url(cls, v: str) -> str:
        if v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        # asyncpg does not understand ?sslmode=; replace with ?ssl=
        v = v.replace("sslmode=", "ssl=")
        return v

    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    MIN_EDGE: float = 0.10
    KELLY_FRACTION: float = 0.25
    MAX_POSITION_PCT: float = 0.05
    INITIAL_BANKROLL: float = 750.0
    AWC_USER_AGENT: str = "WeatherEdge/1.0 (weather-trading-bot; contact@example.com)"
    AWC_RATE_LIMIT_RPS: float = 2.0

    # Multi-provider aviation API keys (empty = provider disabled)
    CHECKWX_API_KEY: str = ""
    AVWX_API_KEY: str = ""
    CHECKWX_DAILY_BUDGET: int = 2000
    IEM_RATE_LIMIT_RPS: float = 1.0
    OGIMET_RATE_LIMIT_RPS: float = 0.2
    NOAA_RATE_LIMIT_RPS: float = 1.0
    AVWX_RATE_LIMIT_RPS: float = 1.0

    # Weather Company v3 observations (airport ICAO stations)
    WX_API_KEY: str = ""  # Empty = WX pipeline disabled
    WX_RATE_LIMIT_RPS: float = 1.5  # ~90/min conservative
    WX_DAILY_BUDGET: int = 10000
    WX_PIPELINE_INTERVAL_MINUTES: int = 1  # Poll every minute
    WX_RETENTION_HOURS: int = 48
    WX_PEAK_CONFIRM_MINUTES: int = 15  # Minutes of decline before declaring peak done

    # Polymarket execution
    POLYMARKET_PRIVATE_KEY: str = ""  # Polygon wallet private key; empty = dry-run
    POLYMARKET_HOST: str = "https://clob.polymarket.com"
    POLYMARKET_CHAIN_ID: int = 137  # Polygon mainnet
    # 0=EOA (raw wallet), 1=Polymarket-Proxy, 2=Gnosis-safe-style proxy.
    # If your wallet has ever logged into polymarket.com, the API expects
    # signature_type=2 and POLYMARKET_FUNDER_ADDRESS set to the proxy
    # address (not the EOA). Leaving these at their defaults works only
    # for fresh wallets that have never touched the Polymarket UI.
    POLYMARKET_SIGNATURE_TYPE: int = 0
    POLYMARKET_FUNDER_ADDRESS: str = ""  # Empty = derive EOA from private key
    AUTO_EXECUTE: bool = False  # Set True to place orders automatically
    DAILY_SPEND_CAP_USD: float = 200.0  # Max total spend per 24h
    MIN_STAKE_USD: float = 5.0  # Skip orders below this amount

    # Unified pipeline
    UNIFIED_PIPELINE_INTERVAL_MINUTES: int = 5

    # Trade filters
    # Side-effective probability floor. After PR-1 the BucketEdge stores
    # `our_probability` in the chosen-side frame, so 0.50 means "trade
    # only when the model is at least a coin flip on the side we're
    # buying". Combined with MIN_EDGE=0.05 this allows entries down to
    # ~0.45 on whichever side has positive edge — the "earlier, lower
    # price" zone the user explicitly asked to unlock.
    MIN_PROBABILITY: float = 0.50
    MIN_ENTRY_PRICE: float = 0.40
    MAX_ENTRY_PRICE: float = 0.97
    MIN_DEPTH_USD: float = 10.0
    MIN_ROUTINE_COUNT: int = 3
    MARKET_CLOSE_BUFFER_MINUTES: int = 30
    MAX_POSITION_USD: float = 200.0
    DEPTH_POSITION_CAP_PCT: float = 0.20

    # Station bias tracking
    DEFAULT_STATION_BIAS_C: float = 1.0
    STATION_BIAS_WINDOW_DAYS: int = 30
    STATION_BIAS_MAX_C: float = 3.0

    # Circuit breakers
    DAILY_LOSS_STOP_USD: float = 200.0
    CONSECUTIVE_LOSS_PAUSE_COUNT: int = 3
    CONSECUTIVE_LOSS_PAUSE_HOURS: int = 2

    # Fast-poll lock loop. Runs between unified-pipeline ticks, re-checking
    # only the EASY lock direction (observed max already clears threshold by
    # margin) so latency from METAR publication to order placement is seconds
    # rather than up to 5 minutes.
    FAST_LOCK_POLL_ENABLED: bool = True
    FAST_LOCK_POLL_INTERVAL_SECONDS: int = 30

    # Lock-rule trader (deterministic physical-condition path)
    LOCK_RULE_ENABLED: bool = True
    LOCK_RULE_MAX_PRICE: float = 0.95
    # Match the unified-pipeline pre-filter so any price the lock path sees is
    # tradeable. Previously 0.30 — blocked cases where a mispriced market was
    # still at single-digit cents despite the outcome being physically locked
    # (a ~10× return on the side we're certain about).
    LOCK_RULE_MIN_PRICE: float = 0.05
    LOCK_MARGIN_F: float = 2.0
    LOCK_POSITION_PCT: float = 0.02
    LOCK_RULE_LOSS_WINDOW_HOURS: int = 72
    LOCK_RULE_LOSS_DISABLE_COUNT: int = 3

    # Open-Meteo forecast
    OPENMETEO_RATE_LIMIT_RPS: float = 2.0

    # Multi-model ensemble (Open-Meteo models= param). Spread across these
    # models drives the probability-engine sigma instead of the hardcoded
    # hours-based schedule.
    # `ecmwf_ifs04` returns no data on Open-Meteo's current /v1/forecast — use
    # `ecmwf_ifs025`. `meteofrance_seamless` is a Europe-domain regional model
    # and returns severely cold-biased peaks outside Europe (e.g. -8°C vs ECMWF
    # at OPKC/OMDB), so it's excluded by default.
    ENSEMBLE_MODELS: str = (
        "ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless"
    )
    # Inflate raw inter-model spread — NWP ensembles are under-dispersive vs
    # actual forecast error, ~20-30% for surface T.
    ENSEMBLE_SPREAD_MULTIPLIER: float = 1.3
    ENSEMBLE_MIN_SIGMA_F: float = 1.0
    ENSEMBLE_MAX_SIGMA_F: float = 5.0
    # If fewer than this many models returned usable peak-hour data, fall back
    # to the deterministic single-source endpoint.
    ENSEMBLE_MIN_MODELS: int = 3

    # Climate-normal prior. Multi-year per-station per-DOY climatology
    # acts as the Bayesian prior for the daily-max distribution before the
    # forecast Gaussian (likelihood) and METAR observations update it.
    # Ships disabled — backfill the `station_normals` table first via
    # `scripts/backfill_station_normals.py`, sanity-check the values,
    # then flip CLIMATE_PRIOR_ENABLED=true.
    CLIMATE_PRIOR_ENABLED: bool = False
    CLIMATE_NORMAL_YEARS: int = 10
    # Floor on posterior σ after the Bayesian blend — prevents tropical
    # / oceanic stations (low climatological σ) from collapsing the
    # distribution width to ~1°F and over-confidently quoting narrow
    # bracket markets.
    CLIMATE_PRIOR_MIN_SIGMA_F: float = 2.0
    # Reject degenerate priors entirely. A station whose computed
    # std_max_c exceeds this is silently bypassed — it would dilute the
    # forecast rather than anchor it.
    CLIMATE_PRIOR_MAX_SIGMA_F: float = 8.0

    # Residual-slope projection (lever A in the projection-latency redesign).
    # When enabled and the station has at least 3 routine METARs in the 6h
    # window with same-hour forecast cells, ``_project_daily_max`` projects
    # the residual forward at the observed hourly slope instead of decaying
    # the level residual with a 2h halflife. Captures "forecast falling
    # further behind every hour" 1-2 hours earlier than the legacy path.
    # Falls back to the halflife decay when fewer than 3 points are available.
    PROJECTION_RESIDUAL_SLOPE_ENABLED: bool = True


settings = Settings()