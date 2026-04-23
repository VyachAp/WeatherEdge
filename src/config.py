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
    AUTO_EXECUTE: bool = False  # Set True to place orders automatically
    DAILY_SPEND_CAP_USD: float = 200.0  # Max total spend per 24h
    MIN_STAKE_USD: float = 5.0  # Skip orders below this amount

    # Unified pipeline
    UNIFIED_PIPELINE_INTERVAL_MINUTES: int = 5

    # Trade filters
    MIN_PROBABILITY: float = 0.60
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
    ENSEMBLE_MODELS: str = (
        "ecmwf_ifs04,gfs_seamless,icon_seamless,gem_seamless,meteofrance_seamless"
    )
    # Inflate raw inter-model spread — NWP ensembles are under-dispersive vs
    # actual forecast error, ~20-30% for surface T.
    ENSEMBLE_SPREAD_MULTIPLIER: float = 1.3
    ENSEMBLE_MIN_SIGMA_F: float = 1.0
    ENSEMBLE_MAX_SIGMA_F: float = 5.0
    # If fewer than this many models returned usable peak-hour data, fall back
    # to the deterministic single-source endpoint.
    ENSEMBLE_MIN_MODELS: int = 3


settings = Settings()