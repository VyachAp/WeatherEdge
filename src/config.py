from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

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

    # Short-range trading tunables
    SR_MIN_LIQUIDITY: float = 100.0
    SR_MIN_VOLUME: float = 50.0
    SR_MIN_EDGE_DISCOUNT: float = 0.65
    SR_PIPELINE_INTERVAL_MINUTES: int = 60

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

    # Unified pipeline (replaces short_range + wx_rapid when enabled)
    UNIFIED_PIPELINE_ENABLED: bool = False
    UNIFIED_PIPELINE_INTERVAL_MINUTES: int = 5

    # Redesigned trade filters (used by unified pipeline)
    MIN_PROBABILITY: float = 0.60
    MIN_ENTRY_PRICE: float = 0.40
    MAX_ENTRY_PRICE: float = 0.97
    MIN_DEPTH_USD: float = 50.0
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

    # Open-Meteo forecast
    OPENMETEO_RATE_LIMIT_RPS: float = 2.0


settings = Settings()