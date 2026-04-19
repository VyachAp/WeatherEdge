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

    # Weather Underground scraping
    WU_SCRAPE_DELAY_SECONDS: float = 3.0
    WU_SCRAPE_TIMEOUT_MS: int = 15000

    # WU confirmation pipeline
    WU_CONFIRM_ENABLED: bool = True
    WU_CONFIRM_DROP_F: float = 1.0  # Min drop from peak to count as decrease
    WU_CONFIRM_RETRY_MINUTES: int = 15  # Re-scrape interval for WU_WAITING
    WU_CONFIRM_MAX_SCRAPES: int = 8  # Max WU scrapes per station per day

    # Polymarket execution
    POLYMARKET_PRIVATE_KEY: str = ""  # Polygon wallet private key; empty = dry-run
    POLYMARKET_HOST: str = "https://clob.polymarket.com"
    POLYMARKET_CHAIN_ID: int = 137  # Polygon mainnet
    AUTO_EXECUTE: bool = False  # Set True to place orders automatically
    DAILY_SPEND_CAP_USD: float = 200.0  # Max total spend per 24h
    MIN_STAKE_USD: float = 5.0  # Skip orders below this amount


settings = Settings()