from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    DATABASE_URL: str = "postgresql+asyncpg://weather:weather@localhost:5432/weatheredge"

    @field_validator("DATABASE_URL")
    @classmethod
    def normalize_db_url(cls, v: str) -> str:
        if v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    MIN_EDGE: float = 0.10
    KELLY_FRACTION: float = 0.25
    MAX_POSITION_PCT: float = 0.05
    INITIAL_BANKROLL: float = 750.0
    GEFS_CACHE_DIR: str = "/tmp/weather-edge/gfs/"
    GEFS_CACHE_TTL_HOURS: int = 24
    ECMWF_CACHE_DIR: str = "/tmp/weather-edge/ecmwf/"
    ECMWF_CACHE_TTL_HOURS: int = 24


settings = Settings()