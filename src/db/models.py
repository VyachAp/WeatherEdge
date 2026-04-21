import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class TradeDirection(str, enum.Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"


class TradeStatus(str, enum.Enum):
    PENDING = "pending"
    OPEN = "open"
    WON = "won"
    LOST = "lost"


class Market(Base):
    __tablename__ = "markets"

    id = Column(String, primary_key=True)
    question = Column(Text, nullable=False)
    slug = Column(String)
    outcomes = Column(JSONB)
    current_yes_price = Column(Float)
    volume = Column(Float)
    liquidity = Column(Float)
    end_date = Column(DateTime(timezone=True))
    resolution_source = Column(String)
    tags = Column(JSONB)
    parsed_location = Column(String)
    parsed_variable = Column(String)
    parsed_threshold = Column(Float)
    parsed_operator = Column(String)
    parsed_target_date = Column(String)
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    snapshots = relationship("MarketSnapshot", back_populates="market")
    signals = relationship("Signal", back_populates="market")
    trades = relationship("Trade", back_populates="market")


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, ForeignKey("markets.id"), nullable=False)
    yes_price = Column(Float)
    no_price = Column(Float)
    volume = Column(Float)
    liquidity = Column(Float)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    market = relationship("Market", back_populates="snapshots")


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, ForeignKey("markets.id"), nullable=False)
    model_prob = Column(Float, nullable=False)
    market_prob = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)
    direction = Column(Enum(TradeDirection), nullable=False)
    confidence = Column(Float)
    gfs_prob = Column(Float)  # Legacy — always NULL, drop via migration later
    ecmwf_prob = Column(Float)  # Legacy — always NULL, drop via migration later
    aviation_prob = Column(Float)
    wx_prob = Column(Float)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    market = relationship("Market", back_populates="signals")
    trades = relationship("Trade", back_populates="signal")


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=False)
    market_id = Column(String, ForeignKey("markets.id"), nullable=False)
    direction = Column(Enum(TradeDirection), nullable=False)
    stake_usd = Column(Float, nullable=False)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)
    status = Column(Enum(TradeStatus), nullable=False, default=TradeStatus.PENDING)
    opened_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    closed_at = Column(DateTime(timezone=True))

    # Polymarket CLOB execution fields
    order_id = Column(String)  # CLOB order identifier
    token_id = Column(String)  # YES/NO conditional token ID
    fill_price = Column(Float)  # Actual execution price
    filled_size = Column(Float)  # Shares filled
    exchange_status = Column(String)  # live|matched|delayed|unmatched|failed

    signal = relationship("Signal", back_populates="trades")
    market = relationship("Market", back_populates="trades")


class BankrollLog(Base):
    __tablename__ = "bankroll_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    balance = Column(Float, nullable=False)
    peak = Column(Float, nullable=False)
    drawdown_pct = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Aviation weather models
# ---------------------------------------------------------------------------


class MetarObservation(Base):
    __tablename__ = "metar_observations"
    __table_args__ = (
        UniqueConstraint("station_icao", "observed_at", name="uq_metar_station_obs"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    station_icao = Column(String, nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    temp_c = Column(Float)
    dewpoint_c = Column(Float)
    temp_f = Column(Float)
    dewpoint_f = Column(Float)
    wind_speed_kts = Column(Float)
    wind_dir = Column(String)
    wind_gust_kts = Column(Float)
    visibility_m = Column(Float)
    visibility_miles = Column(Float)
    pressure_hpa = Column(Float)
    sky_condition = Column(JSONB)
    ceiling_ft = Column(Integer)
    flight_category = Column(String)
    is_speci = Column(Boolean, default=False)
    raw_metar = Column(Text)
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class ForecastExceedanceAlert(Base):
    __tablename__ = "forecast_exceedance_alerts"
    __table_args__ = (
        UniqueConstraint("station_icao", "observed_at", name="uq_exceedance_station_obs"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    station_icao = Column(String, nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    observed_temp_f = Column(Float, nullable=False)
    forecast_temp_f = Column(Float, nullable=False)
    delta_f = Column(Float, nullable=False)
    forecast_hour_utc = Column(Integer, nullable=False)
    alerted_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class TafForecast(Base):
    __tablename__ = "taf_forecasts"
    __table_args__ = (
        Index("ix_taf_station_issued", "station_icao", "issued_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    station_icao = Column(String, nullable=False)
    issued_at = Column(DateTime(timezone=True), nullable=False)
    valid_from = Column(DateTime(timezone=True), nullable=False)
    valid_to = Column(DateTime(timezone=True), nullable=False)
    periods = Column(JSONB)
    amendment_number = Column(Integer, default=0)
    raw_taf = Column(Text)
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class Pirep(Base):
    __tablename__ = "pireps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String, unique=True)
    observed_at = Column(DateTime(timezone=True))
    lat = Column(Float)
    lon = Column(Float)
    altitude_ft = Column(Integer)
    icing_type = Column(String)
    icing_intensity = Column(String)
    turbulence_type = Column(String)
    turbulence_intensity = Column(String)
    weather = Column(String)
    raw_text = Column(Text)
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class SynopObservation(Base):
    __tablename__ = "synop_observations"
    __table_args__ = (
        Index("ix_synop_wmo_observed", "wmo_id", "observed_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    wmo_id = Column(String, nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    temp_c = Column(Float)
    dewpoint_c = Column(Float)
    pressure_hpa = Column(Float)
    wind_speed_kts = Column(Float)
    wind_dir = Column(Integer)
    precip_mm = Column(Float)
    precip_period_hours = Column(Integer)
    cloud_cover_oktas = Column(Integer)
    raw_synop = Column(Text)
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class WxObservation(Base):
    __tablename__ = "wx_observations"
    __table_args__ = (
        Index("ix_wx_station_valid", "station_icao", "valid_time_utc"),
        UniqueConstraint("station_icao", "valid_time_local", name="uq_wx_station_time"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    station_icao = Column(String, nullable=False)
    units = Column(String, nullable=False, server_default="m")  # "m" (metric) or "e" (imperial)
    valid_time_utc = Column(DateTime(timezone=True), nullable=False)
    valid_time_local = Column(String, nullable=False)  # Dedup key
    temp_c = Column(Float)
    dewpoint_c = Column(Float)
    humidity = Column(Float)
    wind_speed_ms = Column(Float)
    wind_gust_ms = Column(Float)
    wind_dir = Column(Integer)
    pressure_hpa = Column(Float)
    pressure_trend = Column(String)  # "Falling"/"Rising"/"Steady"
    precip_1h_mm = Column(Float)
    precip_6h_mm = Column(Float)
    precip_24h_mm = Column(Float)
    snow_1h_mm = Column(Float)
    snow_24h_mm = Column(Float)
    temp_max_since_7am_c = Column(Float)
    temp_max_24h_c = Column(Float)
    temp_min_24h_c = Column(Float)
    cloud_cover = Column(Integer)
    visibility_km = Column(Float)
    uv_index = Column(Integer)
    fetched_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class StationBias(Base):
    """Daily record of observed vs forecast temperature bias per station."""

    __tablename__ = "station_biases"
    __table_args__ = (
        Index("ix_station_bias_icao_date", "station_icao", "date"),
        UniqueConstraint("station_icao", "date", name="uq_station_bias_day"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    station_icao = Column(String, nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)
    observed_max_c = Column(Float, nullable=False)
    forecast_peak_c = Column(Float, nullable=False)
    bias_c = Column(Float, nullable=False)  # observed - forecast
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class AviationAlert(Base):
    __tablename__ = "aviation_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String, unique=True)
    alert_type = Column(String)
    hazard = Column(String)
    severity = Column(String)
    area = Column(JSONB)
    valid_from = Column(DateTime(timezone=True))
    valid_to = Column(DateTime(timezone=True))
    raw_text = Column(Text)
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))