import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    Date,
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
    # Redemption metadata captured from Gamma at scan time. Gamma drops
    # resolved markets from `GET /markets?id=...`, so we must persist
    # `condition_id`/`neg_risk`/`clob_token_ids` while the market is
    # still active — `bet redeem` reads from here instead of refetching.
    condition_id = Column(String, index=True)  # hex 0x… on-chain conditionId
    neg_risk = Column(Boolean)
    clob_token_ids = Column(JSONB)  # [yes_token_id, no_token_id]
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
    # One signal per (market, direction). UPSERT in scheduler.py refreshes
    # model_prob/market_prob/edge/created_at on each tick.
    __table_args__ = (
        UniqueConstraint("market_id", "direction", name="uq_signals_market_direction"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, ForeignKey("markets.id"), nullable=False)
    model_prob = Column(Float, nullable=False)
    # Engine probability *before* `apply_calibration` was applied. The
    # calibration regression must fit from this, not from `model_prob`,
    # otherwise it learns a correction on top of its own correction.
    # NULL on pre-migration rows; `calibration.py` falls back to `model_prob`.
    raw_model_prob = Column(Float)
    calibrated = Column(Boolean)
    market_prob = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)
    direction = Column(Enum(TradeDirection), nullable=False)
    confidence = Column(Float)
    # Path that produced this signal — 'probability' or 'lock'. Lets
    # post-mortems split realised P&L by path without re-deriving from logs.
    signal_kind = Column(String, nullable=False, default="probability")
    # Lock-rule decision context (NULL for probability-path signals).
    lock_branch = Column(String)  # 'easy_super' | 'easy_standard' | 'hard' | 'range_overshoot' | 'range_undershoot' | 'range_in_window'
    lock_routine_count = Column(Integer)
    lock_observed_max_f = Column(Float)
    # Legacy per-model probability columns — always NULL in the unified
    # pipeline. Kept for schema compatibility (no live readers); drop via
    # migration once we're comfortable with the column-removal path.
    gfs_prob = Column(Float)
    ecmwf_prob = Column(Float)
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
    redeemed_at = Column(DateTime(timezone=True), nullable=True, index=True)

    # Polymarket CLOB execution fields
    order_id = Column(String)  # CLOB order identifier
    token_id = Column(String)  # YES/NO conditional token ID
    fill_price = Column(Float)  # Actual execution price
    filled_size = Column(Float)  # Shares filled
    exchange_status = Column(String)  # live|matched|delayed|unmatched|failed
    # Full str(exc) when order submission raised. Pre-2026-05-20 only the
    # exception class name was stored (in exchange_status, truncated to
    # 50 chars), losing the API response body and making PolyApiException
    # postmortems require trawling DigitalOcean logs.
    exchange_error = Column(Text, nullable=True)

    # Submit-time market context — populated by place_order before the FAK
    # call. Lets post-mortems decompose `fill_price - entry_price` into
    # spread (yes_ask - yes_bid) vs depth-walked (size > top-of-book qty).
    # NULL for old rows and for paths that don't capture (e.g. backtests).
    submit_yes_bid = Column(Float)
    submit_yes_ask = Column(Float)
    submit_depth_usd = Column(Float)
    submit_at = Column(DateTime(timezone=True))

    signal = relationship("Signal", back_populates="trades")
    market = relationship("Market", back_populates="trades")


class EvaluationLog(Base):
    """Append-only telemetry: one row per per-side edge evaluation.

    Captures both PASSING and REJECTED candidates so filter-tightening
    decisions (lower MIN_EDGE? raise MIN_DEPTH_USD?) can be backtested
    against the actual stream of candidates we saw, not just the slice
    that passed. ``signals`` is now de-duplicated to one row per
    ``(market_id, direction)`` and only contains passing edges, so it
    can no longer answer "what did we evaluate today and why did it
    fail" — this table can.

    Volume note: one row per (city × market × tick) — at 8 cities × ~5
    markets/city × 12 ticks/hour ≈ 480 rows/hour ≈ 11.5K/day. Keep
    indexed on ``(market_id, created_at)`` for the per-market
    time-series queries calibration scripts will run.
    """

    __tablename__ = "evaluation_logs"
    __table_args__ = (
        Index("ix_eval_logs_market_created", "market_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, ForeignKey("markets.id"), nullable=False)
    direction = Column(Enum(TradeDirection), nullable=False)
    signal_kind = Column(String, nullable=False)  # 'probability' | 'lock'
    model_prob = Column(Float, nullable=False)
    # Raw engine probability + flag indicating whether calibration was
    # applied this tick. See Signal.raw_model_prob for rationale.
    raw_model_prob = Column(Float)
    calibrated = Column(Boolean)
    market_prob = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)
    passes = Column(Boolean, nullable=False)
    reject_reason = Column(String)  # NULL when passes=True
    depth_usd = Column(Float)
    minutes_to_close = Column(Float)
    routine_count = Column(Integer)
    # Forecast/observation context at evaluation time (added 2026-05-23).
    # Lets filter-tuning recompute the single-bucket NO guards from
    # telemetry alone: the landing-band guard needs forecast_peak_f +
    # current_max_f, and the peak-relative lead gate needs
    # hours_until_peak (minutes_to_close above measures lead to *close*,
    # which differs when close precedes the peak). forecast_sigma_f is the
    # ensemble peak σ — captured to diagnose the σ-collapse root cause
    # (tight model agreement ≠ accuracy) behind the deferred σ-floor work.
    # NULL on lock-path rows without forecast context and on legacy rows.
    forecast_peak_f = Column(Float)
    current_max_f = Column(Float)
    hours_until_peak = Column(Float)
    forecast_sigma_f = Column(Float)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class DecisionLog(Base):
    """Append-only telemetry: one row per per-side trading decision.

    Complements :class:`EvaluationLog`. ``evaluation_logs`` answers "did
    the edge clear ``_check_filters``?" — this table answers "what
    happened next." Outcomes cover the full post-filter pipeline:
    ``signal_written`` / ``trade_pending`` / ``trade_filled`` (success
    branches) and ``dup_blocked_inproc`` / ``dup_blocked_db`` /
    ``stake_below_min`` / ``drawdown_paused`` / ``cluster_cap_hit`` /
    ``cap_exceeded`` / ``no_token_ids`` / ``no_client`` / ``no_fill`` /
    ``order_failed`` (skip / failure branches).

    Today's debug session (2026-05-16) cost ~4 hours because 762
    passing evaluations materialised into 0 Signal rows and we had no
    telemetry on the gap. This table closes that.

    Column name is ``metadata_json`` (not ``metadata``) because
    ``Base.metadata`` is reserved by SQLAlchemy's declarative base.
    """

    __tablename__ = "decision_logs"
    __table_args__ = (
        Index("ix_decision_logs_outcome_created", "outcome", "created_at"),
        Index("ix_decision_logs_market_created", "market_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, ForeignKey("markets.id"), nullable=False)
    direction = Column(Enum(TradeDirection), nullable=False)
    signal_kind = Column(String, nullable=False)  # 'probability' | 'lock'
    outcome = Column(String, nullable=False)
    requested_stake_usd = Column(Float)
    actual_stake_usd = Column(Float)
    dd_multiplier = Column(Float)
    dd_level = Column(String)
    metadata_json = Column(JSONB)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


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
    current_max_f = Column(Float)
    forecast_peak_f = Column(Float)
    projected_max_f = Column(Float)
    metar_trend_rate = Column(Float)
    peak_passed = Column(Boolean)
    alerted = Column(Boolean)
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


class StationNormal(Base):
    """Climatological prior — multi-year mean & std of daily-max
    temperature for one (station, day-of-year) combination.

    Source defaults to the Open-Meteo Archive ERA5 reanalysis. Backfilled
    once via ``scripts/backfill_station_normals.py`` and read on every
    pipeline tick by ``ingestion.station_normals.get_normal`` to seed the
    Bayesian prior before forecast/observation likelihoods update it.
    """

    __tablename__ = "station_normals"
    __table_args__ = (
        UniqueConstraint("station_icao", "day_of_year", name="uq_station_normal_doy"),
        Index("ix_station_normal_icao_doy", "station_icao", "day_of_year"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    station_icao = Column(String, nullable=False)
    day_of_year = Column(Integer, nullable=False)  # 1..366; Feb 29 collapses to Feb 28
    mean_max_c = Column(Float, nullable=False)
    std_max_c = Column(Float, nullable=False)
    sample_years = Column(Integer, nullable=False)
    source = Column(String, nullable=False, default="openmeteo_archive_era5")
    computed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class ForecastArchive(Base):
    """Snapshot of an Open-Meteo blended forecast at the moment a unified
    pipeline tick fetched it.

    One row per (station, target-local-day, fetched_at) — multiple per
    station-day are expected (forecast evolves through the heating cycle).
    Used by the replay-capable backtest in ``src/risk/simulate.py`` to
    score the probability/projection paths against realised daily max.
    Hourly arrays match the ``OpenMeteoForecast`` shape (24 entries each
    by convention).
    """

    __tablename__ = "forecast_archive"
    __table_args__ = (
        Index(
            "ix_forecast_archive_icao_target_fetched",
            "station_icao",
            "target_date_local",
            "fetched_at",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    station_icao = Column(String, nullable=False)
    target_date_local = Column(Date, nullable=False)
    fetched_at = Column(DateTime(timezone=True), nullable=False)
    peak_temp_c = Column(Float, nullable=False)
    peak_hour_utc = Column(Integer, nullable=False)
    peak_temp_std_c = Column(Float, nullable=False, default=0.0)
    model_count = Column(Integer, nullable=False, default=1)
    hourly_temps_c = Column(JSONB, nullable=False)
    hourly_cloud_cover = Column(JSONB, nullable=False)
    hourly_solar_radiation = Column(JSONB, nullable=False)
    hourly_dewpoint_c = Column(JSONB, nullable=False)
    hourly_wind_speed = Column(JSONB, nullable=False)
    hourly_temps_std_c = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class BotState(Base):
    """Key/value durable state for runtime values that must survive a
    process restart (e.g. the consecutive-loss circuit-breaker pause).

    Intentionally generic. Keys are dotted strings like
    ``circuit_breakers.paused_until`` so consumers can namespace their
    own values without a migration per new key. Value is JSONB so we can
    store timestamps, counters, or small dicts without schema churn.
    Treat this table as a tiny config-with-history surface, not as a
    high-throughput store — every read is a primary-key lookup but it
    still hits Postgres, so cache in-process when sensible.
    """
    __tablename__ = "bot_state"

    key = Column(String, primary_key=True)
    value = Column(JSONB)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


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