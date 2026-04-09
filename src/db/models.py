import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
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
    end_date = Column(DateTime)
    resolution_source = Column(String)
    tags = Column(JSONB)
    parsed_location = Column(String)
    parsed_variable = Column(String)
    parsed_threshold = Column(Float)
    parsed_operator = Column(String)
    parsed_target_date = Column(String)
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)

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
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    market = relationship("Market", back_populates="snapshots")


class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_source = Column(String, nullable=False)
    run_time = Column(DateTime, nullable=False)
    valid_time = Column(DateTime, nullable=False)
    location_lat = Column(Float, nullable=False)
    location_lon = Column(Float, nullable=False)
    variable = Column(String, nullable=False)
    ensemble_mean = Column(Float)
    ensemble_std = Column(Float)
    ensemble_members = Column(JSONB)
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, ForeignKey("markets.id"), nullable=False)
    model_prob = Column(Float, nullable=False)
    market_prob = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)
    direction = Column(Enum(TradeDirection), nullable=False)
    confidence = Column(Float)
    gfs_prob = Column(Float)
    ecmwf_prob = Column(Float)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

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
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)

    signal = relationship("Signal", back_populates="trades")
    market = relationship("Market", back_populates="trades")


class BankrollLog(Base):
    __tablename__ = "bankroll_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    balance = Column(Float, nullable=False)
    peak = Column(Float, nullable=False)
    drawdown_pct = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)