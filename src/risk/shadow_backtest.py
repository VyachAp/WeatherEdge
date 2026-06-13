"""Replay harness for the four-stage shadow decision flow.

Runs the (pure) ``shadow_decision.evaluate_shadow`` over historical settled
markets using only point-in-time data — archived forecasts, stored routine
METARs, and stored market-price snapshots — so we get an *immediate* read on the
information-latency thesis instead of waiting weeks for the live shadow ledger
to fill.

For each settled ``MarketResolution`` (the label = ``yes_won``) it reconstructs
the ``WeatherState`` the model would have seen at several points across the
trading day (by feeding it the routine METARs observed up to each point) and
records the model's probability alongside the market price at that instant. The
resulting rows feed the same pure ``cli._calibration_report`` aggregator the
live ``calibration-report`` uses, so the replay and live verdicts are
apples-to-apples.

Fidelity caveats (documented, not hidden):
  * Observations are reconstructed from STORED routine METARs on the target
    UTC day — the same source the resolver mirrors, but reconstructed states
    aren't bit-identical to what the live 6-provider failover saw at the time.
  * ``forecast_residual_f`` is left None (the residual mean-shift is skipped)
    rather than risk a bias-frame mismatch in replay; the dominant observational
    signals — the monotonic observed-max floor and obs_fraction σ-tightening —
    ARE faithfully reconstructed.
  * No historical bid/ask spread is stored, so cost = the snapshot mid (depth is
    treated as ample). Calibration scoring (Brier/log-loss of prob vs outcome)
    is spread-independent; the would-trade EV in replay is therefore optimistic.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, time, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from src.db.models import MarketResolution, MarketSnapshot, MetarObservation
from src.execution.binary_market import operator_class
from src.signals.forecast_archive_replay import (
    archive_to_forecast_fields,
    latest_archive_as_of,
)
from src.signals.shadow_decision import (
    DEFAULT_PARAMS,
    MarketSpec,
    ShadowParams,
    evaluate_shadow,
)
from src.signals.state_aggregator import WeatherState

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Fractions of the day's routine METARs to evaluate at — spans the
# observation-fraction axis (early/mid/late/end-of-day) so the calibration
# report can bucket the thesis test.
DEFAULT_LEAD_POINTS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

_REPLAY_BANKROLL = 500.0      # fixed notional; sizing is informational in replay
_AMPLE_DEPTH = 1_000_000.0    # treat depth as ample (no historical spread/depth)


def _spec_from_resolution(r: MarketResolution) -> MarketSpec | None:
    """Build the °F outcome spec from a settled resolution row.

    ``parsed_threshold`` and ``bucket_*_f`` are already °F (the same frame the
    live edge calculator and shadow model use)."""
    op = r.parsed_operator
    if op in ("above", "at_least", "below", "at_most"):
        if r.parsed_threshold is None:
            return None
        return MarketSpec(operator=op, threshold_f=float(r.parsed_threshold))
    if op in ("exactly", "range", "bracket"):
        if r.bucket_low_f is None or r.bucket_high_f is None:
            return None
        return MarketSpec(operator=op, low_f=float(r.bucket_low_f),
                          high_f=float(r.bucket_high_f))
    return None


def _aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _price_as_of(snapshots: list[tuple[datetime, float]], as_of: datetime) -> float | None:
    """Nearest snapshot ``yes_price`` at or before ``as_of`` (point-in-time).

    ``snapshots`` is a time-ordered list of ``(timestamp, yes_price)`` tuples."""
    best: float | None = None
    cutoff = _aware(as_of)
    for ts, price in snapshots:
        if ts <= cutoff:
            best = price
        else:
            break  # snapshots are time-ordered
    return best


async def run_shadow_backtest(
    session: "AsyncSession",
    *,
    days: int = 60,
    lead_points: tuple[float, ...] = DEFAULT_LEAD_POINTS,
    params: ShadowParams = DEFAULT_PARAMS,
) -> dict:
    """Replay the shadow model over settled markets; return row dicts + skip stats.

    The returned ``rows`` are exactly the dict shape ``cli._calibration_report``
    consumes (``updated_yes`` / ``market_mid`` / ``obs_fraction`` / ``yes_won`` /
    ``operator_class`` / ``would_trade`` / ``side`` / ``effective_cost``), so the
    caller scores them with the same aggregator the live report uses.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    res = (await session.execute(
        select(MarketResolution)
        .where(MarketResolution.resolved_at >= cutoff,
               MarketResolution.target_date_local.isnot(None),
               MarketResolution.station_icao.isnot(None))
        .order_by(MarketResolution.resolved_at)
    )).scalars().all()

    # Batch-load every snapshot for the resolved markets in ONE pass and group
    # by market_id. market_snapshots.market_id is UNINDEXED (FK only) over ~8M
    # rows, so a per-market query is a full seq-scan each time — one scan beats
    # hundreds. Only the three columns we need are pulled.
    market_ids = [r.market_id for r in res]
    snaps_by_market: dict[str, list[tuple[datetime, float]]] = {}
    if market_ids:
        snap_rows = (await session.execute(
            select(MarketSnapshot.market_id, MarketSnapshot.timestamp,
                   MarketSnapshot.yes_price)
            .where(MarketSnapshot.market_id.in_(market_ids),
                   MarketSnapshot.yes_price.isnot(None))
            .order_by(MarketSnapshot.market_id, MarketSnapshot.timestamp)
        )).all()
        for mid, ts, price in snap_rows:
            snaps_by_market.setdefault(mid, []).append((_aware(ts), float(price)))

    rows: list[dict] = []
    skips = {"no_spec": 0, "few_metars": 0, "no_archive": 0, "no_price": 0,
             "markets": 0, "points": 0}

    for r in res:
        spec = _spec_from_resolution(r)
        if spec is None:
            skips["no_spec"] += 1
            continue

        icao = r.station_icao
        day = r.target_date_local
        day_start = datetime.combine(day, time(0, 0), tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)

        metars = (await session.execute(
            select(MetarObservation)
            .where(MetarObservation.station_icao == icao,
                   MetarObservation.observed_at >= day_start,
                   MetarObservation.observed_at < day_end,
                   MetarObservation.is_speci == False,  # noqa: E712
                   MetarObservation.temp_f.isnot(None))
            .order_by(MetarObservation.observed_at)
        )).scalars().all()
        if len(metars) < 3:
            skips["few_metars"] += 1
            continue

        snapshots = snaps_by_market.get(r.market_id, [])

        yes_won = 1 if r.yes_won else 0
        op_cls = operator_class(r.parsed_operator)
        n = len(metars)
        skips["markets"] += 1

        for frac in lead_points:
            k = max(2, math.ceil(frac * n))
            sl = metars[:k]
            as_of = _aware(sl[-1].observed_at)
            current_max_f = max(m.temp_f for m in sl)
            temps = [m.temp_f for m in sl]
            rate = (temps[-1] - temps[0]) / max(1, len(temps) - 1)

            arch = await latest_archive_as_of(session, icao, day, as_of)
            if arch is None:
                skips["no_archive"] += 1
                continue
            fc = archive_to_forecast_fields(arch, as_of)

            price = _price_as_of(snapshots, as_of)
            if price is None:
                skips["no_price"] += 1
                continue

            state = WeatherState(
                station_icao=icao,
                current_max_f=current_max_f,
                metar_trend_rate=rate,
                dewpoint_trend_rate=0.0,
                forecast_peak_f=fc["forecast_peak_f"],
                hours_until_peak=fc["hours_until_peak"],
                solar_declining=False, solar_decline_magnitude=0.0,
                cloud_rising=False, cloud_rise_magnitude=0.0,
                routine_count_today=k,
                forecast_sigma_f=fc["forecast_sigma_f"],
                ensemble_model_count=fc["ensemble_model_count"],
                has_forecast=fc["has_forecast"],
            )

            result = evaluate_shadow(
                state, spec,
                yes_bid=None, yes_ask=None, yes_mid=price,
                depth_yes_usd=_AMPLE_DEPTH, depth_no_usd=_AMPLE_DEPTH,
                minutes_to_close=None, bankroll=_REPLAY_BANKROLL, params=params,
            )
            skips["points"] += 1
            rows.append({
                "updated_yes": result.estimate.updated_yes,
                "prior_yes": result.estimate.prior_yes,
                "market_mid": price,
                "obs_fraction": result.estimate.obs_fraction,
                "yes_won": yes_won,
                "operator_class": op_cls,
                "would_trade": result.decision.should_trade,
                "side": result.decision.side,
                "effective_cost": result.decision.cost.effective_cost,
            })

    return {"rows": rows, "stats": skips}
