"""Tests for the Phase 0 measurement layer (M1-M4) telemetry writers.

Covers the pure bound/divergence logic (M3) plus best-effort upserts for
config epochs (M2), exposure snapshots (M4) and market resolutions (M3).
Async writers are driven via ``asyncio.run`` so the suite doesn't depend
on a particular pytest-asyncio mode.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.signals.config_epoch import _TRACKED_FLAGS, current_flags
from src.signals.market_resolution import divergence_f, implied_max_bounds


def _mkt(op, thr, q="Will the highest temperature in X be above 80F?"):
    return SimpleNamespace(
        id="m1",
        parsed_operator=op,
        parsed_threshold=thr,
        parsed_location="X",
        question=q,
    )


# --- M3 pure logic: implied bounds -----------------------------------------


def test_implied_bounds_above():
    m = _mkt("above", 80.0)
    assert implied_max_bounds(m, True) == (80.0, None)
    assert implied_max_bounds(m, False) == (None, 80.0)


def test_implied_bounds_at_least():
    m = _mkt("at_least", 75.0)
    assert implied_max_bounds(m, True) == (75.0, None)
    assert implied_max_bounds(m, False) == (None, 75.0)


def test_implied_bounds_below():
    m = _mkt("below", 60.0)
    assert implied_max_bounds(m, True) == (None, 60.0)
    assert implied_max_bounds(m, False) == (60.0, None)


def test_implied_bounds_at_most():
    m = _mkt("at_most", 60.0)
    assert implied_max_bounds(m, True) == (None, 60.0)
    assert implied_max_bounds(m, False) == (60.0, None)


def test_implied_bounds_exactly_pins_interval():
    m = _mkt("exactly", 88.0, q="Will the highest temperature in X be 88°F?")
    assert implied_max_bounds(m, True) == (88.0, 88.0)
    # NO on a window gives no tight bound.
    assert implied_max_bounds(m, False) == (None, None)


def test_implied_bounds_none_without_threshold():
    assert implied_max_bounds(_mkt("above", None), True) == (None, None)
    assert implied_max_bounds(_mkt(None, 80.0), True) == (None, None)


# --- M3 pure logic: divergence ---------------------------------------------


def test_divergence_hotter_colder_within_none():
    assert divergence_f(83.0, None, 80.0) == 3.0      # we read hotter
    assert divergence_f(77.0, 80.0, None) == -3.0     # we read colder
    assert divergence_f(81.0, 80.0, 82.0) == 0.0      # consistent
    assert divergence_f(None, 80.0, None) is None     # no observation


# --- M2 flags ---------------------------------------------------------------


def test_current_flags_covers_tracked():
    flags = current_flags()
    assert set(flags) == set(_TRACKED_FLAGS)
    assert "BRACKET_LIKE_NO_DISABLED" in flags
    assert "PER_OPERATOR_CALIBRATION_ENABLED" in flags


# --- M2 writer --------------------------------------------------------------


def test_record_config_epoch_inserts_when_no_prior():
    from src.signals.config_epoch import record_config_epoch

    async def _run():
        session = AsyncMock()
        res = MagicMock()
        res.scalars.return_value.first.return_value = None
        session.execute.return_value = res
        session.add = MagicMock()
        session.flush = AsyncMock()
        await record_config_epoch(session)
        session.add.assert_called_once()
        session.flush.assert_awaited_once()

    asyncio.run(_run())


def test_record_config_epoch_skips_when_unchanged():
    from src.signals.config_epoch import record_config_epoch

    async def _run():
        session = AsyncMock()
        prior = SimpleNamespace(id=7, flags_json=current_flags())
        res = MagicMock()
        res.scalars.return_value.first.return_value = prior
        session.execute.return_value = res
        session.add = MagicMock()
        out = await record_config_epoch(session)
        assert out == 7
        session.add.assert_not_called()

    asyncio.run(_run())


# --- M4 writer --------------------------------------------------------------


def test_record_exposure_snapshot_adds_row():
    from src.signals.exposure_snapshot import record_exposure_snapshot

    async def _run():
        session = AsyncMock()
        res = MagicMock()
        res.all.return_value = [("above", 3), ("exactly", 2), (None, 1)]
        session.execute.return_value = res
        session.add = MagicMock()
        await record_exposure_snapshot(session, equity=1000.0, exposure=200.0)
        session.add.assert_called_once()
        snap = session.add.call_args[0][0]
        assert snap.n_open == 6
        assert snap.n_open_by_class["threshold"] == 3
        assert snap.n_open_by_class["bracket-like"] == 2
        assert snap.n_open_by_class["unknown"] == 1
        assert snap.effective_cap >= 300.0  # USD floor binds at $1000 bankroll
        assert snap.headroom == snap.effective_cap - 200.0

    asyncio.run(_run())


# --- M3 writer --------------------------------------------------------------


def test_record_market_resolution_upserts():
    from src.signals.market_resolution import record_market_resolution

    async def _run():
        session = AsyncMock()
        session.execute = AsyncMock()
        m = _mkt("above", 80.0)
        await record_market_resolution(
            session, m, yes_won=True, station_icao="KXXX",
            routine_metar_max_f=85.0,
        )
        session.execute.assert_awaited_once()

    asyncio.run(_run())


# --- M3 backfill ------------------------------------------------------------


def test_backfill_routine_max_fills_and_recomputes_divergence():
    from src.signals.market_resolution import backfill_routine_max
    from datetime import date

    async def _run():
        # Two rows for the same station-day, different implied bounds.
        # ``above 80`` YES → max ≥ 80 (lower=80); ``below 70`` YES → max ≤ 70.
        r_above = SimpleNamespace(
            resolved_max_lower_f=80.0, resolved_max_upper_f=None,
            routine_metar_max_f=None, divergence_f=None,
        )
        r_below = SimpleNamespace(
            resolved_max_lower_f=None, resolved_max_upper_f=70.0,
            routine_metar_max_f=None, divergence_f=None,
        )
        session = AsyncMock()
        res = MagicMock()
        res.scalars.return_value.all.return_value = [r_above, r_below]
        session.execute.return_value = res

        n = await backfill_routine_max(
            session, station_icao="KXXX", target_date_local=date(2026, 5, 30),
            routine_metar_max_f=84.0,
        )
        assert n == 2
        # 84 is consistent with "max ≥ 80" → divergence 0.
        assert r_above.routine_metar_max_f == 84.0
        assert r_above.divergence_f == 0.0
        # 84 violates "max ≤ 70" by +14 → we read hotter than the resolver.
        assert r_below.divergence_f == 14.0

    asyncio.run(_run())


def test_resolve_trades_records_resolution_on_chain_outcome():
    """resolve_trades calls record_market_resolution on a genuine outcome."""
    from unittest.mock import patch
    from datetime import datetime, timedelta, timezone

    from src.db.models import Market, Trade, TradeStatus, TradeDirection
    from src.resolution import resolve_trades

    async def _run():
        market = Market(
            id="mkt-m3",
            question="Will the highest temperature be above 80F?",
            end_date=datetime.utcnow() - timedelta(hours=1),
            condition_id="0xCOND",
            parsed_threshold=80.0,
            parsed_operator="above",
            parsed_location="New York",
        )
        trade = Trade(
            id=1, market_id="mkt-m3", direction=TradeDirection.BUY_YES,
            stake_usd=10.0, entry_price=0.5, status=TradeStatus.OPEN,
            fill_price=0.5,
        )
        trade.market = market
        session = AsyncMock()
        res = MagicMock()
        res.scalars.return_value.unique.return_value = [trade]
        session.execute.return_value = res

        spy = AsyncMock()
        with patch(
            "src.resolution._build_ctf_readonly",
            new=AsyncMock(return_value=("ctf", "0xfunder")),
        ), patch(
            "src.resolution._query_payout_outcome",
            new=AsyncMock(return_value=True),
        ), patch(
            "src.signals.market_resolution.record_market_resolution", new=spy
        ):
            resolved = await resolve_trades(session)

        assert len(resolved) == 1
        spy.assert_awaited_once()
        # yes_won forwarded as a keyword.
        assert spy.await_args.kwargs["yes_won"] is True

    asyncio.run(_run())
