"""Tests for ``risk.cluster_cap.cluster_stake_used``.

The helper is the anti-correlation guard that prevents independent Kelly
sizing across same-day same-city bracket / exactly buckets from
over-allocating. The contracts that matter:

* Returns 0.0 for threshold (non-bracket-like) markets so they're not
  accidentally clustered.
* Returns 0.0 when ``parsed_location`` or ``end_date`` are missing.
* Otherwise issues a ``SUM(stake_usd)`` filtered by parsed_location +
  end_date.date() + PENDING/OPEN status + non-dry-run exchange_status.

We assert against the compiled SQL because the value is a single
``session.execute`` round-trip with a numeric scalar return — building a
realistic in-memory query result would just re-test the SQLAlchemy
plumbing. The query shape is what makes the cap correct or wrong.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.risk.cluster_cap import cluster_stake_used


_DEFAULT_END_DATE = object()  # sentinel — `None` is a legal value the test
                              # wants to pass through unchanged.


def _market(*, operator: str, location: str | None = "Phoenix",
            end_date=_DEFAULT_END_DATE) -> SimpleNamespace:
    if end_date is _DEFAULT_END_DATE:
        end_date = datetime(2026, 5, 16, 23, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        id="m1",
        parsed_operator=operator,
        parsed_location=location,
        end_date=end_date,
    )


@pytest.mark.asyncio
async def test_threshold_market_short_circuits_to_zero():
    """`above`/`at_least`/`below`/`at_most` are NOT bracket-like — the
    helper must return 0.0 without hitting the database. Otherwise a
    threshold market would inherit cluster exposure from unrelated
    bucket trades on the same day."""
    session = AsyncMock()
    market = _market(operator="above")

    used = await cluster_stake_used(session, market)

    assert used == 0.0
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_missing_location_short_circuits_to_zero():
    """Cluster identity requires both parsed_location and end_date.
    Missing either → no cluster to sum over."""
    session = AsyncMock()
    market = _market(operator="bracket", location=None)

    used = await cluster_stake_used(session, market)

    assert used == 0.0
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_missing_end_date_short_circuits_to_zero():
    session = AsyncMock()
    market = _market(operator="exactly", end_date=None)

    used = await cluster_stake_used(session, market)

    assert used == 0.0
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_bracket_market_issues_aggregated_sum():
    """Sanity check on the happy path: bracket market triggers a
    ``session.execute`` whose scalar result is returned as a float."""
    session = AsyncMock()
    result = MagicMock()
    result.scalar = MagicMock(return_value=42.5)
    session.execute.return_value = result

    market = _market(operator="bracket")
    used = await cluster_stake_used(session, market)

    assert used == 42.5
    session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_null_sum_coalesces_to_zero():
    """No trades in the cluster → DB returns NULL → helper returns 0.0
    (the COALESCE in the SQL plus the explicit ``or 0.0`` guard cover
    both row-empty and value-null cases)."""
    session = AsyncMock()
    result = MagicMock()
    result.scalar = MagicMock(return_value=None)
    session.execute.return_value = result

    used = await cluster_stake_used(session, _market(operator="range"))

    assert used == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("operator", ["bracket", "range", "exactly"])
async def test_all_bracket_like_operators_query_database(operator):
    """Each operator the executor / probability path treats as
    bracket-like must trigger the DB query — none can be silently
    skipped."""
    session = AsyncMock()
    result = MagicMock()
    result.scalar = MagicMock(return_value=10.0)
    session.execute.return_value = result

    used = await cluster_stake_used(session, _market(operator=operator))

    assert used == 10.0
    session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_filters_on_location_date_and_pending_open():
    """The compiled SQL must restrict to:
    1. Same parsed_location
    2. Same end_date.date()
    3. Trade.status IN (PENDING, OPEN)
    4. exchange_status IS NULL OR != 'dry_run'

    All four filters protect cap math. Inspecting the rendered SQL is
    the most surgical way to lock this in without standing up a real DB."""
    captured: dict = {}

    async def fake_execute(stmt):
        captured["stmt"] = stmt
        result = MagicMock()
        result.scalar = MagicMock(return_value=0.0)
        return result

    session = AsyncMock()
    session.execute = fake_execute

    market = _market(
        operator="bracket",
        location="Austin",
        end_date=datetime(2026, 5, 17, 4, 0, tzinfo=timezone.utc),
    )
    await cluster_stake_used(session, market)

    # Compile the captured statement to SQL text + the bound parameter
    # set. Asserting on string fragments is brittle but the alternative
    # (introspecting the ClauseElement tree) is far worse — we want a
    # failing test to point clearly at "lost a filter".
    compiled = captured["stmt"].compile(
        compile_kwargs={"literal_binds": True}
    )
    sql = str(compiled).lower()
    assert "markets.parsed_location" in sql
    assert "austin" in sql
    assert "date(markets.end_date)" in sql
    assert "trades.status in" in sql
    assert "'pending'" in sql
    assert "'open'" in sql
    # Dry-run exclusion: either NULL or != 'dry_run'
    assert "dry_run" in sql
