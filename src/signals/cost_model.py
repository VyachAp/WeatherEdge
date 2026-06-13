"""First-class trade-cost model — the friction the old edge calc ignored.

The legacy decision path computed ``edge = our_prob - buy_price`` and stopped
there: no fee, no size-dependent slippage, no accounting for the fact that we
*cross* the spread on every fill (we buy the ask / ``1 - bid``, never the mid).
That made every trade look ~half-a-spread better than it actually fills, a
structural every-trade drag (see the decision-pipeline diagnosis).

This module makes friction explicit and inspectable. It is pure (no I/O, no DB)
and operator-agnostic — it knows nothing about weather, only about prices, book
depth, and order size. ``net_edge = prob - effective_cost`` is the single signed
number the new decision rule keys off.

Convention: the *buy price* passed in is the side's actual taker cost — ``ask``
for a YES buy, ``1 - bid`` for a NO buy — so the spread-crossing cost is already
embedded in it. ``effective_cost`` adds the frictions NOT in that price:

    effective_cost = buy_price + fee + expected_slippage

* ``fee`` — proportional taker fee (Polymarket is ~0 today, but model it so a
  fee regime change is a one-line config flip, not a silent loss).
* ``expected_slippage`` — the average extra price paid walking the book when the
  order is large relative to top-of-book depth. A simple, documented linear
  impact model; deliberately conservative (over-estimating cost only ever makes
  us trade less, never more).
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Defaults (overridable per-call; mirrored into Settings when wired) ------

# Polymarket taker fee as a fraction of notional. Effectively 0 on weather
# markets today; kept as a first-class term so a fee regime is a config flip.
DEFAULT_FEE_RATE: float = 0.0

# Linear price-impact coefficient: the fraction of the order that sits *beyond*
# top-of-book depth is charged this much extra average price per unit
# over-depth. 0.5 ≈ "walking a uniformly-deep book costs you half the over-depth
# ratio in price." Conservative; tune down once we have fill telemetry.
DEFAULT_SLIPPAGE_COEFF: float = 0.5

# Hard ceiling on modelled slippage (price units). A pathologically thin book
# shouldn't make effective_cost exceed 1.0 and produce nonsense edges.
MAX_SLIPPAGE: float = 0.10

_EPS = 1e-9


@dataclass(frozen=True)
class CostBreakdown:
    """Decomposed cost of taking one side, for logging + post-hoc analysis."""

    buy_price: float          # side taker cost (ask for YES, 1-bid for NO)
    half_spread: float        # (ask - bid) / 2 — informational; already in buy_price
    fee: float                # proportional taker fee, price units
    expected_slippage: float  # size-vs-depth book-walk, price units
    effective_cost: float     # buy_price + fee + expected_slippage, clipped ≤ 1.0


def expected_slippage(
    stake_usd: float,
    depth_usd: float | None,
    *,
    coeff: float = DEFAULT_SLIPPAGE_COEFF,
    max_slippage: float = MAX_SLIPPAGE,
) -> float:
    """Average extra price paid walking the book for an order of ``stake_usd``.

    Linear impact: the portion of the order that exceeds top-of-book depth is
    charged ``coeff`` per unit over-depth ratio. Returns 0 when the order fits
    inside visible depth. Unknown/zero depth → max slippage (assume the worst,
    so we never *under*-cost a trade into a book we can't see).
    """
    if stake_usd <= 0:
        return 0.0
    if depth_usd is None or depth_usd <= _EPS:
        return max_slippage
    over_depth_ratio = max(0.0, (stake_usd - depth_usd) / depth_usd)
    return min(max_slippage, coeff * over_depth_ratio)


def effective_cost(
    *,
    buy_price: float,
    bid: float | None = None,
    ask: float | None = None,
    depth_usd: float | None = None,
    stake_usd: float = 0.0,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_coeff: float = DEFAULT_SLIPPAGE_COEFF,
) -> CostBreakdown:
    """Full all-in cost of taking one side at ``stake_usd``.

    ``buy_price`` is the side's taker cost (already spread-crossed). ``bid`` /
    ``ask`` are the side's own quotes (used only to report the informational
    ``half_spread``). All frictions beyond the buy price are added on top.
    """
    half_spread = 0.0
    if bid is not None and ask is not None and ask >= bid:
        half_spread = (ask - bid) / 2.0
    fee = fee_rate * buy_price
    slip = expected_slippage(stake_usd, depth_usd, coeff=slippage_coeff)
    total = min(1.0, buy_price + fee + slip)
    return CostBreakdown(
        buy_price=buy_price,
        half_spread=half_spread,
        fee=fee,
        expected_slippage=slip,
        effective_cost=total,
    )


def net_edge(prob: float, cost: CostBreakdown) -> float:
    """Signed edge after all frictions: ``prob - effective_cost``.

    Positive means the side is +EV per $1 staked at the modelled fill cost.
    This is the single number the new bet-decision rule thresholds against —
    replacing ``our_prob - ask`` (which silently dropped fee + slippage).
    """
    return prob - cost.effective_cost
