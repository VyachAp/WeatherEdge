"""Replay a price-band edge policy against the realised book.

Motivation (2026-06-06 perf audit): per-trade EV is U-shaped in the *effective
price of the side bought*. Over the last 60d the surviving (non-killed) book is
+EV at the extremes and -EV in the middle:

    .40-.60 (deep value)  +0.086 EV/$1
    .60-.85 (the valley)  -EV    <-- model-overconfidence band, where the bot
    .85-1.0 (near-lock)   +0.098 EV/$1     currently fishes (avg NO price 0.75)

This script replays candidate band policies over already-fired, settled trades
and reports the counterfactual volume / win% / PnL / EV shift.

IMPORTANT — what this can and cannot measure:
  * It can only evaluate *tightening* policies (removing valley trades or raising
    the edge bar there), because it scores trades that actually fired and settled
    against their REAL outcomes. It cannot tell us about trades we'd newly add.
    That is fine: the proposed policy is a tightening.
  * `edge` is read from the (deduped, last-refresh) signals row, so it is an
    approximation of the edge at fire time. The block-valley policy (P1) does not
    depend on edge and is the robust headline; the edge-threshold variants are
    illustrative and test whether "more edge in the valley" is real or just
    inflated overconfidence.

Run:  python -m scripts.replay_price_band_policy [--days 60]
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from src.db.engine import async_session

# --- The policy (pure; promote this into binary_market_edge / _check_filters) ---

VALLEY_LOW = 0.60
VALLEY_HIGH = 0.85


def in_valley(effective_price: float) -> bool:
    """The -EV overconfidence band on the effective price of the side bought."""
    return VALLEY_LOW <= effective_price < VALLEY_HIGH


def band_min_edge(effective_price: float, *, base_min_edge: float, valley_min_edge: float) -> float:
    """Required edge for a trade at this price band.

    Outside the valley: the normal global floor. Inside: a raised bar
    (use float('inf') to block the valley entirely).
    """
    return valley_min_edge if in_valley(effective_price) else base_min_edge


def policy_keeps(effective_price: float, edge: float, *, base_min_edge: float, valley_min_edge: float) -> bool:
    return edge >= band_min_edge(
        effective_price, base_min_edge=base_min_edge, valley_min_edge=valley_min_edge
    )


# --- Replay harness ---

QUERY = text(
    """
    SELECT t.entry_price, t.pnl, t.stake_usd, t.status::text AS status,
           s.edge, coalesce(s.lock_branch,'') AS lock_branch,
           m.parsed_operator AS op, t.direction::text AS dir
    FROM trades t
    JOIN signals s ON s.id = t.signal_id
    JOIN markets m ON m.id = t.market_id
    WHERE t.fill_price IS NOT NULL
      AND t.status IN ('WON','LOST')
      AND t.entry_price IS NOT NULL
      AND t.opened_at > now() - make_interval(days => :days)
    """
)


def is_go_forward(row) -> bool:
    """Classes that are still live today (exclude already-killed exactly/range NO)."""
    killed_no = row["op"] in ("exactly", "range", "bracket") and row["dir"] == "BUY_NO"
    killed_lock = row["lock_branch"] in ("range_undershoot", "range_overshoot")
    return not killed_no and not killed_lock


def summarize(rows, keep_fn) -> dict:
    kept = [r for r in rows if keep_fn(r)]
    n = len(kept)
    won = sum(1 for r in kept if r["status"] == "WON")
    pnl = sum(r["pnl"] or 0.0 for r in kept)
    stake = sum(r["stake_usd"] or 0.0 for r in kept)
    return {
        "n": n,
        "won": won,
        "win_pct": (100.0 * won / n) if n else 0.0,
        "pnl": pnl,
        "stake": stake,
        "ev": (pnl / stake) if stake else 0.0,
    }


def fmt_table(title, rows, base_min_edge):
    policies = [
        ("baseline (as-fired)", lambda r: True),
        ("P1 block valley", lambda r: not in_valley(r["entry_price"])),
        ("P2 valley edge>=0.15", lambda r: policy_keeps(r["entry_price"], r["edge"], base_min_edge=base_min_edge, valley_min_edge=0.15)),
        ("P3 valley edge>=0.20", lambda r: policy_keeps(r["entry_price"], r["edge"], base_min_edge=base_min_edge, valley_min_edge=0.20)),
        ("P4 valley edge>=0.25", lambda r: policy_keeps(r["entry_price"], r["edge"], base_min_edge=base_min_edge, valley_min_edge=0.25)),
    ]
    base = summarize(rows, policies[0][1])
    print(f"\n### {title}  (n={base['n']} resolved/filled)\n")
    print("| policy | n | Δn | win% | PnL | ΔPnL | EV/$1 | stake |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|")
    for name, fn in policies:
        s = summarize(rows, fn)
        dn = s["n"] - base["n"]
        dpnl = s["pnl"] - base["pnl"]
        print(
            f"| {name} | {s['n']} | {dn:+d} | {s['win_pct']:.0f}% | "
            f"${s['pnl']:+.2f} | ${dpnl:+.2f} | {s['ev']:+.3f} | ${s['stake']:.0f} |"
        )


async def main(days: int):
    async with async_session() as session:
        result = await session.execute(QUERY, {"days": days})
        rows = [dict(m) for m in result.mappings().all()]

    base_min_edge = 0.05  # current settings.MIN_EDGE floor used at fire time
    print(f"# Price-band edge policy replay — last {days}d")
    print(f"\nValley band = effective price [{VALLEY_LOW}, {VALLEY_HIGH}).")

    fmt_table("FULL universe (incl. already-killed classes)", rows, base_min_edge)
    fmt_table("GO-FORWARD universe (what's live today)", [r for r in rows if is_go_forward(r)], base_min_edge)

    # Band x EV anchor on go-forward, so the policy's target is visible
    gf = [r for r in rows if is_go_forward(r)]
    print("\n### Go-forward EV by band (anchor)\n")
    print("| band | n | win% | PnL | EV/$1 |")
    print("|---|--:|--:|--:|--:|")
    bands = [("value [.40-.60)", lambda r: r["entry_price"] < 0.60),
             ("valley [.60-.85)", lambda r: in_valley(r["entry_price"])),
             ("lock [.85-1.0]", lambda r: r["entry_price"] >= 0.85)]
    for name, fn in bands:
        s = summarize(gf, fn)
        print(f"| {name} | {s['n']} | {s['win_pct']:.0f}% | ${s['pnl']:+.2f} | {s['ev']:+.3f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    args = ap.parse_args()
    asyncio.run(main(args.days))
