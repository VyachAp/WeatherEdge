"""One-off diagnostic: is the shadow model overconfident-but-directional, or broken?

Reuses run_shadow_backtest's returned rows (no second slow query) and dissects
HOW the model is wrong vs the market — confidence distribution, directional
agreement, and the worst-loss exemplars.
"""

import asyncio
import math
from collections import Counter

from src.db.engine import async_session
from src.risk.shadow_backtest import run_shadow_backtest


def _ll(p, y):
    p = min(1 - 1e-6, max(1e-6, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


async def main():
    async with async_session() as s:
        out = await run_shadow_backtest(s, days=60)
    rows = out["rows"]
    n = len(rows)
    print(f"rows={n}")

    # 1. Confidence distribution of the model vs market.
    def band(p):
        if p < 0.05: return "<.05"
        if p < 0.20: return ".05-.20"
        if p < 0.50: return ".20-.50"
        if p < 0.80: return ".50-.80"
        if p < 0.95: return ".80-.95"
        return ">.95"
    mc = Counter(band(r["updated_yes"]) for r in rows)
    kc = Counter(band(r["market_mid"]) for r in rows)
    print("\nconfidence bands  model | market")
    for b in ["<.05", ".05-.20", ".20-.50", ".50-.80", ".80-.95", ">.95"]:
        print(f"  {b:>8}  {mc.get(b,0):5} | {kc.get(b,0):5}")

    # 2. Directional agreement: when model says >0.5, does outcome agree more
    #    often than not? And does model move the SAME direction as the market?
    base_rate = sum(r["yes_won"] for r in rows) / n
    model_acc = sum((r["updated_yes"] > 0.5) == bool(r["yes_won"]) for r in rows) / n
    mkt_acc = sum((r["market_mid"] > 0.5) == bool(r["yes_won"]) for r in rows) / n
    same_dir = sum((r["updated_yes"] > 0.5) == (r["market_mid"] > 0.5) for r in rows) / n
    print(f"\nbase rate P(yes_won)={base_rate:.3f}")
    print(f"model sign-accuracy={model_acc:.3f}  market sign-accuracy={mkt_acc:.3f}  model/market same-side={same_dir:.3f}")

    # 3. Mean prob by actual outcome — a calibrated model should separate them.
    yes = [r for r in rows if r["yes_won"] == 1]
    no = [r for r in rows if r["yes_won"] == 0]
    def mean(xs, k): return sum(x[k] for x in xs) / len(xs) if xs else float("nan")
    print(f"\nwhen yes_won=1 (n={len(yes)}): model mean updated_yes={mean(yes,'updated_yes'):.3f}  market mean={mean(yes,'market_mid'):.3f}")
    print(f"when yes_won=0 (n={len(no)}):  model mean updated_yes={mean(no,'updated_yes'):.3f}  market mean={mean(no,'market_mid'):.3f}")

    # 4. Worst-loss exemplars.
    worst = sorted(rows, key=lambda r: _ll(r["updated_yes"], r["yes_won"]), reverse=True)[:12]
    print("\nworst-loss rows (model confidently wrong):")
    print("  updated_yes  prior_yes  market  yes_won  obs_frac  class")
    for r in worst:
        print(f"  {r['updated_yes']:.3f}        {r['prior_yes']:.3f}      {r['market_mid']:.3f}   {r['yes_won']}        {r['obs_fraction']:.2f}     {r['operator_class']}")


asyncio.run(main())
