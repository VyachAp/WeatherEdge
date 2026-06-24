"""Perf-review digest renderer + calibration-squash detector wrapper.

Pure (no DB / no click / no top-level I/O). ``render_perf_digest`` builds a
string; the squash detector is a thin lazy-import wrapper around the
scheduler's pure detector so the orchestrator can reuse it without importing
the scheduler at module load.
"""

from __future__ import annotations


def _detect_calibration_squash_local(eval_rows: list[dict], fill_count: int):
    """Thin re-export wrapper so the orchestrator reuses the scheduler's pure
    detector without importing it at module load (keeps `cli` light)."""
    from src.scheduler import _detect_calibration_squash
    return _detect_calibration_squash(eval_rows, fill_count)


def render_perf_digest(result: dict, *, markdown: bool = True) -> str:
    """Render a ``perf_review_result`` dict as a compact digest.

    ``markdown=True`` → Telegram MarkdownV2 (dynamic values escaped via
    ``_escape_md2``); ``markdown=False`` → plain text for the console. Leads
    with the PnL headline, the dominant throttle, regressions, flip-ready
    flags, then anomalies — the same shape we'd hand-write each review.
    """
    from src.execution.alerter import _escape_md2

    e = _escape_md2 if markdown else (lambda x: str(x))
    bold = (lambda t: f"*{t}*") if markdown else (lambda t: t)
    bullet = "• "
    L: list[str] = []

    pnl = result.get("pnl") or {}
    _since = result.get("since")
    _win = str(result.get("window_days")) + "d"
    if _since:
        _win += " since " + str(_since)[:10]
    L.append(f"\U0001f4ca {bold('Perf review')} — {e(_win)}")
    wp = f"{pnl['win_pct']*100:.0f}%" if pnl.get("win_pct") is not None else "n/a"
    ev = f"{pnl['ev_per_usd']:+.3f}" if pnl.get("ev_per_usd") is not None else "n/a"
    pnl_s = f"${pnl.get('pnl', 0.0):+,.2f}"
    L.append(
        f"\U0001f4b0 PnL {e(pnl_s)} over {e(pnl.get('n', 0))} resolved "
        f"({e(wp)} win, EV/$1 {e(ev)}); {e(pnl.get('n_open', 0))} open"
    )

    regs = [k for k, v in (result.get("regression") or {}).items() if v.get("regressed")]
    if regs:
        L.append(f"⚠️ Regressed vs prior {e(str(result.get('window_days'))+'d')}: {e(', '.join(regs))}")

    th = result.get("throttle") or {}
    if th.get("dominant_throttle"):
        fr = f"{th['dominant_throttle_frac']*100:.0f}%" if th.get("dominant_throttle_frac") is not None else "n/a"
        L.append(f"\U0001f6a6 Top throttle: {e(th['dominant_throttle'])} ({e(fr)} of decisions)")

    losers = [c for c in (result.get("loss_classes") or []) if c.get("pnl", 0) < 0][:2]
    for c in losers:
        cls = c.get("lock_branch") or c.get("signal_kind") or "?"
        cpnl_s = f"${c['pnl']:+,.2f}"
        cwin_s = f"{c['win_pct']*100:.0f}%"
        L.append(
            f"\U0001f4c9 Loss class {e(cls)}: {e(cpnl_s)} / {e(c['n'])} "
            f"({e(cwin_s)} win)"
        )

    gates = result.get("flip_gates") or []
    ready = [g for g in gates if g.get("verdict") == "ready"]
    if ready:
        L.append(bold("Flip-ready:"))
        for g in ready:
            meas = g.get("measured")
            meas_s = f" (measured {meas})" if meas is not None else ""
            L.append(f"{bullet}{e(g['flag'])} → {e(str(g['proposed_value']))}{e(meas_s)}")
    else:
        insf = sum(1 for g in gates if g.get("verdict") == "insufficient-data")
        L.append(f"No flags flip-ready ({e(insf)} awaiting data).")

    for a in (result.get("anomalies") or []):
        L.append(f"\U0001f6a8 {e(a.get('kind'))}: {e(a.get('detail'))}")

    sep = "\n" if markdown else "\n"
    return sep.join(L)
