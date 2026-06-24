"""Forecast-error aggregator (Phase 3, 2026-06-24).

Pure number-crunching for ``forecast-error-report``. Aggregates
``forecast_error_daily`` rows into the per-lead-time error/RMSE curve — the
empirical input the deferred ``SIGMA_LEAD_TIME_SLOPE_F_PER_HR`` should be fit to:
RMSE should rise with ``lead_bucket_h`` (forecasts are noisier further out), and
the bias (mean signed error) flags any systematic forecast warm/cool offset.
"""

from __future__ import annotations

from collections import defaultdict


def _rmse(vals: list[float]) -> float:
    return (sum(v * v for v in vals) / len(vals)) ** 0.5 if vals else 0.0


def _summarize_forecast_error(
    rows: list[dict], *, prefer_resolved: bool = True
) -> list[dict]:
    """Per-lead-bucket forecast-error stats from ``forecast_error_daily`` rows.

    Scores ``error_vs_resolved_f`` when present (the truer target — the Phase-1
    resolved point), falling back to ``error_vs_metar_f`` per row when
    ``prefer_resolved`` and no resolved error exists. Returns a list of
    ``{lead_bucket_h, n, mean_error, rmse, bias_sign, mean_sigma}`` sorted by
    ``lead_bucket_h`` asc — so the RMSE-rises-with-lead curve reads top to bottom.
    """
    groups: dict[int, list[tuple[float, float | None]]] = defaultdict(list)
    for r in rows:
        lead = r.get("lead_bucket_h")
        if lead is None:
            continue
        err = None
        if prefer_resolved and r.get("error_vs_resolved_f") is not None:
            err = r["error_vs_resolved_f"]
        elif r.get("error_vs_metar_f") is not None:
            err = r["error_vs_metar_f"]
        if err is None:
            continue
        groups[int(lead)].append((float(err), r.get("forecast_sigma_f")))

    out: list[dict] = []
    for lead in sorted(groups):
        pairs = groups[lead]
        errs = [e for e, _s in pairs]
        sigmas = [float(s) for _e, s in pairs if s is not None]
        n = len(errs)
        mean_error = sum(errs) / n
        out.append({
            "lead_bucket_h": lead,
            "n": n,
            "mean_error": mean_error,
            "rmse": _rmse(errs),
            "bias_sign": "warm" if mean_error > 0 else ("cool" if mean_error < 0 else "flat"),
            "mean_sigma": (sum(sigmas) / len(sigmas)) if sigmas else None,
        })
    return out
