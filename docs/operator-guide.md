# WeatherEdge operator guide

What to expect when running the bot day-to-day.

## 1. Before you start

| Check | Where | Notes |
|---|---|---|
| `.env` | repo root | `POLYMARKET_PRIVATE_KEY` empty ⇒ dry-run. `AUTO_EXECUTE=false` (default) ⇒ signals log + alert, no orders. `DAILY_SPEND_CAP_USD` — start at $50. |
| DB up | `src/db/engine.py` | Schema auto-creates on first run. |
| Telegram bot | `.env` | `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` needed or alerts are silently dropped. |

Start: `python -m src.cli run`

## 2. What happens on boot

`job_startup` fires once:
- Telegram alerter task begins — queued messages flush at ~1/sec.
- Drawdown state reloads from `BankrollLog`.
- One `job_scan_markets` call — ingests Polymarket weather markets into `Market` + `MarketSnapshot`.

You'll see `[BOOT]`-style log lines and a "bot started" Telegram ping if wired.

## 3. Recurring cadence

| Job | Every | Does |
|---|---|---|
| `job_scan_markets` | 15 min | Refresh market list + prices via Gamma API |
| `job_unified_pipeline` | 5 min (`UNIFIED_PIPELINE_INTERVAL_MINUTES`) | Main loop — aggregate state, run lock rule, score markets, maybe trade |
| `job_fast_lock_poll` | 30 s (`FAST_LOCK_POLL_INTERVAL_SECONDS`) | Bulk METAR refresh; fire EASY-direction lock trades on new routine METARs AND re-run forecast-exceedance projection from cached forecast/bias (no extra HTTP) |
| `job_resolve_trades` | 5 min (30 s offset) | Settle expired markets (chain-gated on-chain `payoutNumerators`), push each new resolution |
| `job_daily_settlement` | 22:00 UTC | Bankroll/drawdown bookkeeping, station-bias recording, weekly calibration, daily summary, redeem nudge, calibration-squash diagnostic, counterfactual knowledge snapshot. **Does not** resolve trades or clear caches |

## 4. Per-pipeline tick (every 5 min)

Logs will look like this per station:

```
[EDDM] state: max=54°F, trend=+2.4°F/hr, forecast_peak=57°F in 2.5h, solar_declining=False, cloud_rising=False, routine_count=22
[EDDM] exceedance row: obs=52.0°F @ ... vs forecast@11Z=48.3°F (same_hour_delta=+3.7°F) | max=54.0°F, forecast_peak=57.0°F, projected=57.2°F, trend=+4.3°F/hr, slope=+0.45°F/hr n=4, peak_passed=False, alerted=True
```

`projected` is the daily-max projection (halflife-decayed residual + damped trend carry — `_project_daily_max`). `slope=…` is the residual-slope fit over the last 6h (diagnostic only — it no longer feeds the projection; the slope-based "v2" projector was removed 2026-06-24, see `docs/graveyard.md`); `n=…` is the routine count used. `slope` may be `n/a` when unavailable.

For each market you'll either see a skip line (price-edge-of-book, bias runaway, circuit breaker) or a scoring line with `edge=…`, `our_prob=…`, and either a trade or a "filter X failed" reason.

A second log shape comes from the fast-poll loop:

```
[fast-poll EDDM] new routine METAR obs=2026-04-25T15:20:00+00:00 temp=58.5°F max=58.5°F | 2 market(s)
[EDDM] LOCK YES <market_id>: margin=2.5°F, price=0.91, stake=$15.00 (raw=$15.00, dd_mult=1.00) | market-day max 58.5°F >= threshold 56°F + 2.0°F margin (above); routine_count=12 (min 3)
[EDDM] exceedance row: obs=58.5°F @ ... projected=60.2°F, ..., alerted=True
```

The `exceedance row` line firing from a fast-poll tick (within the same second as the METAR landing) is the cadence-gap fix — a fresh routine triggers projection within 30s instead of waiting up to 5 min for the next unified pipeline tick. Reuses the cached forecast/bias from the last unified tick (no extra HTTP). After a process restart, expect the first 5 minutes of fast-poll alerts to be silent until the next unified tick warms the cache — by design.

The `(min 3)` annotation on a LOCK line shows the routine-count gate that was applied. **Super-margin EASY** locks (overshoot ≥ 2× `LOCK_MARGIN_F` = 4°F by default) fire at routine_count=2 instead of 3 — daily max is monotonic, two confirming hot obs are enough.

If `LOCK_RULE_ENABLED=false` or `FAST_LOCK_POLL_ENABLED=false`, the LOCK + fast-poll lines disappear and both the lock path AND the fast-poll projection check stay inert (the projection check is gated on the same fast-poll job).

## 5. Telegram alerts you should expect

### 🔒 LOCK trade (when the lock-rule path fires)

Fired by `_try_lock_rule_trade` when the daily max already mathematically locks YES (above-threshold) or NO (below-threshold). One Telegram per fill: side, threshold, observed margin, stake, price, top reason. No probability — it's deterministic.

### 🌡 "Daily max set to beat forecast" (informational)

With the current tuning:

- **At most once per station per 30 min** (`ALERT_COOLDOWN`).
- Fires when the projected daily max beats the forecast peak by >1°F (~0.56°C).
- Projection (`_project_daily_max`): `α·current_residual + (1-α)·forecast_peak`, `α=exp(-h/2)`, plus a damped carry of the obs-vs-forecast trend residual; falls back to a pre-residual linear blend when there's no current forecast. (The slope-based "v2" projector was removed 2026-06-24 — it overshot the warming concavity; see `docs/graveyard.md`.)
- Capped at `forecast_peak + 5°F`; observed max is always a floor.
- Routine count gate: `>=3` normally, but **`>=2` when same-hour residual >1°F** (the strong-residual fast path — saves 30-60 min on hot mornings when only 2 routines exist but obs is already obviously hot).
- Won't fire after peak (short-trend-based `_peak_passed` heuristic).
- Fast-poll runs the projection check within 30s of a fresh routine — the alert no longer waits up to 5 min for the next unified tick.

Typical daily volume (10-city config): **5–15 pushes/day**.

### 💰 Trade alerts (when `AUTO_EXECUTE=true`)

One per filled order: direction, price, size, edge, market link.

### ⚠️ Circuit breakers

- Daily loss > $200 → "halted for today" message.
- 3 consecutive losses → "paused 2h" message.

### 📊 Daily summary (22:00 UTC)

Trades resolved, P&L, running bankroll, which stations updated bias, weekly calibration on Sundays.

## 6. Where state lives

| Thing | Table |
|---|---|
| Market list + live snapshots | `markets`, `market_snapshots` |
| Each decision | `signals` (even if no trade placed) |
| Every order | `trades` — status `PENDING`/`OPEN`/`WON`/`LOST` |
| Declined-but-won learnings | `bot_state` `knowledge.counterfactual.latest` + `runtime/knowledge_counterfactual.json` (see `counterfactual-report`) |
| Exceedance history | `forecast_exceedance_alerts` — every near-miss, with `alerted` flag showing which went to Telegram |
| Per-station bias | `station_biases` |
| Bankroll curve | `bankroll_logs` |

For backtesting/analysis, `forecast_exceedance_alerts` contains both the alerted and cooldown-suppressed rows — bucket by `alerted` to see the true positive rate over time. The `projected_max_f` column is the single `_project_daily_max` (halflife) value.

## 7. Failure modes to recognize

| Symptom | Likely cause | Action |
|---|---|---|
| `Could not fetch orderbook for token X after 3 attempts` (×2) | Resolved market with dead token | Benign — the pipeline skips it next |
| `METAR history fetch failed for ICAO` | All 6 providers rate-limited | Usually transient; station degrades to `current_max_f = forecast_peak_f` for that tick |
| `bias runaway for ICAO (\|bias\|>3°C)` | Forecast has drifted vs. obs for 30+ days | Station skipped that tick; investigate the station/forecast provider |
| `circuit breaker: daily loss stop` | Today's P&L < -$200 | Bot resumes tomorrow automatically |
| `[fast-poll <ICAO>]` lines firing repeatedly without trades | Lock decided but filter rejected (depth, close buffer); dedup prevents retry | Expected — `_locked_markets_fired_today` resets at 22:00 UTC |
| No `[fast-poll <ICAO>] exceedance row:` lines for the first ~5 min after restart | `_state_cache` empty until first unified tick warms it for that station | Expected — wait for the next unified pipeline tick |
| Three lock losses in 72 h → lock path inert | Auto-disable via `LOCK_RULE_LOSS_DISABLE_COUNT` | Investigate the resolved trades (`scripts/inspect_loss.py`); manually re-enable in `.env` |
| No Telegram traffic for >30 min during a weather event | Alerter queue may be stuck | Restart; queue is in-process only |

To kill the lock path in an emergency: set `LOCK_RULE_ENABLED=false` in `.env` and restart. The probability path is unaffected. To kill *just* the fast-poll loop (both 30-s lock + projection) while keeping main-pipeline locks: `FAST_LOCK_POLL_ENABLED=false`.

## 8. First-day checklist

1. Run dry-run (`AUTO_EXECUTE=false`) for **one full day** — watch the Telegram volume and log lines.
2. Sanity-check: pick one alert, open the Wunderground history page for that ICAO, confirm the obs/trend in the alert match. The DB row (`forecast_exceedance_alerts`) has everything you need to audit.
3. Skim `[fast-poll …]` lines for unexpected lock activity — confirm any 🔒 LOCK alerts correspond to a real new routine METAR that crosses threshold + margin.
4. After 22:00 UTC, verify the daily summary arrived and `bankroll_logs` got a new row.
5. Only then flip `AUTO_EXECUTE=true` with a low `DAILY_SPEND_CAP_USD`.

## 9. Knobs you'll likely touch

All in `.env` (see `src/config.py`):

- `MIN_EDGE`, `MIN_PROBABILITY` — directional trade filters
- `KELLY_FRACTION`, `MAX_POSITION_PCT`, `MAX_POSITION_USD` — directional sizing
- `LOCK_RULE_ENABLED`, `LOCK_MARGIN_F`, `LOCK_POSITION_PCT`, `LOCK_RULE_MIN_PRICE`, `LOCK_RULE_MAX_PRICE` — lock-path knobs (super-margin EASY at routine_count=2 triggers when overshoot ≥ 2× `LOCK_MARGIN_F`)
- `FAST_LOCK_POLL_ENABLED`, `FAST_LOCK_POLL_INTERVAL_SECONDS` — fast-poll cadence (gates BOTH lock + projection paths)
- `ENSEMBLE_MODELS`, `ENSEMBLE_SPREAD_MULTIPLIER`, `ENSEMBLE_MIN_MODELS` — probability-engine σ
- `DAILY_SPEND_CAP_USD` — hard 24h limit
- `UNIFIED_PIPELINE_INTERVAL_MINUTES` — directional-path cadence

Alert-specific tunables are `Settings` fields bound at import in `src/signals/forecast_exceedance.py` (`EXCEEDANCE_MAX_OVERSHOOT_F`, `EXCEEDANCE_ALERT_COOLDOWN_MINUTES`, `EXCEEDANCE_DELTA_THRESHOLD_F`, `EXCEEDANCE_MIN_ROUTINE_COUNT`, `EXCEEDANCE_STRONG_RESIDUAL_*`, `EXCEEDANCE_RESIDUAL_DECAY_HALFLIFE_H`, `EXCEEDANCE_RESIDUAL_TREND_CARRY_K`). Set via `.env` + restart.
