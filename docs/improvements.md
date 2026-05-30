# Improvements log

A running list of hypotheses, experiments, and follow-ups for WeatherEdge.
Captures the *forward-looking* stuff: things considered but not shipped,
ideas to revisit when data accumulates, optimisations that didn't make
the cut today. Things that *have* shipped live in git history + CLAUDE.md
gotchas, not here.

**Status tags:**
- `[backlog]` — proposed, not started
- `[in-flight]` — actively testing or partially implemented
- `[done]` — kept here only when the outcome is worth remembering (e.g.
  "tried X, didn't work because Y"). Otherwise prune to git history.
- `[rejected]` — decided against; include the *why* so it isn't
  re-proposed.

**Entry template:**

```
## [status] Short action-oriented title
**Why:** what triggers this idea / what hypothesis we're testing
**Success criteria:** how we'll know it worked
**Effort:** rough size (hours / days / week)
**Leverage:** volume / signal-quality / risk / observability
**Files:** likely touchpoints (omit if speculative)
**Notes:** anything else (data, links, open questions)
```

---

## [in-flight] Monitor stuck-exposure fix + post-Phase-E observation window

**Why:** On 2026-05-17 the bot was discovered silenced for ~24h —
`decision_logs` (Phase C dividend) immediately revealed 100% of 419
decisions over 24h hit `stake_below_min` with `dd_multiplier=1.0`.
Root cause: 16 OPEN+delayed+null-fill trades from May 13–14 pinned
the exposure cap at $226/$110. Phase E shipped 5 changes:
`admin reconcile-stuck` (E1), `resolve_trades` catch-22 fallback (E2),
widened `job_reconcile_orders` lookback (E3), exposure-cap Telegram
warning (E4), `MAX_EXPOSURE_USD_FLOOR=$300` (E5). Need clean
attribution window before more changes.
**Success criteria:** by 2026-05-24, confirm: (a) operator ran
`admin reconcile-stuck` and exposure dropped from $226 → ~$0;
(b) `decision_logs.outcome` distribution shifts from 100%
`stake_below_min` to a healthy mix (`trade_filled`/`trade_pending`/
`dup_blocked_db` dominant); (c) the Telegram exposure-cap warning
hasn't fired (or has fired and was actionable); (d) `job_reconcile_orders`
catches any new delayed-no-fill trades within hours, not days.
**Effort:** ~5 min/day morning checks.
**Leverage:** measurement.
**Notes:** if exposure starts pinning again, query `decision_logs`
filtered to `outcome='stake_below_min'` and look at `dd_level`
(drawdown-induced) vs `requested_stake_usd > 0` (Kelly returned
non-zero but exposure cap ate it).

## [done] 2026-05-16 monitoring window — superseded

Earlier monitoring entry for the 2026-05-16 deploy (Phase A-D) is
folded into the entry above; same observation pattern, same queries.
Outcome: scheduler ran cleanly overnight, surfaced the exposure-cap
silencer in `decision_logs` in 30 seconds (vs the 4-hour debug that
caused Phase C to exist). Phase C's first dividend.

---

## [backlog] Investigate OPEN-promotion bug for delayed orders

**Why:** CLAUDE.md gotcha says `_update_fill_details` keeps
`exchange_status='delayed'` trades at `status=PENDING` when the order
isn't yet retrievable. But the 16 stuck May-13/14 trades had
`status=OPEN AND exchange_status='delayed' AND fill_price IS NULL` —
something promoted them despite no confirmed fill. Once E1-E5
shipped, the operational symptom (exposure pinning) is fixed, but
this is the deeper bug that allowed the bad state.
**Success criteria:** identify the code path that flips `status` to
OPEN before `_update_fill_details` has confirmed a fill. Likely
candidate: `src/execution/lock_rule_executor.py` line ~339 where
`actual_stake = trade.stake_usd or 0.0 > 0` is checked. If `stake_usd`
retains the requested value (instead of being zeroed on no-fill),
the OPEN branch fires. Add a `logger.warning` at the OPEN-promotion
site that triggers when `fill_price is None`. Watch one day of logs.
**Effort:** 2-3h (trace + targeted fix + test).
**Leverage:** prevention (eliminates one source of stuck-exposure
backlog so reconcile-stuck rarely needs to be run).
**Files:** `src/execution/lock_rule_executor.py`, `src/execution/polymarket_client.py::_update_fill_details`.

## [backlog] Per-station lock-rule auto-disable

**Why:** Calibration data showed bimodal lock-rule performance — RJTT /
WIII / NZWN / RKSI hit ~95% win rate, while RPLL / SAEZ / ZBAA sit
near 50-60%. Current global `LOCK_RULE_LOSS_DISABLE_COUNT=3` over 72h
is too crude: one bad streak on the weak stations would mute the path
entirely, including the high-confidence stations.
**Success criteria:** per-station loss tracker prunes the marginal
stations during a streak without halting the strong ones. Measure: 14d
lock-rule cohort win rate post-deploy ≥ pre-deploy.
**Effort:** 4h code + tests.
**Leverage:** volume + risk (preserves throughput on strong stations).
**Files:** `src/risk/circuit_breakers.py` (new `_lock_losses_by_icao`
dict + helper), `src/execution/lock_rule_executor.py::try_lock_rule_trade`
gate before sizing.

## [backlog] Re-enable BRACKET_MARKETS_ENABLED

**Why:** Disabled 2026-05-08 after live data showed −21pp
overconfidence on bracket markets (memory:
`project_bracket_disabled_2026-05-08`). Calibration is now on +
per-station σ floor lands variance correctly per Phase D. Worth
revisiting once we have 4 weeks of clean threshold-market Brier
scores.
**Success criteria:** 4 consecutive weeks of threshold-market Brier
< 0.10 → re-enable for a 2-week measurement window with
`CLUSTER_STAKE_CAP_USD=100` already capping cluster risk. If bracket
trades' Brier stays < 0.15 over those 2 weeks, keep enabled.
**Effort:** 5 min config + 4 weeks of patience.
**Leverage:** universe expansion (~30% of weather markets are brackets
per the 2026-05-16 scan: 297 bracket / 1617 total).
**Files:** `.env` (`BRACKET_MARKETS_ENABLED=true`).

## [backlog] Weekly station_calibration_report cadence

**Why:** Calibration drifts, station performance varies, bias-runaway
stations need pruning. Currently ad-hoc — should be a routine.
**Success criteria:** Friday afternoon habit + reading the markdown,
adjusting `_EXCLUDED_ICAOS` if any station hits |bias|>2.5°C or
proj_RMSE > 6°F.
**Effort:** 30 min/week of attention.
**Leverage:** signal-quality (catches stations decaying out of model
range before they bleed PnL).
**Files:** `scripts/station_calibration_report.py` (already exists).

## [done 2026-05-23] Aggregation CLI for decision_logs funnel

**Shipped:** `python -m src.cli decisions-report --days N [--signal-kind …]
[--operator threshold|bracket-like]` mirrors `evals_report`. Outputs:
outcome counts overall, outcomes split by operator class, and the
binding-constraint (`size_reason`) breakdown for `stake_below_min` /
`drawdown_paused` from `metadata_json`.
**First finding (7d prod):** `dup_blocked_inproc` 63%, **`stake_below_min`
27%** (dominant real throttle; `size_reason` "below $5 minimum" leads),
`trade_filled` only 2%. The $5 floor at small bankroll is eating most
passing decisions — see the threshold-floor entry below.
**Files:** `src/cli.py::decisions_report`.

## [backlog] Bound the keyword-scan fallback offset configurably

**Why:** `fetch_weather_markets` hard-codes `MAX_OFFSET=10000` for the
legacy keyword scan after the Phase A6 rewrite. Should be a setting
so future Gamma API changes can be tuned without code.
**Success criteria:** `POLYMARKET_LEGACY_SCAN_MAX_OFFSET` in
`src/config.py`; default 10000; can be set to 0 to skip the fallback
entirely (saves ~100 API calls per scan).
**Effort:** 15 min.
**Leverage:** maintenance.
**Files:** `src/ingestion/polymarket.py`, `src/config.py`.

## [backlog] Drop dead Signal columns

**Why:** `Signal.gfs_prob` / `ecmwf_prob` / `aviation_prob` / `wx_prob`
are always NULL per CLAUDE.md gotcha — no live readers. Schema bloat
hurts nothing today but Adds Confusion to anyone reading the model.
**Success criteria:** alembic migration drops the four columns; tests
still pass.
**Effort:** 30 min.
**Leverage:** code clarity (low priority).
**Files:** new alembic migration, `src/db/models.py::Signal`.

## [code shipped 2026-05-23, awaiting .env enable] Operator-aware MIN_PROBABILITY / MIN_EDGE (threshold-only)

**Why:** `MIN_PROBABILITY=0.85` / `MIN_EDGE=0.10` were tightened for the
2026-05-08 *bracket* crisis; threshold markets were never the source of
that bleed but inherit the strict global floors.
**Shipped:** `THRESHOLD_MIN_PROBABILITY` / `THRESHOLD_MIN_EDGE` Settings
(both `None` = no-op) + `_check_filters` `min_edge`/`min_probability`
overrides + `binary_market_edge` threading them for threshold ops only.
Bracket-like ops keep the strict globals + the three single-bucket NO
guards. Deploying is a no-op until set via `.env`.
**Data (7d prod, `evals-report --operator threshold` recoverable band,
ground truth = routine-METAR daily max):**
- baseline already-passing threshold: 93.6% won, +0.183 EV/$1.
- **prob≥0.78 (edge held 0.10): +122 evals, 91.0% won, +0.568 EV/$1** ← sweet spot.
- prob≥0.78, edge≥0.05: +144 evals, 92.4% won, +0.540 EV.
- prob≥0.75: ~76-79% won (degrading); prob≥0.70: 66% won (marginal).
**Recommendation:** set `THRESHOLD_MIN_PROBABILITY=0.78`; optionally
`THRESHOLD_MIN_EDGE=0.05`. Do NOT go below 0.78. Note: "newly-pass" counts
are eval ROWS (deduped per tick downstream), so distinct-market volume is
lower; win-rate/EV per-row are valid since a market's outcome is constant.
**Blocker on realized volume:** `decisions-report` shows `stake_below_min`
is the dominant downstream throttle ($5 floor at ~$487 bankroll). Loosening
filters recovers eval-volume but a chunk stalls before becoming a trade —
pair with a sizing review (KELLY_FRACTION / MIN_STAKE_USD / bankroll).
**Files:** `src/config.py`, `src/signals/edge_calculator.py`, `.env` (enable).

## [active 2026-05-23] Quantity throttle is the exposure cap saturated by LIVE bets (capital-bound, not a bug)

**Diagnosis (7d prod, `decisions-report` + ad-hoc):** 27% of decisions die at
`stake_below_min` — but `requested_stake_usd ≈ $0.00` for 98.6% of them (p90
= 0.00), not "just under $5". So **raising KELLY_FRACTION or lowering
MIN_STAKE_USD does nothing** — Kelly sizes to ~0 *upstream* of the floor.
Cause: the **exposure cap is fully deployed by legitimate in-flight bets**.
equity ≈ $414, `MAX_EXPOSURE_PCT × bankroll = $103 < $300 floor` so cap =
$300; 28 OPEN hold ~$275 → only **~$25-32 room** split across all per-tick
candidates → each sizes to ~0 → `stake_below_min`. **Correction to an earlier
read: these are NOT stuck.** Inspecting the 28: 23 are *today's* markets (end
12:00 UTC, awaiting end-of-local-day resolution), 5 are *tomorrow's*; only **1
is genuinely stale** (Wellington, target Apr 29, ~577h past end, $7 — clean
via `admin reconcile-stuck`). So this is the **risk cap working as designed**
at a small bankroll, not the stuck-OPEN silencer. Capital recycles daily as
positions resolve → headroom returns and the newly-unlocked threshold band
fills into it.
**Notable side-observations:**
- All 28 OPEN have `fill_price=NULL` (separate, minor reconcile gap — fill
  backfill not landing; doesn't affect exposure but blinds slippage analysis).
- ~22 of 28 OPEN are `exactly` (bracket-like) bets — bracket-like YES is
  eating most of the exposure budget, crowding out threshold trades. Possible
  capital-allocation lever (cap bracket-like exposure share?).
- 248 PENDING `exception:PolyApiException` over the window but **0 in last
  24h** — the May-2026 CLOB submission-failure class, self-resolved.
**Levers (genuine, since it's capital not a bug):** (a) grow bankroll;
(b) raise `MAX_EXPOSURE_USD_FLOOR` (trades drawdown safety for volume; risky
when floor approaches/exceeds equity); (c) faster capital recycling (resolve +
`bet redeem` cadence so freed capital re-enters sooner); (d) cap bracket-like
exposure share so threshold trades aren't crowded out. The threshold-filter
loosening is correct + EV but its realized volume is gated by available
headroom, which is intraday-saturated at this bankroll.
**Files:** `src/risk/kelly.py`, `src/config.py`, `src/resolution.py`.

## [backlog] Investigate Polymarket Gamma listing recovery

**Why:** Phase A6 rewrote `fetch_weather_markets` to slug-enumerate
because Gamma's `/markets?tag=weather` silently returned generic
markets starting 2026-05-15. If Polymarket fixes the listing
behaviour, the keyword + tag scans (kept as defensive fallbacks)
would be more complete than slug enumeration (which depends on
hardcoded city + day patterns and only catches `highest-temperature-in-*`
events).
**Success criteria:** weekly probe of
`/markets?active=true&closed=false&tag=weather&limit=100` — if it
starts returning weather markets again, demote slug enumeration to
fallback.
**Effort:** 5 min/week to probe; if needed, 30 min to flip the
strategy ordering.
**Leverage:** discovery breadth (catches `lowest-temperature-*` or
`avg-temperature-*` if Polymarket ever ships them).
**Files:** `src/ingestion/polymarket.py`.

## [backlog] Persist station_rmse for backtest reproducibility

**Why:** Phase D's `get_station_rmse` is an in-process TTL cache
computed on-the-fly. Replay-capable backtests in `src/risk/simulate.py`
won't see the same RMSE values the live bot used at that historical
tick. For now this doesn't matter (the global floor was the prior
behaviour) but if we want to A/B the floor strategies on replay, the
historical RMSE per (station, date) would need to be persisted.
**Success criteria:** new `StationRmseSnapshot` table written nightly
by `job_daily_settlement`; backtest reads from there instead of
recomputing.
**Effort:** 4-6h (migration + nightly write + backtest plumbing).
**Leverage:** measurement (only matters if we run rigorous A/B
backtests on the σ floor).
**Files:** `src/db/models.py`, new alembic migration,
`src/scheduler/__init__.py::job_daily_settlement`, `src/risk/simulate.py`.

## [backlog] POLYMARKET_USE_NEW_EXCHANGES verification

**Why:** Config notes that Polymarket migrated to new exchange
contracts (pUSD collateral). The flag monkey-patches py-clob-client
to sign for the new exchange; without it orders are rejected with
`order_version_mismatch`. Worth verifying the current value is
correct for the live wallet before any larger trading-volume push.
**Success criteria:** one successful FAK on the new exchange this
week confirms the flag is right.
**Effort:** 5 min.
**Leverage:** risk (avoid `order_version_mismatch` halting trades
after a Polymarket-side change).
**Files:** `.env` (`POLYMARKET_USE_NEW_EXCHANGES`).

## [backlog] °C resolver/observation divergence on `exactly` markets

**Why:** Root cause behind the `exactly`-market bleed fixed surgically
on 2026-05-22 (max-lead gate + `RANGE_OVERSHOOT_LOCK_ENABLED=False`).
The `range_overshoot` NO lock fires only when our observed daily max is
already ≥ window-high + 4°F — which *should* be near-certain — yet it
won only 56% (18 trades, -$57.61), **spread across many °C cities**
(Toronto, Buenos Aires, Shanghai, Guangzhou, Busan, Seoul, Beijing…).
That pattern means our `routine_history` daily max reads systematically
*hotter* than Polymarket's resolver. Suspects: SPECI observations
leaking into `routine_history` (resolution mirrors **routine** METARs
only), ICAO ≠ Polymarket resolver-station mismatch (cf. excluded
VHHH/LLBG), or °C rounding at the bucket boundary. Fixing this would
let us re-enable the overshoot lock AND remove the overconfidence the
probability path also suffers on these markets.
**Success criteria:** for a sample of overshoot-LOST `exactly` trades,
reconcile our recorded daily max against Wunderground's published max
for the resolver station; classify the gap (SPECI leakage / station
mismatch / rounding); quantify per-city.
**Effort:** 4-6h (audit query + per-trade Wunderground cross-check;
overlaps with the never-built `scripts/audit_lock_observed_max.py`).
**Leverage:** revenue (unlocks re-enabling overshoot + de-biases the
probability path) + correctness.
**Files:** `src/ingestion/aviation/` (routine vs SPECI filter),
`src/signals/mapper.py` (`CITY_ICAO` resolver-station mapping),
`src/signals/state_aggregator.py::_routine_daily_max`.

## [partly-done] Tune EXACTLY_MAX_LEAD_HOURS once live data accrues

**Why:** Shipped 2026-05-22 at 12.0h based on the historical PnL cliff
(0-12h lead +$57 / 12-24h -$126 on the probability path).
**Done 2026-05-23:** the gate now measures lead to the forecast **peak**
(`state.hours_until_peak`, falling back to time-to-close), fixing the
close-vs-peak mismatch that let pre-dawn Amsterdam bets through (close
12:00 UTC, peak 14:00 UTC). **Still open:** re-fit the 12h cutoff with ≥3
weeks of post-gate live `evaluation_logs` (10h vs 12h vs require-past-peak).
Now unblocked: `evaluation_logs` carries `forecast_peak_f` / `current_max_f`
/ `hours_until_peak` / `forecast_sigma_f` (migration `p6q7r8s9t0u1`,
2026-05-23), so the lead gate and landing-band margins can be recomputed and
re-fit directly from telemetry.
**Effort:** 1-2h analysis + `.env` change.
**Leverage:** revenue (marginal tuning).
**Files:** `src/config.py` (`EXACTLY_MAX_LEAD_HOURS`),
`src/signals/edge_calculator.py::binary_market_edge`.

## [done] 2026-05-23 Single-bucket NO guards (landing band + overconfidence cap)

**Why:** The `exactly` probability-path losses concentrate in the
`model_prob ≥ 0.999` band (-$163.84 / 179 trades): each single-°C bucket
is evaluated as an independent binary, so a tight Gaussian centered on an
over-forecast peak makes every neighbouring bucket look near-impossible →
NO passes on all of them (Amsterdam 2026-05-23 traced case: NO on
27/28/29°C while the blend was pinned at 30°C with σ collapsed to ~1.1°C).
**Shipped:** `binary_market_edge` now takes `forecast_peak_f` /
`current_max_f` / `hours_until_peak` and applies, NO-side only on
bracket-like ops: (1) **landing-band** guard — refuse NO on a window
overlapping `[current_max − margin, max(forecast_peak, current_max) + margin]`
(`SINGLE_BUCKET_NO_BAND_MARGIN_F=1.0`), collapsing to the observed max once
past peak; (2) **overconfidence cap** — floor `our_prob_yes` so NO ≤
`SINGLE_BUCKET_MAX_NO_PROB=0.92`; (3) peak-relative lead gate (above). All
three Amsterdam bets reject under their historical inputs (verified replay).
**Still to confirm:** that the ≥0.999 NO band disappears from live
`evaluation_logs` over the coming weeks.
**Files:** `src/signals/edge_calculator.py::binary_market_edge`,
`src/config.py`, `src/scheduler/__init__.py` (call site),
`tests/test_edge_calculator.py::TestSingleBucketNoGuards`.

## [backlog] Lead-time-aware σ floor (deferred from the 2026-05-23 NO-guards work)

**Why:** Root cause one layer below the single-bucket NO guards. In
`_compute_sigma`, tight inter-model ensemble agreement collapses σ to the
global `ENSEMBLE_MIN_SIGMA_F=2.0°F` (the hours-based schedule only applies
as a ×0.5 *soft* floor). On 2026-05-23 EHAM had σ≈1.1°C **11.5 h before
peak** because 4 models agreed — but inter-model agreement ≠ accuracy
(forecast oscillated 28.8–30.7°C tick-to-tick that morning). A σ floor
that scales with `hours_until_peak` would widen distributions far from
peak and de-bias overconfident NO/YES across **all** cities and market
types — including the profitable threshold path, which is why it was
deferred from the surgical NO-guard fix.
**Status note (2026-05-30):** while this entry is pending,
`BRACKET_LIKE_NO_DISABLED=True` is the operational stop-loss in the live
`.env`. Reactivating bracket-like NO is part of this entry's exit
criteria: after the σ-floor change ships, run `evals-report --operator
bracket-like` and flip the flag back to False only when the tuner's
baseline (all-passing) row shows +EV. Until then, every bracket-like-NO
fill the bot would have placed is reachable as a rejected row in
`evaluation_logs` with the master-switch reason — the learning signal
continues, only the cash bleed stops.
**Success criteria:** `_compute_sigma` floors σ to the full hours-based
schedule (not ×0.5) when far from peak; backtest-v2 calibration
(Brier / per-bucket) on threshold markets does not regress; `evals-report
--operator bracket-like` baseline row flips to +EV; `BRACKET_LIKE_NO_DISABLED`
flipped back to False.
**Effort:** 2-3h + backtest-v2 gate.
**Leverage:** correctness/risk (global calibration), but must not de-tune
the working threshold path.
**Files:** `src/signals/probability_engine.py::_compute_sigma`,
`src/config.py`.
---

## [backlog] `exactly` NO: close the °C resolver divergence (deeper than the knobs)

**Date:** 2026-05-26.

**Context.** After the 2026-05-26 data-day fix (negative-UTC cities no longer
bet the next day's market early) and the single-bucket NO knob retune
(`SINGLE_BUCKET_NO_BAND_MARGIN_F` 1.0→2.5, `SINGLE_BUCKET_MAX_NO_PROB`
0.92→0.85), the `exactly` NO class is the residual loss source.

**Key finding from `evals-report --operator bracket-like`.** Scored against
our **routine-METAR daily max on the corrected target day**, the currently-passing
bracket-like NO evals win ~81% (+0.072 EV/$1) — but the *real fills* over the
same window won only ~61%. That ~20pp gap is **resolver divergence**: our
routine-METAR daily max diverges from Polymarket's resolver (same root cause
that disabled `range_overshoot`). It means the telemetry validator's ground
truth is partly circular (the bot's NO conviction is built from the same data
source it's scored against), so the margin/cap grid can't cleanly separate
real winners from losers — the candidates mostly just trade volume for a tiny
EV delta. The 2.5/0.85 retune is therefore a *risk-reduction / volume cut*,
not a calibration fix.

**Two deeper levers (either could make `exactly` NO genuinely +EV):**
1. **Resolver-divergence correction.** Quantify, per station (and per °C vs °F),
   the signed gap between our routine-METAR daily max and the resolved outcome
   (from settled trades), then shift/penalise single-bucket NO conviction by it.
   Needs a reliable resolved-max source per market — the cleanest is back-solving
   from settled trade WON/LOST across all buckets of an event.
2. **Lead-time-aware σ floor** (already logged separately): a tight ensemble far
   from peak collapses σ and over-excludes neighbouring buckets. Raising the σ
   floor as a function of `hours_until_peak` widens single-bucket probs so fewer
   overconfident NO bets pass. `backtest-v2` *does* exercise `compute_distribution`,
   so this one is replay-validatable (unlike the `binary_market_edge` guards).

**Effort:** medium; both need a per-station resolved-max ground truth that isn't
the routine-METAR feed. Until then the knobs are the best available lever.
**Files:** `src/signals/edge_calculator.py::binary_market_edge`,
`src/signals/probability_engine.py::_compute_sigma`, `src/cli.py`
(`_single_bucket_no_band_section` — extend ground truth to settled-trade-derived
resolved max).
