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

## [in-flight] Automated perf-review loop (Layer 1 deterministic + Layer 2 analyst)

**Why:** For ~2 months the "evaluate DB/results → propose flag/code change"
loop has been run by hand each session. The data backbone already exists
(M1–M4 telemetry + the pure report aggregators in `cli.py` + codified
flip-gate criteria), so the loop is automatable. Shipped 2026-06-08,
**propose-only** (human approves every flip — the project's measure-before-flip
discipline is non-negotiable on a live trading bot).
**What shipped:**
- *Layer 1 (deterministic, in-bot):* `cli.perf_review_result` + the pure fns
  `_realized_pnl` (phantom-LOST-safe) / `_throttle_breakdown` / `_loss_classes`
  / `_window_regression` / `_flip_gate_verdicts`, the `perf-review` command
  (`--days/--json/--push/--write-artifact`), `render_perf_digest`,
  `_persist_perf_artifact` (bot_state row + gitignored `runtime/*.json`), and
  `perf-propose-push`. Scheduler `job_perf_review(days)` registered for
  daily/3d/7d (gated by `PERF_REVIEW_ENABLED`, default False), throttled per
  cadence via a `bot_state` cooldown, staggered after the 22:00 settlement.
- *Layer 2 (LLM analyst, out-of-bot):* `docs/agents/perf_analyst_prompt.md` —
  reads the Layer-1 artifact + this file + memory, evaluates each flip-gate,
  drafts the exact `.env`/code change with evidence, and writes ONLY to this
  file + `~/.claude` memory + a Telegram push. Never edits `.env`/`src`.
**Success criteria:** (a) `PERF_REVIEW_ENABLED=true` in `.env` + restart;
(b) the three cadences push a digest on schedule without spamming;
(c) the Layer-2 analyst (scheduled via the `schedule`/routines skill or system
cron) produces a flip-proposal Telegram + an improvements.md/memory entry each
cadence, with a clean `git diff` of `.env`/`src`;
(d) at least one real flip decision (e.g. a `VALLEY_MIN_EDGE` set or a
`stake_below_min`-blocked threshold loosening) is surfaced from the digest
rather than a manual query.
**Effort:** shipped; remaining = enable in `.env` + stand up the Layer-2
schedule (mechanism TBD: routines vs host cron — needs prod-DB reachability).
**Leverage:** observability + iteration-velocity.
**Files:** `src/cli.py`, `src/scheduler/__init__.py`, `src/config.py`,
`docs/agents/perf_analyst_prompt.md`, `tests/test_perf_review.py`.
**Notes:** Layer-1 flip-gate inputs are best-effort cheap proxies; the
threshold recoverable-band number is left to the analyst (it runs
`evals-report --operator threshold`). The gate verdicts are advisory —
`insufficient-data` below the codified n (30/30/50/8/20).

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

## [backlog] PolyApiException "not enough balance / allowance" — wallet pre-flight check

**Date:** 2026-05-30.

**Why:** 140 `exchange_status LIKE 'exception:%'` rows in the last 14 days
(7-25/day, peaked 35 on 2026-05-28). All 96 with captured `exchange_error`
text are the same class — `not enough balance / allowance: balance: X,
order amount: Y`, where balance ∈ {2.5, 0.6, 0.7, 8.2, 9.3} pUSD and order
amount ∈ {$5, $9, $15, $17, $20}.

**Root cause: transient wallet depletion during burst placement, not
allowance.** Confirmed by three signals: (a) the CLOB-reported balance is
*constant* across each burst (e.g., May 28 19:26-19:28 had 13 consecutive
failures, all reporting `balance: 2502932`) — a static allowance cap
wouldn't vary day-to-day, and a depleting wallet would show monotonic
decrement. Both rule out allowance and rule in "the wallet has $X free
when the burst begins, the burst tries to spend more than $X, the CLOB
rejects the orders past the limit"; (b) at the May 28 19:26 burst start
the bot's DB view had 568 OPEN/PENDING positions totalling $6,655 stake
on a $300 bankroll — the real subset of those reserving wallet pUSD
explains the gap; (c) `get_wallet_balance()` is defined in
`polymarket_client.py:196` but has **zero callers** — the bot never
pre-checks wallet spendable before placing an order.

**Fix path:**
1. **Pre-flight check in `place_order`**: call `get_wallet_balance()`
   before submitting; if `spendable < stake_usd`, skip with new
   `decision_log` outcome `OUTCOME_INSUFFICIENT_BALANCE`. The orders that
   would have failed now don't fire and the +EV signal moves to a
   smaller-stake fallback or the next tick.
2. **Drop `_WALLET_BALANCE_TTL_SEC`** from 300s → ~30s (or invalidate
   the cache after every successful placement so the next placement sees
   the post-reservation reality).
3. **Optional next iteration**: in-process `_reserved_balance` counter
   that decrements at submit, increments on fill/cancel — eliminates the
   TTL race entirely.

**Success criteria:** `exception:PolyApiException` count drops to ≈0 in
the 24h after deploy. The `decision_logs.outcome` distribution gains
some `insufficient_balance` rows (expected and informative — replaces
silent failures with visible skips). Submission-failure circuit breaker
(`SUBMIT_FAIL_PAUSE_COUNT=5/10min`) stops firing.

**Effort:** 2-4h including tests.

**Leverage:** ~$100/14d of foregone profit recovered indirectly (the
losing orders weren't truly "wasted" — they were burst-overcommit; the
real win is stopping the silent silencer pattern and freeing
investigation cycles).

**Files:** `src/execution/polymarket_client.py::place_order`,
`src/execution/polymarket_client.py::get_wallet_balance` (add caller,
shorten TTL), `src/signals/decision_log.py` (new outcome const),
`tests/test_polymarket_client.py` (new test class).

**Companion bug worth a separate entry:** the 568 OPEN/PENDING tracked
positions at burst time vs $300 bankroll suggests phantom-OPEN trades
(local DB out of sync with on-chain). Same class as
[[project-phantom-losses-2026-05-20]] but for the OPEN status. The
balance-pre-flight fix sidesteps the symptom; the phantom-OPEN cleanup
is a separate accounting hygiene task.

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

## [analysis backlog] Wire `forecast_archive` corpus into backtests + discovery

**Why:** every successful `aggregate_state` tick writes one `ForecastArchive`
row (~17k/day at 60 stations) capturing the blended Open-Meteo forecast,
intentionally accumulated as a research substrate. **No live reader yet** —
`simulate_distribution_pipeline` still uses the `forecast_peak_f = mid_max + 2.0`
placeholder (`src/risk/simulate.py:251`) instead of replaying real forecasts.

The corpus is the asset: every passing tick captures (a) what the blended
forecast looked like at that time-to-peak, (b) how it evolved across the
heating cycle, (c) per-station inter-model spread (`peak_temp_std_c`,
`hourly_temps_std_c`). Joined to `MetarObservation` actual daily maxes and
to `Signal`/`Trade` outcomes, this is enough surface for several distinct
investigations:

1. **Replay-capable backtest** — feed real archived forecasts into
   `compute_distribution`, score the resulting bucket distribution against
   the realised METAR daily max. Replaces the synthetic placeholder in
   `simulate.py`. Validates probability-engine changes against ground truth
   without rerunning live ticks.
2. **Forecast-evolution post-mortems** — when a market mis-resolves (lock
   fires NO and the day later overshoots, or a high-confidence YES loses),
   replay the forecast trajectory through the day to see which signal
   collapsed. Currently the only available trace is the per-tick log file.
3. **Calibration of `_compute_sigma`** — compare archived `peak_temp_std_c`
   to realised forecast error per station and lead-time bucket. Directly
   feeds the deferred lead-time-aware σ-floor work (separate backlog entry).
4. **Per-station model-source attribution** — `peak_temp_std_c` +
   `model_count` together let us estimate which station/lead-time bins
   benefit from ensemble vs deterministic-only blends. Could drive a
   per-station model-weight policy.
5. **Same-day update edge** — within a day, forecast updates often
   precede METAR moves by 30–60 min. The corpus lets us measure that
   lead consistently per station and decide whether to treat forecast
   deltas as a tradable signal in their own right.

**Success criteria (any subset):** at least one of the above produces a
report or a code path that meaningfully reads from `ForecastArchive`,
justifying the per-tick write cost from results rather than from intent.

**Effort:** medium-to-large. Each investigation is its own ~1-day exercise;
the shared infra (a JSONB → `OpenMeteoForecast` adapter + a query helper
that returns the latest-as-of-T archived snapshot per (station, target-day))
is a few hours and unlocks all five.

**Leverage:** signal-quality (replay validation), observability (post-mortems),
research (calibration / model-source / forecast-delta).

**Files:** `src/risk/simulate.py::simulate_distribution_pipeline`,
new `src/signals/forecast_archive_replay.py` (suggested home for the
adapter + query helpers), `src/cli.py` (new `backtest-replay` or extended
`backtest-v2`), `src/ingestion/forecast_archive.py` (docstring update if
the replay path materialises).

**Notes (audit pointer):** flagged 2026-05-30 during the per-module audit
pass (Module 4). The audit initially treated this as cruft because all
three of (CLAUDE.md, the model docstring, the module docstring) claimed
the data was "consumed by `simulate_distribution_pipeline`" — which was
not true. Documentation has been corrected to frame the writer as
forward-looking telemetry; this entry exists so a future audit pass
does not re-flag it. Drop the corpus only if a future analysis pass
concludes the dimensions captured are insufficient to deliver any of
the above.

## [climate-prior backlog] Write `scripts/backfill_station_normals.py` to bootstrap the Bayesian prior

**Why:** the climate-prior path through `probability_engine._apply_climate_prior`
is fully wired end-to-end (model, persistence, in-process cache, leap-day DOY
mapping, state-aggregator gate, full Bayesian posterior math with σ-floor
protection, 6+ tests in `test_probability_engine.py` and 7 in
`test_station_normals.py`). The ONE missing piece is the backfill loader that
populates `station_normals` — without it, `get_normal()` always returns `None`,
`state.climate_prior_mean_f/_std_f` stay `None`, and `_apply_climate_prior`
short-circuits on the `prior_mean is None` guard. `CLIMATE_PRIOR_ENABLED=False`
as a result.

**What it would do:** a one-shot script that, for each ICAO in `CITY_ICAO`
(plus optional --stations override), pulls multi-decade daily-max temperature
from an archival reanalysis source (Open-Meteo Archive `era5_seamless` is the
intended default per the model docstring), aggregates by DOY (leap-day mapping
is already handled by `_doy_for_lookup`), and writes one row per (icao, doy)
via `upsert_normal()`. ~50-100 LOC.

**Sketch of the call:**
```
GET https://archive-api.open-meteo.com/v1/archive
    ?latitude=<lat>&longitude=<lon>
    &start_date=1995-01-01&end_date=<today minus 1y>
    &daily=temperature_2m_max&timezone=auto
```
Then group by DOY → `mean()`, `std()`, `count` → upsert. For ~60 stations and
30 years of data that is ~60 API calls and ~22k rows total. Throttle to stay
inside Open-Meteo Archive rate limits (probably one call per station, all years
in one window).

**Sanity checks (run inline before upsert):** mean within `[-40, 50]°C`; std
within `[0.5, 15]°C`; sample_years >= 20 per DOY. Reject rows that fail; log
the ICAO so it can be excluded from the prior path.

**Success criteria:**
- `station_normals` has ≥ `0.95 * 60 * 365` rows after the script runs once.
- A spot-check: `python -m src.cli ...` (or a one-off REPL) calls
  `get_normal(session, "KAUS", date.today())` and returns a non-None value
  with `sample_years ≥ 25`, `mean_max_c` within ±5°C of the climatology
  documented externally (e.g. NOAA station normals page).
- After flipping `CLIMATE_PRIOR_ENABLED=true` in `.env` and restarting, the
  probability-engine reasoning trail (visible via `python debug_pipeline.py`)
  contains the line emitted by `_apply_climate_prior` showing prior_mean,
  prior_std, and posterior_mean/posterior_std.
- One week of live data: `evals-report --operator threshold` shows the
  recoverable-band `prob∈[0.78,0.85)` win rate has not degraded (the prior
  should tighten or shift conviction, not blow it up).

**Effort:** half a day for the script + a few hours of monitoring after the
first live week.

**Leverage:** signal-quality. Direct fix for the `exactly` NO σ-collapse
failure class — a tight ensemble far from peak should be anchored by the
climate prior, which is exactly what this blend does. Pair with the
`[backlog] Lead-time-aware σ floor` work for a more complete fix.

**Files:** new `scripts/backfill_station_normals.py`,
`src/ingestion/station_normals.py` (no changes — the upsert helper is
ready), `.env` (flip the flag after first successful run).

**Notes:** flagged 2026-05-30 during the per-module audit pass (Module 5).
The audit initially treated the unpopulated table as cruft because three
places (`station_normals.py` docstring, `StationNormal` model docstring,
`CLIMATE_PRIOR_ENABLED` Settings comment) all carried a stub TODO
referencing the missing script. Docs have been corrected to frame this
as a planned bootstrap step pointing here; this entry exists so a future
audit pass does not re-flag it. The wiring is the asset; only the data is
missing.

---

## [backlog] Segmented / isotonic calibration for the near-lock band

**Why:** The 2026-06-08 no-trade investigation found the single linear pooled
calibration (n=594, slope 0.71) over-penalizes the high-confidence near-lock
band: it maps raw 1.0 → ~0.78, so a threshold trade needs raw prob ≥ ~0.955 to
clear the `THRESHOLD_MIN_PROBABILITY=0.75` floor after calibration. The pooled
fit is dragged down by the mid-confidence overconfident losers (84% of the
training data is the now-disabled bracket-like-NO class), so the genuinely
+EV near-lock trades (~95% realized win) are suppressed along with the bad
mid-confidence ones. A single global slope can't express "near-locks are
well-calibrated but mid-confidence is overconfident."

**Success criteria:** a calibrator (piecewise/isotonic, or a per-confidence-band
fit) that leaves the near-lock band (raw ≥ ~0.95) ≈ identity while still
discounting the mid band — validated via shadow telemetry (`shadow_json`) before
flipping, and `decisions-report` showing near-lock fills resume without the
mid-band -EV bets returning.

**Effort:** ~1-2 days; gated on more clean post-σ-fix data (threshold class is
n=92 and degenerate today — see `[backlog] Lead-time-aware σ floor`).

**Leverage:** signal-quality + volume (recovers the +EV near-lock trades the
blunt linear fit currently squashes).

**Files:** `src/signals/calibration.py` (fit + apply), shadow wiring in
`src/scheduler/__init__.py`, `tests/test_calibration.py`.

**Notes:** Do NOT pursue by forcing volume (lower prob floor / disable
calibration) — the 2026-06-08 analysis confirmed those bets are ~break-even/-EV
(realized NO win ~70% at ~0.75 prices). The honest fix order is: (1) σ-floor to
make raw probs honest, (2) re-fit calibration on the cleaner data, (3) only then
consider segmentation if the linear fit still mis-serves the tails. The
`_detect_calibration_squash` daily diagnostic (job_daily_settlement section 7,
shipped 2026-06-08) surfaces when this squash is actively gating trades.

---

## [done 2026-06-18] Profitability reframing + measurement foundation (Levers 1 & 2 shipped)

**Why:** A 31-agent read-only diagnostic + independent spot-checks reframed the
"bot is losing -$101/30d" alarm: ~93-98% of that headline is a **settling legacy
tail** (`exactly`-NO opened pre-05-30 + `range_undershoot` pre-06-02, both already
gated). The truly-**live** book (opened ≥06-01, phantom-safe) is **-$21.42/44,
statistically ~zero**. The kill switches worked. The real blocker on *further*
profitability is **broken measurement**: (a) reporting steered on a legacy-polluted
headline; (b) `calibration-report` labels were starved to **8 distinct markets of
978 evaluated (0.8%)** because `record_market_resolution` only fired for bot-traded
markets.

**Shipped (measure-before-flip foundation, zero capital risk):**
- **Lever 1** — `perf-review --since YYYY-MM-DD` (absolute opened-window start).
  `--since 2026-05-30` reads the live book (-$18/72) vs the rolling -$101/30d.
  `perf_review_result(since=)`, digest header label, skips artifact write.
- **Lever 2 (keystone)** — `resolution.label_resolved_markets`, called from
  `job_daily_settlement` before the straggler sweep: labels every expired,
  on-chain-resolvable market the shadow flow scored (in `shadow_ledger`, not yet in
  `market_resolutions`) via on-chain `payoutNumerators`; UMA-unreported → skipped.
  Scoped to shadow-evaluated (~864 backlog), capped 600 chain calls/run. Validated
  by a rolled-back live trial (14 labeled / 40 chain calls).

**Phase-2 "closest gated live lever" verdict (read-only, 2026-06-18) — NONE is
ready to flip; all three are blocked:**
- **L4 lead-time σ-floor** (`SIGMA_FLOOR_LEAD_TIME_ENABLED`): the σ arm bites only
  far from peak (`shadow_json.sigma.delta` p50=0, p75=+0.55°F @ lead p75 9.9h), but
  `evals-report --operator bracket-like` baseline is still **-0.085 EV/$1** (survivors
  win 61.8%, below break-even). Re-enabling bracket-like-NO would still be -EV, and
  the live loss tail is at/past-peak threshold-NO where all σ-floor arms are no-ops.
  Closest in instrumentation, but **not over the line**; stays ship-dark/validate.
- **L5 conviction-lock** (`LOCK_CONVICTION_SIZING_ENABLED`): EASY-YES threshold locks
  don't fire — but the cause is **economic, not a bug**: post-fix lock-YES evals reject
  overwhelmingly on **`price 1.00 outside [0.05,0.95]`** (381 of the rejects). By the
  time the bot's margin-confirmed EASY-YES lock triggers (`current_max ≥ threshold + 2°F`,
  ≥3 routines), Polymarket has already priced YES at 1.00 — **no latency edge left**.
  Conviction-sizing targets an opportunity set that doesn't exist at tradeable prices;
  flipping it fires ~0×. (`above` markets are also vanishingly rare in the universe: 42
  vs 24,690 `exactly`.)
- **L6 depth-cap**: throttled volume is majority **YES** post-fix (56 YES / 29 NO in
  `stake_below_min`), not 73% NO (that figure was a pre-fix-window artifact). Binding
  size_reason is "below $5 minimum" (72/85) on a small/exposure-saturated bankroll. A
  targeted relaxation is *less* dangerous than the synthesis thought (recoverable side
  is no longer the -EV NO book) — but still gated on proving the post-fix YES EV first
  (n=4 fills today is far too thin).

**CORRECTION to the earlier "all-NO book is the deeper root" framing — it was a stale
90d-window artifact.** The all-NO DIRECTION skew was the 06-14 depth-probed-at-mid bug
([[project_yes_depth_mid_bug_2026-06-14]]), and it is **already fixed**: pre-fix, BUY_YES
passed **0 of 362,906** probability evals (100% vetoed); post-fix BUY_YES passes *more
than* BUY_NO (prob 94 vs 34; lock 18 vs 4) and is the **majority of fills** (4 YES / 2 NO).
The 90d lock counts (0 EASY-YES) were dominated by pre-fix history. So direction is solved.

**Actual current state + real next investigation:** the book is now (a) **direction-fixed**,
(b) **capital-throttled** (`stake_below_min` "below $5 minimum" on a $300-456 exposure-
saturated bankroll), (c) **EV-unproven post-fix** (only 6 fills in 4d, net -$16.64, 3W/3L).
The binding question is no longer "why all NO" but **"do we have any edge on the markets we
CAN trade (price ∈ [0.05,0.95])?"** — the easy near-lock edge is priced away (point 5 above),
so the bot's remaining edge lives in the mid-confidence band where it has historically been
overconfident. The L2 label decoupling (8 → ~100+/day) is exactly what's needed to answer
this from `calibration-report`/`resolution-report` once a settlement cycle runs. **Do not**
flip L4/L5/L6 or force volume until the post-fix EV is proven on real labels.
