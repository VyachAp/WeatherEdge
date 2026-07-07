# Mastering Playbook — WeatherEdge self-improvement loop

> Created 2026-06-24, right after the data-enrichment phase (Phase 1
> `station_day_resolutions`, Phase 2 `metar_reprice_snapshots`, Phase 3
> `forecast_error_daily`). This is the durable reference for the *mastering*
> phase: turning that data into a small number of **data-proven** config moves,
> consistently, gym-style — steady gains, never a profit switch.

---

## 0. What "mastering" means here

Enrichment gave us instruments; mastering is using them to walk a specific
backlog of **dark config flags** (all default-off, see §3) from *unproven* →
*data-proven flip OR explicitly rejected*, one validated step per cycle, never
regressing. The loop is **propose-only**: it produces evidence + a recommendation;
a human flips the flag. Success is consistency, not a jackpot.

**The one question the whole arc must settle:** is our edge the *model*
(information latency) or the *price band* (near-lock prices) — and does it exist
on the markets we can actually trade (effective price ∈ [0.05, 0.95])? The thesis
says latency; the 2026-06-18/19 findings say the latency is small/priced-away and
the realized edge is the near-lock price band, not the model. Iteration 3 settles
this fork; Iterations 1–2 build the ground truth needed to trust that verdict.

---

## 1. Standing principles (violating these has burned us before)

1. **Measure-before-flip.** Every dark flag has a shadow-telemetry gate AND a
   backtest/OOS gate (§3). Never flip live without the gate turning green. The
   2026-05-31 per-operator-calibration shadow check *prevented a live mistake*
   (the n=55 class fit was degenerate) — that's the bar.
2. **Phantom-safe PnL.** `LOST + fill_price IS NULL` is accounting cleanup
   (pnl=0), not a realized loss. Always filter `fill_price NOT NULL` and use the
   phantom-safe aggregators (`perf-review`, `_realized_pnl`). Running
   `admin reconcile-stuck --include-lost` once swung 30d realized by ~$500.
3. **Read the clean book, not the rolling window.** A rolling `--days` window
   straddles settled legacy losers from killed classes. Use
   `perf-review --since <post-fix date>` (absolute start). The depth-bug fix was
   2026-06-14; the kill-switches landed 2026-05-30 — anchor accordingly.
4. **De-circularize truth.** `evals-report` scores vs our *own* routine-METAR max
   (circular for °C cities). Prefer Phase-1 `resolved_max_point_f` /
   `market_resolutions.yes_won` as ground truth wherever the join exists.
5. **Always split** by operator class (`threshold` vs `bracket-like`), by
   **effective price band**, and by **`obs_fraction`**. The aggregate hides the
   signal — every real finding so far lived in a sub-cohort.
6. **Per-station heterogeneity is where the edge lives.** RCTP reads cool, KJFK
   is noisy, most stations sit at the ±0.3°F °C-rounding floor. Station-level
   trust drives conviction sizing and exclusions.
7. **Small-n discipline.** calibration-report buckets were once n≈8 → untestable.
   Gate every cohort at a min-n and *print the effective n*. Don't act on n<10
   stations or n<30 price/obs buckets.
8. **One epoch per flip.** Before flipping anything, confirm a `ConfigEpoch` is
   recorded (it is, at startup when tracked flags change) and scope before/after
   reads with `--since-epoch` so the comparison window is clean.

---

## 2. The mastering dashboard (the standing ritual — run each cycle)

| Report (command) | Answers | Data ready? | Feeds decision |
|---|---|---|---|
| `resolver-truth-report --days 120` | Per-station signed divergence (ours − resolved point); trust map | **Now** (731 station-days backfilled) | `LOCK_BIG_SIZE_STATIONS` whitelist, `_EXCLUDED_ICAOS` |
| `calibration-report --days 60 [--json]` | Does the shadow model beat the market baseline (Brier/log-loss) per `obs_fraction` bucket? | **Now** (labels richer post-backfill) | Shadow-model promotion (the thesis test) |
| `perf-review --since <date> --json` | Phantom-safe realized PnL, throttle funnel, loss classes, regression | **Now** | Binding-throttle + which classes bleed |
| `counterfactual-report --days 60 --min-resolved 30` | Declined cohorts that were +EV in hindsight (reject-reason/price/station) | **Now** | Which filter is over-tight |
| `decisions-report --days 30` | Post-filter funnel; which throttle binds (`stake_below_min`, `cap_exceeded`…) | **Now** | Sizing/capital levers |
| `exposure-report --days 30` | Per-tick headroom; is the cap saturating? | **Now** | `MAX_EXPOSURE_USD_FLOOR`, capital add |
| `forecast-error-report --days 30 [--lead H]` | RMSE-by-lead vs claimed σ; over/under-confidence | **After ≥3 settlements** | `SIGMA_LEAD_TIME_SLOPE_F_PER_HR` |
| `shadow-report --key sigma` | σ-floor counterfactual delta far-from-peak | **Now** (telemetry on) | `SIGMA_FLOOR_LEAD_TIME_ENABLED` |
| `latency-report --days 14` | Does the market move toward our new METAR, how fast, by `obs_fraction`? | **After ~1–2 weeks** (flag just enabled) | The model-vs-price-band fork |
| `evals-report --operator bracket-like` | Does the bracket-like-NO baseline turn +EV under a change? | **Now** | `BRACKET_LIKE_NO_DISABLED` re-enable |
| `valley-report --days 30` | P1/P2 price-band policy OOS EV | **Now** | `VALLEY_BLOCK_ENABLED` / `VALLEY_MIN_EDGE` |
| `lock-conviction-report --days 14` | Conviction EASY-YES fire funnel + settlement | **Now** (empty until flag on) | `LOCK_CONVICTION_SIZING_ENABLED` health |

---

## 3. Candidate-flag map (the backlog the loop walks toward)

Every row is default-off/dark. The **gate** is what must turn green before
proposing the flip. Order roughly follows the iterations.

| Flag (current) | Unlocks | Gate (must all hold) |
|---|---|---|
| `LOCK_CONVICTION_SIZING_ENABLED` + `LOCK_BIG_SIZE_STATIONS` | Real size on the bot's one near-certain edge | `resolver-truth-report`: whitelisted stations show \|mean\|≲0.3°F, std≲1.0, n≥10, no outlier |
| `SIGMA_FLOOR_LEAD_TIME_ENABLED` (+ fitted `SIGMA_LEAD_TIME_SLOPE_F_PER_HR`) | Honest σ far from peak; root-cause fix for σ-collapse | `forecast-error-report` RMSE>>σ at high lead; `shadow-report --key sigma` positive far-from-peak; `backtest-v2` no Brier regression |
| `BRACKET_LIKE_NO_DISABLED` → False | Re-open bracket-like NO volume | σ-floor live AND `evals-report --operator bracket-like` baseline (all-passing) row +EV |
| `RANGE_UNDERSHOOT_LOCK_ENABLED` → True | Re-open range-undershoot lock | Same as above (it's the same σ-too-tight overconfidence class) |
| Shadow-model promotion (per `obs_fraction` bucket) | Route live capital to `shadow_decision` | `calibration-report` `model_beats_market` True in high-obs buckets, stays False far from peak; `latency-report` shows a tradeable window |
| `PER_OPERATOR_CALIBRATION_ENABLED` | Un-squash the 0.78–0.85 threshold band | `shadow-report` shows class fit un-squashes the band non-degenerate (guardrail already blocks runaway fits) |
| `VALLEY_BLOCK_ENABLED` / `VALLEY_MIN_EDGE` | Cut the mid-price overconfidence valley | `valley-report` P1/P2 cohort +EV OOS |
| `CLIMATE_PRIOR_ENABLED` | Bayesian climate prior (σ tightening) | Prereq: `scripts/backfill_station_normals.py` populates `station_normals` (separate task) |

---

## 4. The first three iterations

Each iteration = **one question → exercises (exact commands) → what to look for
→ decision criteria (the gate) → output artifact → pitfalls.** Append the result
to §5 so the next iteration relies on it.

### Iteration 1 — Edge Map (ready NOW)

**Question:** Where do we actually have +EV on the *tradeable* book (effective
price ∈ [0.05, 0.95]), and which stations are trustworthy enough to size up on?

**Exercises**
1. `python -m src.cli resolver-truth-report --days 120` → build the **station
   trust map**. Buckets: *trustworthy* (\|mean\|≤0.3°F, std≤1.0, n≥10),
   *biased* (\|mean\|≥1.0 — candidates to exclude/down-weight),
   *noisy* (std≥2.0). Seed data already in §5.
2. `python -m src.cli calibration-report --days 60 --json` → does the shadow
   model beat the market baseline in **any** `obs_fraction` bucket? Record
   per-bucket n; ignore buckets with n<30. This is the foundational thesis read
   on the now-richer labels (06-18 label-decouple + this backfill).
3. `python -m src.cli perf-review --since 2026-06-14 --json` → phantom-safe
   realized PnL on the post-depth-fix book + the binding throttle (expect
   `stake_below_min` / Kelly<$5, not filters). Confirm "live book ≈ break-even,
   not bleeding."
4. `python -m src.cli counterfactual-report --days 60 --min-resolved 30` →
   which declined cohorts were +EV in hindsight (by reject-reason / price-band /
   station). The 30d run previously flagged the `edge` floor as #1 missed-winner.
5. Realized-PnL-by-price-band (the 2026-06-19 split): confirm <0.80 still loses
   vs 0.80–1.00 wins, now with more data → is `PROBABILITY_MIN_ENTRY_PRICE=0.80`
   still the right cut, or has the break moved?

**Decision criteria (propose; human flips)**
- Curate `LOCK_BIG_SIZE_STATIONS` from the trust map → if ≥2–3 stations pass the
  gate, propose `LOCK_CONVICTION_SIZING_ENABLED=true` with that whitelist.
- Propose `_EXCLUDED_ICAOS` additions for any *biased* station beyond RCTP.
- Propose a `PROBABILITY_MIN_ENTRY_PRICE` adjustment only if the band break moved.

**Output artifact:** "Edge Map v1" — station trust ranking + a
(price-band × obs_fraction × operator) realized-EV table + the binding-throttle
diagnosis. Write to §5.

**Pitfalls:** ±0.3°F is the °C-rounding floor (noise, not signal); small-n
stations (KJFK n=3) are not actionable; calibration buckets can still be thin —
print n.

### Iteration 2 — σ honesty → unlock killed classes (after ≥3–7 nightly settlements)

**Question:** Is our forecast uncertainty honest (claimed σ vs realized RMSE) by
lead time, and does fitting the lead-time σ-floor turn the killed bracket-like-NO
/ range-undershoot classes +EV?

**Readiness check:** `forecast-error-report --days 30` returns a non-trivial
table (target ≥150 station-day-lead rows; first rows appear at the first 22:00
UTC settlement after 2026-06-24).

**Exercises**
1. `forecast-error-report --days 30` overall + per high-volume station → the
   **RMSE-by-lead curve**. Compare `rmse` to `mean_sigma`: if RMSE >> mean_σ at
   high lead, the ensemble is overconfident far from peak (σ-collapse confirmed
   from data, not theory).
2. **Fit** `SIGMA_LEAD_TIME_SLOPE_F_PER_HR` ≈ slope of RMSE vs lead_bucket_h
   across the curve (use `error_vs_resolved_f` where present, else
   `error_vs_metar_f`).
3. `shadow-report --key sigma` (telemetry already on) → positive `sigma.delta`
   concentrated on far-from-peak evals, joined to bracket-like-NO outcomes.
4. `backtest-v2 --days 30` A/B with the proposed slope (replay now exercises the
   ensemble branch via `forecast_archive_replay`) → no Brier/cal regression.
5. `evals-report --operator bracket-like` → does the baseline (all-passing) row
   turn **+EV** under the σ-floor? This is the explicit re-enable gate.

**Decision criteria (gated, ordered)**
- Propose `SIGMA_LEAD_TIME_SLOPE_F_PER_HR=<fit>` + `SIGMA_FLOOR_LEAD_TIME_ENABLED=true`
  if (1)+(3)+(4) hold.
- THEN, only if (5) shows +EV: propose `BRACKET_LIKE_NO_DISABLED=False` and
  `RANGE_UNDERSHOOT_LOCK_ENABLED=True`.

**Output artifact:** fitted σ-floor value + go/no-go on the two killed classes,
with the before/after EV numbers. Write to §5.

**Pitfalls:** don't fit on <1 week of forecast-error data; `error_vs_resolved_f`
is the truer target but NULL on un-laddered station-days (fall back to METAR);
backtest-v2 uses reconstructed obs (decide the flip from shadow+evals, not
backtest alone).

### Iteration 3 — Latency thesis verdict & model-promotion fork (after ~2 weeks)

**Question:** Is the information-latency edge *real and tradeable* (does the
market move toward our new METAR, fast or slow, and bigger at high `obs_fraction`)
— and therefore, do we promote the shadow model per bucket, or conclude the edge
is just the price band?

**Readiness check:** `latency-report --days 14` shows `n_measurable` ≥ ~100
paired groups (needs `REPRICE_SNAPSHOT_ENABLED=true`, on since 2026-06-24).

**Exercises**
1. `latency-report --days 14` → mean `yes_mid` move by `obs_fraction`. **Thesis
   signature:** positive move that *rises* with `obs_fraction`. Read `mean_span`:
   if the move completes in <30s the edge is priced away before we can act
   (confirms the 2026-06-18 "latency small" finding); if it takes minutes there's
   a tradeable window.
2. Join reprice moves to Phase-1 resolved truth (which side was actually right) →
   when we held info advantage, did the market move *our* way (realized, not just
   the model's belief)?
3. `calibration-report --days 30` re-run on the now-much-larger labeled
   `shadow_ledger` → does `model_beats_market` turn True in high-`obs_fraction`
   buckets AND stay False far from peak (the exact thesis fingerprint)?
4. Cross-check against the price-band edge: is the latency edge *separate from* or
   *the same as* the near-lock-price edge? (06-18/19 suspect they're the same
   thing — the near-lock price IS the priced-in latency.)

**Decision criteria (the fork)**
- **If** (3) green in high-obs buckets AND (1)/(2) show a tradeable window →
  propose routing live capital to `shadow_decision` **per high-obs_fraction
  bucket only** (graduated promotion).
- **Else** → conclude the edge is the price band + station trust, not the model.
  Stop investing in the latency model; double down on price-band concentration
  (Iter 1 levers) + conviction sizing on trustworthy stations (Iter 1/2).

**Output artifact:** latency-thesis verdict + the capital-routing decision (the
biggest fork in the project). Write to §5.

**Pitfalls:** thin early data — don't conclude below n_measurable≈100; the move
may be noise; we're not the only METAR reader, so a market move on the same
public obs is *confounded* with our latency edge — that's exactly why (3)'s
obs_fraction-bucketed calibration (not raw moves) is the decisive test.

---

## 5. Findings log (append-only — this is what future cycles rely on)

### 2026-07-07 — Iteration 3 close: the fork is decided — edge is the PRICE BAND, not the model

**Iteration 3 fork VERDICT — thesis REFUTED, decisively.** The information-latency
MODEL is not an edge source. shadow_ledger⋈market_resolutions (30d, 15,200 deduped
rows): model(updated_yes) beats market(market_mid) in 0/5 obs_fraction buckets; overall
Brier model 0.1523 vs market 0.0969; gap WIDEST near/at peak (obs_fraction==1.0: model
0.1462 vs market 0.0641 = market 2.3x better with the day fully observed — the exact
inverse of a latency fingerprint). Model under-predicts YES (mean P 0.067–0.104 vs base
0.173–0.220) → the all-NO book is base-rate riding, not mispricing. Independently,
metar_reprice_snapshots (21d, 85.5k groups): market drift toward truth ~1–2c / 55–59%
directional ≤ the ~1–1.8c spread; T0 hook fires ~11min late; no tradeable standalone
latency window. DECISION: stop investing in the model; do NOT route live capital to
shadow_decision in any bucket. Realized edge = near-lock price band (0.80+); keep steering
via PROBABILITY_MIN_ENTRY_PRICE=0.80 + lock discipline. Revival gate (unlikely): isotonic
recal + refinement > market in a high-obs bucket at n≥500.

**Iteration 2 — σ-floor STAYS ON; killed classes STAY KILLED.** forecast_error_daily (30d,
n=1988): realized RMSE 1.8–2.4× ensemble σ at every lead (0h 3.24 vs 1.84; 12h 3.79 vs
1.69), forecast cool by −0.85°F. Overconfidence is real → KEEP SIGMA_FLOOR_LEAD_TIME_ENABLED.
Do NOT lower SIGMA_LEAD_TIME_SLOPE toward the fitted 0.046 °F/hr — the miss is LEVEL not
slope; tightening σ resurrects the NO bleed (the empirical lever, if pursued, is a higher
base σ floor + a −0.85°F cool-bias correction, evaluated separately). Re-enable gate FAILS:
bracket-like-NO baseline (real shadow quotes, n=44) +0.081 EV/$1 at NO winrate 84.1% ≈ 83.7%
base rate = no edge. HOLD BRACKET_LIKE_NO_DISABLED=False and RANGE_UNDERSHOOT_LOCK_ENABLED=True.
(Initial +3.32 EV/$1 was an artifact of 1−market_prob mispricing the NO leg.) Unblock: +EV
with winrate clearly > 83.7% base rate on n≥100 real-quote settled markets.

**Post-06-30-flip verdict — HARD-lock kill CONFIRMED (0 fills since); drag migrated to the
probability path.** Post-flip book −$24.33/3 fills = the entire loss is one Shanghai
at_least BUY_NO @0.85 $27.52 LOST — the SAME forecast-cool NO as the killed HARD-lock, now
firing in the probability path (which lacked the has_forecast guard). prob-path above/at_least
BUY_NO 60d: n=59, −$13.12, EV/$ −0.021; σ-NULL/forecast-failed subset n=51 −$14.61 (holds
3 of 4 big losses, all fpk==cmax); live-forecast subset ~flat +$1.48/n8. past-peak(htp=0)
is the LOSING subset (EV −0.120) — "require past-peak" is backwards. Bankroll $318.96 /
peak $360.66 / dd 11.6% (NOT grown — down from peak); Shanghai = 8.6% of bankroll = binding
MAX_POSITION_PCT=0.10, maxed by a σ-NULL prob→1.0.

**SHIPPED this cycle (commit pending; propose-only flags default-preserve, flip in live env):**
- A1 (DONE, code): `PROBABILITY_THRESHOLD_NO_REQUIRE_FORECAST` (default False; live env True) → has_forecast=True
  guard on prob-path above/at_least BUY_NO in `binary_market_edge` (threaded `state.has_forecast`).
  Root cause of the forecast-cool-NO drag; +~$0.24 to +$1.8/day; verified against state.has_forecast, not σ-NULL.
- A2 (proposed, not yet coded): per-trade stake sub-cap (flat-$ or 5%) on above/at_least BUY_NO — tail-bound
  the residual live-forecast pre-peak miss (Guangzhou −14.40) A1 can't catch.
- A3 (proposed cleanup): retire the inert LOCK_CONVICTION whitelist (0 EASY-YES ever; zero-PnL, dormant large-size path).

**Validate-first:** V1 tighten/disable ALL prob-path threshold BUY_NO (−0.082 EV/$ even in the 0.80+ band;
PROBABILITY_MIN_ENTRY_PRICE=0.80 does NOT rescue it) — gate on post-06-27 σ-floor fills, n=53 too thin to kill
unilaterally. A1 is the surgical subset; A1 ships first.

**Hold:** no station exclusion (none clears RCTP bar; losses don't track divergence);
NEAR_PEAK_FLOOR_UP off (throttle admits only −EV cohorts); shadow-model promotion HOLD indefinitely.

**Honest state: no proven edge. Minimize losses; small-stakes iteration book.** Book run-rate ~−$1.5 to
−$2/day on the surviving −EV NO cohorts. Next cycle: (1) confirm A1's live-forecast subset stays ≥ breakeven
on the next ~2–3wk of fills; (2) re-run the V1 threshold-NO EV split on post-06-27 σ-floor fills; (3) decide
A2/A3; (4) if keeping the model alive at all, run the isotonic-recal refinement test, else formally retire
shadow_decision.

### 2026-06-24 — Iteration 1.5: lock-path drag → NO ACTION (thin-sample + already-mitigated)

The clean-window lock drag (−$15.66/n5, −0.364 EV/$1) that looked alarming is
**small-sample noise**, not a structured bleed. Branch breakdown of all 57
phantom-safe lock fills (`/tmp/lock_branch_breakdown.py`, win% vs price = break-even):

| branch | n | win% | BE% | gap | pnl | EV/$1 | status |
|---|--:|--:|--:|--:|--:|--:|---|
| range_undershoot NO | 35 | 71 | 81 | −9pp | −$26.52 | −0.102 | **already gated off** (working) |
| range_overshoot NO | 9 | 89 | 90 | −1pp | −$3.70 | −0.048 | **already removed** (working) |
| hard NO | 9 | 67 | 73 | −6pp | −$2.14 | −0.033 | live — thin, mildly overconfident |
| range_in_window YES | 1 | 0 | 40 | — | −$9.28 | — | n=1 anecdote |
| easy_super NO | 1 | 0 | 55 | — | −$7.28 | — | n=1 resolver divergence |
| easy_standard NO | 2 | 100 | 95 | +5pp | +$0.81 | +0.053 | **the genuine edge, +EV** |

- **The big lock losses are already handled:** `range_undershoot` (−$26.52, n35) is
  gated off and `range_overshoot` (−$3.70) removed — the kill-switches are confirmed
  working. The clean-window −$15.66 is 4 trades: one `range_in_window` anecdote +
  3 `hard` NO, and the worst (Taipei 06-20, −$9.34) is **already addressed by the
  RCTP exclusion (06-21)**.
- **`hard` NO is the only live concern, and it's premature to gate:** n=9, −$2.14
  all-time, −6pp overconfident, win-small/lose-big shape — the SAME forecast-overconfidence
  class as the killed `bracket-like-NO` / `range_undershoot`, whose documented cure is
  the **lead-time-aware σ floor (Iteration 2)**, not a permanent kill. Gating it now (would
  need a new code flag — none exists) on n=9 would preempt that fix and act on noise. With
  its worst contributor (Taipei) already RCTP-excluded, go-forward `hard` is ~break-even.
- **`conviction`-lock is inert** — 0 fires/60d (EASY-YES prices to 1.00, never fires). So
  the `LOCK_BIG_SIZE_STATIONS` whitelist (incl. the questionable KATL n=2/−0.54°F) is
  **moot** — no amplification, no action. Confirms Edge Map's "the certain locks price away."
- **`easy_standard` NO is +EV** (the real observed-lock edge), just low-volume.

**Action: none this iteration.** Add `hard` NO to the **σ-floor re-measure cohort** — when
Iteration 2 re-scores the killed forecast-NO classes via `evals-report --operator
bracket-like`, re-measure `hard`-NO EV in the same pass (same root cause). Watch
`range_in_window` (n=1). The disciplined outcome: *don't gate HARD on n=5/n=9.*

### 2026-06-24 — `DEPTH_POSITION_CAP_PCT` scoping (`scripts/replay_depth_cap.py`)

**Recommendation: raise `DEPTH_POSITION_CAP_PCT` 0.20 → 0.30 (env flip, no code),
monitor, then step to 0.40.** The depth cap is decisively the binding throttle, and
raising it has near-zero realized-loss downside.

- **E1 — the gate PASSED.** Of 190 throttled probability decisions (since 06-05,
  with kelly+depth metadata), reconstructing `kelly_stake = kelly_pct×equity` vs
  `depth_cap_stake = 0.20×depth_usd`: **78% are depth-limited** — 66% *pure depth*
  (Kelly wanted ≥$5, the thin book cut it → a good trade we can't size) + 12%
  depth+kelly; only **22% kelly-bound** (genuinely small edge — a raise won't/shouldn't
  help those). So the depth cap throttles *good* trades on illiquid books. Lever confirmed.
- **E2 — volume unlock (of the 190 throttled, clean window):** 30% → 68 fire via Kelly
  (up to 117 with the now-re-enabled floor-up); 40% → 94 (152); 50% → 126 (189);
  60% → 126 (plateau — past ~50% the depth cap stops binding). Current book is ~29
  fills/19d, so even 30% roughly **2–3×'s** the unlocked candidate pool. Σ stake stays
  well inside the idle ~$575 cap headroom at 30–40%.
- **E3a — price slippage is NOT a cost.** Book-walk slippage (price paid − best ask)
  is **~0¢ at every depth-fraction bin** (n=258) — the FAK limit bounds the walk; we
  fill at-or-below the quoted price even at >20% consumption. So a raise does **not**
  erode the +0.113 EV/$1.
- **E3b — fill-rate is the only real cost, and it's a *missed-volume* cost, not a loss.**
  Clean-window fill-rate is **0.83 on $25–100 books** (n=23, thin); **<$25 books are
  unobservable** (today's cap throttled them to $0 before they ever became orders) —
  so fill-rate there can only be learned by raising the cap. (The 60d fill-rate 0.24–0.32
  is **contaminated** by the resolved pre-05-30 balance-race `exception:*` era + delayed
  orders — ignore it.) A no-fill costs nothing (FAK cancels), so even at 83% the raise
  nets large positive volume.
- **No code change needed** — E3a shows no slippage problem, so the flat env raise is
  the right tool; the depth-aware-rule fallback is NOT triggered. Floor-up will
  re-activate (watch `floored_up_n > 0` post-flip as a confirmation).
- **Rollout:** flip env 0.20→0.30 → after ~3–5 days `perf-review --since <flip-date>`
  (fills/day up, prob EV/$1 still clearly +, `no_fill` rate acceptable) → step to 0.40.
  Hold ≤0.40–0.50 (volume plateaus past 0.50; consuming >50% of a thin book raises
  cancel risk). **Kill criterion:** new-fill EV/$1 → ~0 or fill-rate craters → revert env.
- **⚠ Also found: `MIN_EDGE` doc/prod drift** — prod app spec runs `MIN_EDGE="0.05"`
  (RUN_TIME), but CLAUDE.md's gotcha says the live `.env` is `0.10`. The DO env wins, so
  the bot runs at **0.05**. Correct CLAUDE.md, and re-read the counterfactual `edge`-floor
  cohort knowing the true floor is 0.05.

### 2026-06-24 — Clean baseline (the current book, `perf-review --since 2026-06-05`)

The honest current book, post-valley-block, legacy tail stripped. **This is the
number to steer on.**

| Cut | n | win% | PnL | EV/$1 |
|---|--:|--:|--:|--:|
| **Overall** | 29 | 76% | **+$6.02** | **+0.026** |
| Probability path | 24 | 79% | **+$21.68** | **+0.113** |
| Lock path | 5 | 60% | −$15.66 | −0.364 |

(Prior equal window = −$89.70 = the settling legacy tail. The clean book is +EV.)

- **The probability path is the engine** (+0.113 EV/$1). **The lock path is the
  only drag** — −$15.66 on 5 fills, all `HARD` (−$6.76/3, forecast NO locks) +
  `range_in_window` (−$8.90/2, YES range locks). Insight: the clean EASY-YES
  monotonic locks (the thesis's "near-certain edge") price to 1.00 and don't fire,
  so what *does* fire on the lock path is its **weakest** branches. → a future
  iteration should look at gating/sizing `HARD` + `range_in_window` down.
- **Volume is the hard binding constraint:** 29 fills vs **216 `stake_below_min`**
  (~88% suppressed), `floored_up_n 0` (floor-up inert), exposure cap idle
  (**~$575 free of $600**, `low_headroom_frac 0.0`). Lever = **`DEPTH_POSITION_CAP_PCT`**
  (the flip-gate names it explicitly). The edge is fine; the bot barely trades.
- **Valley block working** — ≈0 valley fills in-window. `VALLEY_MIN_EDGE` flip-gate
  still reads "ready" but is **redundant** (P1 block already covers the valley and
  wins precedence — do NOT also set P2).
- σ-floor arm still widening (`sigma.delta` p50 0.079, n=189k) but gated on
  bracket-like EV → Iteration 2.

**Steer-on number going forward: `perf-review --since 2026-06-05` (overall
+0.026, prob path +0.113). Next highest-leverage live lever: `DEPTH_POSITION_CAP_PCT`.**

### 2026-06-24 — Valley flip is ALREADY LIVE + verified (correction to Edge Map v1)

Verifying the "READY" valley lever revealed it was **already applied and is
working** — so the Iteration-1 recommendation is **done, not pending**.

- **Path of the valley fills (the open question):** of the 25 go-forward valley
  [.60,.85) fills, **24 (96%) are probability-path**, all `BUY_NO`
  (`at_least`-NO ×21, `at_most`-NO ×3); 1 is a +EV lock trade. So the prob-path
  valley block targets exactly the −EV cohort. ✓
- **`VALLEY_BLOCK_ENABLED="true"` in the prod app spec** (doctl), and it is
  **effective**: the last prob-path valley fill was **2026-06-05**, while
  non-valley prob fills continued through **2026-06-22** → valley fishing stopped
  ~06-05 while the rest of the book kept firing. (It is *not* a tracked
  config-epoch flag, so `epochs` shows no diff — verify it via the fill-timestamp
  split, not `epochs`.)
- **Reconciliation:** the post-block prob path = **+0.077 EV/$1** (Edge Map v1,
  `perf-review --since 2026-06-14`, a window entirely after the 06-05 block) ≈ the
  replay's **+0.074** P1-block projection. The 60d replay's −0.044 valley / +0.036
  go-forward is dragged down by **pre-06-05 valley losers that no longer fire** —
  do NOT read the 60d aggregate as the current book; use `--since 2026-06-05`.
- **⚠ Doc/prod drift:** prod spec has **`MIN_EDGE="0.05"`**, but CLAUDE.md/memory
  claim the live `.env` is `0.10`. Confirm the true live floor before scoping the
  counterfactual `edge`-floor missed-winner cohort (its size depends on it).

**Net:** Iteration 1's one "ready" lever is already harvested. The active
candidates that remain are (a) `DEPTH_POSITION_CAP_PCT` (unlock the $5-min sizing
throttle on the now-clean +EV book) and (b) the `MIN_EDGE` mini-investigation
(after resolving the 0.05-vs-0.10 drift). Re-baseline the book with
`perf-review --since 2026-06-05` next cycle.

### 2026-06-24 — Edge Map v1 (Iteration 1 complete)

**Bottom line.** The **live (go-forward) book is +EV** (+$27 / 82 fills, 74% win,
**+0.036 EV/$1**). The edge is the **price band + capital discipline, NOT the
model** — the central fork now leans hard to "price band." The binding
constraint is **sizing (the $5-min stake), not filters and not the exposure cap**
(idle headroom ~$573 of a $600 cap). One lever is **flip-ready** (valley policy);
the rest need Iteration 2/3 data.

**1. The central fork — answered early: edge ≠ the model.**
`calibration-report --days 60` on **261k** shadow rows (post-backfill labels):
`model_beats_market = false` **overall AND in all four obs_fraction buckets**,
including the high-obs `0.75-1.00` bucket (n=149505) where the thesis predicts the
model should win — model Brier 0.195 vs market 0.128, and a blown log-loss
(model 1.51 vs market 0.40 → the model is **overconfident**, not better-informed).
`would_trade` EV/$1 = **−0.008** (break-even at best). → **Do NOT promote the
shadow model.** The overconfidence is exactly what Iteration 2's σ-floor targets,
so the thesis isn't dead — but capital stays off the model until calibration-report
flips in the high-obs bucket. This confirms [[project_price_band_concentration_2026-06-19]]
on a 30×-larger, clean sample.

**2. The edge map (price band × path), go-forward universe (n=82, last 60d).**
`replay_price_band_policy --days 60`:

| band (effective price) | n | win% | PnL | EV/$1 |
|---|--:|--:|--:|--:|
| value [.40–.60) | 26 | 58% | +$14.69 | **+0.073** |
| valley [.60–.85) | 25 | 68% | −$10.87 | **−0.044** |
| lock [.85–1.0] | 31 | 94% | +$23.14 | **+0.074** |

The U-shape (edge at both extremes, −EV mid valley) is confirmed live. **P1 block
valley** lifts the go-forward book to +$37.83 / **+0.074 EV/$1** (n=57). NOTE the
value band [.40–.60) is **+EV** — so `PROBABILITY_MIN_ENTRY_PRICE=0.80` is likely
**too blunt** (it's a prob-path floor; the value-band winners are mostly lock-path
cheap-but-certain bets, which it correctly leaves alone — but verify the valley
−EV trades' path before assuming a prob-path valley block catches them).
(Full universe n=315 is −$190 / −0.066 — that's the settling **legacy tail** of
killed classes; ignore for live decisions.)

**3. Binding constraint = sizing, not filters, not the cap.**
`perf-review --since 2026-06-14`: probability path **+$9.25 / 15, 80% win,
+0.077 EV/$1**; the −$8 window headline is **4 lock trades** (`range_in_window` +
`HARD`, −$17.5, n=4 — tiny, watch not act). Exposure: equity ~$455, cap $600,
**headroom ~$573, low_headroom_frac 0.0** → cap NOT binding. Throttle:
`stake_below_min` 138 ("below $5 minimum" ×106), **`floored_up_n` 0** (floor-up
still inert). Flip-gate `NEAR_PEAK_FLOOR_UP`: *"inert — depth cap (20%×book) binds
tighter than the floor on <$25 books → raise `DEPTH_POSITION_CAP_PCT` instead."*
**The lever to unlock volume on the +EV book is `DEPTH_POSITION_CAP_PCT`, not the
floor.**

**4. Counterfactual missed-winners (60d, full universe — candidates, not flips).**
`counterfactual-report`: the **`edge` floor** is the #1 missed-winner — 1447
resolved declined, 75% won @ avg 0.68 → **+0.149 EV/$1** (the live `.env` runs
`MIN_EDGE=0.10`). `depth` floor +0.225 (n=35, small). Crucially the **`price`
floor rejections were correctly −EV** (−0.100, n=177) — the price floor is doing
its job. `stake_below_min` throttled 273 resolved 72%-win trades (capital-bound,
again). The `edge`-floor signal is hindsight over killed classes too — treat as a
**dedicated mini-investigation** (lowering MIN_EDGE also admits −EV trades), not a
flip.

**5. Ready / not-ready levers (propose-only).**
- **READY — valley policy.** perf-review flip-gate `VALLEY_MIN_EDGE`: **ready**,
  measured 0.118, proposed 0.15 (n=27), corroborated by the replay. Recommend
  **`VALLEY_BLOCK_ENABLED=true`** (P1, robust, no tuned param — cleanest go-forward
  lift) as the first flip; `VALLEY_MIN_EDGE=0.15` (P2) is the higher-PnL but
  n=27-tuned alternative. **Caveat:** valley policy is probability-path only —
  verify the −EV valley fills aren't lock-path before expecting the full lift.
- **CANDIDATE — `DEPTH_POSITION_CAP_PCT` up.** Unlocks the dominant `stake_below_min`
  throttle on the +EV book. Needs a slippage-risk check before proposing a value.
- **NOT-READY — σ-floor.** `SIGMA_FLOOR_LEAD_TIME_ENABLED` arm IS widening
  (`sigma.delta` p50 = 0.12, n=97600) but stays conjunctively gated on bracket-like
  EV, which is unmeasurable while `BRACKET_LIKE_NO_DISABLED=true`. → Iteration 2
  (forecast-error fit + `evals-report --operator bracket-like`).
- **HOLD — conviction-lock / station whitelist.** Lock path is −EV (n=4) and the
  EASY-YES edge prices to 1.00 (untradeable). Lower priority than valley + sizing.
- **DO NOT PROMOTE — shadow model** (see #1).

**6. Station trust (from `resolver-truth-report --days 120`).** See the seed entry
below — RCTP −1.57 (excluded), KJFK −2.49 (n3), bulk within ±0.3°F (no hot bias on
431 station-days). No new exclusion warranted from this pass; whitelist deferred
with conviction-lock (#5 HOLD).

**Caveats:** go-forward n=82 is still thin — treat band EVs as directional, not
precise. The legacy tail dominates any rolling-window aggregate; always read
`--since`. calibration's "model worse near peak" is robust (n=149505) and is the
firmest single conclusion here.

**Next:** (a) decide the **valley flip** (READY); (b) scope the **MIN_EDGE
mini-investigation** and the **`DEPTH_POSITION_CAP_PCT`** slippage check; (c)
Iteration 2 once `forecast_error_daily` has ≥3–7 settlements.

### 2026-06-24 — Iteration 1 seed (from the Phase-1 backfill verification)
`resolver-truth-report --days 120`: 731 station-days, **431 (59%) with a
continuous point divergence** (vs 175/535 in the old YES-pinned-only audit).
Per-station (ours − resolved point, °F):
- **Biased/exclude-watch:** RCTP −1.57 (n9, already in `_EXCLUDED_ICAOS`),
  KJFK −2.49 (n3 — small, watch), UUEE −0.84 (n11), LFPG −0.60 (n10).
- **Noisy (high std):** ZGSZ std2.43, EGLL std2.17, MPTO std1.49 — *not*
  conviction-whitelist material despite small means.
- **Trustworthy (whitelist candidates):** the bulk sit at \|mean\|≤0.15 with
  std≤0.3 (°C rounding floor) — e.g. RPLL, EHAM, WMKK, LEMD, EDDM, ZBAA, SAEZ,
  RJTT, LTAC, EFHK, ZUCK, OPKC, RKSI (n17), RKPK (n14, but max +3.6 → check the
  outlier day). Confirm n≥10 + no outlier day before whitelisting for
  `LOCK_BIG_SIZE_STATIONS`.
- **Headline holds:** no systematic hot bias — most stations within ±0.3°F. The
  "we read hotter than the resolver" theory stays unsupported, now on 431
  measurable station-days. Do NOT build a °C correction.

_Next: run Iteration 1 exercises 2–5 to complete Edge Map v1._

<!-- Append new findings below this line, newest first. -->

### 2026-06-27 — σ-floor arm flipped live; today's Guangzhou −$14 loss is the live failure-mode confirmation

**Prompt:** today's book = 1 WON + 1 LOST (2 fills). Act now or wait?

**Verdict: today is noise on the aggregate — but it surfaced the exact failure mode
the σ-floor targets, so I flipped that one (already-green-gated) arm and held everything
else.**

1. **Don't steer on today.** `perf-review --since 2026-06-05` moved +$6.02/29 (06-24) →
   **−$6.33/31** — a ~$12 swing on ~2 net fills, still 74% win. Principles #2/#7: a
   2-trade day is below any action threshold. No edge regression. Top throttle is now
   `dup_blocked_inproc` (57%); loss class `hard` −$6.76/3 (the documented lock drag).
2. **The −$14.40 loss (trade 1700) is a textbook known failure.** Probability-path
   `BUY_NO` on "Guangzhou ≥35°C", priced **0.881**, opened **08:30 local (~6h
   pre-peak)** — a near-lock *forecast cool-side* bet on a **noisy** station (ZGSZ std
   2.43). Normal Kelly sized it ~4% of equity. This is precisely the overconfident
   *far-from-peak threshold-NO* the lead-time σ-floor arm tempers (widens σ at ~6h lead
   → lower NO confidence/edge → decline or down-size), while staying **dormant at/after
   peak** (genuine observed locks untouched). n=1 anecdote, but the canonical shape.
3. **Prod-config drift discovered (vs the 06-24/26 log):** `DEPTH_POSITION_CAP_PCT=0.30`
   is **already applied** (the flip-ready rec was pulled), `THRESHOLD_MIN_PROBABILITY=0.75`,
   `MIN_EDGE=0.05`. Volume did **not** surge after the depth raise — still 1–4 fills/day,
   with the 06-23→26 zero-trade days — so the binding constraint is the `KELLY_PROB_CAP`
   near-lock dead-zone, **not** depth. → **do NOT step DEPTH to 0.40.**
4. **`SIGMA_FLOOR_LEAD_TIME_ENABLED` was absent from the prod spec → still default False**
   (last ACTIVE deploy 06-24). The 06-26 recommendation (gates 1/3/4 GREEN) had not been
   applied. Today's loss *updates* the 06-26 "negligible live effect" caveat: the prob
   path **is** firing pre-peak forecast-NO at near-lock prices, so the arm now has a real,
   beneficial live effect (it discriminates pre-peak forecast-NO from past-peak observed
   locks — exactly the bad sub-cohort).

**Action taken:** flipped **`SIGMA_FLOOR_LEAD_TIME_ENABLED=true`** in the prod app spec
(slope left at default 0.3) via `doctl apps update`; redeploy triggered 2026-06-27
~17:03 UTC. A fresh `ConfigEpoch` will mark the boundary. **Decoupled from the killed
classes** (contra the 2026-05-30 graveyard "pair them" note): flip the σ-floor *alone*
as the low-risk prerequisite; `BRACKET_LIKE_NO_DISABLED` / `RANGE_UNDERSHOOT_LOCK_ENABLED`
stay killed (gate 5 unsamplable until the forward-forecast-archive code task lands).
Graveyard entry updated to match.

**Held this cycle (no change):** DEPTH 0.30→0.40 (volume didn't respond; wrong lever);
`NEAR_LOCK_CONVICTION` / `KELLY_PROB_CAP` bypass (gate behind Iter 2 σ-floor re-pricing);
shadow-model promotion (loses in all obs buckets); Iteration 3 latency verdict (needs
~2 weeks of `latency-report` data).

**Next cycle:** verify the arm bites via `shadow-report --key sigma` + spot-check new
pre-peak threshold-NO evals show wider σ; after ~3–5 days `perf-review --since 2026-06-27`
to confirm the at/past-peak +EV book is unharmed. **Kill criterion:** if it suppresses
the working at/past-peak book (should be dormant there), revert the env var.

### 2026-06-26 — Iteration 2.0: σ-honesty diagnostic (ready-now / near-peak half) → σ-floor arm is GREEN on its own gates, but its payoff stays blocked

**Question:** Is the ensemble σ honest (claimed vs realized RMSE) by lead time, and
does the lead-time σ-floor arm (`SIGMA_FLOOR_LEAD_TIME_ENABLED`) pass its own gates?

**Readiness caveat (important — corrects the §4 trigger):** the raw "≥150 rows" bar is
met (1,885), but the data is starved exactly where the *killed-class re-enable* needs it.
`forecast_archive` only starts snapshotting a target-day **~18h before peak** (markets
appear ~18-24h before close; future-days skipped) — recent-week coverage: 12-18h lead =
207 station-days but **18-24h = 7, 24-36h = 4, 36h+ = 0**. The 18-48h regime that gates
bracket-like-NO is **structurally unsamplable by waiting** (needs a code change to archive
multi-day-ahead forecasts). So Iteration 2 splits: the **near-peak (0-18h) half is ready
now**; the far-from-peak half is blocked. This entry is the ready half.

**1. σ-collapse is real — but it's a LEAD-TIME effect, not a level problem.** The naive
RMSE/σ_raw ratio (1.7-2.4×) **overstates** it — the engine already applies ×1.3 +
a 2.0 floor. Net of that, vs the **effective** engine σ = `max(2.0, σ_raw×1.3)`, on
resolved truth:

| lead | n | realized RMSE | eff. engine σ | overconfidence |
|---:|--:|--:|--:|--:|
| 0h (peak) | 480 | 3.03°F | 2.64 | **1.15×** (≈calibrated) |
| 6h | 428 | 3.45°F | 2.66 | 1.30× |
| 12h | 293 | 3.80°F | 2.55 | **1.49×** |

The engine is ~honest **at peak**; overconfidence **grows with lead** (1.15→1.49× over
12h). → the right lever is the **lead-time arm**, NOT a flat σ-floor raise (which would
over-pad the well-calibrated peak). RMSE-vs-lead slope ≈ 0.064°F/hr (intercept 3.05).

**2. The dark arm at its DEFAULT slope 0.3 is roughly right** (corrects a mid-investigation
error — the naive RMSE-slope vs arm-slope comparison is invalid; the arm lifts a *floor*
below the ensemble σ and only bites where it exceeds it). Live σ-shadow telemetry
(`shadow-report --key sigma`, 379k evals, the ground truth of what the arm does):

| lead band | n | σ delta | live σ → with-arm |
|---|--:|--:|---|
| past peak / 0-6h | 49k | **0** | dormant (correct) |
| 6-12h | 42k | +0.49 | 2.60 → 3.09 |
| 12h+ | 15k | **+1.81** | 2.36 → **4.17** |

`with_arm` 4.17 at 12h+ ≈ realized RMSE ~4.0. The shape is exactly right (dormant near
peak, lifts far-from-peak to realized error). A gentler slope (0.15) would under-pad; 0.3
is the better-supported value.

**3. Per-station heterogeneity is large** (`error_vs_resolved`, n≥15 at peak): NZWN/ZGSZ
overconfident even at peak (raw σ 0.8-1.0 → eff 2.0 floor still < RMSE 2.8-3.4 — the classic
confident-ensemble collapse); RKSI/RKPK/RJTT/LTAC blow up with lead (rmse12 5.8-7.4);
ZSPD/SBGR/FACT stay flat (rmse12 ≈ rmse0). A **global** arm is the pragmatic first step;
per-station lead slopes are a later refinement.

**4. backtest-v2 A/B (21d, 90% real-forecast replay): NEUTRAL — passes "no regression."**
arm OFF: cal-err 0.0383 / Brier 0.053130. arm ON (0.3): cal-err 0.0383 / Brier 0.053077.
Identical — the Brier metric is dominated by near-peak resolution where the arm is dormant,
so it neither confirms nor refutes. Decide from shadow + forecast-error, not backtest
(matches the standing §3 caveat).

**Gate status — `SIGMA_FLOOR_LEAD_TIME_ENABLED` (the arm flip, decoupled from killed classes):**
- (1) RMSE >> σ at high lead → ✓ **GREEN** (1.49× at 12h, growing)
- (3) shadow positive far-from-peak → ✓ **GREEN** (+1.81°F at 12h+, dormant near peak)
- (4) backtest no Brier regression → ✓ **PASS** (neutral)
- (5) `evals-report --operator bracket-like` baseline +EV → ✗ **BLOCKED** (bracket-like-NO
  disabled; 18-48h fit unsamplable)

**Recommendation (propose-only):** flip **`SIGMA_FLOOR_LEAD_TIME_ENABLED=True` (slope 0.3)**
now — gates 1/3/4 are green, the shape is data-honest, and the downside is nil (it only
widens σ far from peak → tempers overconfident far-from-peak bets). **But it is
foundation-laying, not volume-unlocking in the current config:** the far-from-peak bets it
would temper are *already* suppressed (bracket-like-NO killed, prob path near-lock-
concentrated), so its live effect today ≈ negligible. Its real role is as the **prerequisite
that must be live before the killed classes can be reconsidered**. The bracket-like-NO /
range-undershoot re-enable (gate 5) stays blocked pending: (a) the forward-forecast-archive
code task (sample 18-48h), then (b) ~2-3 weeks accumulation, then (c) re-run gate 5.

**Action:** propose the arm flip; **schedule the forward-forecast-archive code task** as the
true unblocker of Iteration 2's second half. Per-station lead slopes = future refinement.

### 2026-06-26 — Zero-trade days (Jun 24 & 25): the volume floor hit literal zero — mechanical, not a silencer

**Question that prompted it:** no trades registered on 2026-06-24 or -25. Why?

**Verdict:** *not* a silencer, crash, or filter regression — it's the binding-constraint
story (§5 "binding constraint = sizing") biting all the way to **0**. The pipeline ran
fine (~30k evals/day) and the **edge filters + KELLY_PROB_CAP correctly declined a
near-empty, marginal opportunity set.** Whether to "fix" it is a real question, not a
clear bug (see below).

**The funnel (production DB, UTC):**

| day | evals | passed | decisions | trades |
|---|--:|--:|---|--:|
| Jun 23 | 29,954 | 7 | 1 filled, 5 `stake_below_min`, 1 dup | **1** |
| Jun 24 | 30,686 | **1** | 1 `stake_below_min` | **0** |
| Jun 25 | 32,892 | **0** | — | **0** |
| Jun 26 (part) | 10,201 | 0 | — | 0 |

**Two compounding causes:**

1. **Sizing dead-zone — `KELLY_PROB_CAP=0.90` vs the near-lock band.** The probability
   path now passes *almost exclusively* near-lock `BUY_NO` (model_prob ≈ 1.0), because
   `PROBABILITY_MIN_ENTRY_PRICE=0.80` concentrates it there. But Kelly clamps model_prob
   to 0.90 first, so **any bet priced ≥ 0.90 computes negative edge → $0 stake**
   (`size_reason:"no edge"`, `kelly_pct:0`). Of 22 passing prob-path NO evals Jun 20-26,
   **18 were priced ≥ 0.90 → all sized $0**; only the 4 priced < 0.90 (0.872-0.899) could
   fire (those are the Jun 22-23 fills). Jun 24's single pass was priced **0.938** → $0.
   This is exactly [[project_near_lock_conviction_2026-06-20]] biting to zero. The lever
   that bypasses the cap (`NEAR_LOCK_CONVICTION_SIZING_ENABLED`) is **OFF** — and its
   rebuilt gate (has_forecast + observational lock, **betting-hot direction only**:
   YES on above/at_least, NO on at_most/below) would *not* even cover these anyway,
   since the passes are mostly **`at_least`-NO** (the cool/forecast side), not the hot
   monotonic-lock side.

2. **Pass rate itself decayed to 0.** Passes fell 10→3→2→7→1→**0** (Jun 20→25). Reject
   mix Jun 24-26: `prob < 0.85` 40,748 (model not confident enough — bracket-like strict
   floor / YES side), `price 1.00 outside band` 9,384 (EASY-YES locks & near-resolved
   markets — latency already priced away), `edge < min` 23,767 (negative/low). The
   tradeable near-lock NO band *below* the 0.90 sizing ceiling simply **didn't
   materialize** on the 24-25th — natural market efficiency on the razor-thin band the
   strategy has concentrated onto.

**Silencers explicitly ruled out:** exposure cap idle (headroom **$593 / $600**, 1 open
trade, $7 exposure); `dd_level` normal on the relevant decisions (4 transient `paused`
rows exist but aren't the cause); the throttle reason is `"no edge"`/`kelly_pct=0`, **not**
`drawdown_paused` or `cap_exceeded`. Equity ~$361.

**Is this even a problem?** Partly no. The only things passing on those days were near-lock
`at_least`-NO bets priced ≥ 0.90 — risk ~94¢ to make ~6¢, and the forecast/cool-NO family
is the documented **lock-path drag** (§5 06-24). KELLY_PROB_CAP declining them is arguably
*correct*. So "0 trades" here ≈ the strategy correctly passing a marginal day — a **volume**
problem, not an edge problem, fully consistent with "the edge is fine; the bot barely trades."

**Levers (propose-only; none fixes this dead-zone cleanly):**
- `DEPTH_POSITION_CAP_PCT` 0.20→0.30 (the §5 06-24 flip-ready rec) unlocks **depth-bound**
  `stake_below_min` throttles — it does **NOT** touch the `KELLY_PROB_CAP` "no edge"
  dead-zone, so it would not have produced trades on 24-25. Still the right next flip for
  *good* volume; just don't expect it to fill these days.
- The only thing that would have sized Jun 24's pass (0.938, model 0.999) is a
  `KELLY_PROB_CAP` bypass for genuine near-lock NO — i.e. broadening
  `NEAR_LOCK_CONVICTION` beyond the hot-only direction. **Do not** propose that yet: it
  means deliberately buying 0.90+ forecast-NO for ~6¢ edge, the exact marginal/drag cohort
  Iteration 2's σ-floor is meant to re-price first. **Gate it behind Iteration 2.**

**Action: none.** Logged as the expected tail of price-band concentration × `KELLY_PROB_CAP`.
Re-confirm volume returns once `DEPTH_POSITION_CAP_PCT` is raised; revisit the dead-zone only
after the σ-floor (Iter 2) makes the near-lock forecast-NO band trustworthy.
