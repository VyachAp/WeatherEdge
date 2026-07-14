# Mastering Playbook — WeatherEdge self-improvement loop

> Created 2026-06-24, right after the data-enrichment phase (Phase 1
> `station_day_resolutions`, Phase 2 `metar_reprice_snapshots`, Phase 3
> `forecast_error_daily`). This is the durable reference for the *mastering*
> phase: turning that data into a small number of **data-proven** config moves,
> consistently, gym-style — steady gains, never a profit switch.

---

## Mission & nightly charter (added 2026-07-09)

The nightly orchestrator agent (`docs/agents/nightly_orchestrator_prompt.md`)
runs this playbook every night. Its charter:

- **Mission:** drive WeatherEdge to a **consistently profitable autonomous
  service**, one *data-proven* step per night, **propose-only** (a human flips
  every `.env`/trading change). Success is consistency, not a jackpot.
- **Per-night value contract:** each session yields **≥1** of
  { a flip-ready proposal with evidence · a validated-or-killed hypothesis · a
  new loss-class / station-trust / cohort insight · a new backlog experiment } —
  **or** an explicit *null-result* note naming *what was checked, why nothing
  moved, and which data / sample-n is the wall*. **No silent empty nights.**
  Tokens must always buy at least the location of the wall.
- **Tracks (phased):**
  - **Track A — now.** Master the weather bot by walking the §3 dark-flag→gate
    map to *proven-flip or explicitly-killed*, one gate per night, per §4.
  - **Track B — gated, later.** Whole-Polymarket reconnaissance (survey all
    events/markets for new automated-betting edges). **Do not start** until
    Track A hits the stability trigger. Build it in-house against the existing
    Gamma/CLOB primitives + the `deep-research` skill — *not* a third-party
    skill (see `docs/improvements.md` "[backlog] Whole-Polymarket opportunity
    discovery"; the 2026-07-09 ultracode skills audit found nothing worth
    installing).
- **Stability trigger (noted, not built):** when phantom-safe realized PnL is
  positive for N consecutive days on the clean book, downgrade the loop to
  monitoring-only and consider spinning up Track B. Design deferred.
- **Skills stance:** add no GitHub Claude Code skills. Rely on built-ins
  (`deep-research`, `schedule`/`loop`, `dataviz`+Artifact, `code-review`/
  `verify`/`simplify`) layered on this in-house loop. Steal *techniques* into
  §1 (e.g. the noise-floor variance gate), never install wrappers that fight the
  propose-only / measure-before-flip discipline.

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

### 2026-07-10 — THE EDGE IS FOUND, PROVED, AND SHIPPED (default-off): `bucket_overshoot`

**The central fork of this playbook is settled. The edge is neither the model nor the price band —
it is MATHEMATICAL CERTAINTY + SPEED.** Three months of tuning a forecast model chased an edge that
never existed, while the one real edge sat gated off in `docs/graveyard.md`.

**What was ruled out (each with data, not argument):**
1. **No model edge.** The market's mid beats `shadow_decision`'s posterior on Brier in *every*
   `obs_fraction` bucket, including 0.8–1.0 (0.1224 vs 0.1698, n=243k ticks). Do not promote the
   shadow model. Ever, on this evidence.
2. **No taker edge anywhere.** One bet per market, event-clustered, 906 station-days: buying YES is
   −9% to −17%/$1 in every price band; buying NO is −1% to −2.5%. The overround + spread is the house.
3. **No dutch book.** Full 11-market ladders sum to a median **1.019** at mid (~2% overround). Any
   spread destroys it.
4. **A textbook favorite–longshot bias exists** (bucket at ask 0.20–0.30 wins 20.9%; at 0.60–0.70
   wins 66.0% vs 63.3% priced) but it is not harvestable: depth collapses exactly where the edge is
   (median $5.59 at ask 0.60–0.70, $1.20 at 0.80–0.90).

**What IS the edge.** The daily max is monotonic. Once the running routine-METAR max for the market's
station-local day climbs a full °C above a bucket's top, that bucket **can never win**. On trusted
stations that rule is right **3577 / 3580 times (99.92%)**. The market agrees — eventually. It
collapses the dead bucket to ~0.005 a **median 2.07 min after the METAR's observation time**. We hold
that METAR at a **median 2.70 min**. We lose the race 59% of the time; the other 41% is the business.

**Validated properly** (the first three attempts were wrong — see the methodology notes):
3,580 rule-triggered candidates → priced from **`clob.polymarket.com/prices-history` (1-min)** at the
bot's **real `fetched_at`** → scored against **on-chain resolution** → **event-clustered** bootstrap.

| policy | bets | events | losses | EV/$1 | 95% CI |
|---|---|---|---|---|---|
| ship (max_cost 0.93, trusted stations, d≥1°C, rc≥3) | 63 | 52 | 2 | **+0.911** | [+0.532, +1.463] |
| …out-of-sample (target day > 2026-06-05) | 55 | 45 | **0** | **+1.013** | [+0.613, +1.632] |

Survives: leave-one-station-out (+0.28…+0.49), a +5¢ slippage haircut (+0.56), and a 5× worse
resolver-divergence rate (+0.73). Dies at a 60 s decision delay (+0.10, CI spans 0) — **speed IS the
edge.** Capacity is not binding: live books offer a median **$685** within 3¢ of top for the
0.20–0.40 bucket prices. Sizing sim ($500 bankroll, 5%/trade, quarter-Kelly, depth-capped):
**+$25/day, max DD 9.8%.**

**Three methodology traps this cycle burned, all now in memory + CLAUDE.md:**
- **`market_snapshots.yes_price` (Gamma) is stale in the repricing window** — corr **0.31**, bias
  **+0.142** vs the real book (the bot's own `shadow_ledger` bid/ask tracks it at corr 0.974). Pooled
  over a market's life they agree at 0.99 — *that pooled check is the trap*. It made a fake **+98%
  EV** appear. Never price execution from snapshots.
- **`shadow_ledger` is censored** at `yes_bid≥0.99 / yes_ask≤0.01`, so post-collapse rows vanish and a
  naive `merge_asof` silently matches a *pre-collapse* quote.
- **`market_resolutions` only labels shadow-evaluated markets** (57% of the ladder). Select bets by the
  rule, then fetch ground truth from Gamma directly.

**The graveyard was wrong.** `range_overshoot` was killed 2026-06-24 on "56% / 18 trades / −$57.61".
Its real phantom-safe book: **9 fills, 8W/1L (88.9%), −$3.70**. The single loss was Buenos Aires under
the **pre-2026-05-26 wrong-day bug** (`resolve_target_local_day` returned title-day−1 for −UTC cities),
not °C resolver divergence. It also bought NO at an average **0.901** — it paid the edge away.
*Generalized lesson: before killing a branch, filter `fill_price NOT NULL` and check whether its losses
predate a known input bug. A rule can be correct while its inputs are wrong.*

**Also fixed: `job_fast_lock_poll` was structurally dead.** `fetch_latest_metars` returns only each
station's most recent METAR → 1-point `routine_history` → `evaluate_lock`'s `routine_count < 2` floor →
**no fast-poll lock has ever fired.** (This, not price, is why the conviction-lock path shows 0 fires.)
Now merges the day's history from `state_aggregator._state_cache`.

**Shipped (default-off, awaiting operator flip):** `BUCKET_OVERSHOOT_LOCK_ENABLED=false`,
`BUCKET_OVERSHOOT_MARGIN_C=1.0`, `BUCKET_OVERSHOOT_MAX_COST=0.93`,
`BUCKET_OVERSHOOT_EXCLUDED_STATIONS="ZGSZ,MPTO,EGLL,UUEE"` (the four stations with
`P(divergence ≥ 1°C) > 5%`; every other station with n≥15 is ≤4.2% — a clean natural break).
823 tests pass; mypy unchanged; branch verified to fire on 274/274 historical opportunities and, on
live data, to flag 118 dead buckets and correctly refuse all 118 as already-repriced.

**Next lever, in priority order:**
1. **Cut METAR ingestion lag — but NOT by adding providers.** Measured 2026-07-10 (10-s polling across
   a METAR issue boundary, 8 stations): **AWC and NOAA are the same upstream feed** — identical
   first-appearance *second* on 13 of 14 observations; NOAA won once, by 32 s. A multi-provider racer
   buys ≈0. **Publication is the floor and it is per-station**: WSSS 1.09 min, OPKC 1.65, VILK 2.80,
   RKSI 5.00, ZGGG 5.31, WMKK 8.70. (Note those are exactly the stations our winning bets come from —
   the edge *is* the fast-publishing stations.) The "1-min lag ⇒ 682 bets, 5×" counterfactual in the
   EV-vs-lag table is therefore **not attainable in general**; it is attainable only on the
   fast-publishing subset. Our only controllable slack is `FAST_LOCK_POLL_INTERVAL_SECONDS`
   (30 s → 10 s, done) plus decision→order time.
2. Place the order within ~15 s of detection (EV +0.43 at 15 s, +0.28 at 30 s, dead at 60 s).
3. Re-check the station-exclusion list monthly from `station_day_resolutions` (it is ex-ante and
   walk-forward computable; do not tune it on bet outcomes).


### 2026-07-09 (late) — The `MIN_EDGE` mini-investigation is CLOSED: the "#1 missed-winner" cohort is a **first-failing-filter attribution artifact**. Do NOT lower `MIN_EDGE`.

**Gate advanced tonight: the recurring `counterfactual-report` `edge`-floor lead → KILLED (not
"insufficient data" — structurally invalid as measured).** This lead has topped the report twice
(06-24 Edge Map "#1 missed-winner", 1447 resolved / 75% won / +0.149 EV/$1; and again tonight at 7d:
642 resolved / 82% won @ avg px 0.72 / **+0.195 EV/$1**, edge +10pp). It has never converted because
the metric cannot support the inference. Four independent defects, each verified against source +
prod this cycle:

1. **First-failing-filter attribution (the fatal one).** `_check_filters`
   (`src/signals/edge_calculator.py:435`) tests **`edge` FIRST**, before `probability` and `price`.
   So `reject_reason="edge"` is not "trades only `MIN_EDGE` blocked" — it is the **catch-all bucket for
   every side the model saw nothing on**, never tested against the downstream floors. An edge-reject
   implies `our_prob ≤ side_price + MIN_EDGE` = `≤ 0.77` at the cohort's avg price 0.72. Those rows
   would then be re-rejected by `MIN_PROBABILITY=0.85` (bracket-like) / `THRESHOLD_MIN_PROBABILITY=0.75`
   (threshold), and by `PROBABILITY_MIN_ENTRY_PRICE=0.80` (avg px 0.72 < 0.80). **Lowering `MIN_EDGE`
   recovers ≈none of this cohort.** The report's counterfactual ("this filter left money") is only
   valid if the *remaining* filters are re-run on the cohort — they are not.
2. **…and what little it WOULD admit is the losing half.** Back-solving the reported aggregates
   (n_resolved=642, win 82%, EV/$1 +0.195 ⇒ ~526 wins / ~116 losses): mean winner return 0.459 ⇒
   **winners' avg price ≈ 0.686**; losers' avg price ≈ **0.876**. The winners sit *below* the 0.80
   entry-price floor and the losers *above* it. So relaxing `MIN_EDGE` while `PROBABILITY_MIN_ENTRY_PRICE
   =0.80` stands would admit predominantly the **losing** sub-cohort. (Arithmetic from rounded printed
   aggregates — directional, but the sign is robust.)
3. **Rows are not independent; effective n is station-days.** Cohort rows are per-`(market, direction)`,
   but every market on a station-day is a deterministic function of **one** number (that day's resolved
   max). `resolver-truth-report --days 7` (run tonight): **255 station-days resolved**. So the 642
   "independent" rows collapse to **≤255 clusters** (~2.5 rows/cluster, within-cluster ρ≈1).
   `cohort_stats` treats them as 642 independent equal-stake bets → SE understated.
4. **The by-station table is a silent max-of-N selection.** `cli.py::_table` renders
   `[:8]` of cohorts **sorted by `ev_per_dollar × n_resolved` desc** with no note that ~40+ stations were
   dropped. So it structurally displays the *winning tail*. Tonight it printed **MPTO +1.316 EV/$1** —
   on n=33 rows = **7 station-days** (7 independent draws), and **CYYZ +0.286 on 6**. That is exactly the
   max-order-statistic of ~50 noisy small-n cohorts, not a signal. Violates §1's own "no silent caps".

**Also found: `by_op_class` and `by_price_band` are computed and thrown away.** `mine_rejected` builds
all four cohorts but `counterfactual_report` renders only `by_reason` / `by_station` / `by_outcome`. The
two most decision-relevant splits (killed bracket-like vs threshold; the valley price band) never reach
the operator. Cheap fix, high value.

**Verdict:** `MIN_EDGE` stays **0.05**. Close the "MIN_EDGE mini-investigation" (open since 06-24) as
**rejected on methodology**, not as unproven. Treat *every* counterfactual "missed winner" cohort as
**non-actionable until re-run through the downstream filters** and given a station-day-clustered noise
floor. This is the generalized lesson: **a reject-reason cohort names the first gate that fired, not the
only gate that would.**

**Smallest validating step (if ever revisited), propose-only:** (a) report `n_clusters` (distinct
station-days) beside `n_resolved`, and suppress "missed winner" headlines below `n_clusters ≥ 30`;
(b) add the station-day-clustered bootstrap / label-permutation baseline from
`docs/improvements.md [backlog] Noise-floor variance gate`; (c) add a `filters_rerun` pass that replays
the surviving `_check_filters` chain on a cohort before calling it recoverable; (d) render `by_op_class`
+ `by_price_band`, and label the top-8 truncation.

**Report tooling wall (last cycle's TODO #1) — DIAGNOSED, and it is NOT a pathological join.** The three
"hangs" are plain **O(raw rows) client-side transfers**; they complete if you wait:
- `counterfactual-report`: `knowledge_base.fetch_counterfactual_rows` streams **every**
  `evaluation_logs.passes=False` row since cutoff (~30k/day) into Python and dedups there
  (`latest_rej[(mid,dir)] = …`). Measured tonight: **1d = 38s, 7d = 151s** ⇒ 30d ≈ 10 min, 60d ≈ 20 min.
  Fix: `DISTINCT ON (market_id, direction) … ORDER BY market_id, direction, created_at DESC` in SQL.
- `calibration-report` (`cli.py:4932`): `select(ShadowLedger, …)` fetches the **whole ORM entity** —
  including the `feature_json` JSONB WeatherState snapshot + `decision_reason` — for every row in the
  window, while using **8 scalar columns**. Fix: column projection.
- `latency-report` (`cli.py:4573`): `select(MetarRepriceSnapshot)` whole entity, uses 7 columns.
- `perf-review --since 2026-06-05`: **completed in ~22 min** tonight (0 bytes for 20 of them). Slow, not broken.
All four are pure read-path changes — no schema, no behavior. This unblocks the shadow-retire +
latency verdicts that have been stale since 07-07.

**⚠ `perf-review`'s `flip_gates[].current_value` reads the LOCAL `Settings`, not prod.** Tonight's
artifact reports `SIGMA_FLOOR_LEAD_TIME_ENABLED=false`, `THRESHOLD_MIN_PROBABILITY=0.78`,
`NEAR_PEAK_FLOOR_UP_ENABLED=true` — **all three contradict doctl-authoritative prod** (true / 0.75 /
false). An analyst trusting the digest would propose flipping a flag that is already on. Always reconcile
against `doctl apps spec get`. (Related: `VALLEY_MIN_EDGE` again reads "ready" — it is **redundant**, P1
block is live and wins precedence. Do not set P2.)

**A1 epoch boundary now EXISTS.** `epoch.current_id=7`, started **2026-07-09T09:42 UTC**, `flags_diff =
{PROBABILITY_THRESHOLD_NO_REQUIRE_FORECAST: null→true}` — last cycle's `config_epoch._TRACKED_FLAGS`
change is deployed and the boundary was written. Future A1 scoring should use `--since-epoch 7` rather
than date-only windowing. **A1 remains PROVISIONAL/unscored:** book unchanged since last cycle
(clean book since 06-05 still **n=37, 73% win, −$34.40, −0.102 EV/$**; prob path n=29 −0.052; lock
n=8 −0.326 from legacy `hard` n=4 −$12.02 + `range_in_window` n=4 −$8.03). **Zero** new settled
post-07-07 threshold-NO fills. Wall: need ~10; at the current ~1 fill/day overall that is **~2–3 more
weeks**.

**Binding throttle unchanged:** `dup_blocked_inproc` 431/790 (54.6%), `stake_below_min` 283, size_reasons
"below $5 minimum" 175 + "no edge" 110 (the `KELLY_PROB_CAP` dead-zone). Exposure **99% idle**
(headroom p50 $592.9 / $600, `low_headroom_frac` 0.0). Same story as 06-24 → 07-09.

**🚨 The nightly Telegram sink is a silent no-op.** `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` exist
but are **empty** in the operator Mac's `.env` (they're only in the DO app spec, where the *bot*
runs). `perf-propose-push` printed `[no Telegram credentials — not pushed]` and **exited 0**. The
nightly loop would look healthy while its operator-facing sink is dead — the project's recurring
silent-silencer class. This cycle's durable sinks are docs + memory only. Fix: populate the keys on
the Mac **and** make `perf-propose-push --push` exit non-zero without credentials.

**⚠ Un-authored `src/` edit during this run.** An uncommitted `Index("ix_eval_logs_created",
"created_at")` appeared in `src/db/models.py` mid-session (not written by the agent; left in place).
It targets exactly the slowness diagnosed above — but a bare `models.py` index with **no Alembic
migration** is the documented `bot_state` / `gfs_prob` failure class, and on a ~2M-row live table it
needs `CREATE INDEX CONCURRENTLY`. Needs an operator decision + a migration.

**Next-cycle TODO:**
1. **Propose-and-apply (operator) the 4 read-path report fixes** above — they are the prerequisite for
   re-running `calibration-report --days 60` / `latency-report --days 14` and closing the stale
   shadow-retire + latency verdicts.
2. **Score A1 with `--since-epoch 7`** once ~n≥10 post-07-07 threshold-NO settle (~2–3 weeks).
3. **Warm level-correction (~+0.7°F to `forecast_peak_f`)** — still the empirically-supported σ-honesty
   follow-up (the miss is a LEVEL cool bias, not a slope). Unstarted.
4. Armed conviction-lock path (`CYYZ,SBGR,RKSI`): still 0 fires. Open question to disarm entirely.

### 2026-07-09 — Post-A1 monitoring cycle: A1 live+biting (unscored), A3 CANCELLED (armed not dead), config drift reconciled

**Thin-data monitoring cycle, 2 days after the 07-07 close.** Ran the dashboard + two custom
splits (A1 live-forecast subset, V1 threshold-NO band) against prod. Every post-07-07 cohort is
single-digit n — all verdicts are directional/provisional by principle #7. Central conclusion
unchanged: **no proven edge; minimize losses.** The one hard action this cycle was a **config-reality
correction** that cancelled the planned A3.

**A1 is LIVE and BITING, but unscored (n=0 settled).** `doctl apps spec get` (app 4b020323,
authoritative) confirms `PROBABILITY_THRESHOLD_NO_REQUIRE_FORECAST=true` in prod — commit `ff7f6bc`
IS deployed, despite there being **no A1 ConfigEpoch boundary** (the flag wasn't epoch-tracked — a
telemetry gap, not a deploy miss). Post-07-07 the guard fired **3** `threshold NO without forecast
(degenerate state)` rejects and admitted **0** σ-NULL `above`/`at_least` BUY_NO fills — exactly the
design. But settled post-07-07 trades of this class = **0**, so A1's go-forward EV is unmeasurable
this cycle. **Verdict: PROVISIONAL** (deployed + biting confirmed; not yet scored). Re-check after
~1–2 weeks of settlements.
- **Wrinkle worth recording:** on the 06-05 book (phantom-safe), the σ-NULL degenerate cohort A1 now
  kills was only mildly −EV (−0.091 EV/$, **n=11**), while the *live-forecast* above/at_least NO
  subset A1 lets through is the deeper bleeder (−0.44 to −0.53 EV/$, **n=2+2, directional**). Read
  honestly: the **whole** above/at_least NO class looks −EV forecast-or-not, but every cohort is n<10.
  A1 removes the clearly-broken degenerate tail; it does **not** by itself make the class +EV. Points
  toward V1, pending volume.

**V1 (disable all prob-path threshold BUY_NO) → HOLD.** Post-06-27 threshold BUY_NO is entirely in
the 0.80+ band (**n=4**, 50% win, avg NO px 0.872, −0.55 EV/$), **zero** <0.80 fills —
`PROBABILITY_MIN_ENTRY_PRICE=0.80` is fully binding, no cheap-NO leakage. The 0.80+ band does **not**
rescue threshold-NO, and directional evidence leans toward tightening, but n=4 is 25× short of the
n≥100 kill bar. HOLD.

**A2 (per-trade stake sub-cap on above/at_least BUY_NO) → HOLD (would be inert).** No new post-07-07
live-forecast pre-peak tail loss appeared (n=0), AND `decisions-report` (30d, n=354) proves per-trade
`MAX_POSITION_PCT` is **not** the binding constraint — **zero** `cap_exceeded`, zero
MAX_POSITION_PCT-limited rows. A per-trade sub-cap as specced would throttle nothing. If a tail loss
appears, the correct lever is the `KELLY_PROB_CAP`/$5-min mechanics, not the per-trade cap. HOLD.

**A3 (retire the "inert" LOCK_CONVICTION whitelist) → CANCELLED. Premise refuted — the path is
ARMED, not dead.** The plan assumed doubly-dead (flag False + empty whitelist, per code defaults +
stale docs). `doctl` shows the **opposite live**: `LOCK_CONVICTION_SIZING_ENABLED=true` and
`LOCK_BIG_SIZE_STATIONS=CYYZ,SBGR,RKSI,KATL`. The conviction-lock path is **dormant-but-armed** — it
has never fired (0 EASY-YES fires/14d) only because EASY-YES locks price to ~1.00, outside the
[0.05,0.95] band ("the certain locks price away", confirmed again). **Do NOT retire it** — that would
disable a live feature on a false premise. Instead, a **new recommendation for the operator** (a
*de-arming* env change, outside the "apply gated-green flips" mandate — needs an explicit decision):
the armed path is a latent black-swan large-size surface (Kelly-by-price up to 15% of bankroll on one
fill) with **zero demonstrated benefit** (0 fires) and includes **KATL (n=2, unverified** in the trust
map). Minimal move: **trim KATL** from the whitelist; conservative move consistent with
"don't amplify, minimize losses": **disarm entirely** (`LOCK_CONVICTION_SIZING_ENABLED=false`) until a
station-trust map finds EASY-YES locks systematically filling *below* ~0.95 (a real discount to a
verified-monotonic outcome — none exists today).
**Action taken 2026-07-09 (operator-directed):** trimmed **KATL** from the live whitelist via
`doctl apps update` → `LOCK_BIG_SIZE_STATIONS=CYYZ,SBGR,RKSI` (the three trust-map-clean stations);
`LOCK_CONVICTION_SIZING_ENABLED` left `true`. The path stays armed on verified stations but still won't
fire until an EASY-YES lock prices below ~0.95. Full disarm remains the fallback if that never happens.

**Shadow-decision → RETIRE-as-promotion-candidate stands, but on STALE evidence.** The 07-07 fork
(edge = price band, not the model) holds; the last clean `calibration-report` (07-07) had
`model_beats_market` in **0/5** obs buckets, worst near peak, no n≥500 winner. But **this cycle's
`calibration-report` (60d) and `latency-report` (14d) were both killed 2× under prod-DB contention**,
and `counterfactual-report` hangs at prod scale (0 bytes, pathological join). So the retirement rests
on 07-07 data, not a fresh run. Keep `shadow_ledger`/`shadow-report` σ telemetry (still healthy). The
only sanctioned revival route remains an **isotonic-recal refinement test** (not yet run).

**σ-floor is healthy and stays ON.** `shadow-report --key sigma` (n=218,869): `sigma.delta`
p25/p50/p75 = 0/0/0 — firing only on the <25% far-from-peak tail (>~10h) at ~+0.1°F, dormant near
peak, as designed. `forecast-error-report` (30d) confirms genuine under-dispersion (RMSE **1.74–2.29×**
mean σ at n≥746) and, crucially, the miss is a **LEVEL cool bias (~−0.75°F, flat across 0/6/12h lead),
NOT a SLOPE** (the 18/24h warm flip is n<30 — ignore). → do **not** grow the lead-time slope; the
empirically-supported candidate (separate, validate-first) is a small **warm level correction**
(~+0.7°F) to `forecast_peak_f`, which would tighten the whole above/at_least-NO class the honest way.

**Clean book (since 06-05, phantom-safe) still mildly −EV, shallow.** Overall **n=37**, 73% win,
−$34.40, **−0.102 EV/$**. Prob path **n=29, −0.052** (borderline); Lock path **n=8, −0.326** — the
worse drag, from 4 legacy HARD fills opened before the 06-30 gate (−$12.02) + `range_in_window`
(n=4, −$8.03). Post-σ-floor (since 06-27) is n=8 total, too thin. The σ arm did not suppress
prob-path volume.

**Binding throttle = capital-sizing, not filters/cap (unchanged).** `decisions-report` (30d, n=354):
`stake_below_min`=51.7%; sizing stalls split "no edge" 94 (the `KELLY_PROB_CAP` dead-zone zeroing the
>0.90 near-lock band) + "below $5 min" 89. Exposure 99% idle (median headroom $593/$600, 0.0% of ticks
below $5). The bot barely trades because Kelly can't size the near-lock band it's concentrated onto —
same story as 06-24/06-26.

**Resolver truth: no new exclusions.** `resolver-truth-report` (120d, 914 point-divergences): no
station crosses |signed mean|≥1.0°F at n≥10. Closest real one is **UUEE −0.66 mean / abs 1.19 / std
1.40 / n26 — WATCH, not exclude**. RCTP (−1.57, n9) already excluded. Whitelist stations CYYZ/SBGR/RKSI
read clean; **KATL is n=2, unverified** (feeds the A3 de-arm recommendation above).

**Config drift reconciled (doctl = ground truth).** Live prod diverges from the docs on two knobs —
now fixed in CLAUDE.md: **`MIN_EDGE=0.05`** (docs said 0.10) and **`THRESHOLD_MIN_PROBABILITY=0.75`**
(docs said 0.78). All other tracked flags match docs (A1 true, σ-floor true, HARD_LOCK false,
RANGE_UNDERSHOOT false, NEAR_PEAK_FLOOR_UP false, PROBABILITY_MIN_ENTRY_PRICE 0.80).

**SHIPPED this cycle (code/docs; no live-config change needed — A1 + σ-floor already live):**
- Made **A1 epoch-tracked** — added `PROBABILITY_THRESHOLD_NO_REQUIRE_FORECAST` to
  `config_epoch._TRACKED_FLAGS`, so the next scheduler startup writes a ConfigEpoch boundary and
  future flips get `--since-epoch` scoping instead of date-only windowing.
- Reconciled the two **CLAUDE.md config drifts** to the doctl-authoritative values.
- **Cancelled the A3 code removal** (armed feature, see above) — instead, **trimmed KATL** (n=2,
  unverified) from the live `LOCK_BIG_SIZE_STATIONS` whitelist via `doctl` per operator direction
  (now `CYYZ,SBGR,RKSI`); the conviction path stays enabled on the trust-clean stations.

**Next-cycle TODO:**
1. **Fix the report tooling that failed at prod scale** — `calibration-report` + `latency-report`
   were killed 2× under DB contention; `counterfactual-report` hangs (pathological join, 0 bytes at
   30d/60d). Run against a replica / off-peak, or profile+shrink the join. The shadow-retire and
   latency verdicts are blocked on this.
2. **Score A1** once ~n≥10 post-07-07 threshold-NO settle — apply the CONFIRM gate (live-forecast
   subset EV/$ ≥ 0). If the live-forecast subset stays −EV, escalate toward **V1**.
3. **Armed conviction-lock path** — KATL trimmed (done 07-09); whitelist now `CYYZ,SBGR,RKSI`. Open
   question: fully disarm (`LOCK_CONVICTION_SIZING_ENABLED=false`) if EASY-YES locks keep pricing to
   ~1.00 and never fire, since it's a large-size risk surface with no demonstrated benefit.
4. **Evaluate the warm level-correction** (~+0.7°F to `forecast_peak_f`) as the σ-honesty follow-up —
   it's the LEVEL fix the forecast-error data actually supports (the σ-floor was the width fix).

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

---

## §5 — Iteration 2026-07-14: the edge was never firing, and the "cheap" candidates were fiction

**Headline:** `bucket_overshoot` — the project's only edge validated against real executable
prices — had **never fired a single lock signal** since going live on 07-10. The bot had placed
**zero trades since 07-09**. Neither was a filter problem. Two independent root causes, both in
the fast path, plus one methodological trap that nearly produced a bad config change.

### 1. We were losing the latency race to ourselves (FIXED, `1e766c0`)

`bucket_overshoot` EV is **+0.46/$1 at 0s decision delay, +0.28 at 30s, dead by 60s** (§5 07-10).
APScheduler runs `job_fast_lock_poll` with `max_instances=1`, so **a tick that overruns its
interval is DROPPED, not queued** — the configured 10s is a floor, not a guarantee. Prod was
dropping **63% of ticks** (83 skipped / 49 run in 22 min); effective poll period **~27s**.

The tick's *fixed* cost was ~24s, all of it O(markets)/O(stations) round-trips that look
harmless in review:

| Phase | Before | After | Cause |
|---|---|---|---|
| `get_active_weather_markets` | ~8s | 0.24-0.46s | **seq scan of all 55,428 `markets` rows, every 10s** (no index). `EXPLAIN ANALYZE` 3,237ms → **0.97ms** |
| `fetch_latest_metars` | ~16s | 2.2-2.9s | one `INSERT` **per observation** — 48 × ~231ms round-trips to the remote managed PG (~11s) |
| `get_token_ids` | 519ms **per market per tick** | 0 | immutable IDs re-fetched from Gamma every tick, though `job_scan_markets` already persists them |
| **total** | **~24s** | **~2.8s** | budget is 10s |

Post-deploy: drop rate **63% → ~15%**, effective interval ~27s → ~11.8s.

**Invariant added to CLAUDE.md: anything on the fast-poll path must be O(1) network
round-trips, not O(markets) or O(stations).** The canary is
`grep -c 'skipped: maximum number of running instances'` in prod logs — nonzero means you are
silently losing the race.

### 2. The lock path was pricing off STALE Gamma (FIXED, same commit)

`job_unified_pipeline` gated the lock path on `price > 0` but **not** on `token_ids`. When
`get_token_ids` returned None (Gamma rate-limit — frequent, per above), `price` fell back to
`market.current_yes_price`: the Gamma snapshot, already documented as stale in exactly the
post-METAR repricing window this path lives in. The executor then wrote `EvaluationLog` rows at
NO costs **that never existed on the book**, with `buy_depth` hardcoded `0.0` — so the only
thing preventing bets at fictional prices was an accidental `depth $0 < $10` veto.

### 3. THE TRAP — and it nearly cost us

Because Gamma's staleness is **directional** (a dying YES bucket still looks expensive ⇒ NO
looks cheap), those contaminated rows form a *beautiful* fake edge. I found 38 "cheap"
bucket_overshoot candidates blocked on depth; **26 of 26 resolved ones would have won, at an
apparent +0.76/$1**. Every number was true. The conclusion was still wrong: the **rule** was
right (the buckets really did die) but the **prices** were fiction — the live CLOB was at
0.91+, and the real tradeable band is ≥0.96. Proof: an eval logged
`market_prob = 0.7050000000000001` while the Gamma snapshot was `0.2950` (1−0.295 = 0.705000…1)
and `shadow_ledger`'s live quote was 0.91.

**Methodological rules banked:**
- **When a backtest hands you a spectacular edge, verify the PRICE source before the OUTCOME
  source.** Outcomes were impeccable here; prices were garbage.
- **A filter that rejects for reason X may be masking a completely different failure Y.** The
  `depth $0` reject reason described the *symptom* (no token to probe), not the disease (no
  live quote at all). Cf. the 07-09 rule: *a reject-reason cohort names the first gate that
  fired, not the only gate that would.* This is the same family, one level deeper.
- Pre-07-14 lock rows in `evaluation_logs` are **contaminated**. Signature:
  `signal_kind='lock' AND depth_usd IS NULL AND reject_reason LIKE 'depth%'`. **Filter them out
  of any lock-EV analysis** — the nightly analyst would otherwise re-derive the phantom edge.

### 4. Also found, deliberately NOT acted on (see `docs/improvements.md`)

- **Open-Meteo is 429 rate-limited in prod** → mass `has_forecast=False` degenerate states.
  Backlogged: `bucket_overshoot` needs no forecast, and the existing guards fail closed.
- **The probability path is structurally self-blocked**: `PROBABILITY_MIN_ENTRY_PRICE=0.80`
  admits exactly the band `KELLY_PROB_CAP=0.90` sizes to **$0 ("no edge")**. 100% of passing
  probability evals in 9 days died as `stake_below_min`. Left alone on purpose — the model
  beats the market in **0 of 5** calibration buckets, so unblocking it just buys more unproven
  bets. (This supersedes the "dead-zone" note above with a confirmed mechanism.)
- A runaway ad-hoc analysis query (3 backends, **3d16h** old) was pinning the DB buffer pool
  *and* blocking `CREATE INDEX CONCURRENTLY` — leaving the index built but `indisvalid=false`,
  i.e. present in `pg_indexes` and **ignored by the planner**. Terminated; index rebuilt valid.

### Next iteration's gate (do NOT tune anything until this is in)

1. `skipped: maximum number of running instances` count → drive to **0**.
2. Live NO-cost distribution at first fire, **post-07-14 rows only**
   (`signal_kind='lock' AND direction='BUY_NO' AND depth_usd IS NOT NULL`). Hypothesis: with a
   ~2.8s tick we now arrive *before* the collapse and see the <0.93 band for the first time.
3. Real fills + realised PnL on `lock_branch='bucket_overshoot'`.

Only then consider `BUCKET_OVERSHOOT_MAX_COST`. Certainty math: EV/$1 = p/c − 1, so at the
trusted-station violation rate (p≈0.9992) c=0.96 → +4.1%, c=0.98 → +2.0%, c=0.99 → +0.9% — all
positive but thin, and thin margins are where resolver divergence and slippage bite.
**Prefer winning the race over paying up.**

### §5 addendum (same day, post-deploy) — the constraint is BOOK EXISTENCE, not price

Verified against live CLOB books for ZGGG's dead buckets (running max 89.6°F / 32°C):

```
"…Guangzhou be 25°C"   YES book: bids=0  asks=58    NO book: asks=0
"…Guangzhou be 26°C"   YES book: bids=0  asks=59    NO book: asks=0
"…Guangzhou be 27°C"   YES book: bids=0  asks=75    NO book: asks=0
"…Guangzhou be 28°C"   YES book: bids=0  asks=86    NO book: asks=0
```

`evaluate_lock` fires `bucket_overshoot` on all of them (rc=28 — the rule is healthy). But the
YES book has **zero bids** (only holders dumping worthless YES), and by the binary-CLOB mirror
invariant *no YES bids ⇔ no NO asks*: **there is nothing to buy.** Nobody sells you a certain
$1 for less. `get_best_bid_ask` correctly returns None on a one-sided book (`if not bids or not
asks`), so the new live-quote gate skips them — that is not a regression, it is the truth.

**Three consequences, and they redirect the whole strategy:**

1. **This is the final proof the pre-07-14 "cheap" rows were fiction.** The old code computed a
   NO cost of "0.705" from the stale Gamma price for markets whose NO book is *empty*. There
   was never anything to buy at 0.705 — or at any price.
2. **Raising `BUCKET_OVERSHOOT_MAX_COST` is a DEAD END. Do not propose it.** The binding
   constraint is not the price being too high; it is that no offer exists at all. A higher cap
   buys nothing when the book is empty. (Supersedes the "consider raising MAX_COST" line in the
   §5 main entry above.)
3. **Latency is the ONLY lever, and now the *only* remaining one.** The single tradeable moment
   is the ~2-min window after the METAR, while the NO offers are still standing — exactly the
   window the 07-10 study measured (market collapses at median 2.07 min; we saw the data at
   2.70 min). Everything that shortens time-to-decision is +EV; nothing else moves this edge.

**Therefore the next lever is INGESTION lag, not decision lag.** Decision lag is now ~2.8s/tick
(was ~24s). The remaining ~2.7 min is dominated by **METAR publication** (AWC per-station floor:
WSSS 1.1min, OPKC 1.7, VILK 2.8, WMKK 8.7 — §5 07-10). The 07-10 study also found that cutting
ingestion lag to 1 min would **5× the opportunity set (136 → 682 bets)**. A provider racer was
already measured and REJECTED (AWC and NOAA are the same upstream feed). So the open question
is whether *any* faster source of the routine METAR exists (direct station feed / regional
met-service / SYNOP), not whether we poll harder.

### §5 addendum 2 — OPEN RISK: is `bucket_overshoot` executable at all?

The dead-bucket book finding above cuts deeper than "don't raise MAX_COST". It puts a crack in
the study that valued this edge in the first place.

**The 07-10 study priced 3,580 rule-triggered candidates off `clob.polymarket.com/prices-history`
(1-min fidelity) and concluded EV +0.91/$1.** But prices-history reports **trades / mid-prices —
not the existence of a resting NO offer we could actually hit.** A price series can keep printing
0.90 for a bucket whose NO book has **zero asks**. We have now confirmed that dead buckets end up
in exactly that state (ZGGG: YES bids=0, asks=58-86; NO asks=0), and a full scan of every station
returned **66 lock fires / 66 empty books / 0 executable**.

So the thesis has a load-bearing, still-unverified assumption:

> **In the ~2-min window between the METAR and the market's repricing, does the NO book still
> carry resting asks we can lift?**

If YES → the edge is real, latency is the only lever, and the 07-14 fixes (24s → 2.8s tick) are
exactly right. If NO → the edge is a **mirage**: the price we "would have paid" in the backtest
was never buyable, and no amount of speed helps. This would also retro-explain why
`bucket_overshoot` has fired **zero** trades since going live on 07-10 despite being enabled.

**The instrument (shipped `0dead84`):** a structured `lock_unexecutable` event
(`icao, market_id, lock_branch, side, seconds_since_obs, source`) fires whenever a lock fires but
`get_best_bid_ask` returns None. It cannot be an `EvaluationLog` row — `market_prob`/`edge` are
NOT NULL and inventing a price is the exact stale-Gamma bug we just removed.

**How to read it (do this in a few days):**
```bash
doctl apps logs <app> scheduler --type run --tail 5000 | grep lock_unexecutable
# bucket by seconds_since_obs; compare with the executed lock evals
# (evaluation_logs: signal_kind='lock' AND depth_usd IS NOT NULL, created_at > 2026-07-14)
```
- Fresh kills (**low** `seconds_since_obs`) with **live** books ⇒ edge real, keep cutting latency.
- Fresh kills with **empty** books ⇒ **edge is a mirage — stop paying for it** and re-open the
  "is there any edge at all?" fork in the playbook header.

**Do not tune anything until this resolves.** It is now the single highest-value unknown in the
project — it decides whether the one surviving edge exists.

### §5 addendum 3 — RESOLVED: the edge IS priced-tradeable at fresh kill (~13%). Depth is the last unknown.

Addendum 2's "is this a mirage?" fear was based on n=8 and is **REFUTED**. Measured over 20 days
using the bot's OWN live CLOB quotes from `metar_reprice_snapshots` (corr 0.974 with the book —
NOT Gamma), 9,935 dead-bucket observations on trusted stations, dead-ness computed with the real
`market_range_f` + °C/°F step:

| age since METAR | n | mean live NO cost | min | **≤0.93 (tradeable)** |
|---|---|---|---|---|
| **<3 min (fresh kill)** | **184** | **0.9671** | 0.510 | **24 (13.0%)** |
| 3-10 min | 658 | 0.9949 | 0.510 | 8 |
| 10-30 min | 5,089 | 0.9973 | 0.520 | 25 |
| 30-60 min | 3,311 | 0.9918 | 0.370 | 86 |
| >60 min | 693 | 0.9705 | 0.520 | 77 |
| **overall** | **9,935** | 0.9929 | — | 220 (2.2%) |

**The fresh-kill band has the LOWEST mean cost** — the market genuinely has not repriced yet.
So the thesis holds: **the edge is real, it is buyable ~13% of the time, and only inside the
window the 07-14 latency fix made reachable.** The 66/66 empty books seen at 03:55 UTC were all
*long-dead* buckets, correctly skipped — not evidence against the edge.

**New, actionable:** every one of the 24 tradeable fresh kills was a **European** station —
EDDM (Munich ×9), EFHK (Helsinki ×5), EHAM (Amsterdam ×3), LTFM (Istanbul ×2), LIMC (Milan) —
none from the Asian stations that dominate our lock volume. European books are thinner and slower
to reprice. **Before sizing up there, check divergence *variance* per §5 07-10** (mean is not the
risk; EGLL is already excluded for exactly this reason, and EHAM is the σ-collapse station).

**THE LAST UNKNOWN — depth.** A cheap NO *price* is worthless without a resting NO *size*, and
**NO-side depth is recorded nowhere**: `metar_reprice_snapshots.depth_no_usd` is NULL in all
859,455 rows (the writer only passes `depth_yes_usd`), and `shadow_ledger` has no such column at
all (`jobs.py:580` computes `_no_depth_for_market()` for `record_shadow_decision`, which never
persists it). So history cannot answer it.

It resolves itself from here: the executor probes the NO book at the true NO ask for any candidate
that clears the 0.93 cost gate and writes it to `evaluation_logs.depth_usd`. **Read it in a few
days:**
```sql
SELECT round(market_prob::numeric,2) no_cost, depth_usd, passes, reject_reason
FROM evaluation_logs
WHERE signal_kind='lock' AND direction='BUY_NO'
  AND depth_usd IS NOT NULL AND created_at > '2026-07-14'
ORDER BY created_at DESC;
```
- depth ≥ `MIN_DEPTH_USD` on the cheap candidates ⇒ **the edge is real and fillable — scale it.**
- depth ~0 ⇒ the cheap prices are dust quotes; the edge is unfillable and we stop paying for it.

### §5 addendum 4 — the FULL funnel, measured. Depth is real; the DEPTH CAP is the next lever.

The depth question is **answered** (no need to wait days). Live probe of 81 `exactly` markets
sitting in the tradeable NO band (0.05-0.93), NO book probed at its **true ask** (1 - yes_bid):

- **61/81 (75%) carry ≥ $10 of resting NO size.** Median NO depth: **EU $32, ASIA $58**; the top
  of the book reaches $300-780 (ZGGG $784, OPKC $560, RJTT $422, WSSS $407, EPWA $332, EDDM $321).
- **The books are real, not dust. The edge is FILLABLE.**

**But our own sizing throws most of it away.** `size_locked_position` caps a flat lock at
**15% of depth**, then the CAUTION drawdown multiplier halves it, against a `MIN_TRADE_USD=$5`
floor:

```
stake = min(LOCK_POSITION_PCT×bankroll, MAX_POSITION_USD/2, 0.15×depth) × dd_mult
      = min(0.05×$306,            $100,           0.15×depth) × 0.5
⇒ 0.15 × depth × 0.5 ≥ $5   ⇒   depth must be ≥ $67
```

- **35 of the 61 fillable opportunities (57%) size to $0** purely on the depth cap.
- Even on the **$784** book, the stake is **$7.65** — the 5%×$306×0.5 arm binds, not depth.

**End-to-end funnel per dead bucket** (post-latency-fix):
`13% priced ≤0.93` × `75% have depth` × `43% survive the depth cap` ≈ **4%** become a **$7.65** bet.

This is the **same finding as §5 06-03** ("floor-up is inert; the true cause is the DEPTH CAP;
the real lever is raising the cap, not the floor") — now confirmed for the **lock** path with
hard numbers. Note `DEPTH_POSITION_CAP_PCT=0.30` (live) only governs the **probability** path;
the flat lock path uses the hardcoded **0.15** in `size_locked_position`, and
`LOCK_DEPTH_CAP_PCT_BIG=0.50` only applies to *conviction* (EASY-YES, whitelisted) locks —
`bucket_overshoot` is a NO branch and gets neither.

**Candidate levers (PROPOSE-ONLY — operator chose "leave sizing as-is, measure a week"):**
1. **Give `bucket_overshoot` the 0.50 depth cap** (it already exists as `LOCK_DEPTH_CAP_PCT_BIG`).
   Justification: this is a **certainty** bet (0.08% violation rate on trusted stations), so the
   usual reason for a tight depth cap — adverse selection / being wrong about the price — does
   not apply. Walking further into a book whose every level is ≤ `BUCKET_OVERSHOOT_MAX_COST=0.93`
   stays +EV by construction.
2. **Probe depth up to `BUCKET_OVERSHOOT_MAX_COST`, not just at the best ask.** We currently size
   against depth at the *best* NO ask; levels at 0.91/0.92/0.93 are all still +EV and invisible.
3. Clear the CAUTION multiplier (dd 15.2% of a stale $360.66 peak) — but that is a risk decision,
   not a measurement one.

**Do not flip these blind.** The week's data now has the instrument it needs: `depth_no_usd` is
recorded at the kill (fast-poll T0, shipped this session) and `evaluation_logs.depth_usd` carries
the executor's own probe for every candidate clearing the 0.93 gate.

### §5 addendum 5 — two more silent vetoes on the same edge (shipped `ee951c3`)

Chasing "why is the depth cap binding?" surfaced two further throttles, both starving the edge
of *size* rather than blocking it outright.

**1. FLOAT EPSILON — the third member of the depth-probe bug family.**
Callers derive the NO buy price as `1.0 - yes_bid`. In IEEE-754 that lands a hair *below* the
tick the book quotes for ~20% of the 1-cent grid (`1.0 - 0.32 == 0.6799999999999999 < 0.68`).
`_compute_depth` tested `ask_price <= price`, and asks ascend — so excluding the best ask
excludes **every** level and depth returns **exactly 0**. `depth >= MIN_DEPTH_USD` then vetoed
markets with thousands of dollars resting on them.

- Measured live: **36 of 269** tradeable-band markets returned **$0 depth at their own best
  ask** — several holding **$5k-13k**. After the fix (`+ _PRICE_EPS`, 5e-5): **0 of 267**.
- **I had already "refuted" this hypothesis earlier in the session** by testing it against
  `evaluation_logs` — but those were the **stale-Gamma contaminated rows**, synthetic prices
  that never went through a float subtraction. *A hypothesis tested against a corrupted dataset
  is not tested.* Re-run it against live quotes.
- This is the **third** time `_compute_depth` has silently vetoed real liquidity (06-14 mid,
  06-18 mid, 07-14 epsilon). **RULE: treat any `depth == 0.0` as a suspected bug, not a fact.**

**2. `bucket_overshoot` now probes to `BUCKET_OVERSHOOT_MAX_COST`, not the best ask** — and the
FAK's `max_slippage_cents` is widened to the same ceiling so the size we measure is the size we
can take. It is a certainty bet already cost-bounded by the cap, so **every resting level at or
below it is +EV by construction**; sizing against the top of the book alone left **$664,695** of
NO liquidity unreachable across 269 markets. Opportunities able to clear `MIN_TRADE_USD`:
**179/269 → 265/269**.

**Cumulative effect of this session on the edge's size funnel** (bankroll $306, CAUTION 0.5×):

| throttle | before | after |
|---|---|---|
| fast-poll ticks reaching the fresh-kill window | 37% (63% dropped) | ~89% |
| depth probe returns a true value | 87% (36/269 spuriously $0) | 100% |
| liquidity visible to the sizer | best ask only | every level ≤ 0.93 (+$665k) |
| opportunities clearing `MIN_TRADE_USD` | 179/269 | **265/269** |
| lock depth cap | 0.15 (needed depth ≥ $67) | 0.50 |

**Still unmoved:** the stake itself is capped at **$7.65** by `LOCK_POSITION_PCT (5%) × $306 ×
0.5 (CAUTION)` — depth is no longer the binding arm on any book. Raising *that* is a risk
decision (the 15.2% drawdown is real, not a stale peak), so it stays with the operator.

### §5 addendum 6 — first read of the `lock_unexecutable` instrument: thesis INTACT

First 81 events after the fixes (all `source: unified`, i.e. the 5-min tick):

```
branches : bucket_overshoot 60, easy_super 16, easy_standard 5
seconds_since_obs on empty-book locks (n=81):
   min = 458s (7.6 min)   median = 1637s (27 min)   max = 1677s
   FRESH (<180s): 0 / 81
```

**Zero empty-book locks are fresh kills.** Every one is a bucket that died 7.6+ minutes ago
(median 27 min). This is precisely the post-collapse state the thesis predicts — the NO offers
are withdrawn *after* the market reprices — and it is **not** evidence against the edge. The
"is bucket_overshoot a mirage?" question (addendum 2) is now answered from live production
telemetry as well as from history: **no.**

The fast-poll path (the one that reaches the <3-min window) has logged **no** `lock_unexecutable`
yet, i.e. no new routine METAR has killed a bucket since the restart. That is the event to wait
for, and it is now fully instrumented end-to-end:

| stage | where it lands |
|---|---|
| lock fires, book already empty | `lock_unexecutable` log event + `seconds_since_obs` |
| lock fires, live book | `evaluation_logs` row with a real `depth_usd` |
| quote + NO depth at the instant of the kill | `metar_reprice_snapshots` (`depth_no_usd`, fast-poll T0) |
| sized + placed | `decision_logs` + `Trade` |

**The kill criterion is now explicit.** If fast-poll starts logging `lock_unexecutable` with
**low** `seconds_since_obs` (fresh kills that are *already* unbuyable), the edge is dead at any
latency and we stop paying for it. If instead those fresh kills produce `evaluation_logs` rows
with real depth, the edge is real and the only remaining lever is **ingestion lag** (METAR
publication, ~2.7 min — see §5 07-10; a provider racer was already measured and rejected).

### §5 addendum 7 — FIRST fresh-kill datum: the book is DEEP, the PRICE is gone. Station lag is the lever.

The first `metar_reprice_snapshots` row with `depth_no_usd` populated (fast-poll T0 = a lock
fired on a live book):

```
station        : WMKK (Kuala Lumpur)
METAR observed : 05:00:00Z      we saw it 514s later  (8.6 min)
market         : "…highest temperature in Kuala Lumpur…"  [exactly]
live quote     : yes_bid 0.001 / yes_ask 0.002   =>  NO cost 0.999
NO-side DEPTH  : $406.80
```

**Two conclusions, both important:**

1. **The NO book is NOT empty just after a collapse — it holds $407 at 0.999.** The empty-book
   state we kept hitting arrives *much* later (the 81 stale `lock_unexecutable` events had a
   median age of **27 min**). So the binding constraint on a fresh kill is **PRICE, not
   liquidity**: the size is there, we just arrive after the reprice. This refines addendum 3 —
   depth was never the real problem on fresh kills.

2. **8.6 min is exactly WMKK's known METAR publication floor** (§5 07-10: WSSS 1.1 min, OPKC 1.7,
   VILK 2.8, **WMKK 8.7**). The market collapses dead buckets at a median **2.07 min**. On WMKK
   we are ~6.5 min late *before our code runs at all* — **the race is structurally unwinnable
   there at any tick speed.** Our 24s→2.8s tick fix cannot help a station whose data is 8.6 min
   old on arrival.

**⇒ NEXT LEVER: prioritise / gate stations by METAR PUBLICATION LAG, not by divergence alone.**
Decision lag is solved (2.8s tick, 0% dropped ticks). The remaining ~2.7 min is publication, and
it is **per-station**. The winnable set is the low-lag stations (WSSS 1.1, OPKC 1.7, VILK 2.8);
WMKK at 8.7 min is a guaranteed loss that still costs us CLOB calls and log noise.

**Concrete next step (measure first, as always):** we now record `seconds_since_obs` on every
fast-poll T0 snapshot AND on every `lock_unexecutable` event. Build the per-station lag
distribution from `metar_reprice_snapshots.seconds_since_obs`, then:
- compute, per station, `P(NO cost ≤ 0.93 | fresh kill)` against that station's lag;
- expect a sharp cliff around the market's ~2 min reprice time;
- consider a `BUCKET_OVERSHOOT_MIN_LAG_STATIONS` allow-list (or simply exclude stations whose
  median publication lag exceeds ~2-3 min) — the same shape as
  `BUCKET_OVERSHOOT_EXCLUDED_STATIONS`, but gating on **speed** rather than **divergence**.

Do NOT hand-pick from one datum. WMKK is one snapshot; build the distribution first.

### §5 addendum 8 — THE STATION SPEED MAP: only 6 trusted stations can win the race at all

Measured from `metar_observations` (`fetched_at - observed_at`, routine only, 7d, n=50+ per
station). **This is the clean instrument** — an earlier attempt using
`metar_reprice_snapshots.seconds_since_obs` was confounded by the 5-min unified-tick cadence and
gave WSSS 8.5 min. The clean numbers reproduce the known publication floors (§5 07-10: WSSS 1.1,
OPKC 1.7, WMKK 8.7 → measured 1.4 / 1.8 / 8.8), so it is trustworthy.

Market collapses a dead bucket at a median **2.07 min** after the observation. Our ingestion lag
must beat that or the bucket is already priced ~1.00 when we see it.

| median lag | station | status |
|---|---|---|
| **0.5** | SBGR | ✅ winnable |
| **1.0** | OEJN | ✅ winnable |
| 1.1 | MPTO | ⛔ excluded (divergence 30.8%) |
| **1.4** | WSSS | ✅ winnable |
| **1.5** | MMMX | ✅ winnable |
| **1.8** | OPKC | ✅ winnable |
| **1.8** | EFHK | ✅ winnable |
| 1.8 | LLBG | ⛔ excluded (`_EXCLUDED_ICAOS`) |
| 2.7-3.8 | EHAM, VHHH, LFPG, EPWA, RPLL, EDDM | marginal (p10 sometimes beats it) |
| 4.6-8.8 | VILK, EGLL, RKSI, NZWN, LEMD, RKPK, LTFM, ZBAA, LIMC, ZSPD, **ZGGG**, ZGSZ, RCTP, **CYYZ**, **RJTT**, **WMKK** | ❌ structurally LOST |

**TRUSTED + WINNABLE SET (6): `SBGR, OEJN, WSSS, MMMX, OPKC, EFHK`.**

**Why this reframes everything:** our lock volume is dominated by Asian stations — ZGGG (6.0),
RKSI (4.9), RJTT (8.0), ZBAA (5.9), WMKK (8.8) — and **every one of them is structurally lost.**
We arrive 3-7 minutes after the market repriced. The 24s→2.8s tick fix cannot help a station
whose data is already 6 minutes old on arrival. This is why the first fresh-kill datum (WMKK,
8.6 min, NO already at 0.999 with $407 of depth) looked the way it did: **the book was deep, the
price was gone.**

Note `LOCK_BIG_SIZE_STATIONS = CYYZ,SBGR,RKSI` — two of those three (CYYZ 7.9, RKSI 4.9) are in
the structurally-lost set. That whitelist was curated on *divergence*, and for any speed-sensitive
branch it is mostly inert.

**Do NOT rush to gate.** `BUCKET_OVERSHOOT_MAX_COST` already refuses the repriced ones on price,
so hopeless stations cost only a few CLOB calls, not EV — and several "lost" stations have a p10
under 2 min (UUEE 1.2, RPLL 1.2, VHHH 1.8), i.e. a real fast tail we'd throw away with a
median-based exclusion. What this map *does* change:
1. **Calibrate expectations.** Realistic fill flow comes from **6 stations**, not 45. Judge the
   coming week's data against that denominator, not the whole universe.
2. **Where to look.** Any station-level analysis of the edge should start with those six.
3. **The real lever is now unambiguous: cut INGESTION lag** (§5 07-10 estimated 5× the
   opportunity set at 1-min ingestion). A multi-provider racer was already measured and rejected
   (AWC and NOAA share an upstream feed), so the open question is whether a genuinely different
   source exists for the slow stations — regional met service, SYNOP, direct station feed.

### §5 addendum 9 — the instrument is now DURABLE (a log line is not an instrument)

`lock_unexecutable` was shipped as a **log event** — and prod log volume rotates the tail within
minutes. Two consecutive `doctl apps logs --tail 3000` calls returned **disjoint** windows: one
showed 81 events, the next showed zero. The single most important number in the project was being
written to a buffer that throws it away, and the nightly analyst reading it days later would have
found nothing.

Fixed (`34a642a`): the fast-poll fresh-kill path now also writes a `metar_reprice_snapshots` row
with **`yes_bid IS NULL`** — which reads exactly as *"a lock fired here and the book was empty"* —
carrying station, market, `metar_observed_at` and `seconds_since_obs`. No migration (all those
columns were already nullable).

**THE KILL CRITERION — run this in a few days:**

```sql
-- Empty-book locks: were any of them FRESH kills on a FAST station?
SELECT s.station_icao,
       count(*)                                             AS empty_book_locks,
       count(*) FILTER (WHERE s.seconds_since_obs < 180)    AS fresh_and_unbuyable,
       round(min(s.seconds_since_obs)/60.0, 1)              AS min_age_min
FROM metar_reprice_snapshots s
WHERE s.yes_bid IS NULL                 -- the empty-book marker
  AND s.created_at > '2026-07-14'
GROUP BY 1
ORDER BY 3 DESC;
```

- `fresh_and_unbuyable > 0` **on a fast station** (`SBGR, OEJN, WSSS, MMMX, OPKC, EFHK` — see
  addendum 8) ⇒ the bucket is *already unbuyable within 3 minutes* even where we beat the reprice
  ⇒ **the edge is a mirage at any latency. Stop paying for it.**
- All-stale (as observed so far: **0 of 81 fresh, min age 458s**) ⇒ thesis intact; the empty book
  is just the post-collapse state, and the lever remains **ingestion lag**.

And the positive side of the same question:

```sql
-- Fresh kills that DID have a live book: the actual opportunity set.
SELECT station_icao, round(seconds_since_obs) age_s,
       round((1 - yes_bid)::numeric, 3) AS no_cost, depth_no_usd
FROM metar_reprice_snapshots
WHERE yes_bid IS NOT NULL AND depth_no_usd IS NOT NULL   -- fast-poll T0 rows
  AND created_at > '2026-07-14'
ORDER BY created_at DESC;
```
`no_cost <= 0.93` AND `depth_no_usd >= 10` on a fast station = a bet we should have taken.
Cross-check against `evaluation_logs` (`signal_kind='lock'`, `depth_usd IS NOT NULL`) and `trades`.

### §5 addendum 10 — END-TO-END validation against a live CLOB book (no order, no DB write)

Drove the **real** `try_lock_rule_trade` against a **real** live orderbook, with only
`place_order` / persistence / alerter mocked and a synthetic dead-bucket state:

```
market    : "…highest temperature in Jeddah be 36°C…"   station OEJN (winnable, 1.0min lag)
live book : yes_bid=0.09  yes_ask=0.12   =>  NO cost 0.910
eval      : passes=True   reject=None
DEPTH     : $62.00        <- REAL CLOB probe, epsilon-fixed, measured up to the 0.93 cap
ORDER     : stake $7.65   entry 0.910
FAK limit : 0.910 + 2.0c = 0.9300      <- exactly at BUCKET_OVERSHOOT_MAX_COST, no breach
```

**All five fixes compose.** The stake is bound by the **bankroll arm**
(`LOCK_POSITION_PCT 5% × $306 × 0.5 CAUTION = $7.65`), *not* by depth — exactly as the funnel
predicted once the depth cap moved 0.15 → 0.50. The pipeline will place a bet the moment a fresh
kill lands on a live book.

This is a harness test (the bucket was forced dead on a market that is still live), so it proves
the **code path**, not the edge. The edge itself is still gated on the natural event — see the
durable kill-criterion queries in addendum 9.

### §5 addendum 11 — ⚠️ CORRECTION: my kill criterion was BROKEN, and the corrected one is flashing red

**The criterion in addendum 9 is INVALID. Do not use it. The "0 of 81 fresh → thesis intact"
verdict in addendum 6 is therefore also unearned — it was measuring the wrong thing.**

**The bug:** `seconds_since_obs` is the age of the **latest METAR**, not the age of the **KILL**.
`bucket_overshoot` fires on *every* bucket below the running max, on *every* new METAR, for the
rest of the day — a bucket that died at 09:00 still re-fires at 15:00 with a "fresh" 60-second
METAR age. So a `lock_unexecutable` row with `seconds_since_obs = 55s` is almost always a
**long-dead bucket re-firing**, whose empty book is entirely expected and says nothing.

Confirmed: of 17 "fresh (<3min)" empty-book locks, **14 had an overshoot of 2.4-12.6°F past the
dead threshold** — i.e. the max crossed those buckets hours ago.

**CORRECTED CRITERION — discriminate on OVERSHOOT, not METAR age:**

```sql
-- overshoot = observed_max - (bucket_top + step)   [step = 1.8°F for °C markets, 1.0°F for °F]
-- overshoot <= ~1.8°F (one METAR heating step) => the max JUST crossed => a TRUE fresh kill.
-- Compute in Python with execution.binary_market.market_range_f + market_unit;
-- SQL alone cannot parse the bucket. See scratchpad/isfresh.py.
SELECT s.station_icao, s.seconds_since_obs, s.new_observed_max_f, m.question
FROM metar_reprice_snapshots s JOIN markets m ON m.id = s.market_id
WHERE s.yes_bid IS NULL              -- empty book
  AND s.created_at > '2026-07-14'
  AND s.seconds_since_obs < 180;     -- necessary, NOT sufficient — then filter on overshoot
```

**FIRST CORRECTED READ — and it is bad news:** 3 of the 17 were TRUE fresh kills
(overshoot ≤ 1.8°F). **All three had an EMPTY book:**

| station | METAR age | overshoot | book |
|---|---|---|---|
| **WSSS** (FAST, 1.4min median lag) | **1.3 min** | 0.6°F — just crossed | **EMPTY** |
| RPLL | 0.9 min | 0.6°F | **EMPTY** |
| VILK | 2.8 min | 1.8°F | **EMPTY** |

**WSSS is the damning one.** We had the METAR **1.3 min** after observation — inside the 2.07-min
reprice window, i.e. *we won the race* — the bucket had **just** died, and there was **nothing to
buy.**

**NEW LEADING HYPOTHESIS — the market FRONT-RUNS the kill.** Traders do not need the killing
METAR. Once the temperature is visibly climbing toward a bucket, they withdraw their NO offers on
it *before* it is mathematically dead. If that is what is happening, the offers are gone by the
time our certainty condition triggers, and **no amount of speed helps — the edge does not exist**,
and the §5 07-10 study's "~19% still priced NO < 0.99" was an artifact of pricing off **stale
Gamma** (see the contamination gotcha), not a real executable opportunity.

**n = 3. Do NOT act on this yet.** But it is now the leading hypothesis, it directly contradicts
the thesis the whole `bucket_overshoot` programme rests on, and it is cheap to settle:

1. Let `metar_reprice_snapshots` accumulate (writer is live and durable).
2. Re-run the **overshoot-filtered** query in a few days.
3. **Decision rule:** if TRUE fresh kills (overshoot ≤ 1.8°F) on FAST stations
   (`SBGR/OEJN/WSSS/MMMX/OPKC/EFHK`) keep showing an empty book at n ≥ 15-20, **the edge is a
   mirage — kill `BUCKET_OVERSHOOT_LOCK_ENABLED` and stop spending on latency.** If instead some
   show a live book at NO ≤ 0.93, the edge is real and rare, and ingestion lag remains the lever.

**Meta-lesson, third time this session:** *verify the instrument before believing the
measurement.* I built the kill criterion, it returned a clean "thesis intact", and it was
measuring the wrong variable. Same family as the `yaml.safe_load` key artifact and the epsilon
hypothesis "refuted" against stale-Gamma rows.

### §5 addendum 12 — ✅ DEFINITIVE: the edge is REAL and speed is worth 5×. (Addendum 11's alarm was noise.)

Addendum 11 raised the alarm on **n=3**. That was premature. Redone properly on 20 days, with the
**corrected** fresh-kill discriminator (overshoot ≤ 1.8°F, *not* METAR age) **cross-tabbed against
quote freshness** — because a "fresh kill" observed through a 5-min unified tick is priced with a
5-min-old quote, which is exactly the confound that produced the wrong answer twice:

| quote age at a **TRUE** fresh kill | n | empty book | mean NO cost | **buyable ≤ 0.93** |
|---|---|---|---|---|
| **< 3 min — what the latency fix now gives us** | 115 | 3 (2.6%) | **0.974** | **13 → 11.3%** |
| 3-10 min | 379 | 4 | 0.992 | 8 → 2.1% |
| > 10 min | 4,731 | 0 | 0.993 | 97 → 2.1% |

**1. The market does NOT front-run the kill.** Only **2.6%** of true fresh kills have an empty
book (and 0.13% across all 5,225 true fresh kills). The 3 empty books in addendum 11 were an
unlucky micro-sample. **Hypothesis refuted.**

**2. The edge is REAL.** With a fresh kill *and* a fresh quote, **11.3%** of dead buckets are
buyable at ≤ 0.93, at a mean NO cost of **0.974**.

**3. Speed is worth 5×.** 11.3% buyable inside 3 minutes vs **2.1%** after. This is the first
uncontaminated measurement of the core thesis, and it says the entire 2026-07-14 latency programme
(tick 24s → 2.8s, dropped ticks 63% → 0%) buys a **5× larger opportunity set** — and that the
station speed map (addendum 8) is the right next lever, because only 6 stations can put us in the
<3 min cell at all.

**Superseded:** the earlier "13% buyable at fresh kill" (addendum 3) split on METAR age and was
wrong-by-luck; the corrected figure for the same cell is **11.3%** — materially the same
conclusion, now for the right reason. The "2.3% overall" figure is the *unconditional* rate across
all fresh kills regardless of quote age, and it is the number to beat.

**I was wrong twice today on this exact question** — once reassuring ("0/81 fresh, thesis intact",
on a criterion that measured the wrong variable), once alarming ("mirage", on n=3). Both errors had
the same root: **believing an instrument I had not verified.** The rule stands: *verify the
measuring device before acting on the measurement* — and when a result is decision-changing, find
the confound before you write the conclusion.

### §5 addendum 13 — INGESTION LEVER CLOSED: no provider beats AWC. The universe IS the 6 fast stations.

Addendum 8 named ingestion lag as the next lever and asked whether a *genuinely different* METAR
source exists for the slow stations. **Answer: no — not among the six providers in this codebase.**

**Experiment (2026-07-14, live):** watched a real METAR cycle and recorded when each provider
FIRST served the new 06:30Z observation.

| station | AWC (current) | NOAA | OGIMET |
|---|---|---|---|
| RJTT | **12.7 min** | 13.1 min | not within window |
| WMKK | **13.1 min** | 13.1 min | not within window |

They arrive **essentially simultaneously** (AWC ahead by 0.4 min), and a snapshot probe showed
AWC / NOAA / OGIMET returning **identical `observed_at`** for every station. This confirms
empirically what §5 07-10 asserted: they share an upstream feed. IEM was 30-60 min *behind*
throughout.

**And the publication lag itself is 8-13 min for these stations** — against a **2.07 min** market
reprice. The information does not exist anywhere in time. **No amount of engineering on our side
fixes that.** Switching providers, adding a racer, or polling harder are all dead ends.

**⇒ THE UNIVERSE IS THE 6 FAST STATIONS: `SBGR, OEJN, WSSS, MMMX, OPKC, EFHK`.** That is not a
limitation to be engineered around — it is the shape of the opportunity. Judge all future
`bucket_overshoot` volume, fill-rate and PnL against that denominator. The remaining upside is
(a) making sure we win *every* race on those six (decision lag is now 2.8s, so we do), and
(b) sizing those wins properly (done: `BUCKET_OVERSHOOT_DEPTH_CAP_PCT`).

**Bonus catch — a data-integrity bug found by this experiment.** OGIMET served `METAR OEJN NIL`
(station published nothing) and `parse_raw_metar` **fabricated** `observed_at = now()` for it,
making it look like a 0.0-minute-old observation — briefly appearing to be a provider beating AWC
by 25 minutes. `observed_at` is load-bearing for this entire edge (`seconds_since_obs`, kill
freshness, and fast-poll's `_last_routine_seen` watermark), and the OGIMET parser sorts by
`observed_at` DESC — so the phantom sorted to the FRONT as "the latest METAR" and would have
pushed the dedup watermark to the present, **suppressing every genuinely new METAR behind it.**
Fixed (`ee4ae49`): a report with no DDHHMMZ group is NIL or malformed — drop it, never stamp it.

**That is the FOURTH instrument artifact this session** (yaml-hex key, stale-Gamma prices, the
METAR-age kill criterion, and now a fabricated observation time). Every one of them produced a
confident, plausible, wrong conclusion. The discipline that caught all four was the same:
**look at the raw input before believing the derived number.**
