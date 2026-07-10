from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    DATABASE_URL: str = ""

    @field_validator("DATABASE_URL")
    @classmethod
    def normalize_db_url(cls, v: str) -> str:
        if v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        # asyncpg does not understand ?sslmode=; replace with ?ssl=
        v = v.replace("sslmode=", "ssl=")
        return v

    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    # Edge threshold gate in `_check_filters`. The historical hardcode in
    # `edge_calculator.py` was 0.05 and was the value actually applied
    # for years; the older `MIN_EDGE=0.10` default here was unused. Default
    # restored to 0.05 to match observed paper-trade behavior; tune after
    # calibration backtest is wired (see Phase 1 in the revenue plan).
    MIN_EDGE: float = 0.05
    KELLY_FRACTION: float = 0.25
    MAX_POSITION_PCT: float = 0.05
    # Hard ceiling on the probability passed into the Kelly formula.
    # Defensive bound against probability-engine overconfidence: at p=1.0
    # Kelly bets 100% of bankroll; at p=0.90 it bets a reasonable
    # fraction. Recorded model_prob is untouched — cap is sizing-only.
    # See `src/risk/kelly.py::size_position`.
    KELLY_PROB_CAP: float = 0.90
    INITIAL_BANKROLL: float = 750.0
    AWC_USER_AGENT: str = "WeatherEdge/1.0 (weather-trading-bot; contact@example.com)"
    AWC_RATE_LIMIT_RPS: float = 2.0

    # Multi-provider aviation API keys (empty = provider disabled)
    CHECKWX_API_KEY: str = ""
    AVWX_API_KEY: str = ""

    # Polymarket execution
    POLYMARKET_PRIVATE_KEY: str = ""  # Polygon wallet private key; empty = dry-run
    POLYMARKET_HOST: str = "https://clob.polymarket.com"
    POLYMARKET_CHAIN_ID: int = 137  # Polygon mainnet
    # 0=EOA (raw wallet), 1=Polymarket-Proxy, 2=Gnosis-safe-style proxy.
    # If your wallet has ever logged into polymarket.com, the API expects
    # signature_type=2 and POLYMARKET_FUNDER_ADDRESS set to the proxy
    # address (not the EOA). Leaving these at their defaults works only
    # for fresh wallets that have never touched the Polymarket UI.
    POLYMARKET_SIGNATURE_TYPE: int = 0
    POLYMARKET_FUNDER_ADDRESS: str = ""  # Empty = derive EOA from private key
    # Polymarket migrated to new exchange contracts (collateral=pUSD) in 2026.
    AUTO_EXECUTE: bool = False  # Set True to place orders automatically
    DAILY_SPEND_CAP_USD: float = 200.0  # Max total spend per 24h
    MIN_STAKE_USD: float = 5.0  # Skip orders below this amount

    # Near-peak floor-up (2026-05-28). At a small bankroll, fractional Kelly
    # (further capped by KELLY_PROB_CAP) sizes high-confidence near-peak bets
    # below MIN_STAKE_USD, so they're dropped despite passing every filter —
    # the dominant volume throttle once drawdown is healthy. When enabled, a
    # passing edge that is (a) a threshold operator (NOT exactly/range/bracket
    # — those stay validate-first) and (b) high-confidence OR near the forecast
    # peak gets its stake floored up instead of dropped. The floor is applied
    # *inside* the sizer (after every exposure/USD/depth cap) so it can never
    # breach a cap, and it is *pre-drawdown-multiplier* (CAUTION/RECOVERY 0.5×
    # still applies on top — set the floor to 2× MIN_STAKE_USD if you want it
    # to survive half-sizing). Default disabled = no behavior change.
    NEAR_PEAK_FLOOR_UP_ENABLED: bool = False
    NEAR_PEAK_FLOOR_STAKE_USD: float = 5.0  # floor target; doubles as the "bigger stake" knob
    NEAR_PEAK_FLOOR_UP_MIN_PROB: float = 0.97  # confidence arm (recorded prob, not Kelly-capped)
    NEAR_PEAK_FLOOR_UP_MAX_HOURS: float = 2.0  # near-peak arm: |hours_until_peak| ≤ this

    # Near-lock conviction sizing (probability path). KELLY_PROB_CAP=0.90 clamps
    # the model prob before Kelly, so a bet bought above 0.90 looks negative-edge
    # → sized to $0 ("no edge"). That structurally forbids the near-lock band the
    # PROBABILITY_MIN_ENTRY_PRICE=0.80 floor is meant to harvest (prices 0.90–0.97).
    # When ENABLED, a passing **threshold** bet that is a GENUINE observational
    # monotonic lock — has_forecast AND the target-day-anchored observed max has
    # already won the bet (max only rises) by NEAR_LOCK_CONVICTION_MARGIN_F — sizes
    # against PROB_CAP (0.99) instead of KELLY_PROB_CAP. The whole cap cascade
    # (per-trade MAX_POSITION_PCT, exposure, USD, depth) still bounds it. Restricted
    # to the "already hot, betting hot" direction (YES on at_least/above, NO on
    # at_most/below); the forecast direction is excluded — see
    # near_lock_conviction_eligible. Rebuilt 2026-06-21 after the original
    # prob/hours proxy fired 96% on degenerate forecast-failed states (the lock
    # path correctly refuses those). Default off — re-enable only after validation.
    # GRAVEYARD: NEAR_LOCK_CONVICTION_SIZING_ENABLED — docs/graveyard.md#near_lock_conviction_sizing_enabled--prob-cap-bypass-for-near-lock-threshold-bets (status: suspended)
    NEAR_LOCK_CONVICTION_SIZING_ENABLED: bool = False
    NEAR_LOCK_CONVICTION_PROB_CAP: float = 0.99  # Kelly prob cap for eligible bets
    NEAR_LOCK_CONVICTION_MARGIN_F: float = 2.0  # observed max must clear threshold by this (hedges resolver divergence)

    # Unified pipeline
    UNIFIED_PIPELINE_INTERVAL_MINUTES: int = 5

    # Trade filters
    # Side-effective probability floor. Raised from 0.50 → 0.85 after the
    # 2026-05-08 calibration audit: live data showed the 0.6-0.85 model_prob
    # band realizing ~25-50% win rate (vs ~80% predicted) on bracket
    # markets, the dominant bleed. Combined with APPLY_CALIBRATION the
    # raw 0.99 predictions are corrected down toward true ~0.85 and
    # gated here, dropping marginal NO bets at the modal bucket where
    # sigma overconfidence had been inventing edge.
    MIN_PROBABILITY: float = 0.85
    MIN_ENTRY_PRICE: float = 0.40
    MAX_ENTRY_PRICE: float = 0.97
    # Probability-path-only entry-price floor (None = use the global
    # MIN_ENTRY_PRICE). Realized June book splits sharply by the price of the
    # side bought: everything < ~0.80 lost (−$42 / 26 trades), the 0.80–1.00
    # near-lock band won 94% (+$17 / 18). Setting this to 0.80 concentrates the
    # probability path on the band where it demonstrably wins. Scoped so the
    # LOCK path keeps firing cheap-but-certain bets (it has its own
    # LOCK_RULE_MIN_PRICE gate and never wants a high floor). None = no change.
    PROBABILITY_MIN_ENTRY_PRICE: float | None = None
    # Require a live forecast (`state.has_forecast`) for probability-path
    # above/at_least BUY_NO ("forecast-cool NO"). Default False preserves
    # behavior; the live env sets True. Read live off `settings`.
    #
    # Why (2026-07-07 mastering audit, workflow wf_beb30c80-6a3): the remaining
    # post-HARD-lock-kill drag is the probability-path above/at_least BUY_NO
    # cohort — the SAME forecast-cool bet as the killed HARD-lock, still firing
    # because the probability path lacks the `has_forecast` guard the HARD-lock
    # path enforces. 3 of the 4 big losses (Sao Paulo −$6.92, Moscow −$6.20,
    # Shanghai −$27.52) are degenerate Open-Meteo-FAILED states (has_forecast
    # False → forecast_peak_f == current_max_f, σ NULL) where the model prob
    # collapses to ~1.0 and Kelly sizes the max — betting a forecast signal when
    # there is no forecast. σ-NULL subset −$14.61/60d incl. the −$27.52 tail;
    # the live-forecast subset is ~flat. Same class as the conviction-gate
    # degenerate-state bug (see memory project_conviction_degenerate_state_bug_2026-06-21).
    # NOT "require past-peak" — the data shows past-peak is the *losing* subset.
    PROBABILITY_THRESHOLD_NO_REQUIRE_FORECAST: bool = False
    # Operator-aware overrides for the two filters that were globally
    # tightened during the 2026-05-08 bracket-overconfidence crisis
    # (MIN_PROBABILITY 0.50→0.85, MIN_EDGE→0.10). Threshold markets
    # (above/at_least/below/at_most) were never the source of that bleed —
    # all the 2026-05-22/23 quality cuts target bracket-like ops — yet they
    # inherit the strict global floors. When set, `binary_market_edge`
    # applies these to threshold ops ONLY; bracket-like ops keep the strict
    # global MIN_PROBABILITY / MIN_EDGE plus the three single-bucket NO
    # guards. `None` (default) = no override = current behavior, so deploying
    # is a no-op until a value is set via .env after telemetry validation
    # (see evals-report --operator threshold recoverable-band analysis).
    THRESHOLD_MIN_PROBABILITY: float | None = None
    THRESHOLD_MIN_EDGE: float | None = None
    MIN_DEPTH_USD: float = 10.0
    MIN_ROUTINE_COUNT: int = 3
    MARKET_CLOSE_BUFFER_MINUTES: int = 30
    MAX_POSITION_USD: float = 200.0
    DEPTH_POSITION_CAP_PCT: float = 0.20

    # Absolute USD floor on the per-tick exposure cap. The exposure cap
    # math in ``src/risk/kelly.py`` is ``max(MAX_EXPOSURE_PCT × bankroll,
    # MAX_EXPOSURE_USD_FLOOR)``. At small bankroll the percent cap binds
    # too tightly (e.g. $441 × 25% = $110 = only ~5 simultaneous $22
    # positions), so a single batch of stuck OPEN trades silences the
    # bot (incident 2026-05-17). The floor "fires" only when bankroll
    # < ``floor / MAX_EXPOSURE_PCT`` (≈ $1200 at the defaults); above
    # that the percent cap binds again. Per-trade caps
    # (``MAX_POSITION_PCT``, ``MAX_POSITION_USD``) and the drawdown
    # state machine still gate runaway risk.
    MAX_EXPOSURE_USD_FLOOR: float = 300.0

    # Same-day same-city bracket/exactly cluster total stake cap. Outcomes
    # across buckets of one bracket are anti-correlated — only one bucket
    # can win — so each bucket's Kelly stake is mis-priced as if
    # independent. The cap (default $100, half of MAX_POSITION_USD) sums
    # already-staked open/pending stakes on the same parsed_location +
    # target local day before sizing a new bucket, refusing the new bet
    # when the cluster total would exceed the cap. Set to 0.0 to disable.
    CLUSTER_STAKE_CAP_USD: float = 100.0

    # Bracket markets (multi-bucket "between X-Y°F on day D" questions)
    # are gated off after live data showed -21pp model overconfidence
    # vs realised win rate, leaving the strategy at -1.5% breakeven
    # margin. Exactly/threshold markets are unaffected. Re-enable via
    # `.env` once sigma + calibration changes have ≥2 weeks of paper-
    # trade evidence behind them.
    # GRAVEYARD: BRACKET_MARKETS_ENABLED — docs/graveyard.md#bracket_markets_enabled--evaluate-bracketrange-multi-outcome-markets (status: suspended)
    BRACKET_MARKETS_ENABLED: bool = False

    # Station bias tracking
    DEFAULT_STATION_BIAS_C: float = 1.0
    STATION_BIAS_WINDOW_DAYS: int = 30
    STATION_BIAS_MAX_C: float = 3.0

    # Linear calibration. When True, the unified pipeline refreshes a
    # linear (slope, intercept) fit from resolved signals every tick and
    # applies it to the chosen side's probability before edge filtering.
    # Flipped True by default 2026-05-08 after live data showed +10-21pp
    # overconfidence on bracket/exactly markets. The fit needs at least
    # `MIN_CALIBRATION_SAMPLES=50` resolved trades; below that, callers
    # see the raw probability unchanged. See `src/signals/calibration.py`.
    APPLY_CALIBRATION: bool = True
    # Per-operator-class calibration split (added 2026-05-30). When True,
    # ``apply_calibration`` looks up class-specific (slope, intercept) for
    # ``"threshold"`` vs ``"bracket-like"`` ops, falling back to the pooled
    # fit when the class lacks ``MIN_CALIBRATION_SAMPLES`` resolved trades
    # of its own. When False (default), all calls use the pooled fit —
    # bit-for-bit identical to pre-2026-05-30 behavior.
    #
    # Motivation: the global pooled fit is dominated by 0.95+-confidence
    # threshold rows, so the linear slope pulls 0.78-0.85 raw predictions
    # toward ≥0.85 — directly squashing the loosened
    # ``THRESHOLD_MIN_PROBABILITY=0.78`` band (0 fills observed in
    # ``signals.model_prob ∈ [0.78, 0.85)`` over the last 7+ days). Per-
    # class fitting lets the threshold band train on its own win-rate curve
    # without bracket-like's 0.99-dominated tail distorting it.
    # GRAVEYARD: PER_OPERATOR_CALIBRATION_ENABLED — docs/graveyard.md#per_operator_calibration_enabled--split-calibration-fit-by-operator-class (status: suspended)
    PER_OPERATOR_CALIBRATION_ENABLED: bool = False

    # Shadow calibration telemetry (M1 / Phase 1, 2026-05-31). Pure
    # telemetry — CANNOT affect any trade. When True (default), every
    # probability-path eval stamps ``evaluation_logs.shadow_json`` with
    # both the pooled and the per-class calibrated probability for that
    # raw input, so ``shadow-report`` can show whether flipping
    # ``PER_OPERATOR_CALIBRATION_ENABLED`` would actually un-squash the
    # threshold [0.78,0.85) band BEFORE the flip touches live trading.
    # This is the "measure before flip" gate for the calibration split.
    SHADOW_CALIBRATION_ENABLED: bool = True

    # Shadow lead-time-σ telemetry (Phase 2, 2026-05-31). Pure telemetry —
    # CANNOT affect any trade. When True (default), every ensemble-branch
    # probability-path eval stamps ``evaluation_logs.shadow_json.sigma``
    # with the live σ and the σ the lead-time arm WOULD produce, so
    # ``shadow-report --key sigma`` can size the effect of flipping
    # ``SIGMA_FLOOR_LEAD_TIME_ENABLED`` across the full lead range that live
    # evals span (backtest-v2 snapshots mid-day and under-samples the tail).
    SHADOW_SIGMA_LEADTIME_ENABLED: bool = True

    # Automated performance review (Layer 1 of the auto-evaluation loop).
    # When True, setup_scheduler registers daily / 3-day / 7-day jobs that
    # compute a perf digest, persist a JSON artifact for the Layer-2 analyst,
    # and push a Telegram summary (throttled per cadence via bot_state).
    # Propose-only — the jobs never change config. Default off so a deploy
    # doesn't start pushing until the operator opts in.
    PERF_REVIEW_ENABLED: bool = False

    # Daily "why no trade" funnel digest (universe → evaluated → passed →
    # traded). Propose-only; default off until the operator opts in.
    NO_TRADE_REVIEW_ENABLED: bool = False

    # Information-latency measurement (Phase 2, 2026-06-24). When True, the
    # scheduler snapshots the market YES quote + depth the instant a new routine
    # METAR is detected (fast-poll T0) and on the following unified ticks
    # (T+5/T+10), keyed to the METAR's observed_at, into ``metar_reprice_snapshots``.
    # Diffing yes_mid across those rows measures how fast the market reprices
    # toward information we already hold — the realized latency edge. Best-effort,
    # places no orders, never touches the live paths. Default off because it adds
    # write volume; flip after `alembic upgrade head`. Retention is capped by
    # REPRICE_SNAPSHOT_RETENTION_DAYS at daily settlement.
    REPRICE_SNAPSHOT_ENABLED: bool = False
    REPRICE_SNAPSHOT_RETENTION_DAYS: int = 30

    # Circuit breakers
    DAILY_LOSS_STOP_USD: float = 200.0
    CONSECUTIVE_LOSS_PAUSE_COUNT: int = 3
    CONSECUTIVE_LOSS_PAUSE_HOURS: int = 2

    # Drawdown monitor thresholds (fraction of peak). Overridable so a small
    # iteration-mode bankroll isn't perma-PAUSED by a stale daily peak. Defaults
    # match the long-standing module constants in src/risk/drawdown.py — leave
    # unset for identical behavior. CAUTION → 0.5× size, PAUSE → 0.0× (no trades).
    DRAWDOWN_CAUTION_THRESHOLD: float = 0.10
    DRAWDOWN_PAUSE_THRESHOLD: float = 0.20

    # Submission-failure circuit breaker (2026-05-20). When the CLOB
    # rejects ``N`` orders in a row within the recent window, pause new
    # submissions for ``SUBMIT_FAIL_PAUSE_MINUTES`` and Telegram-alert.
    # Prevents the May 2026 failure mode: 115 PolyApiException rows in
    # one week silently piling up while the bot kept retrying every
    # 5-min tick, generating decision-log noise + phantom-PENDING rows
    # that later got swept LOST. Detected via ``trades.exchange_status
    # LIKE 'exception:%'`` over the recent window.
    SUBMIT_FAIL_PAUSE_COUNT: int = 5
    SUBMIT_FAIL_PAUSE_WINDOW_MINUTES: int = 10
    SUBMIT_FAIL_PAUSE_MINUTES: int = 30

    # Fast-poll lock loop. Runs between unified-pipeline ticks, re-checking
    # only the EASY lock direction (observed max already clears threshold by
    # margin) so latency from METAR publication to order placement is seconds
    # rather than up to 5 minutes.
    FAST_LOCK_POLL_ENABLED: bool = True
    # 10 s, not 30 s. The `bucket_overshoot` edge is a seconds-scale race: the
    # market collapses a dead bucket a median 2.07 min after the METAR's
    # observation time, and realized EV falls from +0.46/$1 at zero decision
    # delay to +0.28 at 30 s and to noise (+0.10, CI spans 0) at 60 s. The poll
    # interval is pure additive latency on top of publication.
    #
    # Measured 2026-07-10 (10 s polling across an issue boundary): AWC and NOAA
    # are the SAME upstream feed — identical first-appearance second on 13 of 14
    # observations — so racing providers buys nothing. Publication itself is the
    # floor and it is per-station (WSSS 1.1 min, OPKC 1.7, VILK 2.8, RKSI 5.0,
    # WMKK 8.7). Our only controllable slack is this interval.
    #
    # Cost: one bulk AWC call per tick over the active-bucket ICAOs (batched 20
    # per request), i.e. ~6 calls/min. `_throttle()` in `ingestion.aviation`
    # still bounds concurrency.
    FAST_LOCK_POLL_INTERVAL_SECONDS: int = 10

    # Order reconciliation job — polls PENDING/OPEN trades whose
    # `fill_price` is still NULL (delayed orders that posted to the CLOB
    # but the matching engine hasn't filled yet) and updates fill data
    # from the live order endpoint. Without this, slippage analytics
    # remain blind. Disable by setting to 0.
    ORDER_RECONCILE_INTERVAL_MINUTES: int = 5
    ORDER_RECONCILE_LOOKBACK_HOURS: int = 24

    # ``resolve_trades`` catch-22 grace window. When ``_refresh_market_price``
    # returns None for an OPEN trade whose market ended more than this many
    # hours ago, fall back to: (a) mark LOST if ``fill_price IS NULL`` (the
    # order never landed on-chain — no real position), (b) leave OPEN with a
    # warning if ``fill_price`` is populated (operator should run
    # ``admin reconcile-stuck`` for on-chain payout settlement). Without
    # this fallback, the CLOB drops resolved markets and trades stay OPEN
    # forever, pinning the ``MAX_EXPOSURE_PCT`` cap (incident 2026-05-17).
    RESOLVE_NO_PRICE_GRACE_HOURS: int = 4

    # Stuck-OPEN sweep + heartbeat in ``job_reconcile_orders``.
    #
    # When ``check_order_status`` keeps returning None for a trade row
    # (CLOB endpoint dropped the order after market resolution; see the
    # docstring in ``polymarket_client.check_order_status``), the row
    # stays ``status=OPEN`` with ``fill_price IS NULL`` indefinitely,
    # pinning the ``max(MAX_EXPOSURE_PCT × bankroll,
    # MAX_EXPOSURE_USD_FLOOR)`` cap. Incident 2026-05-18 silenced the
    # bot for 12+ hours because the 22:00-UTC daily-settlement
    # 80%-of-cap alert is too coarse to catch the failure inside the
    # 24h window.
    #
    # The 5-minute reconcile job logs a high-signal warning when the
    # stuck-trade set is non-empty past this grace, and pushes a
    # Telegram alert when either the count threshold or the relative
    # exposure-fraction threshold trips. ``BotState`` row keyed
    # ``reconcile.stuck_alert_last_pushed_at`` enforces the cooldown
    # so the queue isn't spammed every tick during a multi-hour
    # silence.
    #
    # The job itself NEVER marks trades LOST — that requires on-chain
    # ``balanceOf`` + ``payoutNumerators`` which is the operator's
    # ``admin reconcile-stuck`` flow.
    STALE_OPEN_RECONCILE_GRACE_HOURS: int = 4
    STUCK_ALERT_MIN_COUNT: int = 5
    STUCK_ALERT_EXPOSURE_FRACTION: float = 0.50
    STUCK_ALERT_COOLDOWN_HOURS: int = 4

    # Lock-rule trader (deterministic physical-condition path)
    LOCK_RULE_ENABLED: bool = True
    LOCK_RULE_MAX_PRICE: float = 0.95
    # Match the unified-pipeline pre-filter so any price the lock path sees is
    # tradeable. Previously 0.30 — blocked cases where a mispriced market was
    # still at single-digit cents despite the outcome being physically locked
    # (a ~10× return on the side we're certain about).
    LOCK_RULE_MIN_PRICE: float = 0.05
    LOCK_MARGIN_F: float = 2.0
    LOCK_POSITION_PCT: float = 0.02

    # --- Conviction-weighted lock sizing (default-off; see the plan) ---------
    # When enabled, an EASY-YES lock on a trusted (whitelisted) station is
    # sized by fractional Kelly against its price + branch certainty, capped at
    # LOCK_MAX_POSITION_PCT_*, instead of the flat LOCK_POSITION_PCT. EASY-YES
    # is the only truly monotonic (daily max can't decrease) and resolver-
    # divergence-clean lock class — every other branch / direction keeps the
    # flat 2% path, as do off-whitelist stations.
    LOCK_CONVICTION_SIZING_ENABLED: bool = False
    # Comma-separated ICAO whitelist that gets conviction sizing (empty =
    # nobody, even with the flag on). Curate from `resolution-report`: stations
    # with decent N, |mean divergence| ≈ 0, and no large outlier.
    LOCK_BIG_SIZE_STATIONS: str = ""
    LOCK_KELLY_FRACTION: float = 0.25
    # Assumed true win-prob per branch (Phase-3 divergence audit: unbiased
    # ±0.28°F abs-mean; a 4°F super-margin ≈ 14× that spread).
    LOCK_WIN_PROB_SUPER: float = 0.99
    LOCK_WIN_PROB_STANDARD: float = 0.97
    # Hard concentration caps (fraction of bankroll) — the backstop a single
    # black-swan wrong-station resolution can never exceed, whatever Kelly says.
    LOCK_MAX_POSITION_PCT_SUPER: float = 0.15
    LOCK_MAX_POSITION_PCT_STANDARD: float = 0.07
    # FAK walk ceiling for conviction lock orders (≤ LOCK_RULE_MAX_PRICE): the
    # order sweeps every ask at/below this price to deploy the requested size.
    LOCK_WALK_MAX_PRICE: float = 0.95
    # Relaxed depth cap for conviction locks (vs the flat path's hard 0.15) so
    # the Kelly stake isn't pre-throttled to a sliver of top-of-book before the
    # order even walks.
    LOCK_DEPTH_CAP_PCT_BIG: float = 0.50

    # Multi-model ensemble (Open-Meteo models= param). Spread across these
    # models drives the probability-engine sigma instead of the hardcoded
    # hours-based schedule.
    # `ecmwf_ifs04` returns no data on Open-Meteo's current /v1/forecast — use
    # `ecmwf_ifs025`. `meteofrance_seamless` is a Europe-domain regional model
    # and returns severely cold-biased peaks outside Europe (e.g. -8°C vs ECMWF
    # at OPKC/OMDB), so it's excluded by default.
    ENSEMBLE_MODELS: str = (
        "ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless"
    )
    # Inflate raw inter-model spread — NWP ensembles are under-dispersive vs
    # actual forecast error, ~20-30% for surface T.
    ENSEMBLE_SPREAD_MULTIPLIER: float = 1.3
    # Raised 1.0 → 2.0 (2026-05-08): a 1°F floor produces probability
    # distributions that under-disperse vs realised forecast error,
    # generating spurious NO-side edge on bracket modal-bucket bets at
    # the 0.7-0.8 prob range. 2°F matches the rough 12h-out RMS of
    # surface-T forecasts and aligns with CLIMATE_PRIOR_MIN_SIGMA_F.
    ENSEMBLE_MIN_SIGMA_F: float = 2.0
    ENSEMBLE_MAX_SIGMA_F: float = 5.0
    # If fewer than this many models returned usable peak-hour data, fall back
    # to the deterministic single-source endpoint.
    ENSEMBLE_MIN_MODELS: int = 3

    # Lead-time-aware σ floor (2026-05-30). Concrete failure motivating
    # this knob: Amsterdam 2026-05-23, σ collapsed to ~1.1°C at h=11.5
    # pre-peak because 4 ensemble models agreed tightly. Adjacent
    # single-°C buckets (27/28/29) all looked near-impossible → NO
    # triggered on all three; -$215 / 14d on `exactly` BUY_NO. The
    # additive arm enforces ``floor ≥ slope × hours_until_peak`` so a
    # confident ensemble at long lead can't pin σ below physical
    # forecast-error variance. ``SIGMA_HOURS_FLOOR_MULTIPLIER`` exposes
    # the previously-hardcoded 0.5× soft floor on ``_hours_based_sigma``
    # so the moderate-lead band (2-4h) can be tightened independently of
    # the additive arm. Defaults preserve pre-2026-05-30 behavior
    # bit-for-bit: master flag off zeros the additive arm and the 0.5
    # multiplier matches the prior constant.
    # GRAVEYARD: SIGMA_FLOOR_LEAD_TIME_ENABLED — docs/graveyard.md#sigma_floor_lead_time_enabled--lead-time-aware-σ-floor-probability-engine-gaussian (status: suspended)
    SIGMA_FLOOR_LEAD_TIME_ENABLED: bool = False
    SIGMA_LEAD_TIME_SLOPE_F_PER_HR: float = 0.3
    SIGMA_HOURS_FLOOR_MULTIPLIER: float = 0.5

    # Climate-normal prior. Multi-year per-station per-DOY climatology
    # acts as the Bayesian prior for the daily-max distribution before the
    # forecast Gaussian (likelihood) and METAR observations update it.
    # Ships disabled. Enabling requires backfilling the `station_normals`
    # table, sanity-checking the values, then flipping this to true. The
    # `scripts/backfill_station_normals.py` loader does not exist yet — see
    # `docs/improvements.md` [climate-prior backlog] for the concrete spec.
    # GRAVEYARD: CLIMATE_PRIOR_ENABLED — docs/graveyard.md#climate_prior_enabled--bayesian-climate-prior-blend-probability-engine (status: suspended)
    CLIMATE_PRIOR_ENABLED: bool = False
    # Floor on posterior σ after the Bayesian blend — prevents tropical
    # / oceanic stations (low climatological σ) from collapsing the
    # distribution width to ~1°F and over-confidently quoting narrow
    # bracket markets.
    CLIMATE_PRIOR_MIN_SIGMA_F: float = 2.0
    # Reject degenerate priors entirely. A station whose computed
    # std_max_c exceeds this is silently bypassed — it would dilute the
    # forecast rather than anchor it.
    CLIMATE_PRIOR_MAX_SIGMA_F: float = 8.0

    # ------------------------------------------------------------------
    # Forecast-exceedance alert tunables
    # ------------------------------------------------------------------
    # Used by ``src/signals/forecast_exceedance.py``. Previously module-level
    # constants; moved here so the operator can override via .env without code
    # edits (process restart still required — Pydantic loads once at boot).
    EXCEEDANCE_THRESHOLD_F: float = 0.5
    EXCEEDANCE_DELTA_THRESHOLD_F: float = 1.0
    EXCEEDANCE_MIN_ROUTINE_COUNT: int = 3
    EXCEEDANCE_STRONG_RESIDUAL_DELTA_F: float = 1.0  # 2 × EXCEEDANCE_THRESHOLD_F
    EXCEEDANCE_STRONG_RESIDUAL_MIN_ROUTINES: int = 2
    EXCEEDANCE_EXTRAPOLATION_HOURS_CAP: float = 3.0
    EXCEEDANCE_PEAK_TOLERANCE_F: float = 0.5
    EXCEEDANCE_EXTRAPOLATION_HALFLIFE_H: float = 2.0
    EXCEEDANCE_MAX_OVERSHOOT_F: float = 5.0
    EXCEEDANCE_DEWPOINT_NUDGE_F: float = 0.5
    EXCEEDANCE_ALERT_COOLDOWN_MINUTES: int = 30
    EXCEEDANCE_RESIDUAL_DECAY_HALFLIFE_H: float = 2.0
    EXCEEDANCE_RESIDUAL_TREND_CARRY_K: float = 0.5

    # ------------------------------------------------------------------
    # Post-peak trend-carry tunables (shared by forecast_exceedance and
    # probability_engine — single source of truth so alert and trading
    # Gaussian move in lockstep when Open-Meteo's nominal peak was too
    # early).
    # ------------------------------------------------------------------
    POST_PEAK_HOURS_CAP: float = 1.5
    POST_PEAK_TREND_CARRY_K: float = 0.75
    POST_PEAK_MIN_TREND_F_PER_HR: float = 0.5
    POST_PEAK_MAX_SHIFT_F: float = 3.0  # probability_engine center-shift cap

    # ------------------------------------------------------------------
    # Range/`exactly` lock-rule gates (lock_rules.py). Tighter than
    # threshold markets because range overshoot/undershoot enters at
    # ~0.95 — each loss costs ~$1/$1 staked, so breakeven needs ≥95%
    # accuracy.
    # ------------------------------------------------------------------
    RANGE_LOCK_MIN_ROUTINES: int = 4
    RANGE_LOCK_MARGIN_MULTIPLIER: float = 2.0

    # Max lead time (hours before close) for bracket-like markets
    # (exactly / range / bracket) in the PROBABILITY path. Live data
    # (2026-05-22): exactly-market probability trades lose -$126 in the
    # 12-24h lead band but make +$57 in the 0-12h band — far from peak the
    # Gaussian collapses P(a single ~2°F bucket) → ~0, manufacturing NO
    # edge that empirically reverts. Edges evaluated earlier than this are
    # rejected. Threshold markets (above/at_least/below/at_most) are
    # unaffected.
    EXACTLY_MAX_LEAD_HOURS: float = 12.0

    # Defense-in-depth for single-°C / single-bucket (exactly / range /
    # bracket) NO bets in the PROBABILITY path. Root cause behind the
    # -$163.84 / 179-trade `model_prob ≥ 0.999` NO loss class (many °C
    # cities; e.g. Amsterdam 2026-05-23 bet NO on 27/28/29°C while the
    # blended forecast was pinned at 30°C with a 1.1°C-wide Gaussian).
    #
    # SINGLE_BUCKET_NO_BAND_MARGIN_F — half-width of the "plausible landing
    # band" margin (°F) around [observed_max, forecast_peak]. A NO bet on a
    # single-bucket window overlapping that band is refused: never bet
    # against a bucket the day could still plausibly land in. The band
    # collapses toward the observed max once past peak, so genuinely
    # out-of-reach NO bets still fire.
    #
    # Tightened 1.0 → 2.5°F (2026-05-26). The `exactly` NO class was the #1
    # real-money loss source (61% win vs ~78% breakeven on fills). 2.5°F ≈
    # 1.4°C, so the band now refuses NO on the forecast/observed bucket AND
    # its immediate °C neighbours — a hedge against the ±~1°C divergence
    # between our routine-METAR daily max and Polymarket's resolver (the same
    # divergence that killed the removed `range_overshoot` lock). Validated via `evals-report
    # --operator bracket-like` (the single-bucket-NO-guard-tuning section):
    # margin 2.5 / cap 0.85 gave the best survivor EV while cutting ~58% of NO
    # volume.
    SINGLE_BUCKET_NO_BAND_MARGIN_F: float = 2.5
    # SINGLE_BUCKET_MAX_NO_PROB — hard ceiling on NO-side confidence for
    # single-bucket windows (floors `our_prob_yes` before the NO side is
    # computed). Caps tail overconfidence independent of lead time so the
    # ≥0.999 band can't recur. Tightened 0.92 → 0.85 (2026-05-26): with
    # MIN_EDGE=0.10 this effectively restricts NO to prices ≤ ~0.75 (better
    # risk/reward), pruning the marginal near-certain NO bets that bled.
    SINGLE_BUCKET_MAX_NO_PROB: float = 0.85

    # (The `range_overshoot` NO lock branch + its RANGE_OVERSHOOT_LOCK_ENABLED
    # flag were removed 2026-06-24 — lost -$57.61 / 56% win on °C resolver
    # divergence; see docs/graveyard.md.)

    # The range_undershoot lock branch (NO when observed max undershoots the
    # range low by 2× margin AND no-more-heating) is, like every bucket/range
    # NO class, model-OVERCONFIDENT — not a resolver divergence (2026-06-03
    # audit). It won 71.4% but its avg NO price 0.808 implies an 80.8%
    # break-even → ~9pp overconfident → -$26.52 / 35 trades over 30d, the same
    # win-small/lose-big shape as the killed `exactly`-NO. Measurable resolver
    # divergence is unbiased (±0.1°F), so a °C correction would not help; the
    # real cure is the deferred lead-time-aware σ-floor (σ is too tight on
    # narrow windows). Flag read live off `settings` (like overshoot); default
    # True preserves behavior, the live `.env` sets it False to gate the bleed.
    # Re-enable only after `SIGMA_FLOOR_LEAD_TIME_ENABLED` lands and the class
    # turns +EV. range_in_window (YES) is unaffected.
    # GRAVEYARD: RANGE_UNDERSHOOT_LOCK_ENABLED — docs/graveyard.md#range_undershoot_lock_enabled--range_undershoot-no-lock-branch (status: suspended)
    RANGE_UNDERSHOOT_LOCK_ENABLED: bool = True

    # Master switch for the HARD lock branch (observed max << threshold AND
    # `_no_more_heating` ⇒ NO for above/at_least, YES for below/at_most). Read
    # live off `settings` (like RANGE_UNDERSHOOT). Default True preserves
    # behavior; the live env sets it False.
    #
    # Why suspend: the HARD branch is the #1 active lock bleed (2026-06-30 prod
    # audit, workflow wf_3496e753-f0b). All 10 fills ever are BUY_NO: wins are
    # tiny (+$0.7–1.9; the one big win was a cheap NO@0.29), losses are large
    # (−$5.3 to −$9.3). Every loss carried a 14–20°F lock margin yet still lost
    # — the deterministic forecast missed the actual max by >14°F, a forecast
    # catastrophe that no margin/σ-floor tweak fixes (the lead-time σ-floor only
    # touches the probability engine, not `_no_more_heating`). Net −$16.97/30d
    # (−$7.63/30d even excluding the now-_EXCLUDED Taipei/RCTP). Same
    # win-small/lose-big shape as the killed range_undershoot / bracket-like NO.
    # Re-enable only after the >14°F forecast-miss class is explained (resolver
    # station verification at e.g. LTFM/OEJN) and the branch turns +EV.
    HARD_LOCK_ENABLED: bool = True

    # Minimum YES buy-price for the `range_in_window` (YES) lock branch. The
    # only realized fill ever was a 0.40-priced "exactly 27°C" YES gamble that
    # LOST −$9.28 (2026-06-30 audit) — a low-conviction cheap bet, not a lock.
    # Gating it to near-certain prices keeps the branch honest. Read live off
    # `settings`; 0.0 disables the gate (pre-audit behavior).
    RANGE_IN_WINDOW_MIN_YES_PRICE: float = 0.80

    # ------------------------------------------------------------------
    # `bucket_overshoot` — buy NO on a temperature bucket the day's max has
    # already passed. Restores (correctly) the `range_overshoot` branch that was
    # removed 2026-06-24 on a mis-attributed loss: its real phantom-safe book was
    # −$3.70 / 9 trades / 89% win, and its single loss was the **pre-2026-05-26
    # wrong-day bug** on a −UTC city, not °C resolver divergence. See
    # docs/graveyard.md + memory `project_dead_bucket_edge_2026-07-10`.
    #
    # The rule is mathematically certain, not a forecast: the daily max is
    # monotonic, so once the running routine-METAR max for the market's
    # station-local day exceeds the bucket top by `BUCKET_OVERSHOOT_MARGIN_C`
    # degrees C, that bucket can never win. The only way to lose is resolver
    # divergence (our station reads hotter than Polymarket's resolver), measured
    # at P(div >= 1°C) = 4.1% globally and ~0-3% on trusted stations.
    #
    # Validated 2026-07-10 on 3,580 rule-triggered candidates priced off real
    # CLOB 1-min history (`clob.polymarket.com/prices-history`) at the bot's real
    # METAR arrival time, scored against on-chain resolution, event-clustered:
    #   63 bets / 52 station-days, 2 losses, EV +0.91/$1 [95% CI +0.53, +1.46]
    #   out-of-sample (target day > 2026-06-05): 55 bets, 0 losses, EV +1.01/$1
    # NOTE: `market_snapshots.yes_price` (Gamma) is STALE in the repricing window
    # and must never be used to backtest this — see
    # memory `project_gamma_snapshot_stale_2026-07-10`.
    #
    # SPEED IS THE EDGE. The market reprices a median 2.1 min after the METAR's
    # observation time; we receive it at a median 2.7 min. EV survives a ~30 s
    # decision delay and dies by ~60 s, so this branch must fire from
    # `job_fast_lock_poll` (30 s), never from the 5-min unified tick.
    BUCKET_OVERSHOOT_LOCK_ENABLED: bool = False

    # Degrees C the running max must exceed the bucket top by. 1.0 == "the max is
    # already a full °C above this bucket" — the minimal certainty condition,
    # since routine METARs report whole °C.
    BUCKET_OVERSHOOT_MARGIN_C: float = 1.0

    # Max price paid for the NO side. Above this the market has already
    # repriced and the residual edge no longer covers resolver-divergence risk:
    # at cost 0.93 break-even needs a 7% loss rate vs the ~1.5% observed, and
    # the policy stays +EV (+0.73/$1) even if divergence were 5x worse. Raising
    # it to 0.99 adds volume but makes >1/3 of bets -EV under that stress.
    BUCKET_OVERSHOOT_MAX_COST: float = 0.93

    # Stations whose resolver diverges from our METAR feed too often to bet
    # certainty on. Chosen ex-ante from `station_day_resolutions`:
    # P(divergence >= 1°C) — ZGSZ 34.5%, MPTO 30.8%, EGLL 26.9%, UUEE 14.8%;
    # every other station with n>=15 is <= 4.2% (a clean natural break).
    # These four produced 11 of the 16 rule violations observed.
    BUCKET_OVERSHOOT_EXCLUDED_STATIONS: str = "ZGSZ,MPTO,EGLL,UUEE"

    @property
    def bucket_overshoot_excluded(self) -> set[str]:
        return {
            s.strip().upper()
            for s in (self.BUCKET_OVERSHOOT_EXCLUDED_STATIONS or "").split(",")
            if s.strip()
        }

    # Master switch for bracket-like NO (`exactly`/`range`/`bracket` BUY_NO)
    # in the PROBABILITY path. When True, `binary_market_edge` rejects any
    # otherwise-passing bracket-like NO edge with reason
    # ``"bracket-like NO disabled (sigma recalibration pending)"``.
    #
    # Why: live data (2026-05-30) all-time -$260 / -7.4% ROI on 383 trades,
    # last 14d -$346 / -14.3% ROI on 263 trades. The model is structurally
    # overconfident on single-bucket NO (each °C bucket evaluated as an
    # independent binary; tight ensemble agreement collapses σ far from
    # peak; landing-band/cap/lead guards block specific failure modes but
    # the surviving NO evals still score -0.023 EV/$1 per the live
    # ``evals-report --operator bracket-like`` single-bucket-NO tuning
    # grid). The proper fix is the lead-time-aware σ floor backlogged in
    # ``docs/improvements.md`` (deferred from the 2026-05-23 NO-guards work
    # because it would also retune the working threshold path). Until that
    # ships, every additional bracket-like-NO fill is paid-for lab data we
    # already have.
    #
    # Scope: probability path only. The lock path (`evaluate_lock` →
    # `_evaluate_range_lock`) does NOT go through `binary_market_edge`, so
    # `range_in_window`/`range_undershoot` YES locks keep firing. Threshold ops
    # (above/at_least/below/at_most) are unaffected on both sides.
    #
    # Default False = no behavior change. Flip True in `.env` to stop the
    # bleed; flip back when the σ-recalibration backlog item lands and the
    # evals-report tuner shows survivors at +EV.
    # GRAVEYARD: BRACKET_LIKE_NO_DISABLED — docs/graveyard.md#bracket_like_no_disabled--kill-switch-for-bracket-like-exactlyrangebracket-buy_no-in-the-probability-path (status: suspended)
    BRACKET_LIKE_NO_DISABLED: bool = False

    # --- Price-band edge policy (2026-06-06 perf audit) -----------------
    # Per-trade EV is U-shaped in the effective price of the side bought:
    # +EV at the extremes (deep-value [.40,.60) and near-lock [.85,1.0]) and
    # -EV in the mid "overconfidence valley" where the Gaussian's mild
    # over-tightness bites (60d go-forward book: valley -0.054 EV/$1 vs the
    # extremes +0.086 / +0.096; the bot's realised avg NO price ~0.75 sits in
    # the valley). Two layers in `binary_market_edge`, applied to the chosen
    # side under the `reason is None` guard — probability path only, since the
    # lock path doesn't route through `binary_market_edge`:
    #   P1 (VALLEY_BLOCK_ENABLED): block any trade whose side-price lands in
    #      [VALLEY_PRICE_LOW, VALLEY_PRICE_HIGH). Robust, no tuned parameter —
    #      the interim guard. In-sample 60d: EV/$1 0.033→0.093, win 75→81%,
    #      volume -40%.
    #   P2 (VALLEY_MIN_EDGE): instead require a raised edge floor in the valley.
    #      Higher PnL than P1 (keeps the +EV high-edge valley trades) but the
    #      0.15 threshold is in-sample-tuned on n=8 removed trades — validate
    #      via SHADOW_VALLEY_* on out-of-sample data before flipping. None=off.
    # Precedence: P1 (block) wins when both are set. Both default to no-op.
    VALLEY_PRICE_LOW: float = 0.60
    VALLEY_PRICE_HIGH: float = 0.85
    # GRAVEYARD: VALLEY_BLOCK_ENABLED / VALLEY_MIN_EDGE — docs/graveyard.md#valley_block_enabled--valley_min_edge--price-band-overconfidence-valley-policy (status: suspended)
    VALLEY_BLOCK_ENABLED: bool = False
    VALLEY_MIN_EDGE: float | None = None

    # Shadow telemetry for the P2 refinement (measure-before-flip). Pure —
    # stamps evaluation_logs.shadow_json.valley with {in_valley, eff_price,
    # edge, p2_would_block, p2_min_edge} so a future report can join valley
    # evals to their resolved outcome (via market_id — no fired trade needed)
    # and confirm the edge>=SHADOW_VALLEY_MIN_EDGE split is +EV before
    # VALLEY_MIN_EDGE is set live. Runs regardless of VALLEY_BLOCK_ENABLED
    # (blocked valley evals are still logged, so their counterfactual outcome
    # stays recoverable).
    SHADOW_VALLEY_POLICY_ENABLED: bool = True
    SHADOW_VALLEY_MIN_EDGE: float = 0.15


settings = Settings()