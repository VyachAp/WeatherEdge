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
    CHECKWX_DAILY_BUDGET: int = 2000
    IEM_RATE_LIMIT_RPS: float = 1.0
    OGIMET_RATE_LIMIT_RPS: float = 0.2
    NOAA_RATE_LIMIT_RPS: float = 1.0
    AVWX_RATE_LIMIT_RPS: float = 1.0

    # Weather Company v3 observations (airport ICAO stations)
    WX_API_KEY: str = ""  # Empty = WX pipeline disabled
    WX_RATE_LIMIT_RPS: float = 1.5  # ~90/min conservative
    WX_DAILY_BUDGET: int = 10000
    WX_PIPELINE_INTERVAL_MINUTES: int = 1  # Poll every minute
    WX_RETENTION_HOURS: int = 48
    WX_PEAK_CONFIRM_MINUTES: int = 15  # Minutes of decline before declaring peak done

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
    # Set True after depositing to pUSD via the polymarket.com UI. The flag
    # monkey-patches py-clob-client to sign for the new exchanges instead of
    # the old USDC.e ones; without it every order is rejected with
    # ``order_version_mismatch``.
    POLYMARKET_USE_NEW_EXCHANGES: bool = False
    AUTO_EXECUTE: bool = False  # Set True to place orders automatically
    DAILY_SPEND_CAP_USD: float = 200.0  # Max total spend per 24h
    MIN_STAKE_USD: float = 5.0  # Skip orders below this amount

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

    # Circuit breakers
    DAILY_LOSS_STOP_USD: float = 200.0
    CONSECUTIVE_LOSS_PAUSE_COUNT: int = 3
    CONSECUTIVE_LOSS_PAUSE_HOURS: int = 2

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
    FAST_LOCK_POLL_INTERVAL_SECONDS: int = 30

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
    LOCK_RULE_LOSS_WINDOW_HOURS: int = 72
    LOCK_RULE_LOSS_DISABLE_COUNT: int = 3

    # Open-Meteo forecast
    OPENMETEO_RATE_LIMIT_RPS: float = 2.0

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

    # Climate-normal prior. Multi-year per-station per-DOY climatology
    # acts as the Bayesian prior for the daily-max distribution before the
    # forecast Gaussian (likelihood) and METAR observations update it.
    # Ships disabled — backfill the `station_normals` table first via
    # `scripts/backfill_station_normals.py`, sanity-check the values,
    # then flip CLIMATE_PRIOR_ENABLED=true.
    CLIMATE_PRIOR_ENABLED: bool = False
    CLIMATE_NORMAL_YEARS: int = 10
    # Floor on posterior σ after the Bayesian blend — prevents tropical
    # / oceanic stations (low climatological σ) from collapsing the
    # distribution width to ~1°F and over-confidently quoting narrow
    # bracket markets.
    CLIMATE_PRIOR_MIN_SIGMA_F: float = 2.0
    # Reject degenerate priors entirely. A station whose computed
    # std_max_c exceeds this is silently bypassed — it would dilute the
    # forecast rather than anchor it.
    CLIMATE_PRIOR_MAX_SIGMA_F: float = 8.0

    # Residual-slope projection (lever A in the projection-latency redesign).
    # When enabled and the station has at least 3 routine METARs in the 6h
    # window with same-hour forecast cells, ``_project_daily_max`` projects
    # the residual forward at the observed hourly slope instead of decaying
    # the level residual with a 2h halflife. Captures "forecast falling
    # further behind every hour" 1-2 hours earlier than the legacy path.
    # Falls back to the halflife decay when fewer than 3 points are available.
    PROJECTION_RESIDUAL_SLOPE_ENABLED: bool = True

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
    EXCEEDANCE_RESIDUAL_SLOPE_MIN_POINTS: int = 3
    EXCEEDANCE_RESIDUAL_SLOPE_HOURS_CAP: float = 3.0
    EXCEEDANCE_RESIDUAL_SLOPE_MAX_F_PER_HR: float = 1.5

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
    SINGLE_BUCKET_NO_BAND_MARGIN_F: float = 1.0
    # SINGLE_BUCKET_MAX_NO_PROB — hard ceiling on NO-side confidence for
    # single-bucket windows (floors `our_prob_yes` before the NO side is
    # computed). Caps tail overconfidence independent of lead time so the
    # ≥0.999 band can't recur.
    SINGLE_BUCKET_MAX_NO_PROB: float = 0.92

    # The range_overshoot lock branch (NO when observed max overshoots the
    # range high by 2× margin) lost -$57.61 across 18 trades (56% win) on
    # exactly markets — systematic resolver/observation divergence (our
    # routine-METAR daily max reads hotter than Polymarket's resolver), not
    # a margin problem (2× was already applied). Disabled by default; the
    # range_undershoot (+$8, 100%) and range_in_window branches are kept.
    RANGE_OVERSHOOT_LOCK_ENABLED: bool = False


settings = Settings()