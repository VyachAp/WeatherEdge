"""Tests for the per-bucket edge calculator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.signals.edge_calculator import BucketEdge
from src.signals.probability_engine import BucketDistribution


def _make_dist(probs: dict[int, float], current_max: int = 75) -> BucketDistribution:
    return BucketDistribution(
        current_max_f=current_max,
        probabilities=probs,
        reasoning=["test"],
    )


# Multi-bucket bracket-evaluation path (`compute_edges`) was retired
# 2026-05-30 — see docs/graveyard.md. Bracket/range/exactly ops are
# evaluated via `binary_market_edge`'s single-bucket window branch
# (covered by `TestBinaryMarketEdge` below). Filter coverage now flows
# entirely through `_check_filters` directly + `binary_market_edge`.


class TestCheckFiltersMinRoutineOverride:
    """`_check_filters` accepts a `min_routine_count` override so the
    lock-rule path can fire on 2 routines for super-margin EASY locks
    without the scheduler-side gate undoing the relaxation."""

    def test_default_uses_settings_min_routine_count(self):
        from src.signals.edge_calculator import _check_filters

        # routine_count=2 with default min (3) → rejected. Prob bumped
        # above MIN_PROBABILITY=0.85 so it doesn't short-circuit the
        # earlier probability filter.
        reason = _check_filters(
            edge=0.2, prob=0.9, price=0.7,
            routine_count=2, minutes_to_close=120, depth=100.0,
        )
        assert reason is not None
        assert "routine count" in reason

    def test_override_to_two_allows_two_routines(self):
        from src.signals.edge_calculator import _check_filters

        reason = _check_filters(
            edge=0.2, prob=0.9, price=0.7,
            routine_count=2, minutes_to_close=120, depth=100.0,
            min_routine_count=2,
        )
        assert reason is None

    def test_override_still_rejects_below_override(self):
        from src.signals.edge_calculator import _check_filters

        reason = _check_filters(
            edge=0.2, prob=0.9, price=0.7,
            routine_count=1, minutes_to_close=120, depth=100.0,
            min_routine_count=2,
        )
        assert reason is not None
        assert "routine count 1 < 2" in reason


class TestBucketEdgeDirection:
    """`BucketEdge.direction` is BUY_YES by default (back-compat) but is
    set explicitly by `_binary_market_edge` based on which side of the
    market the model disagrees with."""

    def test_default_direction_is_yes(self):
        from src.db.models import TradeDirection

        e = BucketEdge(
            bucket_value=80, our_probability=0.7, market_price=0.5,
            edge=0.2, passes=True, reject_reason=None,
        )
        assert e.direction == TradeDirection.BUY_YES

    def test_explicit_no_direction(self):
        from src.db.models import TradeDirection

        e = BucketEdge(
            bucket_value=80, our_probability=0.7, market_price=0.45,
            edge=0.25, passes=True, reject_reason=None,
            direction=TradeDirection.BUY_NO,
        )
        assert e.direction == TradeDirection.BUY_NO


def _eval_binary(
    *, question, threshold_f, op, our_prob_in_window,
    yes_bid, yes_ask, forecast_peak_f, current_max_f, hours_until_peak,
    end_hours=5, routine_count=5,
):
    """Drive `binary_market_edge` for a single market with the new
    forecast/observation context kwargs that feed the single-bucket NO
    guards. Mass is placed inside the bucket window (for bracket-like ops)
    or at/above threshold (for threshold ops) so `our_prob_yes` is known.
    """
    from src.execution.binary_market import market_range_f
    from src.signals.edge_calculator import binary_market_edge

    market = SimpleNamespace(
        id="m1", question=question,
        parsed_threshold=threshold_f, parsed_operator=op,
        current_yes_price=(yes_bid + yes_ask) / 2,
        end_date=None, outcomes=["Yes", "No"],
    )
    rng = market_range_f(market)
    probs: dict[int, float] = {}
    if rng is not None:
        low, high = rng
        window = list(range(low, high + 1))
        per = our_prob_in_window / len(window)
        for b in window:
            probs[b] = per
        probs[low - 12] = max(0.0, 1.0 - our_prob_in_window)
    else:
        thr = int(round(threshold_f))
        probs = {thr: our_prob_in_window, thr - 1: 1.0 - our_prob_in_window}

    dist = BucketDistribution(
        current_max_f=int(current_max_f), probabilities=probs, reasoning=["test"],
    )
    end_time = datetime.now(timezone.utc) + timedelta(hours=end_hours)
    return binary_market_edge(
        dist, market, end_time, routine_count=routine_count,
        depth_yes=100.0, depth_no_fn=lambda: 100.0,
        yes_bid=yes_bid, yes_ask=yes_ask,
        forecast_peak_f=forecast_peak_f, current_max_f=current_max_f,
        hours_until_peak=hours_until_peak,
    )


class TestSingleBucketNoGuards:
    """Three layered guards stop the incoherent "NO on every adjacent
    single-°C bucket" pattern (e.g. Amsterdam 2026-05-23: NO on 27/28/29°C
    while the blended forecast was pinned at 30°C). All guards are no-ops
    for threshold ops and when the forecast context isn't supplied.
    """

    Q28 = "Will the highest temperature in Amsterdam be 28°C on May 23"
    Q30 = "Will the highest temperature in Amsterdam be 30°C on May 23"

    @pytest.fixture(autouse=True)
    def _disable_master_switch(self, monkeypatch):
        """Pin the bracket-like-NO master switch OFF for this class — these
        tests cover the Layer 1-3 guards, not the master switch on top.
        Needed because the live `.env` ships with the switch flipped on.
        """
        from src.config import settings
        monkeypatch.setattr(settings, "BRACKET_LIKE_NO_DISABLED", False)

    def test_no_inside_landing_band_rejected(self):
        from src.db.models import TradeDirection

        # forecast 30°C (86.5°F), obs 23°C (73°F), 6.6h pre-peak. The 28°C
        # window (82-83°F) sits inside the landing band [72, 87.5]°F.
        edge = _eval_binary(
            question=self.Q28, threshold_f=82.4, op="exactly",
            our_prob_in_window=0.05, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=86.5, current_max_f=73.0, hours_until_peak=6.6,
        )
        assert edge.direction == TradeDirection.BUY_NO
        assert edge.passes is False
        assert "landing band" in (edge.reject_reason or "")

    def test_no_above_band_passes(self):
        from src.db.models import TradeDirection

        # forecast 25.6°C (78°F), obs 23°C (73°F), 2h to peak. The 30°C
        # window (86°F) is clear of the band [72, 79]°F → a genuinely
        # out-of-reach NO bet still fires.
        edge = _eval_binary(
            question=self.Q30, threshold_f=86.0, op="exactly",
            our_prob_in_window=0.10, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=78.0, current_max_f=73.0, hours_until_peak=2.0,
        )
        assert edge.direction == TradeDirection.BUY_NO
        assert edge.passes is True
        assert edge.reject_reason is None

    def test_overconfidence_cap_clamps_no_prob(self):
        # P(window)=0 would manufacture NO=1.0; the cap floors YES to
        # 1 - SINGLE_BUCKET_MAX_NO_PROB, so the NO side can't exceed the cap.
        # Reads the live setting so it tracks the config (cap was 0.92, now 0.85).
        from src.config import settings

        edge = _eval_binary(
            question=self.Q30, threshold_f=86.0, op="exactly",
            our_prob_in_window=0.0, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=78.0, current_max_f=73.0, hours_until_peak=2.0,
        )
        assert edge.our_probability == pytest.approx(settings.SINGLE_BUCKET_MAX_NO_PROB)

    def test_widened_margin_rejects_bucket_that_narrow_band_allowed(self, monkeypatch):
        from src.config import settings
        from src.db.models import TradeDirection

        # forecast 25.6°C (78°F), obs 73°F, 2h to peak, bucket 80°F. Under a
        # 1.0°F margin the band is [72, 79] and 80 squeaks through; under the
        # tightened 2.5°F margin the band is [70.5, 80.5] and 80 is refused.
        # Pin the margin so the test is independent of the shipped default.
        q80 = "Will the highest temperature in Amsterdam be 80°F on May 23"

        monkeypatch.setattr(settings, "SINGLE_BUCKET_NO_BAND_MARGIN_F", 1.0)
        edge = _eval_binary(
            question=q80, threshold_f=80.0, op="exactly",
            our_prob_in_window=0.05, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=78.0, current_max_f=73.0, hours_until_peak=2.0,
        )
        assert edge.direction == TradeDirection.BUY_NO
        assert edge.passes is True

        monkeypatch.setattr(settings, "SINGLE_BUCKET_NO_BAND_MARGIN_F", 2.5)
        edge = _eval_binary(
            question=q80, threshold_f=80.0, op="exactly",
            our_prob_in_window=0.05, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=78.0, current_max_f=73.0, hours_until_peak=2.0,
        )
        assert edge.passes is False
        assert "landing band" in (edge.reject_reason or "")

    def test_peak_relative_lead_gate_uses_hours_until_peak(self):
        # Only 2h to CLOSE but 13h to PEAK → rejected. Bucket is clear of the
        # band, so it's unambiguously the peak-relative lead gate firing
        # (the close-vs-peak bug that let pre-dawn Amsterdam bets through).
        edge = _eval_binary(
            question=self.Q30, threshold_f=86.0, op="exactly",
            our_prob_in_window=0.10, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=78.0, current_max_f=73.0, hours_until_peak=13.0,
            end_hours=2,
        )
        assert edge.passes is False
        assert "lead" in (edge.reject_reason or "")
        assert "peak" in (edge.reject_reason or "")

    def test_threshold_market_unaffected_by_guards(self):
        from src.db.models import TradeDirection

        # at_least market, 13h to peak, forecast context supplied: threshold
        # ops bypass the lead gate, the landing-band guard, and the cap.
        edge = _eval_binary(
            question="Will the highest temperature in Amsterdam be 82°F or higher",
            threshold_f=82, op="at_least",
            our_prob_in_window=0.90, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=86.0, current_max_f=73.0, hours_until_peak=13.0,
        )
        assert edge.direction == TradeDirection.BUY_YES
        assert edge.passes is True


class TestCheckFiltersThresholdOverrides:
    """`_check_filters` accepts `min_edge` / `min_probability` overrides so
    `binary_market_edge` can run threshold ops on the (optional) looser
    THRESHOLD_MIN_* floors while bracket-like ops keep the strict globals.
    None defaults must reproduce the global-setting behavior exactly."""

    def test_min_probability_override_loosens(self):
        from src.signals.edge_calculator import _check_filters

        # prob 0.80 fails the global 0.85 floor but passes a 0.78 override.
        assert _check_filters(
            edge=0.2, prob=0.80, price=0.6,
            routine_count=5, minutes_to_close=120, depth=100.0,
        ) is not None  # default 0.85 floor rejects
        assert _check_filters(
            edge=0.2, prob=0.80, price=0.6,
            routine_count=5, minutes_to_close=120, depth=100.0,
            min_probability=0.78,
        ) is None

    def test_min_edge_override_loosens(self):
        from src.signals.edge_calculator import _check_filters
        from src.config import settings

        # edge 0.06 fails when the global floor is 0.10 (the live .env value)
        # but passes a 0.05 override. Skip the negative leg if the active
        # global floor already admits 0.06 (default 0.05 build).
        if settings.MIN_EDGE > 0.06:
            assert _check_filters(
                edge=0.06, prob=0.95, price=0.6,
                routine_count=5, minutes_to_close=120, depth=100.0,
            ) is not None
        assert _check_filters(
            edge=0.06, prob=0.95, price=0.6,
            routine_count=5, minutes_to_close=120, depth=100.0,
            min_edge=0.05,
        ) is None

    def test_none_defaults_reproduce_global(self):
        from src.signals.edge_calculator import _check_filters
        from src.config import settings

        # Explicit None == omitted == global settings. prob just below the
        # global floor must reject identically with and without the kwargs.
        prob = settings.MIN_PROBABILITY - 0.01
        a = _check_filters(
            edge=0.5, prob=prob, price=0.3,
            routine_count=5, minutes_to_close=120, depth=100.0,
        )
        b = _check_filters(
            edge=0.5, prob=prob, price=0.3,
            routine_count=5, minutes_to_close=120, depth=100.0,
            min_edge=None, min_probability=None,
        )
        assert a == b
        assert a is not None and "probability" in a


class TestCheckFiltersMinEntryPriceOverride:
    """`_check_filters` accepts a `min_entry_price` override so the probability
    path can concentrate on the near-lock band (PROBABILITY_MIN_ENTRY_PRICE)
    while the lock path keeps the global low floor and can buy cheap-but-certain
    contracts. None default reproduces the global MIN_ENTRY_PRICE."""

    def test_override_rejects_below_scoped_floor(self):
        from src.signals.edge_calculator import _check_filters

        # A side priced 0.70 passes the global 0.40 floor but is rejected by a
        # 0.80 override — the probability-path concentration.
        assert _check_filters(
            edge=0.2, prob=0.95, price=0.70,
            routine_count=5, minutes_to_close=120, depth=100.0,
        ) is None  # global 0.40 floor admits 0.70
        reason = _check_filters(
            edge=0.2, prob=0.95, price=0.70,
            routine_count=5, minutes_to_close=120, depth=100.0,
            min_entry_price=0.80,
        )
        assert reason is not None and "price" in reason

    def test_lock_path_keeps_cheap_bets(self):
        from src.signals.edge_calculator import _check_filters

        # The lock path passes no override → cheap-but-certain bet at 0.50 is
        # still allowed even though the probability path floor would be 0.80.
        assert _check_filters(
            edge=0.4, prob=0.99, price=0.50,
            routine_count=5, minutes_to_close=120, depth=100.0,
            min_entry_price=None,
        ) is None

    def test_none_defaults_reproduce_global(self):
        from src.signals.edge_calculator import _check_filters
        from src.config import settings

        price = settings.MIN_ENTRY_PRICE - 0.05
        a = _check_filters(
            edge=0.5, prob=0.99, price=price,
            routine_count=5, minutes_to_close=120, depth=100.0,
        )
        b = _check_filters(
            edge=0.5, prob=0.99, price=price,
            routine_count=5, minutes_to_close=120, depth=100.0,
            min_entry_price=None,
        )
        assert a == b
        assert a is not None and "price" in a


class TestBinaryMarketEdgeThresholdFloors:
    """`binary_market_edge` applies THRESHOLD_MIN_* to threshold ops ONLY.
    Bracket-like ops keep the strict global floors (and their NO guards)."""

    def test_threshold_passes_under_loosened_floor(self, monkeypatch):
        from src.config import settings
        from src.db.models import TradeDirection

        monkeypatch.setattr(settings, "THRESHOLD_MIN_PROBABILITY", 0.78)
        monkeypatch.setattr(settings, "THRESHOLD_MIN_EDGE", 0.05)
        # above-80 YES at prob 0.80: fails the default 0.85 global floor,
        # passes the 0.78 threshold override. yes_ask 0.62 → edge 0.18.
        edge = _eval_binary(
            question="Will the highest temperature be above 80F",
            threshold_f=80.0, op="above",
            our_prob_in_window=0.80, yes_bid=0.60, yes_ask=0.62,
            forecast_peak_f=85.0, current_max_f=70.0, hours_until_peak=3.0,
        )
        assert edge.direction == TradeDirection.BUY_YES
        assert edge.passes is True

    def test_threshold_rejected_without_override(self, monkeypatch):
        from src.config import settings
        from src.db.models import TradeDirection

        # Pin the overrides to None so this exercises the documented
        # "no override" path regardless of what the live .env sets.
        monkeypatch.setattr(settings, "THRESHOLD_MIN_PROBABILITY", None)
        monkeypatch.setattr(settings, "THRESHOLD_MIN_EDGE", None)
        # Same trade, default floors (THRESHOLD_MIN_* unset = None): the
        # 0.80 prob is below the global 0.85 floor → rejected.
        edge = _eval_binary(
            question="Will the highest temperature be above 80F",
            threshold_f=80.0, op="above",
            our_prob_in_window=0.80, yes_bid=0.60, yes_ask=0.62,
            forecast_peak_f=85.0, current_max_f=70.0, hours_until_peak=3.0,
        )
        assert edge.direction == TradeDirection.BUY_YES
        assert edge.passes is False
        assert "probability" in (edge.reject_reason or "")

    def test_bracket_like_unaffected_by_threshold_override(self, monkeypatch):
        from src.config import settings
        from src.db.models import TradeDirection

        monkeypatch.setattr(settings, "THRESHOLD_MIN_PROBABILITY", 0.78)
        monkeypatch.setattr(settings, "THRESHOLD_MIN_EDGE", 0.05)
        # YES on an `exactly` window at prob 0.80, near peak (no lead gate),
        # YES side (no landing-band/NO-cap). The threshold override must NOT
        # apply → still rejected by the strict global 0.85 floor.
        edge = _eval_binary(
            question="Will the highest temperature be 80F on May 23",
            threshold_f=80.0, op="exactly",
            our_prob_in_window=0.80, yes_bid=0.60, yes_ask=0.62,
            forecast_peak_f=80.0, current_max_f=79.0, hours_until_peak=2.0,
        )
        assert edge.direction == TradeDirection.BUY_YES
        assert edge.passes is False
        assert "probability" in (edge.reject_reason or "")


class TestBracketLikeNoMasterSwitch:
    """`BRACKET_LIKE_NO_DISABLED=True` rejects any otherwise-passing
    bracket-like NO edge while σ recalibration is pending. Threshold ops,
    `exactly` YES, and the off-by-default state are all unaffected.
    """

    Q30 = "Will the highest temperature in Amsterdam be 30°C on May 23"

    def test_off_by_default_passes_exactly_no(self, monkeypatch):
        from src.config import settings
        from src.db.models import TradeDirection

        # Pin OFF so the assertion is independent of any future default flip.
        monkeypatch.setattr(settings, "BRACKET_LIKE_NO_DISABLED", False)
        # forecast 25.6°C (78°F), obs 73°F, 2h to peak, NO on the 30°C window
        # (86°F) — clear of the landing band, passes Layers 1-3.
        edge = _eval_binary(
            question=self.Q30, threshold_f=86.0, op="exactly",
            our_prob_in_window=0.10, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=78.0, current_max_f=73.0, hours_until_peak=2.0,
        )
        assert edge.direction == TradeDirection.BUY_NO
        assert edge.passes is True
        assert edge.reject_reason is None

    def test_master_switch_blocks_exactly_no(self, monkeypatch):
        from src.config import settings
        from src.db.models import TradeDirection

        monkeypatch.setattr(settings, "BRACKET_LIKE_NO_DISABLED", True)
        # Same NO edge as above — would pass every layer, but the master
        # switch rejects with its own reason.
        edge = _eval_binary(
            question=self.Q30, threshold_f=86.0, op="exactly",
            our_prob_in_window=0.10, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=78.0, current_max_f=73.0, hours_until_peak=2.0,
        )
        assert edge.direction == TradeDirection.BUY_NO
        assert edge.passes is False
        assert "bracket-like NO disabled" in (edge.reject_reason or "")

    def test_master_switch_preserves_specific_reject_reason(self, monkeypatch):
        from src.config import settings

        # Master switch ON, but the edge is already rejected by Layer 2
        # (landing band): bucket 82°F sits inside [72, 87.5]°F. The original
        # landing-band reason must win — the master switch only marks
        # *otherwise-passing* residual NO evals.
        monkeypatch.setattr(settings, "BRACKET_LIKE_NO_DISABLED", True)
        edge = _eval_binary(
            question="Will the highest temperature in Amsterdam be 28°C on May 23",
            threshold_f=82.4, op="exactly",
            our_prob_in_window=0.05, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=86.5, current_max_f=73.0, hours_until_peak=6.6,
        )
        assert edge.passes is False
        assert "landing band" in (edge.reject_reason or "")
        assert "sigma recalibration" not in (edge.reject_reason or "")

    def test_master_switch_passes_exactly_yes(self, monkeypatch):
        from src.config import settings
        from src.db.models import TradeDirection

        # Master switch ON, but YES on an `exactly` window at high prob with
        # YES underpriced. The switch must NOT block YES — only NO is killed.
        monkeypatch.setattr(settings, "BRACKET_LIKE_NO_DISABLED", True)
        edge = _eval_binary(
            question=self.Q30, threshold_f=86.0, op="exactly",
            our_prob_in_window=0.90, yes_bid=0.55, yes_ask=0.60,
            forecast_peak_f=86.0, current_max_f=85.0, hours_until_peak=1.0,
        )
        assert edge.direction == TradeDirection.BUY_YES
        assert edge.passes is True
        assert edge.reject_reason is None

    def test_master_switch_passes_threshold_no(self, monkeypatch):
        from src.config import settings
        from src.db.models import TradeDirection

        # Master switch ON, but `at_least` NO is a threshold op — must pass.
        # forecast 70°F, obs 65°F, peak passed → NO on at_least 85°F is the
        # +EV class the switch is designed to preserve. yes_bid=0.20 puts
        # NO buy price at 0.80 with NO edge 0.15 (clears the 0.10 global
        # MIN_EDGE the live .env sets).
        monkeypatch.setattr(settings, "BRACKET_LIKE_NO_DISABLED", True)
        edge = _eval_binary(
            question="Will the highest temperature in Amsterdam be 85F or higher",
            threshold_f=85.0, op="at_least",
            our_prob_in_window=0.05, yes_bid=0.20, yes_ask=0.25,
            forecast_peak_f=70.0, current_max_f=65.0, hours_until_peak=-1.0,
        )
        assert edge.direction == TradeDirection.BUY_NO
        assert edge.passes is True
        assert edge.reject_reason is None


class TestPriceBandValleyPolicy:
    """Price-band edge policy (2026-06-06 audit): per-trade EV is U-shaped in
    the effective price of the side bought, with a -EV "overconfidence valley"
    in [VALLEY_PRICE_LOW, VALLEY_PRICE_HIGH). P1 (block) and P2 (raised edge
    floor) gate the valley; both no-op by default. Side/operator-agnostic.
    """

    @pytest.fixture(autouse=True)
    def _clean_flags(self, monkeypatch):
        # Deterministic baseline: calibration off (so probs/edges are exact),
        # valley flags at their no-op defaults. Individual tests opt in.
        from src.config import settings
        monkeypatch.setattr(settings, "APPLY_CALIBRATION", False)
        monkeypatch.setattr(settings, "VALLEY_BLOCK_ENABLED", False)
        monkeypatch.setattr(settings, "VALLEY_MIN_EDGE", None)
        monkeypatch.setattr(settings, "VALLEY_PRICE_LOW", 0.60)
        monkeypatch.setattr(settings, "VALLEY_PRICE_HIGH", 0.85)

    def _valley_no(self):
        """A threshold at_least NO whose side price (0.70) is in the valley,
        passing every other filter (no_prob 0.90, edge 0.20)."""
        return _eval_binary(
            question="Will the highest temperature in X be at least 75",
            threshold_f=75, op="at_least", our_prob_in_window=0.10,
            yes_bid=0.30, yes_ask=0.32,
            forecast_peak_f=70.0, current_max_f=68.0, hours_until_peak=2.0,
        )

    def test_in_price_valley_is_half_open(self):
        from src.signals.edge_calculator import _in_price_valley
        assert _in_price_valley(0.60) is True
        assert _in_price_valley(0.84) is True
        assert _in_price_valley(0.85) is False   # upper bound exclusive
        assert _in_price_valley(0.599) is False
        assert _in_price_valley(0.55) is False    # deep-value extreme
        assert _in_price_valley(0.90) is False    # near-lock extreme

    def test_baseline_valley_trade_passes_when_flags_off(self):
        edge = self._valley_no()
        from src.db.models import TradeDirection
        assert edge.direction == TradeDirection.BUY_NO
        assert 0.60 <= edge.market_price < 0.85
        assert edge.passes is True

    def test_p1_block_rejects_valley_trade(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "VALLEY_BLOCK_ENABLED", True)
        edge = self._valley_no()
        assert edge.passes is False
        assert "price-valley blocked" in edge.reject_reason

    def test_p2_floor_rejects_low_edge_valley_trade(self, monkeypatch):
        from src.config import settings
        # Edge is 0.20; require 0.25 → blocked.
        monkeypatch.setattr(settings, "VALLEY_MIN_EDGE", 0.25)
        edge = self._valley_no()
        assert edge.passes is False
        assert "price-valley edge" in edge.reject_reason

    def test_p2_floor_passes_high_edge_valley_trade(self, monkeypatch):
        from src.config import settings
        # Edge is 0.20; require 0.15 → the high-edge valley trade survives.
        monkeypatch.setattr(settings, "VALLEY_MIN_EDGE", 0.15)
        edge = self._valley_no()
        assert edge.passes is True

    def test_p1_wins_over_p2_when_both_set(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "VALLEY_BLOCK_ENABLED", True)
        monkeypatch.setattr(settings, "VALLEY_MIN_EDGE", 0.10)  # would pass P2
        edge = self._valley_no()
        assert edge.passes is False
        assert "price-valley blocked" in edge.reject_reason

    def test_extremes_pass_even_when_block_enabled(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "VALLEY_BLOCK_ENABLED", True)
        # Deep-value NO at price 0.55 (yes_bid 0.45) — below the valley, the
        # +EV value band. no_prob 0.90, edge 0.35 → unambiguous pass.
        edge = _eval_binary(
            question="Will the highest temperature in X be at least 75",
            threshold_f=75, op="at_least", our_prob_in_window=0.10,
            yes_bid=0.45, yes_ask=0.47,
            forecast_peak_f=70.0, current_max_f=68.0, hours_until_peak=2.0,
        )
        assert edge.market_price < 0.60
        assert edge.passes is True

    def test_more_specific_reject_reason_wins(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "VALLEY_BLOCK_ENABLED", True)
        # Valley-priced NO (0.70) but edge 0.02 < MIN_EDGE → _check_filters
        # rejects on edge first; the valley guard must not overwrite it.
        edge = _eval_binary(
            question="Will the highest temperature in X be at least 75",
            threshold_f=75, op="at_least", our_prob_in_window=0.28,
            yes_bid=0.30, yes_ask=0.32,
            forecast_peak_f=70.0, current_max_f=68.0, hours_until_peak=2.0,
        )
        assert edge.passes is False
        assert "edge" in edge.reject_reason
        assert "price-valley" not in edge.reject_reason


class TestShadowValleyFields:
    """`shadow_valley_fields` — pure counterfactual for the P2 refinement,
    independent of the live VALLEY_* flags; gated by SHADOW_VALLEY_POLICY_ENABLED."""

    @pytest.fixture(autouse=True)
    def _defaults(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "SHADOW_VALLEY_POLICY_ENABLED", True)
        monkeypatch.setattr(settings, "SHADOW_VALLEY_MIN_EDGE", 0.15)
        monkeypatch.setattr(settings, "VALLEY_PRICE_LOW", 0.60)
        monkeypatch.setattr(settings, "VALLEY_PRICE_HIGH", 0.85)

    def test_valley_low_edge_would_block(self):
        from src.signals.edge_calculator import shadow_valley_fields
        f = shadow_valley_fields(0.70, 0.10)
        assert f["in_valley"] is True
        assert f["p2_would_block"] is True
        assert f["p2_min_edge"] == 0.15

    def test_valley_high_edge_would_not_block(self):
        from src.signals.edge_calculator import shadow_valley_fields
        f = shadow_valley_fields(0.70, 0.20)
        assert f["in_valley"] is True
        assert f["p2_would_block"] is False

    def test_outside_valley_never_blocks(self):
        from src.signals.edge_calculator import shadow_valley_fields
        f = shadow_valley_fields(0.90, 0.01)
        assert f["in_valley"] is False
        assert f["p2_would_block"] is False

    def test_disabled_returns_none(self, monkeypatch):
        from src.config import settings
        from src.signals.edge_calculator import shadow_valley_fields
        monkeypatch.setattr(settings, "SHADOW_VALLEY_POLICY_ENABLED", False)
        assert shadow_valley_fields(0.70, 0.10) is None
