"""Polymarket weather-market scanner.

Fetches active markets from the Gamma API, filters for weather-related
contracts, parses question text into structured fields via regex, and
persists markets + price snapshots to the database.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.db.engine import async_session
from src.db.models import Market, MarketSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA_BASE = "https://gamma-api.polymarket.com"
MARKETS_URL = f"{GAMMA_BASE}/markets"

WEATHER_KEYWORDS: set[str] = {
    "temperature", "°f", "°c",
}

# Compiled pattern for temperature-market matching.
import re as _re
_WEATHER_KW_RE = _re.compile(
    r"\btemperature\b|°[fc]",
    _re.IGNORECASE,
)

WEATHER_TAGS: list[str] = ["weather"]

# Rate-limit: ≤10 requests / second
_rate_semaphore = asyncio.Semaphore(10)
_rate_interval = 1.0  # seconds

# US states for location extraction
_US_STATES = (
    "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|"
    "Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|"
    "Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|"
    "Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|"
    "New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|"
    "Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|"
    "Virginia|Washington|West Virginia|Wisconsin|Wyoming"
)

# Major cities used in weather markets
_CITIES = (
    "New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|"
    "San Diego|Dallas|San Jose|Austin|Jacksonville|San Francisco|Columbus|"
    "Indianapolis|Seattle|Denver|Nashville|Oklahoma City|Portland|Las Vegas|"
    "Memphis|Louisville|Baltimore|Milwaukee|Albuquerque|Tucson|Fresno|"
    "Sacramento|Kansas City|Miami|Atlanta|Omaha|Minneapolis|Tampa|"
    "New Orleans|Cleveland|Orlando|St\\. Louis|Pittsburgh|Cincinnati|"
    "Anchorage|Honolulu|Detroit|Boston|Washington D\\.C\\.|Washington DC"
)

# Month names for date extraction
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|"
    "October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
)

# ---------------------------------------------------------------------------
# Parsed question dataclass
# ---------------------------------------------------------------------------


@dataclass
class ParsedQuestion:
    location: str | None = None
    variable: str | None = None
    threshold: float | None = None
    operator: str | None = None      # above / below / exactly / at_least / at_most
    target_date: str | None = None    # free-text date or range
    matched: bool = False
    raw: str = ""
    pattern_index: int | None = None   # which _PATTERNS entry matched (0-based)
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regex question parsers (≥10 patterns)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern[str], dict[str, str]]] = []


def _p(pattern: str, defaults: dict[str, str] | None = None) -> None:
    """Register a compiled pattern with optional default field values."""
    _PATTERNS.append((re.compile(pattern, re.IGNORECASE), defaults or {}))


# Location pattern: one or more capitalized words, allows "D.C.", "St. Louis"
_L = r"(?P<location>(?:[A-Z][A-Za-z'\.]*(?:\s+(?:of|the|D\.C\.|DC))?(?:\s+[A-Z][A-Za-z'\.]*)*))"


def _loc(name: str = "location") -> str:
    """Return location capture group, optionally with a custom group name."""
    return _L if name == "location" else _L.replace("location", name)


# 1 — Temperature above/below threshold in a city on a date
#     "Will the temperature in Phoenix exceed 115°F on July 15, 2026?"
_p(
    r"temperature\s+in\s+" + _L + r"\s+(?P<operator>exceed|reach|drop below|fall below|go above|go below|hit|surpass)\s+(?P<threshold>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[FC])",
    {"variable": "temperature"},
)

# 2 — High temperature record
#     "Will New York City record a high above 100°F in July 2026?"
_p(
    _L + r"\s+record\s+a\s+(?:high|low)\s+(?P<operator>above|below|at or above|at or below)\s+(?P<threshold>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[FC])",
    {"variable": "temperature"},
)

# 3 — Precipitation / rainfall amount
#     "Will rainfall in Houston exceed 5 inches on September 12?"
_p(
    r"(?:rainfall|precipitation|rain)\s+in\s+" + _L + r"\s+(?P<operator>exceed|surpass|reach|go above|top)\s+(?P<threshold>\d+(?:\.\d+)?)\s*(?:inches|in\b|mm)",
    {"variable": "precipitation"},
)

# 4 — Snowfall amount
#     "Will snowfall in Denver exceed 12 inches in January 2026?"
_p(
    r"snowfall\s+in\s+" + _L + r"\s+(?P<operator>exceed|surpass|reach|top)\s+(?P<threshold>\d+(?:\.\d+)?)\s*(?:inches|in\b|cm)",
    {"variable": "snowfall"},
)

# 5 — Hurricane landfall
#     "Will a hurricane make landfall in Florida in 2026?"
_p(
    r"(?:a\s+)?(?:hurricane|tropical storm|cyclone)\s+(?:make\s+)?landfall\s+in\s+" + _L + r"(?:\s+in\s+(?P<date>\d{4}))?\s*\??",
    {"variable": "hurricane_landfall", "operator": "occurs"},
)

# 6 — Named hurricane category
#     "Will Hurricane Milton reach Category 5?"
_p(
    r"hurricane\s+(?P<name>[A-Z]\w+)\s+(?:reach|become|intensify to|hit)\s+(?:a\s+)?category\s+(?P<threshold>[1-5])",
    {"variable": "hurricane_category", "operator": "at_least"},
)

# 7 — Heat wave / consecutive days above threshold
#     "Will Chicago have 5 consecutive days above 95°F in July?"
_p(
    _L + r"\s+have\s+(?P<days>\d+)\s+consecutive\s+days\s+(?P<operator>above|below)\s+(?P<threshold>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[FC])",
    {"variable": "heat_wave"},
)

# 8 — Freeze / frost
#     "Will there be a freeze in Atlanta before November 15, 2026?"
_p(
    r"(?:a\s+)?(?:freeze|frost|freezing temperatures?)\s+in\s+" + _L + r"(?:\s+(?:before|by|after)\s+(?P<date>[\w\s,]+))?\s*\??",
    {"variable": "freeze", "operator": "occurs", "threshold": "32"},
)

# 9 — Drought declaration
#     "Will a drought be declared in California in 2026?"
_p(
    r"(?:a\s+)?drought\s+(?:be\s+)?(?:declared|announced)\s+in\s+" + _L + r"(?:\s+in\s+(?P<date>[\w\s,]+))?\s*\??",
    {"variable": "drought", "operator": "occurs"},
)

# 10 — Tornado count
#      "Will there be more than 20 tornadoes in Oklahoma in April 2026?"
_p(
    r"(?:more than|at least|fewer than|over|under)\s+(?P<threshold>\d+)\s+tornado(?:e?s)?\s+in\s+" + _L + r"(?:\s+in\s+(?P<date>[\w\s,]+))?\s*\??",
    {"variable": "tornado_count"},
)

# 11 — Wildfire acreage
#      "Will wildfires burn more than 500,000 acres in California in 2026?"
_p(
    r"wildfire[s]?\s+burn\s+(?:more than|over|at least)\s+(?P<threshold>[\d,]+)\s+acres\s+in\s+" + _L + r"(?:\s+in\s+(?P<date>[\w\s,]+))?\s*\??",
    {"variable": "wildfire_acreage", "operator": "above"},
)

# 12 — Flood event
#      "Will there be major flooding in Houston in 2026?"
_p(
    r"(?:major\s+)?flood(?:ing)?\s+in\s+" + _L + r"(?:\s+in\s+(?P<date>[\w\s,]+))?\s*\??",
    {"variable": "flood", "operator": "occurs"},
)

# 13 — Wind speed threshold
#      "Will wind speeds in Miami exceed 100 mph during hurricane season?"
_p(
    r"wind\s+(?:speeds?|gusts?)\s+in\s+" + _L + r"\s+(?P<operator>exceed|surpass|reach|top)\s+(?P<threshold>\d+(?:\.\d+)?)\s*(?:mph|km/h|knots)",
    {"variable": "wind_speed"},
)

# 14 — Generic "above/below X" with degrees
#      "Will it be above 110°F in Death Valley on August 1?"
_p(
    r"(?:be|go|get|stay)\s+(?P<operator>above|below)\s+(?P<threshold>-?\d+(?:\.\d+)?)\s*°\s*(?P<unit>[FC])\s+in\s+" + _L,
    {"variable": "temperature"},
)

# 15 — Record-breaking phrasing
#      "Will 2026 be the hottest year on record in the US?"
_p(
    r"(?P<date>\d{4})\s+be\s+the\s+(?P<superlative>hottest|coldest|wettest|driest|warmest|snowiest)\s+(?:year|month|season|summer|winter)\s+(?:on record\s+)?in\s+(?:the\s+)?" + _L,
    {"variable": "record", "operator": "record_breaking"},
)

# 16a — Single-value high/low temperature market (Yes/No)
#       "Will the highest temperature in Paris be 22°C on April 11?"
#       "Will the highest temperature in Paris be 28°C or higher on April 11?"
#       "Will the lowest temperature in Seoul be 5°C or lower on April 12?"
_p(
    r"(?P<hilo>highest|lowest|high|low)\s+temperature\s+in\s+" + _L +
    r"\s+be\s+(?P<threshold>-?\d+(?:\.\d+)?)\s*°?\s*(?:[FC])?"
    r"(?:\s+or\s+(?P<operator>higher|lower))?"
    r"(?:\s+on\s+(?P<date>[\w\s,]+))?\s*\??",
    {"variable": "temperature", "operator": "at_least"},
)

# 16b — Bracket temperature market (daily high/low)
#       "Highest temperature in Austin on April 8?"
#       "Lowest temperature in Seoul on April 12?"
_p(
    r"(?:highest|lowest|high|low)\s+temperature\s+in\s+" + _L +
    r"(?:\s+on\s+(?P<date>[\w\s,]+))?\s*\??",
    {"variable": "temperature", "operator": "bracket"},
)

# Operator normalization map
_OP_MAP = {
    "exceed": "above", "surpass": "above", "go above": "above", "top": "above",
    "reach": "at_least", "hit": "at_least",
    "drop below": "below", "fall below": "below", "go below": "below",
    "at or above": "at_least", "at or below": "at_most",
    "higher": "above", "lower": "below",
}


def _normalize_operator(raw_op: str | None) -> str | None:
    if raw_op is None:
        return None
    return _OP_MAP.get(raw_op.lower().strip(), raw_op.lower().strip())


# ---------------------------------------------------------------------------
# Bracket outcome parsing (for daily temperature markets)
# ---------------------------------------------------------------------------

_BRACKET_RE = re.compile(
    r"(?P<low>-?\d+(?:\.\d+)?)\s*[-–]\s*(?P<high>-?\d+(?:\.\d+)?)\s*°?\s*[FC]?",
    re.IGNORECASE,
)
_BRACKET_GTE_RE = re.compile(
    r"[≥>=]+\s*(?P<val>-?\d+(?:\.\d+)?)\s*°?\s*[FC]?",
    re.IGNORECASE,
)
_BRACKET_LTE_RE = re.compile(
    r"[≤<=]+\s*(?P<val>-?\d+(?:\.\d+)?)\s*°?\s*[FC]?",
    re.IGNORECASE,
)


def parse_temperature_brackets(
    outcomes: list[str] | None,
) -> list[tuple[float, float]] | None:
    """Parse bracket outcomes into (low_f, high_f) bounds.

    Examples:
        ["65-69°F", "70-74°F", "≥90°F"]
        → [(65, 70), (70, 75), (90, 150)]

    The upper bound is exclusive (+1 from the stated range end).
    Open-ended brackets use -60 / 150 as sentinels.
    Returns None if outcomes cannot be parsed as brackets.
    """
    if not outcomes or len(outcomes) < 2:
        return None

    brackets: list[tuple[float, float]] = []
    for o in outcomes:
        o_stripped = o.strip()
        m = _BRACKET_RE.match(o_stripped)
        if m:
            low = float(m.group("low"))
            high = float(m.group("high")) + 1  # exclusive upper bound
            brackets.append((low, high))
            continue
        m = _BRACKET_GTE_RE.match(o_stripped)
        if m:
            brackets.append((float(m.group("val")), 150.0))
            continue
        m = _BRACKET_LTE_RE.match(o_stripped)
        if m:
            brackets.append((-60.0, float(m.group("val")) + 1))
            continue
        # Outcome doesn't look like a temperature bracket
        return None

    return brackets if brackets else None


def _extract_date_from_text(text: str) -> str | None:
    """Best-effort date/range extraction from trailing question text."""
    # "on July 15, 2026" / "in January 2026" / "in 2026" / "before Nov 1"
    # Try Month Day, Year first, then Month Year, then bare Year
    m = re.search(
        rf"(?:on|in|before|after|by|during|for)\s+"
        rf"((?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}})",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        rf"(?:on|in|before|after|by|during|for)\s+"
        rf"((?:{_MONTHS})\s+\d{{4}})",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        rf"(?:on|in|before|after|by|during|for)\s+"
        rf"((?:{_MONTHS})\s+\d{{1,2}})\b",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r"(?:on|in|before|after|by|during|for)\s+(\d{4})\b",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    # date range "July 1 - July 15, 2026"
    m = re.search(
        rf"((?:{_MONTHS})\s+\d{{1,2}}\s*[-–]\s*(?:{_MONTHS})?\s*\d{{1,2}}(?:,?\s*\d{{4}})?)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


def parse_question(question: str) -> ParsedQuestion:
    """Parse a Polymarket weather-market question into structured fields."""
    for pat_idx, (pattern, defaults) in enumerate(_PATTERNS):
        m = pattern.search(question)
        if not m:
            continue

        groups = m.groupdict()
        location = groups.get("location", defaults.get("location"))
        if location:
            location = location.strip().rstrip("?.,")
            # IGNORECASE makes [A-Z] match lowercase, so the location group
            # may swallow trailing words like " in", " before".  Trim them:
            # keep only the leading run of capitalized words (+ connectors).
            # Strip leading filler words before cleanup (handles "the US ...")
            location = re.sub(
                r"^(?:Will|Does|Is|Can|Has|the)\s+", "", location,
                flags=re.IGNORECASE,
            ).strip()
            # Keep only leading run of capitalized words (case-sensitive
            # so lowercase prepositions like "in", "before" act as stop words).
            loc_match = re.match(
                r"(?:[A-Z][A-Za-z'\.]*(?:\s+(?:of|the|D\.C\.|DC)\b)?(?:\s+[A-Z][A-Za-z'\.]*)*)",
                location,
            )
            if loc_match:
                location = loc_match.group(0).strip()

        raw_op = groups.get("operator") or defaults.get("operator")
        operator = _normalize_operator(raw_op)

        threshold_str = groups.get("threshold", defaults.get("threshold"))
        threshold: float | None = None
        if threshold_str is not None:
            threshold_str = threshold_str.replace(",", "")
            try:
                threshold = float(threshold_str)
            except ValueError:
                pass

        target_date = groups.get("date") or _extract_date_from_text(question)

        return ParsedQuestion(
            location=location,
            variable=groups.get("variable") or defaults.get("variable"),
            threshold=threshold,
            operator=operator,
            target_date=target_date,
            matched=True,
            raw=question,
            pattern_index=pat_idx,
            extras={k: v for k, v in groups.items()
                    if k not in ("location", "variable", "threshold", "operator", "date")
                    and v is not None},
        )

    # Fallback — no pattern matched, try to extract date anyway
    return ParsedQuestion(
        matched=False,
        raw=question,
        target_date=_extract_date_from_text(question),
    )


# ---------------------------------------------------------------------------
# Weather-keyword filter
# ---------------------------------------------------------------------------


def is_weather_market(market: dict[str, Any]) -> bool:
    """Return True if a Gamma API market dict is weather-related."""
    question = market.get("question") or ""
    if _WEATHER_KW_RE.search(question):
        return True
    tags: list[str] = market.get("tags") or []
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict):
                tag = tag.get("label", "")
            if str(tag).lower() in WEATHER_KEYWORDS | {"climate"}:
                return True
    return False


# ---------------------------------------------------------------------------
# Rate-limited HTTP helpers
# ---------------------------------------------------------------------------


async def _throttle() -> None:
    """Simple token-bucket: acquire semaphore, release after interval."""
    await _rate_semaphore.acquire()

    async def _release() -> None:
        await asyncio.sleep(_rate_interval)
        _rate_semaphore.release()

    asyncio.create_task(_release())


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    reraise=True,
)
async def _get_json(client: httpx.AsyncClient, url: str,
                    params: dict[str, Any] | None = None) -> Any:
    await _throttle()
    resp = await client.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Market fetching
# ---------------------------------------------------------------------------


async def fetch_weather_markets(client: httpx.AsyncClient | None = None) -> list[dict[str, Any]]:
    """Fetch active weather markets from the Gamma API.

    Strategy:
      1. Paginate through all active markets, keyword-filter locally.
      2. Also query tag-filtered endpoints for "weather" and "climate".
      3. Deduplicate by market id.
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()

    seen: dict[str, dict[str, Any]] = {}

    try:
        # --- keyword scan: paginate through active markets ---
        offset = 0
        limit = 100
        while True:
            params: dict[str, Any] = {
                "active": "true",
                "closed": "false",
                "limit": limit,
                "offset": offset,
            }
            try:
                batch = await _get_json(client, MARKETS_URL, params=params)
            except Exception:
                logger.exception("Failed to fetch markets at offset %d", offset)
                break

            if not batch:
                break

            for m in batch:
                mid = m.get("id") or m.get("conditionId")
                if mid and mid not in seen and is_weather_market(m):
                    seen[mid] = m

            if len(batch) < limit:
                break
            offset += limit
            logger.debug("Fetched %d markets so far, %d weather", offset, len(seen))

        # --- tag-based queries ---
        for tag in WEATHER_TAGS:
            try:
                batch = await _get_json(client, MARKETS_URL, params={
                    "active": "true", "closed": "false",
                    "tag": tag, "limit": 100,
                })
            except Exception:
                logger.exception("Failed tag query for %s", tag)
                continue
            for m in (batch or []):
                mid = m.get("id") or m.get("conditionId")
                if mid and mid not in seen:
                    seen[mid] = m

    finally:
        if own_client:
            await client.aclose()

    logger.info("Found %d weather markets", len(seen))
    return list(seen.values())


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _market_to_row(raw: dict[str, Any], parsed: ParsedQuestion, now: datetime) -> dict[str, Any]:
    outcomes = raw.get("outcomes")
    if isinstance(outcomes, str):
        outcomes = [outcomes]

    yes_price: float | None = None
    tokens = raw.get("outcomePrices") or raw.get("tokens")
    if isinstance(tokens, list) and tokens:
        first = tokens[0]
        if isinstance(first, dict):
            yes_price = float(first.get("price", 0))
        else:
            try:
                yes_price = float(first)
            except (TypeError, ValueError):
                pass
    if yes_price is None:
        yes_price = _safe_float(raw.get("outcomePrices"))

    return dict(
        id=raw.get("id") or raw.get("conditionId") or "",
        question=raw.get("question", ""),
        slug=raw.get("slug"),
        outcomes=outcomes,
        current_yes_price=yes_price,
        volume=_safe_float(raw.get("volume")),
        liquidity=_safe_float(raw.get("liquidity")),
        end_date=_parse_dt(raw.get("endDate")),
        resolution_source=raw.get("resolutionSource"),
        tags=raw.get("tags"),
        parsed_location=parsed.location,
        parsed_variable=parsed.variable,
        parsed_threshold=parsed.threshold,
        parsed_operator=parsed.operator,
        parsed_target_date=parsed.target_date,
        fetched_at=now,
    )


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, (list, dict)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_dt(val: str | None) -> datetime | None:
    if not val:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    return None


async def ingest_markets(session: AsyncSession, raw_markets: list[dict[str, Any]]) -> int:
    """Parse and upsert markets + create snapshots. Returns count ingested."""
    from collections import Counter

    now = datetime.utcnow()
    count = 0

    # --- parse diagnostic counters ------------------------------------------
    by_pattern: Counter[str] = Counter()     # pattern_index → count
    by_variable: Counter[str] = Counter()    # variable → count
    by_operator: Counter[str] = Counter()    # operator → count
    unmatched_questions: list[str] = []       # every unmatched question
    matched_count = 0

    for raw in raw_markets:
        mid = raw.get("id") or raw.get("conditionId")
        if not mid:
            continue

        parsed = parse_question(raw.get("question", ""))
        row_data = _market_to_row(raw, parsed, now)

        # track parse results
        if parsed.matched:
            matched_count += 1
            by_pattern[f"p{parsed.pattern_index}"] += 1
            by_variable[parsed.variable or "?"] += 1
            by_operator[parsed.operator or "none"] += 1
        else:
            unmatched_questions.append(raw.get("question", "")[:120])

        existing = await session.get(Market, mid)
        if existing:
            for k, v in row_data.items():
                if k != "id":
                    setattr(existing, k, v)
        else:
            session.add(Market(**row_data))

        # snapshot
        session.add(MarketSnapshot(
            market_id=mid,
            yes_price=row_data["current_yes_price"],
            no_price=(1.0 - row_data["current_yes_price"]) if row_data["current_yes_price"] is not None else None,
            volume=row_data["volume"],
            liquidity=row_data["liquidity"],
            timestamp=now,
        ))
        count += 1

    await session.commit()
    logger.info("Ingested %d markets with snapshots", count)

    # --- log parse diagnostic -----------------------------------------------
    logger.info(
        "Parse stats: matched=%d/%d by_pattern=%s",
        matched_count, count, dict(by_pattern),
    )
    logger.info("Parse by_variable=%s by_operator=%s", dict(by_variable), dict(by_operator))
    if unmatched_questions:
        logger.info("Unmatched questions (%d):", len(unmatched_questions))
        for q in unmatched_questions:
            logger.info("  UNMATCHED: %s", q)

    return count


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


async def get_active_weather_markets(
    session: AsyncSession | None = None,
) -> list[Market]:
    """Return all active markets that have parsed weather parameters."""
    own_session = session is None
    if own_session:
        session = async_session()

    try:
        stmt = (
            select(Market)
            .where(Market.parsed_variable == "temperature")
            .where(Market.end_date > datetime.utcnow())
            .order_by(Market.end_date)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())
    finally:
        if own_session:
            await session.close()


async def get_market_history(
    market_id: str,
    session: AsyncSession | None = None,
) -> list[MarketSnapshot]:
    """Return price snapshots for a market, ordered by time."""
    own_session = session is None
    if own_session:
        session = async_session()

    try:
        stmt = (
            select(MarketSnapshot)
            .where(MarketSnapshot.market_id == market_id)
            .order_by(MarketSnapshot.timestamp)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())
    finally:
        if own_session:
            await session.close()


# ---------------------------------------------------------------------------
# Top-level scan entrypoint
# ---------------------------------------------------------------------------


async def scan_and_ingest() -> int:
    """Full scan: fetch from Gamma API → parse → store. Returns count."""
    raw = await fetch_weather_markets()
    if not raw:
        logger.warning("No weather markets found")
        return 0

    async with async_session() as session:
        return await ingest_markets(session, raw)