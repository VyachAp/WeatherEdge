"""Weather Underground history page scraper.

Headlessly fetches the daily history table from wunderground.com
for a given ICAO station and date.  Returns structured daily summary
and hourly observation data for comparison against the WX API.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WuHourlyReading:
    """Single hourly row from the WU history table."""

    time_local: str  # e.g. "2:53 AM"
    temp_f: float | None = None
    dew_point_f: float | None = None
    humidity_pct: float | None = None
    wind_dir: str | None = None
    wind_mph: float | None = None
    wind_gust_mph: float | None = None
    pressure_in: float | None = None
    precip_in: float | None = None
    condition: str | None = None


@dataclass(frozen=True)
class WuDailySummary:
    """Scraped daily history from Weather Underground."""

    station_icao: str
    date: date
    high_f: float | None = None
    low_f: float | None = None
    avg_f: float | None = None
    precip_in: float | None = None
    dew_point_high_f: float | None = None
    dew_point_low_f: float | None = None
    dew_point_avg_f: float | None = None
    humidity_high_pct: float | None = None
    humidity_low_pct: float | None = None
    wind_speed_max_mph: float | None = None
    pressure_in: float | None = None
    visibility_mi: float | None = None
    hourly: list[WuHourlyReading] = field(default_factory=list)
    scraped_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d+\.?\d*")


def _parse_num(text: str) -> float | None:
    """Extract a number from a cell value like '73 °F' or '29.28 in'."""
    text = text.strip()
    if not text or text == "-" or text == "--":
        return None
    m = _NUM_RE.search(text)
    return float(m.group()) if m else None


def _parse_summary_table(rows_text: list[list[str]]) -> dict[str, float | None]:
    """Parse the daily summary table (Table 0) into a flat dict.

    Each row is [label, actual, historic_avg, record] but some rows
    are section headers like 'Temperature (°F)' with no data columns.
    """
    result: dict[str, float | None] = {}
    for cells in rows_text:
        if len(cells) < 2:
            continue
        label = cells[0].strip().lower()
        actual = cells[1].strip() if len(cells) > 1 else ""

        if label.startswith("high temp"):
            result["high_f"] = _parse_num(actual)
        elif label.startswith("low temp"):
            result["low_f"] = _parse_num(actual)
        elif label.startswith("day average"):
            result["avg_f"] = _parse_num(actual)
        elif label.startswith("precipitation") and "past" in label:
            result["precip_in"] = _parse_num(actual)
        elif label == "high" and "dew_point_high_f" not in result:
            result["dew_point_high_f"] = _parse_num(actual)
        elif label == "low" and "dew_point_low_f" not in result:
            result["dew_point_low_f"] = _parse_num(actual)
        elif label == "average" and "dew_point_avg_f" not in result:
            result["dew_point_avg_f"] = _parse_num(actual)
        elif label.startswith("max wind"):
            result["wind_speed_max_mph"] = _parse_num(actual)
        elif label.startswith("visibility"):
            result["visibility_mi"] = _parse_num(actual)
        elif label.startswith("sea level pressure"):
            result["pressure_in"] = _parse_num(actual)
    return result


def _parse_hourly_row(cells: list[str]) -> WuHourlyReading | None:
    """Parse a single hourly observation row.

    Expected columns: Time, Temp, DewPoint, Humidity, WindDir,
    WindSpeed, WindGust, Pressure, Precip, Condition.
    """
    if len(cells) < 10:
        return None

    time_str = cells[0].strip()
    # Skip header rows or non-time rows
    if not time_str or not re.search(r"\d", time_str):
        return None
    # Must look like a time (e.g. "12:53 AM")
    if not re.match(r"\d{1,2}:\d{2}\s*[APap]", time_str):
        return None

    return WuHourlyReading(
        time_local=time_str,
        temp_f=_parse_num(cells[1]),
        dew_point_f=_parse_num(cells[2]),
        humidity_pct=_parse_num(cells[3]),
        wind_dir=cells[4].strip() or None,
        wind_mph=_parse_num(cells[5]),
        wind_gust_mph=_parse_num(cells[6]),
        pressure_in=_parse_num(cells[7]),
        precip_in=_parse_num(cells[8]),
        condition=cells[9].strip() or None,
    )


# ---------------------------------------------------------------------------
# Browser management
# ---------------------------------------------------------------------------

_browser = None
_pw_instance = None
_browser_lock = asyncio.Lock()


async def _get_browser():
    """Lazily start a headless Chromium browser (singleton)."""
    global _browser, _pw_instance
    async with _browser_lock:
        if _browser is not None and _browser.is_connected():
            return _browser

        from playwright.async_api import async_playwright

        _pw_instance = await async_playwright().start()
        _browser = await _pw_instance.chromium.launch(headless=True)
        return _browser


async def close_browser() -> None:
    """Shut down the shared browser instance."""
    global _browser, _pw_instance
    async with _browser_lock:
        if _browser is not None:
            try:
                await _browser.close()
            except Exception:
                pass
            _browser = None
        if _pw_instance is not None:
            try:
                await _pw_instance.stop()
            except Exception:
                pass
            _pw_instance = None


_scrape_semaphore = asyncio.Semaphore(1)
_last_scrape_at: float = 0.0


# ---------------------------------------------------------------------------
# Core scraping
# ---------------------------------------------------------------------------


async def fetch_wu_history(
    station_icao: str,
    target_date: date,
) -> WuDailySummary | None:
    """Scrape WU history page and return structured data.

    Returns None if the page cannot be loaded or parsed.
    """
    import time as _time

    global _last_scrape_at

    url = (
        f"https://www.wunderground.com/history/daily/"
        f"{station_icao}/date/{target_date.year}-{target_date.month}-{target_date.day}"
    )

    async with _scrape_semaphore:
        # Rate-limit: wait between scrapes
        now = _time.monotonic()
        elapsed = now - _last_scrape_at
        delay = settings.WU_SCRAPE_DELAY_SECONDS
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)

        try:
            browser = await _get_browser()
            page = await browser.new_page()
            try:
                result = await _scrape_page(page, url, station_icao, target_date)
            finally:
                await page.close()
        except Exception:
            logger.exception("WU scrape failed for %s %s", station_icao, target_date)
            result = None
        finally:
            _last_scrape_at = _time.monotonic()

    return result


async def _scrape_page(
    page,
    url: str,
    station_icao: str,
    target_date: date,
) -> WuDailySummary | None:
    """Navigate to WU history page and extract data."""
    timeout = settings.WU_SCRAPE_TIMEOUT_MS

    await page.goto(url, wait_until="domcontentloaded", timeout=timeout)

    # Wait for at least one table to render
    try:
        await page.wait_for_selector("table", timeout=timeout)
    except Exception:
        logger.warning("No table rendered for %s %s", station_icao, target_date)
        return None

    # Small extra wait for JS-rendered content
    await asyncio.sleep(2)

    tables = await page.query_selector_all("table")
    if not tables:
        return None

    # --- Parse daily summary (first table) ---
    summary_data: dict[str, float | None] = {}
    if len(tables) >= 1:
        summary_rows = await _extract_table_rows(tables[0])
        summary_data = _parse_summary_table(summary_rows)

    # --- Parse hourly observations (second table) ---
    hourly: list[WuHourlyReading] = []
    if len(tables) >= 2:
        hourly_rows = await _extract_table_rows(tables[1])
        for row_cells in hourly_rows:
            reading = _parse_hourly_row(row_cells)
            if reading is not None:
                hourly.append(reading)

    return WuDailySummary(
        station_icao=station_icao,
        date=target_date,
        high_f=summary_data.get("high_f"),
        low_f=summary_data.get("low_f"),
        avg_f=summary_data.get("avg_f"),
        precip_in=summary_data.get("precip_in"),
        dew_point_high_f=summary_data.get("dew_point_high_f"),
        dew_point_low_f=summary_data.get("dew_point_low_f"),
        dew_point_avg_f=summary_data.get("dew_point_avg_f"),
        wind_speed_max_mph=summary_data.get("wind_speed_max_mph"),
        pressure_in=summary_data.get("pressure_in"),
        visibility_mi=summary_data.get("visibility_mi"),
        hourly=hourly,
    )


async def _extract_table_rows(table) -> list[list[str]]:
    """Extract all <tr> rows from a table as lists of cell text."""
    rows = await table.query_selector_all("tr")
    result: list[list[str]] = []
    for row in rows:
        cells = await row.query_selector_all("td, th")
        cell_texts = []
        for cell in cells:
            text = await cell.inner_text()
            cell_texts.append(text.strip())
        if cell_texts:
            result.append(cell_texts)
    return result


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def compare_wu_vs_wx(
    wu: WuDailySummary,
    wx_high_f: float | None,
    wx_low_f: float | None,
) -> dict[str, float | None]:
    """Compare WU scraped data against WX API observed high/low.

    Returns dict with wu values, api values, and deltas.
    """
    result: dict[str, float | None] = {
        "high_f_wu": wu.high_f,
        "high_f_api": wx_high_f,
        "high_f_delta": None,
        "low_f_wu": wu.low_f,
        "low_f_api": wx_low_f,
        "low_f_delta": None,
    }
    if wu.high_f is not None and wx_high_f is not None:
        result["high_f_delta"] = wu.high_f - wx_high_f
    if wu.low_f is not None and wx_low_f is not None:
        result["low_f_delta"] = wu.low_f - wx_low_f
    return result
