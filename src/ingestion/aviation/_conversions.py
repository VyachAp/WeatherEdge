"""Unit conversion utilities for aviation weather data.

All internal storage is metric. Convert on output based on market question units.
"""

from __future__ import annotations


def c_to_f(temp_c: float | None) -> float | None:
    """Celsius to Fahrenheit."""
    if temp_c is None:
        return None
    return temp_c * 9.0 / 5.0 + 32.0


def f_to_c(temp_f: float | None) -> float | None:
    """Fahrenheit to Celsius."""
    if temp_f is None:
        return None
    return (temp_f - 32.0) * 5.0 / 9.0


def kts_to_mph(kts: float | None) -> float | None:
    """Knots to miles per hour."""
    if kts is None:
        return None
    return kts * 1.15078


def mph_to_kts(mph: float | None) -> float | None:
    """Miles per hour to knots."""
    if mph is None:
        return None
    return mph / 1.15078


def kts_to_kmh(kts: float | None) -> float | None:
    """Knots to kilometers per hour."""
    if kts is None:
        return None
    return kts * 1.852


def kts_to_ms(kts: float | None) -> float | None:
    """Knots to meters per second."""
    if kts is None:
        return None
    return kts * 0.514444


def m_to_miles(m: float | None) -> float | None:
    """Meters to statute miles."""
    if m is None:
        return None
    return m / 1609.34


def miles_to_m(miles: float | None) -> float | None:
    """Statute miles to meters."""
    if miles is None:
        return None
    return miles * 1609.34


def m_to_ft(m: float | None) -> float | None:
    """Meters to feet."""
    if m is None:
        return None
    return m * 3.28084


def hpa_to_inhg(hpa: float | None) -> float | None:
    """Hectopascals (millibars) to inches of mercury."""
    if hpa is None:
        return None
    return hpa * 0.02953


def inhg_to_hpa(inhg: float | None) -> float | None:
    """Inches of mercury to hectopascals."""
    if inhg is None:
        return None
    return inhg / 0.02953


def hpa_to_mmhg(hpa: float | None) -> float | None:
    """Hectopascals to millimeters of mercury."""
    if hpa is None:
        return None
    return hpa * 0.750062


def mm_to_inches(mm: float | None) -> float | None:
    """Millimeters to inches."""
    if mm is None:
        return None
    return mm / 25.4


def inches_to_mm(inches: float | None) -> float | None:
    """Inches to millimeters."""
    if inches is None:
        return None
    return inches * 25.4


def mm_to_cm(mm: float | None) -> float | None:
    """Millimeters to centimeters."""
    if mm is None:
        return None
    return mm / 10.0


def nm_to_km(nm: float | None) -> float | None:
    """Nautical miles to kilometers."""
    if nm is None:
        return None
    return nm * 1.852
