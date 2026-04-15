"""Map active Polymarket temperature markets to aviation forecast probabilities.

Geocodes market locations via a static city lookup, parses target dates,
converts thresholds to SI units, and fetches aviation-based (METAR/TAF)
exceedance probabilities for each market.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from dateutil import parser as dateutil_parser

from src.ingestion import aviation
from src.ingestion.aviation import get_realtime_probability
from src.ingestion.polymarket import get_active_weather_markets

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models import Market

logger = logging.getLogger(__name__)

# Limit concurrent market-mapping coroutines to avoid DB connection exhaustion.
_FETCH_SEMAPHORE = asyncio.Semaphore(8)

# ---------------------------------------------------------------------------
# Geocoding – static lookup for major US cities & state capitals
# ---------------------------------------------------------------------------

CITIES: dict[str, tuple[float, float]] = {
    # Top 100 US cities by population (lat, lon)
    "new york city": (40.7128, -74.0060),
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698),
    "phoenix": (33.4484, -112.0740),
    "philadelphia": (39.9526, -75.1652),
    "san antonio": (29.4241, -98.4936),
    "san diego": (32.7157, -117.1611),
    "dallas": (32.7767, -96.7970),
    "san jose": (37.3382, -121.8863),
    "austin": (30.2672, -97.7431),
    "jacksonville": (30.3322, -81.6557),
    "fort worth": (32.7555, -97.3308),
    "columbus": (39.9612, -82.9988),
    "charlotte": (35.2271, -80.8431),
    "san francisco": (37.7749, -122.4194),
    "indianapolis": (39.7684, -86.1581),
    "seattle": (47.6062, -122.3321),
    "denver": (39.7392, -104.9903),
    "washington": (38.9072, -77.0369),
    "washington dc": (38.9072, -77.0369),
    "nashville": (36.1627, -86.7816),
    "oklahoma city": (35.4676, -97.5164),
    "el paso": (31.7619, -106.4850),
    "boston": (42.3601, -71.0589),
    "portland": (45.5152, -122.6784),
    "las vegas": (36.1699, -115.1398),
    "memphis": (35.1495, -90.0490),
    "louisville": (38.2527, -85.7585),
    "baltimore": (39.2904, -76.6122),
    "milwaukee": (43.0389, -87.9065),
    "albuquerque": (35.0844, -106.6504),
    "tucson": (32.2226, -110.9747),
    "fresno": (36.7378, -119.7871),
    "mesa": (33.4152, -111.8315),
    "sacramento": (38.5816, -121.4944),
    "atlanta": (33.7490, -84.3880),
    "kansas city": (39.0997, -94.5786),
    "colorado springs": (38.8339, -104.8214),
    "omaha": (41.2565, -95.9345),
    "raleigh": (35.7796, -78.6382),
    "long beach": (33.7701, -118.1937),
    "virginia beach": (36.8529, -75.9780),
    "miami": (25.7617, -80.1918),
    "oakland": (37.8044, -122.2712),
    "minneapolis": (44.9778, -93.2650),
    "tulsa": (36.1540, -95.9928),
    "tampa": (27.9506, -82.4572),
    "arlington": (32.7357, -97.1081),
    "new orleans": (29.9511, -90.0715),
    "wichita": (37.6872, -97.3301),
    "cleveland": (41.4993, -81.6944),
    "bakersfield": (35.3733, -119.0187),
    "aurora": (39.7294, -104.8319),
    "anaheim": (33.8366, -117.9143),
    "honolulu": (21.3069, -157.8583),
    "santa ana": (33.7455, -117.8677),
    "riverside": (33.9806, -117.3755),
    "corpus christi": (27.8006, -97.3964),
    "lexington": (38.0406, -84.5037),
    "pittsburgh": (40.4406, -79.9959),
    "anchorage": (61.2181, -149.9003),
    "stockton": (37.9577, -121.2908),
    "cincinnati": (39.1031, -84.5120),
    "saint paul": (44.9537, -93.0900),
    "st. paul": (44.9537, -93.0900),
    "toledo": (41.6528, -83.5379),
    "greensboro": (36.0726, -79.7920),
    "newark": (40.7357, -74.1724),
    "plano": (33.0198, -96.6989),
    "henderson": (36.0395, -114.9817),
    "lincoln": (40.8136, -96.7026),
    "buffalo": (42.8864, -78.8784),
    "jersey city": (40.7178, -74.0431),
    "chula vista": (32.6401, -117.0842),
    "norfolk": (36.8508, -76.2859),
    "detroit": (42.3314, -83.0458),
    "chandler": (33.3062, -111.8413),
    "laredo": (27.5036, -99.5076),
    "madison": (43.0731, -89.4012),
    "lubbock": (33.5779, -101.8552),
    "scottsdale": (33.4942, -111.9261),
    "reno": (39.5296, -119.8138),
    "glendale": (33.5387, -112.1860),
    "gilbert": (33.3528, -111.7890),
    "winston-salem": (36.0999, -80.2442),
    "north las vegas": (36.1989, -115.1175),
    "irving": (32.8140, -96.9489),
    "chesapeake": (36.7682, -76.2875),
    "boise": (43.6150, -116.2023),
    "richmond": (37.5407, -77.4360),
    "spokane": (47.6588, -117.4260),
    "baton rouge": (30.4515, -91.1871),
    "des moines": (41.5868, -93.6250),
    "tacoma": (47.2529, -122.4443),
    "birmingham": (33.5186, -86.8104),
    "salt lake city": (40.7608, -111.8910),
    "rochester": (43.1566, -77.6088),
    "modesto": (37.6391, -120.9969),
    "st. louis": (38.6270, -90.1994),
    "saint louis": (38.6270, -90.1994),
    # State names → capital coordinates
    "alabama": (32.3792, -86.3077),
    "alaska": (58.3005, -134.4197),
    "arizona": (33.4484, -112.0740),
    "arkansas": (34.7465, -92.2896),
    "california": (38.5816, -121.4944),
    "colorado": (39.7392, -104.9903),
    "connecticut": (41.7658, -72.6734),
    "delaware": (39.1582, -75.5244),
    "florida": (30.4383, -84.2807),
    "georgia": (33.7490, -84.3880),
    "hawaii": (21.3069, -157.8583),
    "idaho": (43.6150, -116.2023),
    "illinois": (39.7817, -89.6501),
    "indiana": (39.7684, -86.1581),
    "iowa": (41.5868, -93.6250),
    "kansas": (39.0473, -95.6752),
    "kentucky": (38.1867, -84.8753),
    "louisiana": (30.4515, -91.1871),
    "maine": (44.3106, -69.7795),
    "maryland": (38.9784, -76.4922),
    "massachusetts": (42.3601, -71.0589),
    "michigan": (42.7325, -84.5555),
    "minnesota": (44.9537, -93.0900),
    "mississippi": (32.2988, -90.1848),
    "missouri": (38.5767, -92.1736),
    "montana": (46.5958, -112.0270),
    "nebraska": (40.8136, -96.7026),
    "nevada": (39.1638, -119.7674),
    "new hampshire": (43.2067, -71.5381),
    "new jersey": (40.2206, -74.7699),
    "new mexico": (35.6672, -105.9644),
    "north carolina": (35.7796, -78.6382),
    "north dakota": (46.8083, -100.7837),
    "ohio": (39.9612, -82.9988),
    "oklahoma": (35.4676, -97.5164),
    "oregon": (44.9429, -123.0351),
    "pennsylvania": (40.2732, -76.8867),
    "rhode island": (41.8240, -71.4128),
    "south carolina": (34.0007, -81.0348),
    "south dakota": (44.3683, -100.3510),
    "tennessee": (36.1627, -86.7816),
    "texas": (30.2672, -97.7431),
    "utah": (40.7608, -111.8910),
    "vermont": (44.2601, -72.5754),
    "virginia": (37.5407, -77.4360),
    "west virginia": (38.3498, -81.6326),
    "wisconsin": (43.0731, -89.4012),
    "wyoming": (41.1400, -104.8202),
    # International cities commonly seen on Polymarket
    "seoul": (37.5665, 126.9780),
    "toronto": (43.6532, -79.3832),
    "kuala lumpur": (3.1390, 101.6869),
    "london": (51.5074, -0.1278),
    "tokyo": (35.6762, 139.6503),
    "sydney": (-33.8688, 151.2093),
    "paris": (48.8566, 2.3522),
    "bangkok": (13.7563, 100.5018),
    "singapore": (1.3521, 103.8198),
    "dubai": (25.2048, 55.2708),
    "mumbai": (19.0760, 72.8777),
    "mexico city": (19.4326, -99.1332),
    "berlin": (52.5200, 13.4050),
    "rome": (41.9028, 12.4964),
    "madrid": (40.4168, -3.7038),
    "beijing": (39.9042, 116.4074),
    "shanghai": (31.2304, 121.4737),
    "hong kong": (22.3193, 114.1694),
    "taipei": (25.0330, 121.5654),
    "cairo": (30.0444, 31.2357),
    "lagos": (6.5244, 3.3792),
    "buenos aires": (-34.6037, -58.3816),
    "são paulo": (-23.5505, -46.6333),
    "sao paulo": (-23.5505, -46.6333),
    "jakarta": (-6.2088, 106.8456),
    "istanbul": (41.0082, 28.9784),
    "moscow": (55.7558, 37.6173),
    "johannesburg": (-26.2041, 28.0473),
    "nairobi": (-1.2921, 36.8219),
    "lima": (-12.0464, -77.0428),
    "bogota": (4.7110, -74.0721),
    "montreal": (45.5017, -73.5673),
    "vancouver": (49.2827, -123.1207),
    "osaka": (34.6937, 135.5023),
    "delhi": (28.7041, 77.1025),
    "new delhi": (28.7041, 77.1025),
    "riyadh": (24.7136, 46.6753),
    "doha": (25.2854, 51.5310),
    "athens": (37.9838, 23.7275),
    "lisbon": (38.7223, -9.1393),
    "amsterdam": (52.3676, 4.9041),
    "zurich": (47.3769, 8.5417),
    "stockholm": (59.3293, 18.0686),
    "manila": (14.5995, 120.9842),
    "hanoi": (21.0278, 105.8342),
    "ho chi minh city": (10.8231, 106.6297),
    "tel aviv": (32.0853, 34.7818),
    "ankara": (39.9334, 32.8597),
    "wellington": (-41.2866, 174.7756),
    "lucknow": (26.8467, 80.9462),
    "munich": (48.1351, 11.5820),
    "milan": (45.4642, 9.1900),
    "warsaw": (52.2297, 21.0122),
    "chongqing": (29.4316, 106.9123),
    "wuhan": (30.5928, 114.3055),
    "chengdu": (30.5728, 104.0668),
    "shenzhen": (22.5431, 114.0579),
    "busan": (35.1796, 129.0756),
    "helsinki": (60.1699, 24.9384),
    "panama city": (8.9824, -79.5199),
    "cape town": (-33.9249, 18.4241),
    "jeddah": (21.4858, 39.1925),
    "guangzhou": (23.1291, 113.2644),
    "kolkata": (22.5726, 88.3639),
    "chennai": (13.0827, 80.2707),
    "bangalore": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "karachi": (24.8607, 67.0011),
    "lahore": (31.5204, 74.3587),
    "dhaka": (23.8103, 90.4125),
    "colombo": (6.9271, 79.8612),
    "kuching": (1.5535, 110.3593),
    "penang": (5.4164, 100.3327),
    "auckland": (-36.8485, 174.7633),
    "melbourne": (-37.8136, 144.9631),
    "brisbane": (-27.4698, 153.0251),
    "perth": (-31.9505, 115.8605),
    "osaka": (34.6937, 135.5023),
    "fukuoka": (33.5904, 130.4017),
    "sapporo": (43.0618, 141.3545),
    "prague": (50.0755, 14.4378),
    "budapest": (47.4979, 19.0402),
    "vienna": (48.2082, 16.3738),
    "brussels": (50.8503, 4.3517),
    "oslo": (59.9139, 10.7522),
    "copenhagen": (55.6761, 12.5683),
    "dublin": (53.3498, -6.2603),
    "edinburgh": (55.9533, -3.1883),
    "barcelona": (41.3874, 2.1686),
}


# ---------------------------------------------------------------------------
# ICAO station lookup – maps city names to nearest airport ICAO codes
# ---------------------------------------------------------------------------

CITY_ICAO: dict[str, str] = {
    "new york city": "KJFK",
    "new york": "KJFK",
    "los angeles": "KLAX",
    "chicago": "KORD",
    "houston": "KIAH",
    "phoenix": "KPHX",
    "philadelphia": "KPHL",
    "san antonio": "KSAT",
    "san diego": "KSAN",
    "dallas": "KDFW",
    "san jose": "KSJC",
    "austin": "KAUS",
    "jacksonville": "KJAX",
    "fort worth": "KDFW",
    "columbus": "KCMH",
    "charlotte": "KCLT",
    "san francisco": "KSFO",
    "indianapolis": "KIND",
    "seattle": "KSEA",
    "denver": "KDEN",
    "washington": "KDCA",
    "washington dc": "KDCA",
    "nashville": "KBNA",
    "oklahoma city": "KOKC",
    "el paso": "KELP",
    "boston": "KBOS",
    "portland": "KPDX",
    "las vegas": "KLAS",
    "memphis": "KMEM",
    "louisville": "KSDF",
    "baltimore": "KBWI",
    "milwaukee": "KMKE",
    "albuquerque": "KABQ",
    "tucson": "KTUS",
    "fresno": "KFAT",
    "mesa": "KPHX",
    "sacramento": "KSMF",
    "atlanta": "KATL",
    "kansas city": "KMCI",
    "colorado springs": "KCOS",
    "omaha": "KOMA",
    "raleigh": "KRDU",
    "long beach": "KLGB",
    "virginia beach": "KORF",
    "miami": "KMIA",
    "oakland": "KOAK",
    "minneapolis": "KMSP",
    "tulsa": "KTUL",
    "tampa": "KTPA",
    "arlington": "KDFW",
    "new orleans": "KMSY",
    "wichita": "KICT",
    "cleveland": "KCLE",
    "bakersfield": "KBFL",
    "aurora": "KDEN",
    "anaheim": "KSNA",
    "honolulu": "PHNL",
    "santa ana": "KSNA",
    "riverside": "KRAL",
    "corpus christi": "KCRP",
    "lexington": "KLEX",
    "pittsburgh": "KPIT",
    "anchorage": "PANC",
    "stockton": "KSCK",
    "cincinnati": "KCVG",
    "saint paul": "KMSP",
    "st. paul": "KMSP",
    "toledo": "KTOL",
    "greensboro": "KGSO",
    "newark": "KEWR",
    "plano": "KDFW",
    "henderson": "KLAS",
    "lincoln": "KLNK",
    "buffalo": "KBUF",
    "jersey city": "KEWR",
    "chula vista": "KSAN",
    "norfolk": "KORF",
    "detroit": "KDTW",
    "chandler": "KPHX",
    "laredo": "KLRD",
    "madison": "KMSN",
    "lubbock": "KLBB",
    "scottsdale": "KPHX",
    "reno": "KRNO",
    "glendale": "KPHX",
    "gilbert": "KPHX",
    "winston-salem": "KINT",
    "north las vegas": "KLAS",
    "irving": "KDFW",
    "chesapeake": "KORF",
    "boise": "KBOI",
    "richmond": "KRIC",
    "spokane": "KGEG",
    "baton rouge": "KBTR",
    "des moines": "KDSM",
    "tacoma": "KSEA",
    "birmingham": "KBHM",
    "salt lake city": "KSLC",
    "rochester": "KROC",
    "modesto": "KMOD",
    "st. louis": "KSTL",
    "saint louis": "KSTL",
    # State names → major airport in capital / largest city
    "alabama": "KBHM",
    "alaska": "PANC",
    "arizona": "KPHX",
    "arkansas": "KLIT",
    "california": "KLAX",
    "colorado": "KDEN",
    "connecticut": "KBDL",
    "delaware": "KILG",
    "florida": "KMIA",
    "georgia": "KATL",
    "hawaii": "PHNL",
    "idaho": "KBOI",
    "illinois": "KORD",
    "indiana": "KIND",
    "iowa": "KDSM",
    "kansas": "KICT",
    "kentucky": "KSDF",
    "louisiana": "KMSY",
    "maine": "KPWM",
    "maryland": "KBWI",
    "massachusetts": "KBOS",
    "michigan": "KDTW",
    "minnesota": "KMSP",
    "mississippi": "KJAN",
    "missouri": "KSTL",
    "montana": "KBZN",
    "nebraska": "KOMA",
    "nevada": "KLAS",
    "new hampshire": "KMHT",
    "new jersey": "KEWR",
    "new mexico": "KABQ",
    "north carolina": "KCLT",
    "north dakota": "KFAR",
    "ohio": "KCMH",
    "oklahoma": "KOKC",
    "oregon": "KPDX",
    "pennsylvania": "KPHL",
    "rhode island": "KPVD",
    "south carolina": "KCHS",
    "south dakota": "KFSD",
    "tennessee": "KBNA",
    "texas": "KDFW",
    "utah": "KSLC",
    "vermont": "KBTV",
    "virginia": "KRIC",
    "west virginia": "KCRW",
    "wisconsin": "KMKE",
    "wyoming": "KCPR",
    # International cities
    "seoul": "RKSI",
    "toronto": "CYYZ",
    "kuala lumpur": "WMKK",
    "london": "EGLL",
    "tokyo": "RJTT",
    "sydney": "YSSY",
    "paris": "LFPG",
    "bangkok": "VTBS",
    "singapore": "WSSS",
    "dubai": "OMDB",
    "mumbai": "VABB",
    "mexico city": "MMMX",
    "berlin": "EDDB",
    "rome": "LIRF",
    "madrid": "LEMD",
    "beijing": "ZBAA",
    "shanghai": "ZSPD",
    "hong kong": "VHHH",
    "taipei": "RCTP",
    "cairo": "HECA",
    "lagos": "DNMM",
    "buenos aires": "SAEZ",
    "são paulo": "SBGR",
    "sao paulo": "SBGR",
    "jakarta": "WIII",
    "istanbul": "LTFM",
    "moscow": "UUEE",
    "johannesburg": "FAOR",
    "nairobi": "HKJK",
    "lima": "SPJC",
    "bogota": "SKBO",
    "montreal": "CYUL",
    "vancouver": "CYVR",
    "osaka": "RJBB",
    "delhi": "VIDP",
    "new delhi": "VIDP",
    "riyadh": "OERK",
    "doha": "OTHH",
    "athens": "LGAV",
    "lisbon": "LPPT",
    "amsterdam": "EHAM",
    "zurich": "LSZH",
    "stockholm": "ESSA",
    "manila": "RPLL",
    "hanoi": "VVNB",
    "ho chi minh city": "VVTS",
    "tel aviv": "LLBG",
    "ankara": "LTAC",
    "wellington": "NZWN",
    "lucknow": "VILK",
    "munich": "EDDM",
    "milan": "LIMC",
    "warsaw": "EPWA",
    "chongqing": "ZUCK",
    "wuhan": "ZHHH",
    "chengdu": "ZUUU",
    "shenzhen": "ZGSZ",
    "busan": "RKPK",
    "helsinki": "EFHK",
    "panama city": "MPTO",
    "cape town": "FACT",
    "jeddah": "OEJN",
    "guangzhou": "ZGGG",
    "kolkata": "VECC",
    "chennai": "VOMM",
    "bangalore": "VOBL",
    "hyderabad": "VOHS",
    "karachi": "OPKC",
    "lahore": "OPLA",
    "dhaka": "VGHS",
    "colombo": "VCBI",
    "kuching": "WBGG",
    "penang": "WMKP",
    "auckland": "NZAA",
    "melbourne": "YMML",
    "brisbane": "YBBN",
    "perth": "YPPH",
    "osaka": "RJBB",
    "fukuoka": "RJFF",
    "sapporo": "RJCC",
    "prague": "LKPR",
    "budapest": "LHBP",
    "vienna": "LOWW",
    "brussels": "EBBR",
    "oslo": "ENGM",
    "copenhagen": "EKCH",
    "dublin": "EIDW",
    "edinburgh": "EGPH",
    "barcelona": "LEBL",
}


def icao_for_location(location: str) -> str | None:
    """Resolve a location name to its nearest ICAO station code.

    Tries exact match first, then substring containment.
    """
    key = location.strip().lower()
    if key in CITY_ICAO:
        return CITY_ICAO[key]

    for city, icao in CITY_ICAO.items():
        if key in city or city in key:
            return icao

    logger.warning("icao_for_location: unknown location %r", location)
    return None


def geocode(location: str) -> tuple[float, float] | None:
    """Resolve a location name to (lat, lon) via static lookup.

    Tries exact match first, then substring containment.
    """
    key = location.strip().lower()
    if key in CITIES:
        return CITIES[key]

    # Substring fallback – find the first city whose name contains the query
    for city, coords in CITIES.items():
        if key in city or city in key:
            return coords

    logger.warning("geocode: unknown location %r", location)
    return None


# ---------------------------------------------------------------------------
# Operator normalisation
# ---------------------------------------------------------------------------

OPERATOR_MAP: dict[str, str] = {
    "above": "above",
    "below": "below",
    "at_least": "above",
    "at_most": "below",
    "occurs": "below",  # freeze/frost events → temperature below threshold
}

SUPPORTED_VARIABLES: set[str] = {"temperature", "precipitation", "wind_speed"}

# Aliases map unsupported market variables to NWP-compatible ones.
VARIABLE_ALIASES: dict[str, dict] = {
    "snowfall": {
        "nwp_variable": "precipitation",
    },
    "freeze": {
        "nwp_variable": "temperature",
        "threshold_override": 32.0,  # always 32°F
        "operator_override": "below",
    },
    "heat_wave": {
        "nwp_variable": "temperature",
        "operator_override": "above",
    },
}


def resolve_variable(
    variable: str,
    threshold: float | None,
    operator: str | None,
) -> tuple[str, float | None, str | None]:
    """Resolve market variable to NWP-compatible variable via aliases.

    Returns (resolved_variable, resolved_threshold, resolved_operator).
    """
    alias = VARIABLE_ALIASES.get(variable)
    if alias is None:
        return variable, threshold, operator
    return (
        alias["nwp_variable"],
        alias.get("threshold_override", threshold),
        alias.get("operator_override", operator),
    )


def normalize_operator(op: str) -> str | None:
    """Map a parsed operator to the 'above'/'below' expected by forecast APIs."""
    return OPERATOR_MAP.get(op)


# ---------------------------------------------------------------------------
# Unit conversion  (market units → SI / GRIB units)
# ---------------------------------------------------------------------------


def convert_threshold(value: float, variable: str) -> float:
    """Convert a market threshold from imperial to SI units.

    - temperature: °F → K
    - precipitation: inches → kg/m² (≈ mm)
    - wind_speed: mph → m/s
    """
    if variable == "temperature":
        return (value - 32.0) * 5.0 / 9.0 + 273.15
    if variable == "precipitation":
        return value * 25.4
    if variable == "wind_speed":
        return value / 2.237
    return value


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def parse_target_date(date_str: str) -> datetime | None:
    """Parse a free-text date string into a timezone-aware UTC datetime.

    If only month+year are present (no day), defaults to the 15th.
    Returns ``None`` on failure.
    """
    try:
        now = datetime.now(tz=timezone.utc)
        dt = dateutil_parser.parse(
            date_str, fuzzy=True, default=datetime(now.year, now.month, 15),
        )
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
            dt = dt.replace(hour=23, minute=59, second=59)
        logger.debug("parse_target_date: %r -> %s", date_str, dt.isoformat())
        return dt
    except (ValueError, OverflowError):
        logger.warning("parse_target_date: could not parse %r", date_str)
        return None


# ---------------------------------------------------------------------------
# Market → signal mapping
# ---------------------------------------------------------------------------




@dataclass
class AviationContext:
    """Contextual intelligence from aviation weather sources."""

    taf_amendment_count: int = 0
    speci_events_2h: int = 0
    has_severe_pireps: bool = False
    active_sigmet_count: int = 0
    active_airmet_count: int = 0


@dataclass
class MarketSignal:
    """Intermediate representation linking a market to model probabilities."""

    market_id: str
    question: str
    market_prob: float
    aviation_prob: float | None
    days_to_resolution: int
    hours_to_resolution: float
    market: Market
    aviation_context: AviationContext | None = None


async def map_market(
    market: Market,
    session: AsyncSession | None = None,
) -> MarketSignal | None:
    """Map a single market to aviation forecast probabilities.

    Returns ``None`` when the market cannot be mapped (missing fields,
    unsupported variable, unknown location, or no aviation data).
    """
    # --- validate parsed fields -------------------------------------------
    if not all([
        market.parsed_location,
        market.parsed_variable,
        market.parsed_threshold is not None,
        market.parsed_operator,
        market.parsed_target_date,
    ]):
        missing = []
        if not market.parsed_location:
            missing.append("location")
        if not market.parsed_variable:
            missing.append("variable")
        if market.parsed_threshold is None:
            missing.append("threshold")
        if not market.parsed_operator:
            missing.append("operator")
        if not market.parsed_target_date:
            missing.append("target_date")
        logger.debug(
            "map_market skip market %s: missing fields %s (question=%r)",
            market.id, missing, market.question[:80] if market.question else None,
        )
        return None

    # Resolve variable aliases (snowfall→precipitation, freeze→temperature, etc.)
    variable, threshold, operator_raw = resolve_variable(
        market.parsed_variable,
        market.parsed_threshold,
        market.parsed_operator,
    )

    if variable not in SUPPORTED_VARIABLES:
        logger.debug(
            "map_market skip market %s: unsupported variable %r (raw=%r)",
            market.id, variable, market.parsed_variable,
        )
        return None

    operator = normalize_operator(operator_raw)
    if operator is None:
        logger.debug(
            "map_market skip market %s: unsupported operator %r", market.id, operator_raw,
        )
        return None

    coords = geocode(market.parsed_location)
    if coords is None:
        logger.debug(
            "map_market skip market %s: geocode failed for %r", market.id, market.parsed_location,
        )
        return None
    lat, lon = coords

    target_dt = parse_target_date(market.parsed_target_date)
    if target_dt is None:
        logger.debug(
            "map_market skip market %s: date parse failed for %r",
            market.id, market.parsed_target_date,
        )
        return None

    now = datetime.now(tz=timezone.utc)
    hours_to_resolution = (target_dt - now).total_seconds() / 3600.0
    days_to_resolution = (target_dt - now).days
    if days_to_resolution < 0 and hours_to_resolution < 0:
        logger.debug(
            "map_market skip market %s: expired (target=%s, hours=%.1f)",
            market.id, target_dt.isoformat(), hours_to_resolution,
        )
        return None

    threshold_si = convert_threshold(threshold, variable)

    # --- fetch aviation probability ----------------------------------------
    try:
        aviation_prob = await get_realtime_probability(market)
    except Exception as e:
        logger.warning("Aviation fetch failed for market %s: %s", market.id, e)
        aviation_prob = None

    if aviation_prob is None:
        logger.debug(
            "map_market skip %s: no aviation data (loc=%r, date=%r)",
            market.id, market.parsed_location, market.parsed_target_date,
        )
        return None

    # --- gather aviation context -------------------------------------------
    aviation_ctx = None
    icao = icao_for_location(market.parsed_location)
    if icao is not None:
        ctx_results = await asyncio.gather(
            aviation.taf_amendment_count(icao, hours=24),
            aviation.detect_speci_events(icao, hours=2),
            aviation.has_severe_weather_reports(lat, lon),
            aviation.alerts_affecting_location(lat, lon),
            return_exceptions=True,
        )
        amend = ctx_results[0] if not isinstance(ctx_results[0], BaseException) else 0
        speci = ctx_results[1] if not isinstance(ctx_results[1], BaseException) else []
        severe = ctx_results[2] if not isinstance(ctx_results[2], BaseException) else False
        alerts = ctx_results[3] if not isinstance(ctx_results[3], BaseException) else []
        aviation_ctx = AviationContext(
            taf_amendment_count=amend,
            speci_events_2h=len(speci) if isinstance(speci, list) else 0,
            has_severe_pireps=bool(severe),
            active_sigmet_count=sum(1 for a in alerts if getattr(a, "alert_type", None) == "SIGMET"),
            active_airmet_count=sum(1 for a in alerts if getattr(a, "alert_type", None) == "AIRMET"),
        )

    market_prob = market.current_yes_price or 0.5

    return MarketSignal(
        market_id=market.id,
        question=market.question,
        market_prob=market_prob,
        aviation_prob=aviation_prob,
        days_to_resolution=days_to_resolution,
        hours_to_resolution=hours_to_resolution,
        market=market,
        aviation_context=aviation_ctx,
    )


async def map_bracket_market(
    market: Market,
    session: AsyncSession | None = None,
) -> MarketSignal | None:
    """Map a bracket temperature market to forecast probabilities.

    Bracket markets (e.g. "Highest temperature in Austin on April 8?") have
    outcomes like "70-74°F" instead of a single threshold.  We parse the
    bracket bounds from the outcomes and compute P(low ≤ temp < high).
    """
    from src.ingestion.polymarket import parse_bracket_from_question, parse_temperature_brackets

    if not all([market.parsed_location, market.parsed_variable, market.parsed_target_date]):
        missing = []
        if not market.parsed_location:
            missing.append("location")
        if not market.parsed_variable:
            missing.append("variable")
        if not market.parsed_target_date:
            missing.append("target_date")
        logger.debug(
            "map_bracket skip market %s: missing fields %s (question=%r)",
            market.id, missing, market.question[:80] if market.question else None,
        )
        return None
    if market.parsed_operator != "bracket":
        logger.debug("map_bracket skip market %s: operator is %r not bracket", market.id, market.parsed_operator)
        return None

    brackets = parse_temperature_brackets(market.outcomes)
    if not brackets:
        fallback = parse_bracket_from_question(market.question)
        if fallback is None:
            logger.info(
                "map_bracket skip market %s: no parseable brackets from outcomes=%r or question=%r",
                market.id, market.outcomes, market.question[:80] if market.question else None,
            )
            return None
        brackets = [fallback]

    coords = geocode(market.parsed_location)
    if coords is None:
        logger.debug("map_bracket skip market %s: geocode failed for %r", market.id, market.parsed_location)
        return None
    lat, lon = coords

    target_dt = parse_target_date(market.parsed_target_date)
    if target_dt is None:
        logger.debug("map_bracket skip market %s: date parse failed for %r", market.id, market.parsed_target_date)
        return None

    now = datetime.now(tz=timezone.utc)
    hours_to_resolution = (target_dt - now).total_seconds() / 3600.0
    days_to_resolution = (target_dt - now).days
    if days_to_resolution < 0 and hours_to_resolution < 0:
        logger.debug(
            "map_bracket skip market %s: expired (target=%s, hours=%.1f)",
            market.id, target_dt.isoformat(), hours_to_resolution,
        )
        return None

    # Use the market's YES price — this corresponds to the first outcome bracket
    market_prob = market.current_yes_price or 0.5

    # The first bracket's bounds determine our probability target
    low_f, high_f = brackets[0]

    # Convert bracket bounds from Fahrenheit to Kelvin for NWP models
    low_si = convert_threshold(low_f, "temperature")
    high_si = convert_threshold(high_f, "temperature")

    icao = icao_for_location(market.parsed_location)

    # --- fetch aviation bracket probability --------------------------------
    aviation_prob = None
    if icao is not None:
        try:
            aviation_prob = await aviation.get_bracket_probability(icao, low_f, high_f, target_dt)
        except Exception as e:
            logger.warning("Aviation bracket fetch failed for market %s: %s", market.id, e)

    if aviation_prob is None:
        logger.debug("map_bracket skip %s: no aviation data", market.id)
        return None

    gfs_prob = None
    ecmwf_prob = None

    # --- gather aviation context ---
    aviation_ctx = None
    if icao is not None:
        ctx_results = await asyncio.gather(
            aviation.taf_amendment_count(icao, hours=24),
            aviation.detect_speci_events(icao, hours=2),
            aviation.has_severe_weather_reports(lat, lon),
            aviation.alerts_affecting_location(lat, lon),
            return_exceptions=True,
        )
        amend = ctx_results[0] if not isinstance(ctx_results[0], BaseException) else 0
        speci = ctx_results[1] if not isinstance(ctx_results[1], BaseException) else []
        severe = ctx_results[2] if not isinstance(ctx_results[2], BaseException) else False
        alerts = ctx_results[3] if not isinstance(ctx_results[3], BaseException) else []
        aviation_ctx = AviationContext(
            taf_amendment_count=amend,
            speci_events_2h=len(speci) if isinstance(speci, list) else 0,
            has_severe_pireps=bool(severe),
            active_sigmet_count=sum(1 for a in alerts if getattr(a, "alert_type", None) == "SIGMET"),
            active_airmet_count=sum(1 for a in alerts if getattr(a, "alert_type", None) == "AIRMET"),
        )

    return MarketSignal(
        market_id=market.id,
        question=market.question,
        market_prob=market_prob,
        gfs_prob=gfs_prob,
        ecmwf_prob=ecmwf_prob,
        aviation_prob=aviation_prob,
        days_to_resolution=days_to_resolution,
        hours_to_resolution=hours_to_resolution,
        market=market,
        aviation_context=aviation_ctx,
    )


EXACTLY_BRACKET_HALF_WIDTH = 1.0  # ±1°F around the stated threshold


async def map_exactly_market(
    market: Market,
    session: AsyncSession | None = None,
) -> MarketSignal | None:
    """Map an 'exactly N°F' temperature market to forecast probabilities.

    Converts the point threshold to a narrow bracket (threshold ± 1°F) and
    computes P(low ≤ temp < high) using aviation data, mirroring the bracket
    market approach.
    """
    if not all([
        market.parsed_location,
        market.parsed_variable,
        market.parsed_threshold is not None,
        market.parsed_target_date,
    ]):
        missing = []
        if not market.parsed_location:
            missing.append("location")
        if not market.parsed_variable:
            missing.append("variable")
        if market.parsed_threshold is None:
            missing.append("threshold")
        if not market.parsed_target_date:
            missing.append("target_date")
        logger.debug(
            "map_exactly skip market %s: missing fields %s (question=%r)",
            market.id, missing, market.question[:80] if market.question else None,
        )
        return None

    threshold = market.parsed_threshold
    low_f = threshold - EXACTLY_BRACKET_HALF_WIDTH
    high_f = threshold + EXACTLY_BRACKET_HALF_WIDTH

    coords = geocode(market.parsed_location)
    if coords is None:
        logger.debug("map_exactly skip market %s: geocode failed for %r", market.id, market.parsed_location)
        return None
    lat, lon = coords

    target_dt = parse_target_date(market.parsed_target_date)
    if target_dt is None:
        logger.debug("map_exactly skip market %s: date parse failed for %r", market.id, market.parsed_target_date)
        return None

    now = datetime.now(tz=timezone.utc)
    hours_to_resolution = (target_dt - now).total_seconds() / 3600.0
    days_to_resolution = (target_dt - now).days
    if days_to_resolution < 0 and hours_to_resolution < 0:
        logger.debug(
            "map_exactly skip market %s: expired (target=%s, hours=%.1f)",
            market.id, target_dt.isoformat(), hours_to_resolution,
        )
        return None

    market_prob = market.current_yes_price or 0.5

    icao = icao_for_location(market.parsed_location)

    # --- fetch aviation bracket probability --------------------------------
    aviation_prob = None
    if icao is not None:
        try:
            aviation_prob = await aviation.get_bracket_probability(icao, low_f, high_f, target_dt)
        except Exception as e:
            logger.warning("Aviation exactly fetch failed for market %s: %s", market.id, e)

    if aviation_prob is None:
        logger.debug("map_exactly skip %s: no aviation data", market.id)
        return None

    # --- gather aviation context ---
    aviation_ctx = None
    if icao is not None:
        ctx_results = await asyncio.gather(
            aviation.taf_amendment_count(icao, hours=24),
            aviation.detect_speci_events(icao, hours=2),
            aviation.has_severe_weather_reports(lat, lon),
            aviation.alerts_affecting_location(lat, lon),
            return_exceptions=True,
        )
        amend = ctx_results[0] if not isinstance(ctx_results[0], BaseException) else 0
        speci = ctx_results[1] if not isinstance(ctx_results[1], BaseException) else []
        severe = ctx_results[2] if not isinstance(ctx_results[2], BaseException) else False
        alerts = ctx_results[3] if not isinstance(ctx_results[3], BaseException) else []
        aviation_ctx = AviationContext(
            taf_amendment_count=amend,
            speci_events_2h=len(speci) if isinstance(speci, list) else 0,
            has_severe_pireps=bool(severe),
            active_sigmet_count=sum(1 for a in alerts if getattr(a, "alert_type", None) == "SIGMET"),
            active_airmet_count=sum(1 for a in alerts if getattr(a, "alert_type", None) == "AIRMET"),
        )

    return MarketSignal(
        market_id=market.id,
        question=market.question,
        market_prob=market_prob,
        aviation_prob=aviation_prob,
        days_to_resolution=days_to_resolution,
        hours_to_resolution=hours_to_resolution,
        market=market,
        aviation_context=aviation_ctx,
    )


EXACTLY_BRACKET_HALF_WIDTH = 1.0  # ±1°F around the stated threshold


async def map_exactly_market(
    market: Market,
    session: AsyncSession | None = None,
) -> MarketSignal | None:
    """Map an 'exactly N°F' temperature market to forecast probabilities.

    Converts the point threshold to a narrow bracket (threshold ± 1°F) and
    computes P(low ≤ temp < high) using aviation data, mirroring the bracket
    market approach.
    """
    if not all([
        market.parsed_location,
        market.parsed_variable,
        market.parsed_threshold is not None,
        market.parsed_target_date,
    ]):
        missing = []
        if not market.parsed_location:
            missing.append("location")
        if not market.parsed_variable:
            missing.append("variable")
        if market.parsed_threshold is None:
            missing.append("threshold")
        if not market.parsed_target_date:
            missing.append("target_date")
        logger.debug(
            "map_exactly skip market %s: missing fields %s (question=%r)",
            market.id, missing, market.question[:80] if market.question else None,
        )
        return None

    threshold = market.parsed_threshold
    low_f = threshold - EXACTLY_BRACKET_HALF_WIDTH
    high_f = threshold + EXACTLY_BRACKET_HALF_WIDTH

    coords = geocode(market.parsed_location)
    if coords is None:
        logger.debug("map_exactly skip market %s: geocode failed for %r", market.id, market.parsed_location)
        return None
    lat, lon = coords

    target_dt = parse_target_date(market.parsed_target_date)
    if target_dt is None:
        logger.debug("map_exactly skip market %s: date parse failed for %r", market.id, market.parsed_target_date)
        return None

    now = datetime.now(tz=timezone.utc)
    hours_to_resolution = (target_dt - now).total_seconds() / 3600.0
    days_to_resolution = (target_dt - now).days
    if days_to_resolution < 0 and hours_to_resolution < 0:
        logger.debug(
            "map_exactly skip market %s: expired (target=%s, hours=%.1f)",
            market.id, target_dt.isoformat(), hours_to_resolution,
        )
        return None

    market_prob = market.current_yes_price or 0.5

    icao = icao_for_location(market.parsed_location)

    # --- fetch aviation bracket probability --------------------------------
    aviation_prob = None
    if icao is not None:
        try:
            aviation_prob = await aviation.get_bracket_probability(icao, low_f, high_f, target_dt)
        except Exception as e:
            logger.warning("Aviation exactly fetch failed for market %s: %s", market.id, e)

    if aviation_prob is None:
        logger.debug("map_exactly skip %s: no aviation data", market.id)
        return None

    # --- gather aviation context ---
    aviation_ctx = None
    if icao is not None:
        ctx_results = await asyncio.gather(
            aviation.taf_amendment_count(icao, hours=24),
            aviation.detect_speci_events(icao, hours=2),
            aviation.has_severe_weather_reports(lat, lon),
            aviation.alerts_affecting_location(lat, lon),
            return_exceptions=True,
        )
        amend = ctx_results[0] if not isinstance(ctx_results[0], BaseException) else 0
        speci = ctx_results[1] if not isinstance(ctx_results[1], BaseException) else []
        severe = ctx_results[2] if not isinstance(ctx_results[2], BaseException) else False
        alerts = ctx_results[3] if not isinstance(ctx_results[3], BaseException) else []
        aviation_ctx = AviationContext(
            taf_amendment_count=amend,
            speci_events_2h=len(speci) if isinstance(speci, list) else 0,
            has_severe_pireps=bool(severe),
            active_sigmet_count=sum(1 for a in alerts if getattr(a, "alert_type", None) == "SIGMET"),
            active_airmet_count=sum(1 for a in alerts if getattr(a, "alert_type", None) == "AIRMET"),
        )

    return MarketSignal(
        market_id=market.id,
        question=market.question,
        market_prob=market_prob,
        aviation_prob=aviation_prob,
        days_to_resolution=days_to_resolution,
        hours_to_resolution=hours_to_resolution,
        market=market,
        aviation_context=aviation_ctx,
    )


def _choose_mapper(market: Market):
    """Return the appropriate mapping function for a market."""
    if market.parsed_operator == "bracket":
        return map_bracket_market
    if market.parsed_operator == "exactly":
        return map_exactly_market
    return map_market


async def map_short_range_markets(
    session: AsyncSession | None = None,
) -> list[MarketSignal]:
    """Fetch active weather markets and map only those within 30h to resolution."""
    markets = await get_active_weather_markets(session)
    now = datetime.now(tz=timezone.utc)

    short_range = []
    no_date = 0
    parse_fail = 0
    out_of_range = 0
    for m in markets:
        if not m.parsed_target_date:
            no_date += 1
            continue
        target_dt = parse_target_date(m.parsed_target_date)
        if target_dt is None:
            parse_fail += 1
            continue
        hours_out = (target_dt - now).total_seconds() / 3600.0
        if 0 <= hours_out <= 30:
            short_range.append(m)
        else:
            out_of_range += 1

    logger.info(
        "Short-range pipeline: %d / %d markets within 30h "
        "(no_date=%d, parse_fail=%d, out_of_range=%d)",
        len(short_range), len(markets), no_date, parse_fail, out_of_range,
    )

    # --- pre-fetch aviation data to prime cache (best-effort) ---------------
    try:
        stations = set()
        for m in short_range:
            if m.parsed_location:
                icao = icao_for_location(m.parsed_location)
                if icao:
                    stations.add(icao)

        if stations:
            station_list = sorted(stations)
            logger.info("Pre-fetching aviation data for %d stations: %s", len(station_list), station_list)

            # Run METAR (per-station), TAF (batched), and SIGMET fetches concurrently.
            # 24h METAR covers the 6h window, so no separate 6h call needed.
            metar_coros = [aviation.fetch_metar_history(stn, hours=24) for stn in station_list]
            await asyncio.gather(
                asyncio.gather(*metar_coros, return_exceptions=True),
                aviation.fetch_latest_tafs(station_list),
                aviation._fetch_airsigmets(),
                return_exceptions=True,
            )
        else:
            try:
                await aviation._fetch_airsigmets()
            except Exception:
                logger.warning("Pre-fetch SIGMETs failed")
    except Exception:
        logger.warning("Aviation pre-fetch failed; falling back to per-market fetch")
    # -----------------------------------------------------------------------

    async def _bounded(coro):
        async with _FETCH_SEMAPHORE:
            return await coro

    tasks = [_bounded(_choose_mapper(m)(m, session)) for m in short_range]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    signals: list[MarketSignal] = []
    errors = 0
    skipped = 0
    for r in results:
        if isinstance(r, BaseException):
            logger.error("map_short_range error: %s", r, exc_info=r)
            errors += 1
        elif r is not None:
            signals.append(r)
        else:
            skipped += 1

    logger.info(
        "Short-range mapped %d / %d signals (skipped=%d, errors=%d)",
        len(signals), len(short_range), skipped, errors,
    )
    return signals
