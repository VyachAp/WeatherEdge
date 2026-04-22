"""Geocoding, ICAO lookup, operator normalisation, and date parsing utilities.

Provides static city→coordinate and city→ICAO mappings used by the unified
pipeline and other modules.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# ICAO → IANA timezone lookup
# ---------------------------------------------------------------------------
#
# Polymarket daily-max markets resolve on the LOCAL city day (Wunderground
# convention), not the UTC day. Using UTC-day boundaries for the routine-
# daily-max computation attributes next-local-day METARs to "today" once the
# UTC day rolls over but local time hasn't — producing spurious lock-ins and
# bad probability estimates. See `_routine_daily_max` in state_aggregator.py.

ICAO_TIMEZONE: dict[str, str] = {
    # US Eastern
    "KJFK": "America/New_York", "KEWR": "America/New_York", "KPHL": "America/New_York",
    "KBOS": "America/New_York", "KBWI": "America/New_York", "KDCA": "America/New_York",
    "KCLT": "America/New_York", "KRDU": "America/New_York", "KORF": "America/New_York",
    "KRIC": "America/New_York", "KMIA": "America/New_York", "KTPA": "America/New_York",
    "KJAX": "America/New_York", "KATL": "America/New_York", "KBHM": "America/Chicago",
    "KBNA": "America/Chicago", "KSDF": "America/New_York", "KCVG": "America/New_York",
    "KIND": "America/New_York", "KCMH": "America/New_York", "KCLE": "America/New_York",
    "KDTW": "America/New_York", "KPIT": "America/New_York", "KPVD": "America/New_York",
    "KMHT": "America/New_York", "KBTV": "America/New_York", "KPWM": "America/New_York",
    "KBDL": "America/New_York", "KILG": "America/New_York", "KBUF": "America/New_York",
    "KROC": "America/New_York", "KLEX": "America/New_York", "KCHS": "America/New_York",
    "KGSO": "America/New_York", "KINT": "America/New_York", "KCRW": "America/New_York",
    "KTOL": "America/New_York",
    # US Central
    "KORD": "America/Chicago", "KMKE": "America/Chicago", "KMSN": "America/Chicago",
    "KMSP": "America/Chicago", "KDSM": "America/Chicago", "KSTL": "America/Chicago",
    "KMCI": "America/Chicago", "KOMA": "America/Chicago", "KLNK": "America/Chicago",
    "KICT": "America/Chicago", "KTUL": "America/Chicago", "KOKC": "America/Chicago",
    "KMSY": "America/Chicago", "KBTR": "America/Chicago", "KJAN": "America/Chicago",
    "KMEM": "America/Chicago", "KIAH": "America/Chicago", "KDFW": "America/Chicago",
    "KSAT": "America/Chicago", "KAUS": "America/Chicago", "KFSD": "America/Chicago",
    "KFAR": "America/Chicago", "KCRP": "America/Chicago", "KLIT": "America/Chicago",
    "KLRD": "America/Chicago", "KLBB": "America/Chicago",
    # US Mountain (with DST)
    "KDEN": "America/Denver", "KCOS": "America/Denver", "KABQ": "America/Denver",
    "KSLC": "America/Denver", "KELP": "America/Denver", "KBZN": "America/Denver",
    "KCPR": "America/Denver",
    # Phoenix — Arizona does not observe DST
    "KPHX": "America/Phoenix", "KTUS": "America/Phoenix",
    # US Pacific
    "KLAS": "America/Los_Angeles", "KRNO": "America/Los_Angeles",
    "KLAX": "America/Los_Angeles", "KSAN": "America/Los_Angeles",
    "KSFO": "America/Los_Angeles", "KOAK": "America/Los_Angeles",
    "KSJC": "America/Los_Angeles", "KSMF": "America/Los_Angeles",
    "KFAT": "America/Los_Angeles", "KBFL": "America/Los_Angeles",
    "KPDX": "America/Los_Angeles", "KSEA": "America/Los_Angeles",
    "KGEG": "America/Los_Angeles", "KBOI": "America/Boise",
    "KSCK": "America/Los_Angeles", "KMOD": "America/Los_Angeles",
    "KRAL": "America/Los_Angeles", "KSNA": "America/Los_Angeles",
    "KLGB": "America/Los_Angeles",
    # US non-contiguous
    "PANC": "America/Anchorage", "PHNL": "Pacific/Honolulu",
    # Canada
    "CYYZ": "America/Toronto", "CYUL": "America/Toronto",
    "CYVR": "America/Vancouver",
    # Mexico / Central America
    "MMMX": "America/Mexico_City", "MPTO": "America/Panama",
    # South America
    "SAEZ": "America/Argentina/Buenos_Aires", "SBGR": "America/Sao_Paulo",
    "SPJC": "America/Lima", "SKBO": "America/Bogota",
    # Western Europe
    "EGLL": "Europe/London", "EGPH": "Europe/London", "EIDW": "Europe/Dublin",
    "LFPG": "Europe/Paris", "EHAM": "Europe/Amsterdam", "EBBR": "Europe/Brussels",
    "EDDB": "Europe/Berlin", "EDDM": "Europe/Berlin", "LOWW": "Europe/Vienna",
    "LSZH": "Europe/Zurich", "LIRF": "Europe/Rome", "LIMC": "Europe/Rome",
    "LEMD": "Europe/Madrid", "LEBL": "Europe/Madrid", "LPPT": "Europe/Lisbon",
    # Nordics
    "ESSA": "Europe/Stockholm", "ENGM": "Europe/Oslo",
    "EKCH": "Europe/Copenhagen", "EFHK": "Europe/Helsinki",
    # Eastern Europe
    "EPWA": "Europe/Warsaw", "LKPR": "Europe/Prague", "LHBP": "Europe/Budapest",
    "LGAV": "Europe/Athens", "UUEE": "Europe/Moscow",
    # Middle East / Turkey
    "LTFM": "Europe/Istanbul", "LTAC": "Europe/Istanbul",
    "LLBG": "Asia/Jerusalem", "OMDB": "Asia/Dubai", "OTHH": "Asia/Qatar",
    "OERK": "Asia/Riyadh", "OEJN": "Asia/Riyadh",
    # Africa
    "HECA": "Africa/Cairo", "DNMM": "Africa/Lagos",
    "FAOR": "Africa/Johannesburg", "FACT": "Africa/Johannesburg",
    "HKJK": "Africa/Nairobi",
    # South Asia
    "VABB": "Asia/Kolkata", "VIDP": "Asia/Kolkata", "VECC": "Asia/Kolkata",
    "VOMM": "Asia/Kolkata", "VOBL": "Asia/Kolkata", "VOHS": "Asia/Kolkata",
    "VILK": "Asia/Kolkata",
    "OPKC": "Asia/Karachi", "OPLA": "Asia/Karachi",
    "VGHS": "Asia/Dhaka", "VCBI": "Asia/Colombo",
    # Southeast Asia
    "VTBS": "Asia/Bangkok", "WSSS": "Asia/Singapore",
    "WMKK": "Asia/Kuala_Lumpur", "WMKP": "Asia/Kuala_Lumpur",
    "WBGG": "Asia/Kuala_Lumpur", "WIII": "Asia/Jakarta",
    "VVNB": "Asia/Ho_Chi_Minh", "VVTS": "Asia/Ho_Chi_Minh",
    "RPLL": "Asia/Manila",
    # East Asia
    "RJTT": "Asia/Tokyo", "RJBB": "Asia/Tokyo",
    "RJCC": "Asia/Tokyo", "RJFF": "Asia/Tokyo",
    "RKSI": "Asia/Seoul", "RKPK": "Asia/Seoul",
    "ZBAA": "Asia/Shanghai", "ZSPD": "Asia/Shanghai", "ZUCK": "Asia/Shanghai",
    "ZHHH": "Asia/Shanghai", "ZUUU": "Asia/Shanghai", "ZGSZ": "Asia/Shanghai",
    "ZGGG": "Asia/Shanghai",
    "VHHH": "Asia/Hong_Kong", "RCTP": "Asia/Taipei",
    # Oceania
    "YSSY": "Australia/Sydney", "YMML": "Australia/Melbourne",
    "YBBN": "Australia/Brisbane", "YPPH": "Australia/Perth",
    "NZAA": "Pacific/Auckland", "NZWN": "Pacific/Auckland",
}


def icao_timezone(icao: str) -> ZoneInfo:
    """Return the IANA timezone for an ICAO station.

    Falls back to UTC with a warning when the station is not in the lookup
    table (preserves the legacy UTC-day behavior for unknown stations rather
    than silently producing wrong answers).
    """
    tz_name = ICAO_TIMEZONE.get(icao.upper())
    if tz_name is None:
        logger.warning("icao_timezone: no timezone for %r, falling back to UTC", icao)
        return ZoneInfo("UTC")
    return ZoneInfo(tz_name)


def f_to_c(temp_f: float) -> float:
    return (temp_f - 32.0) * 5.0 / 9.0


def unit_for_station(icao: str) -> str:
    """Return the temperature unit Polymarket markets use for this station.

    K-prefix ICAOs are the contiguous US (every US city in CITY_ICAO uses
    KXXX) and trade in °F; everything else (R, E, L, W, Z, S, …) trades in
    °C. Pacific US stations (PA, PH, PG) aren't in CITY_ICAO yet — extend
    this rule when they're added.
    """
    return "°F" if icao.upper().startswith("K") else "°C"


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


# ---------------------------------------------------------------------------
# Reverse lookup: ICAO → city names
# ---------------------------------------------------------------------------

_ICAO_TO_CITIES: dict[str, list[str]] | None = None


def cities_for_icao(icao: str) -> list[str]:
    """Return all city names that map to the given ICAO station code.

    Lazily inverts the CITY_ICAO dict on first call.
    """
    global _ICAO_TO_CITIES
    if _ICAO_TO_CITIES is None:
        _ICAO_TO_CITIES = {}
        for city, code in CITY_ICAO.items():
            _ICAO_TO_CITIES.setdefault(code, []).append(city)
    return _ICAO_TO_CITIES.get(icao.upper(), [])


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
}

SUPPORTED_VARIABLES: set[str] = {"temperature"}


def normalize_operator(op: str) -> str | None:
    """Map a parsed operator to the 'above'/'below' expected by forecast APIs."""
    return OPERATOR_MAP.get(op)


# ---------------------------------------------------------------------------
# Unit conversion  (market units → SI / GRIB units)
# ---------------------------------------------------------------------------


def convert_threshold(value: float, variable: str) -> float:
    """Convert a market threshold from imperial to SI units.

    - temperature: °F → K
    """
    if variable == "temperature":
        return (value - 32.0) * 5.0 / 9.0 + 273.15
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


