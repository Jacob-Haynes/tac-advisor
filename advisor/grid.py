"""Grid references for the LLM context and SITREPs: MGRS, British National Grid, or lat/lon.

Pure Python (the Pi is offline and the ``mgrs``/``pyproj`` packages are not in this venv).

* **MGRS** (default): WGS84 -> UTM (Krueger series, sub-millimetre in-zone) -> MGRS with the
  standard 100 km letters, 1 m precision, e.g. ``30U XC 12345 67890``. Norway/Svalbard zone
  exceptions included; polar (UPS) latitudes fall back to lat/lon.
* **BNG**: WGS84 -> OSGB36 by the OS 7-parameter Helmert (about 5 m, plenty for a SITREP) ->
  OS National Grid, e.g. ``TQ 38345 84135``. Outside the grid it falls back to MGRS.
* **latlon**: ``51.15000 -1.75000``.

``GRID_FORMAT=mgrs|bng|latlon`` picks the format (default mgrs). ``bearing_distance`` gives
"1.2 km NW" style relative positions.
"""

from __future__ import annotations

import math
import os

# ── ellipsoids ──
_WGS84 = (6378137.0, 1 / 298.257223563)
_AIRY = (6377563.396, 6356256.909)  # a, b

FORMATS = ("mgrs", "bng", "latlon")


def grid_format() -> str:
    f = os.environ.get("GRID_FORMAT", "mgrs").strip().lower()
    return f if f in FORMATS else "mgrs"


# ── UTM / MGRS ──

def utm_zone(lat: float, lon: float) -> int:
    lon = ((lon + 180) % 360) - 180
    zone = int((lon + 180) // 6) + 1
    if 56 <= lat < 64 and 3 <= lon < 12:
        return 32
    if 72 <= lat < 84 and lon >= 0:
        if lon < 9:
            return 31
        if lon < 21:
            return 33
        if lon < 33:
            return 35
        if lon < 42:
            return 37
    return min(zone, 60)


_BANDS = "CDEFGHJKLMNPQRSTUVWX"


def lat_band(lat: float) -> str:
    if lat >= 72:
        return "X"
    return _BANDS[int((lat + 80) // 8)]


def latlon_to_utm(lat: float, lon: float, zone: int | None = None) -> tuple[int, str, float, float]:
    """(zone, band, easting, northing) on WGS84. Valid for -80 <= lat < 84."""
    if not -80 <= lat < 84:
        raise ValueError("latitude outside UTM (polar)")
    zone = zone or utm_zone(lat, lon)
    a, f = _WGS84
    k0 = 0.9996
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    alpha = (n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16,
             13 * n ** 2 / 48 - 3 * n ** 3 / 5,
             61 * n ** 3 / 240)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    phi, lam = math.radians(lat), math.radians(lon) - lon0
    e = math.sqrt(f * (2 - f))
    t = math.sinh(math.atanh(math.sin(phi)) - e * math.atanh(e * math.sin(phi)))
    xi_ = math.atan2(t, math.cos(lam))
    eta_ = math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    xi = xi_ + sum(alpha[j] * math.sin(2 * (j + 1) * xi_) * math.cosh(2 * (j + 1) * eta_) for j in range(3))
    eta = eta_ + sum(alpha[j] * math.cos(2 * (j + 1) * xi_) * math.sinh(2 * (j + 1) * eta_) for j in range(3))
    easting = 500000 + k0 * A * eta
    northing = k0 * A * xi + (10000000 if lat < 0 else 0)
    return zone, lat_band(lat), easting, northing


def to_mgrs(lat: float, lon: float, digits: int = 5) -> str:
    """'30U WB 12345 67890' (``digits`` per axis: 5 = 1 m, 4 = 10 m, 3 = 100 m)."""
    zone, band, e, n = latlon_to_utm(lat, lon)
    s = (zone - 1) % 3  # column letter set
    cols = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")[s]
    rows = "ABCDEFGHJKLMNPQRSTUV"
    col = cols[int(e // 100000) - 1]
    row = rows[(int(n // 100000) + (5 if zone % 2 == 0 else 0)) % 20]
    scale = 10 ** (5 - digits)
    ee = int(e % 100000) // scale
    nn = int(n % 100000) // scale
    return f"{zone}{band} {col}{row} {ee:0{digits}d} {nn:0{digits}d}"


# ── British National Grid ──

def _wgs84_to_osgb36(lat: float, lon: float) -> tuple[float, float]:
    """OS 7-parameter Helmert (WGS84/ETRS89 -> OSGB36), ellipsoid height taken as 0."""
    a, f = _WGS84
    e2 = f * (2 - f)
    phi, lam = math.radians(lat), math.radians(lon)
    nu = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    x = nu * math.cos(phi) * math.cos(lam)
    y = nu * math.cos(phi) * math.sin(lam)
    z = nu * (1 - e2) * math.sin(phi)
    tx, ty, tz = -446.448, 125.157, -542.060
    s = 20.4894e-6
    rx, ry, rz = (math.radians(v / 3600) for v in (-0.1502, -0.2470, -0.8421))
    x2 = tx + (1 + s) * x - rz * y + ry * z
    y2 = ty + rz * x + (1 + s) * y - rx * z
    z2 = tz - ry * x + rx * y + (1 + s) * z
    a2, b2 = _AIRY
    e2b = 1 - (b2 * b2) / (a2 * a2)
    p = math.hypot(x2, y2)
    phi2 = math.atan2(z2, p * (1 - e2b))
    for _ in range(10):
        nu2 = a2 / math.sqrt(1 - e2b * math.sin(phi2) ** 2)
        phi2 = math.atan2(z2 + e2b * nu2 * math.sin(phi2), p)
    return math.degrees(phi2), math.degrees(math.atan2(y2, x2))


def osgb36_to_en(lat: float, lon: float) -> tuple[float, float]:
    """OSGB36 lat/lon -> National Grid easting/northing (OS 'A guide to coordinate systems', C.1)."""
    a, b = _AIRY
    F0 = 0.9996012717
    lat0, lon0 = math.radians(49), math.radians(-2)
    N0, E0 = -100000.0, 400000.0
    e2 = 1 - (b * b) / (a * a)
    n = (a - b) / (a + b)
    phi, lam = math.radians(lat), math.radians(lon)
    sin, cos, tan = math.sin(phi), math.cos(phi), math.tan(phi)
    nu = a * F0 / math.sqrt(1 - e2 * sin * sin)
    rho = a * F0 * (1 - e2) / (1 - e2 * sin * sin) ** 1.5
    eta2 = nu / rho - 1
    dp, sp = phi - lat0, phi + lat0
    M = b * F0 * ((1 + n + 5 / 4 * n ** 2 + 5 / 4 * n ** 3) * dp
                  - (3 * n + 3 * n ** 2 + 21 / 8 * n ** 3) * math.sin(dp) * math.cos(sp)
                  + (15 / 8 * n ** 2 + 15 / 8 * n ** 3) * math.sin(2 * dp) * math.cos(2 * sp)
                  - 35 / 24 * n ** 3 * math.sin(3 * dp) * math.cos(3 * sp))
    I = M + N0
    II = nu / 2 * sin * cos
    III = nu / 24 * sin * cos ** 3 * (5 - tan ** 2 + 9 * eta2)
    IIIA = nu / 720 * sin * cos ** 5 * (61 - 58 * tan ** 2 + tan ** 4)
    IV = nu * cos
    V = nu / 6 * cos ** 3 * (nu / rho - tan ** 2)
    VI = nu / 120 * cos ** 5 * (5 - 18 * tan ** 2 + tan ** 4 + 14 * eta2 - 58 * tan ** 2 * eta2)
    d = lam - lon0
    N = I + II * d ** 2 + III * d ** 4 + IIIA * d ** 6
    E = E0 + IV * d + V * d ** 3 + VI * d ** 5
    return E, N


def en_to_bng(e: float, n: float, digits: int = 5) -> str | None:
    """'TQ 38345 84135', or None outside the 700 km x 1300 km grid."""
    if not (0 <= e < 700000 and 0 <= n < 1300000):
        return None
    e100, n100 = int(e // 100000), int(n // 100000)
    letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # no I
    l1 = (19 - n100) - (19 - n100) % 5 + (e100 + 10) // 5
    l2 = (19 - n100) * 5 % 25 + e100 % 5
    scale = 10 ** (5 - digits)
    return (f"{letters[l1]}{letters[l2]} {int(e % 100000) // scale:0{digits}d} "
            f"{int(n % 100000) // scale:0{digits}d}")


def to_bng(lat: float, lon: float, digits: int = 5) -> str | None:
    """WGS84 lat/lon -> 'SU 12345 67890', or None outside Great Britain's grid."""
    if not (49 <= lat <= 61.5 and -9 <= lon <= 2.5):
        return None
    return en_to_bng(*osgb36_to_en(*_wgs84_to_osgb36(lat, lon)), digits=digits)


# ── what the context and SITREPs use ──

def grid_ref(lat: float | None, lon: float | None, fmt: str | None = None) -> str:
    """The position in the configured format. Never invents a ref: missing/0,0 -> 'no position'."""
    if lat is None or lon is None or (lat == 0 and lon == 0):
        return "no position"
    fmt = fmt or grid_format()
    if fmt == "bng":
        ref = to_bng(lat, lon)
        if ref:
            return ref
        fmt = "mgrs"
    if fmt == "mgrs":
        try:
            return to_mgrs(lat, lon)
        except ValueError:
            pass
    return f"{lat:.5f} {lon:.5f}"


def grid_name(fmt: str | None = None) -> str:
    return {"mgrs": "MGRS", "bng": "BNG (OS National Grid)", "latlon": "lat/lon"}[fmt or grid_format()]


_POINTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def bearing_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    """(metres, true bearing in degrees) from point 1 to point 2 (spherical, fine below ~100 km)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    dist = 2 * 6371008.8 * math.asin(min(1.0, math.sqrt(h)))
    brg = math.degrees(math.atan2(math.sin(dl) * math.cos(p2),
                                  math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)))
    return dist, brg % 360


def relative(lat: float, lon: float, ref_lat: float, ref_lon: float, ref_name: str) -> str:
    """'1.2 km NW of HANGOVER' (position relative to the reference point)."""
    dist, brg = bearing_distance(ref_lat, ref_lon, lat, lon)
    if dist < 50:
        return f"at {ref_name}"
    d = f"{dist:.0f} m" if dist < 1000 else f"{dist / 1000:.1f} km"
    return f"{d} {_POINTS[int((brg + 22.5) // 45) % 8]} of {ref_name}"
