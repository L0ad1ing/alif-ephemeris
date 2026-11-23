"""The Moon's nodes and apsides: mean node, true (osculating) node, mean apogee (Black Moon Lilith),
osculating apogee.

Mean elements are the polynomials of Meeus (Astronomical Algorithms, ch. 47), from the ELP-2000/82 theory.
Osculating elements come from the Moon's geocentric position and velocity: at an instant the Moon moves on a
Kepler ellipse around the Earth; its node is where that ellipse crosses the ecliptic and its apogee is the far
end of the ellipse's long axis.

Longitudes are on the true ecliptic and equinox of date (degrees), like the planets.
"""
from __future__ import annotations

import numpy as np

from . import frames, planets, timescales

GM_EARTH_MOON = 398600.4418 + 4902.800066         # km^3 / s^2 (DE440 values)
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def _t(jd_tt):
    return (np.asarray(jd_tt, dtype=float) - 2451545.0) / 36525.0


def mean_node(jd_tt):
    """Mean ascending node of the Moon's orbit, mean equinox of date (degrees)."""
    t = _t(jd_tt)
    return np.mod(125.0445479 + t * (-1934.1362891 + t * (0.0020754 + t * (1.0 / 467441.0 + t * (-1.0 / 60616000.0)))), 360.0)


def mean_perigee(jd_tt):
    """Mean longitude of the Moon's perigee, mean equinox of date (degrees)."""
    t = _t(jd_tt)
    return np.mod(83.3532465 + t * (4069.0137287 + t * (-0.0103200 + t * (-1.0 / 80053.0 + t * (1.0 / 18999000.0)))), 360.0)


MEAN_INCLINATION = 5.145396          # degrees, mean inclination of the Moon's orbit (Meeus ch. 47)


def mean_apogee_lonlat(jd_tt):
    """Mean lunar apogee on the Moon's inclined mean orbit: (longitude, latitude) in degrees.

    The angle from the ascending node to the apogee is measured along the orbit. Tilting that angle by the
    orbit's inclination gives the point's ecliptic longitude and latitude (latitude up to +-5.1 degrees).
    Longitudes are on the mean equinox of date.
    """
    node = mean_node(jd_tt)
    u = np.radians(mean_perigee(jd_tt) + 180.0 - node)         # argument from the node, along the orbit
    inc = np.radians(MEAN_INCLINATION)
    lon = node + np.degrees(np.arctan2(np.cos(inc) * np.sin(u), np.cos(u)))
    lat = np.degrees(np.arcsin(np.sin(inc) * np.sin(u)))
    return np.mod(lon, 360.0), lat


def mean_apogee(jd_tt):
    """Longitude of the mean lunar apogee (the astrologers' mean Black Moon Lilith), degrees, mean equinox of date."""
    return mean_apogee_lonlat(jd_tt)[0]


def _moon_state_true_ecliptic(spk, whole_tt, frac_tt):
    """Geocentric Moon position (km) and velocity (km/s) in the true ecliptic and equinox of date."""
    whole = np.asarray(whole_tt, dtype=float)
    frac_tt = np.asarray(frac_tt, dtype=float)
    jd_tt = whole + frac_tt
    frac = frac_tt + timescales.tdb_minus_tt(jd_tt) / planets.SECONDS_PER_DAY
    moon_p, moon_v = planets._barycentric(spk, planets.MOON, whole, frac)
    earth_p, earth_v = planets._barycentric(spk, planets.EARTH, whole, frac)
    r, v = moon_p - earth_p, moon_v - earth_v
    m = frames.bias_precession_nutation(jd_tt)
    obl = frames.true_obliquity(jd_tt)
    r = frames.equator_to_ecliptic(np.einsum("...ij,...j->...i", m, r), obl)
    v = frames.equator_to_ecliptic(np.einsum("...ij,...j->...i", m, v), obl)
    return r, v


def _lon(x, y):
    return np.mod(np.arctan2(y, x) * R2D, 360.0)


def true_node(spk, whole_tt, frac_tt=0.0):
    """Osculating ascending node of the Moon (degrees): where the instantaneous orbit crosses the ecliptic going north."""
    r, v = _moon_state_true_ecliptic(spk, whole_tt, frac_tt)
    h = np.cross(r, v)
    # ascending node direction = z x h = (-h_y, h_x, 0)
    return _lon(-h[..., 1], h[..., 0])


def true_node_distance(spk, whole_tt, frac_tt=0.0):
    """Distance (km) from the Earth to the Moon's osculating orbit at its ascending node."""
    r, v = _moon_state_true_ecliptic(spk, whole_tt, frac_tt)
    h = np.cross(r, v)
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    e_vec = np.cross(v, h) / GM_EARTH_MOON - r / rn
    e = np.linalg.norm(e_vec, axis=-1)
    node = np.stack([-h[..., 1], h[..., 0], np.zeros_like(h[..., 0])], axis=-1)
    node = node / np.linalg.norm(node, axis=-1, keepdims=True)
    cos_nu = np.sum(e_vec * node, axis=-1) / e
    p_semi = np.sum(h * h, axis=-1) / GM_EARTH_MOON
    return p_semi / (1.0 + e * cos_nu)


def osculating_apogee(spk, whole_tt, frac_tt=0.0):
    """Osculating lunar apogee (degrees): the far end of the instantaneous orbit's major axis, as a longitude and a latitude."""
    r, v = _moon_state_true_ecliptic(spk, whole_tt, frac_tt)
    h = np.cross(r, v)
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    e = np.cross(v, h) / GM_EARTH_MOON - r / rn            # eccentricity vector points to perigee
    a = -e                                                  # toward apogee
    lon = _lon(a[..., 0], a[..., 1])
    lat = np.arcsin(a[..., 2] / np.linalg.norm(a, axis=-1)) * R2D
    return lon, lat


def interpolated_apogee(spk, jd_tt, window=110.0, step=0.25, perigee=False):
    """Interpolated lunar apogee ("natural Lilith"): the Moon's longitude at each actual apogee, joined by a smooth curve.

    Finds the instants when the Moon is farthest from the Earth within `window` days either side, takes its
    geometric longitude, latitude and distance at those instants on the mean equinox of date, and interpolates
    between them with a cubic spline. Returns (longitude, latitude, distance) in degrees, degrees and AU.
    At the apsides themselves it agrees with the reference to a few hundredths of a degree; between them the two
    interpolate differently and differ by up to 0.2 degrees.
    """
    from scipy.interpolate import CubicSpline
    from . import positions
    jd_tt = float(jd_tt)
    grid = jd_tt + np.arange(-window, window, step)
    w, f = timescales.split_jd(grid)
    kw = dict(light_time=False, deflection=False, aberration=False, nutation=False)
    lon, lat, dist = positions.observe(spk, planets.MOON, w, f, **kw)
    sign = -1.0 if perigee else 1.0                      # the nearest points instead of the farthest ones
    idx = [i for i in range(1, len(grid) - 1) if sign * dist[i] > sign * dist[i - 1] and sign * dist[i] >= sign * dist[i + 1]]
    times, lons, lats, dists = [], [], [], []
    for i in idx:
        y0, y1, y2 = dist[i - 1], dist[i], dist[i + 1]
        t = grid[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * step
        tw, tf = timescales.split_jd(t)
        lo, la, di = positions.observe(spk, planets.MOON, float(tw), float(tf), **kw)
        times.append(t); lons.append(lo); lats.append(la); dists.append(di)
    times = np.array(times)
    lons = np.unwrap(np.array(lons))
    return (float(np.degrees(CubicSpline(times, lons)(jd_tt)) % 360.0),
            float(np.degrees(CubicSpline(times, lats)(jd_tt))), float(CubicSpline(times, dists)(jd_tt)))


def interpolated_perigee(spk, jd_tt, window=110.0, step=0.25):
    """Interpolated lunar perigee ("Priapus"): the Moon's longitude at each actual perigee, joined by a spline.

    Built exactly like `interpolated_apogee`. The reference program's perigee is joined differently (its points jump by
    15 degrees from one perigee to the next, so the joining matters); this one differs from it by a median of 0.3 degrees
    and up to 6 degrees.
    """
    return interpolated_apogee(spk, jd_tt, window, step, perigee=True)
