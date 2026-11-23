"""Calendar dates, Julian Day numbers and time scales (UT, TT, TDB).

Delta-T (TT minus UT) comes from:
  * 1973 to the last published value: observed values from the US Naval Observatory,
    then the USNO prediction table (public domain US government data).
  * Before 1973 (back to 720 BC): the Stephenson, Morrison and Hohenkerk spline model
    (Proc. R. Soc. A 472, 2016; table S15 as updated 2020), the current best published history.
  * After the USNO predictions end, and before 720 BC: the long-term parabola of the same authors,
    joined to the tables with smooth cubic patches.

Pure functions; the data tables are loaded once and are read-only.
"""
from __future__ import annotations

import bisect
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

SECONDS_PER_DAY = 86400.0
MJD_TO_JD = 2400000.5
J2000 = 2451545.0


# -- calendar <-> Julian Day -------------------------------------------------------------

def julian_day(year: int, month: int, day: float, hour: float = 0.0, calendar: str = "auto") -> float:
    """Julian Day number (UT) for a calendar date, following Meeus, Astronomical Algorithms ch. 7.

    `calendar` is "gregorian", "julian" or "auto" (Julian before 1582-10-15, Gregorian from then).
    `day` may be fractional; `hour` adds hours on top.
    """
    y, m = year, month
    if m <= 2:
        y -= 1
        m += 12
    if calendar == "auto":
        gregorian = (year, month, int(day)) >= (1582, 10, 15)
    else:
        gregorian = calendar == "gregorian"
    b = 0
    if gregorian:
        a = y // 100
        b = 2 - a + a // 4
    jd = math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + day + b - 1524.5
    return jd + hour / 24.0


def calendar_date(jd: float, calendar: str = "auto"):
    """(year, month, day, hour) for a Julian Day number (UT)."""
    jd = jd + 0.5
    z = math.floor(jd)
    f = jd - z
    if calendar == "gregorian" or (calendar == "auto" and z >= 2299161):
        alpha = math.floor((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - alpha // 4
    else:
        a = z
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day = b - d - math.floor(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    whole = math.floor(day)
    return int(year), int(month), int(whole), (day - whole) * 24.0


def decimal_year(jd: float) -> float:
    y, m, d, h = calendar_date(jd)
    return y + (m - 0.5) / 12.0 + 0.0 * (d + h)


# -- delta-T -----------------------------------------------------------------------------

def _parabola(y):
    """Long-term trend of delta-T (Stephenson, Morrison, Hohenkerk 2016), seconds."""
    return -320.0 + 32.5 * ((y - 1825.0) / 100.0) ** 2


def _parabola_slope(y):
    return 32.5 * 2.0 * (y - 1825.0) / 10000.0


def _hermite(x0, y0, d0, x1, y1, d1):
    """Cubic through (x0, y0) and (x1, y1) with the given end slopes; returns a function of x."""
    h = x1 - x0

    def f(x):
        t = (np.asarray(x, dtype=float) - x0) / h
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * h * d0
                + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * h * d1)
    return f


def _year_frac(jd: float) -> float:
    """Julian-year coordinate used by the delta-T tables: 2000.0 is JD 2451545.0 (J2000)."""
    return (float(jd) - 1721045.0) / 365.25


@lru_cache(maxsize=1)
def _tables():
    base = Path(__file__).with_name("data")
    jd, dt, unc = [], [], []
    for line in (base / "deltat_observed_usno.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            jd.append(julian_day(y, m, d))
            dt.append(float(parts[3]))
            unc.append(0.001)
    last_obs = jd[-1]
    for line in (base / "deltat_predicted_usno.txt").read_text().splitlines()[1:]:
        parts = line.replace("\t", " ").split()
        if len(parts) >= 3:
            try:
                mjd, val = float(parts[0]), float(parts[2])
            except ValueError:
                continue
            j = mjd + MJD_TO_JD
            if j > last_obs:
                jd.append(j)
                dt.append(val)
                try:
                    unc.append(float(parts[-1]))
                except ValueError:
                    unc.append(1.0)
    return np.array(jd), np.array(dt), np.array(unc)


@lru_cache(maxsize=1)
def _spline_model():
    table = np.load(Path(__file__).with_name("data") / "deltat_smh_splines.npz")["table"]
    lower, upper = table[0], table[1]
    coeffs = table[2:]          # a3, a2, a1, a0 for t in 0..1 across each interval

    def spline(y):
        y = np.asarray(y, dtype=float)
        i = np.clip(np.searchsorted(lower, y, side="right") - 1, 0, len(lower) - 1)
        t = (y - lower[i]) / (upper[i] - lower[i])
        v = coeffs[0][i]
        for c in coeffs[1:]:
            v = v * t + c[i]
        return v

    def slope(y):
        h = 1e-3
        return (spline(y + h) - spline(y - h)) / (2 * h)

    tj, td, _ = _tables()
    y_first = _year_frac(tj[0])
    y_last = _year_frac(tj[-1])
    offset = float(td[0]) - float(spline(y_first))     # make the spline meet the observed table
    ramp_start = y_first - 1.5

    # smooth patches joining the spline table to the long-term parabola
    y_left = float(lower[0])
    left_patch = _hermite(y_left - 800.0, _parabola(y_left - 800.0), _parabola_slope(y_left - 800.0),
                          y_left, float(spline(y_left)), float(slope(y_left)))
    x1 = (y_last + 800.0) // 100.0 * 100.0
    recent = tj >= tj[-1] - 366.0
    last_slope = (td[-1] - td[recent][0]) / ((tj[-1] - tj[recent][0]) / 365.25)
    right_patch = _hermite(y_last, float(td[-1]), float(last_slope), x1, _parabola(x1), _parabola_slope(x1))
    return spline, left_patch, right_patch, y_left, offset, ramp_start, y_first, y_last, x1


@lru_cache(maxsize=1)
def _scalar_model():
    tj, td, _ = _tables()
    table = np.load(Path(__file__).with_name("data") / "deltat_smh_splines.npz")["table"]
    spline, _lp, _rp, y_left, offset, ramp_start, y_first, y_last, x1 = _spline_model()
    return (tj.tolist(), td.tolist(), table[0].tolist(), table[1].tolist(), [c.tolist() for c in table[2:]],
            y_left, offset, ramp_start, y_first)


def _delta_t_scalar(j):
    """delta_t for one date without numpy overhead; None when the date is outside the common range."""
    tj, td, lower, upper, coeffs, y_left, offset, ramp_start, y_first = _scalar_model()
    if tj[0] <= j <= tj[-1]:
        i = bisect.bisect_right(tj, j) - 1
        if i >= len(tj) - 1:
            return td[-1]
        slope = (td[i + 1] - td[i]) / (tj[i + 1] - tj[i])
        return slope * (j - tj[i]) + td[i]
    y = (j - 1721045.0) / 365.25
    if y_left <= y < y_first:
        i = bisect.bisect_right(lower, y) - 1
        i = 0 if i < 0 else len(lower) - 1 if i > len(lower) - 1 else i
        t = (y - lower[i]) / (upper[i] - lower[i])
        v = coeffs[0][i]
        for c in coeffs[1:]:
            v = v * t + c[i]
        w = min(max((y - ramp_start) / (y_first - ramp_start), 0.0), 1.0)
        return v + offset * w
    return None


def delta_t(jd_ut):
    """TT minus UT in seconds for a Julian Day (UT). Accepts a float or an array (fully vectorised)."""
    if np.ndim(jd_ut) == 0:                      # one date: plain Python arithmetic, same formulas as the array path
        value = _delta_t_scalar(float(jd_ut))
        if value is not None:
            return value
    return _delta_t_array(jd_ut)


def _delta_t_array(jd_ut):
    jd = np.asarray(jd_ut, dtype=float)
    tj, td, _ = _tables()
    spline, left_patch, right_patch, y_left, offset, ramp_start, y_first, y_last, x1 = _spline_model()
    y = (jd - 1721045.0) / 365.25
    out = np.empty(jd.shape)
    in_table = (jd >= tj[0]) & (jd <= tj[-1])
    far_left = y < y_left - 800.0
    left = ~far_left & (y < y_left)
    mid = (y >= y_left) & (y < y_first)
    right = ~in_table & (jd > tj[-1]) & (y <= x1)
    far_right = ~in_table & (y > x1)
    out[in_table] = np.interp(jd[in_table], tj, td)
    out[far_left] = _parabola(y[far_left])
    out[left] = left_patch(y[left])
    w = np.clip((y[mid] - ramp_start) / (y_first - ramp_start), 0.0, 1.0)
    out[mid] = spline(y[mid]) + offset * w
    out[right] = right_patch(y[right])
    out[far_right] = _parabola(y[far_right])
    return float(out) if out.ndim == 0 else out


def delta_t_uncertainty(jd_ut) -> float:
    """A rough 1-sigma uncertainty of delta_t(jd_ut) in seconds. An estimate, not a guarantee."""
    tj, _, unc = _tables()
    j = float(jd_ut)
    if tj[0] <= j <= tj[-1]:
        return float(np.interp(j, tj, unc))
    y = _year_frac(j)
    if y > 2033:
        return float(unc[-1] + 0.5 * (y - 2033.75))
    if y >= 1955:
        return 0.1
    if y >= 1800:
        return 1.0
    if y >= 1600:
        return 3.0
    if y >= -720:
        return 10.0 + 0.02 * abs(y - 1600)
    return 600.0


def ut_to_tt(jd_ut):
    """Julian Day (TT) from Julian Day (UT)."""
    return np.asarray(jd_ut, dtype=float) + np.asarray(delta_t(jd_ut)) / SECONDS_PER_DAY


def tdb_minus_tt(jd_tt):
    """TDB minus TT in seconds (periodic terms, Fairhead and Bretagnon; accurate to a few microseconds)."""
    t = (np.asarray(jd_tt, dtype=float) - J2000) / 36525.0   # Julian centuries
    return (0.001657 * np.sin(628.3076 * t + 6.2401)
            + 0.000022 * np.sin(575.3385 * t + 4.2970)
            + 0.000014 * np.sin(1256.6152 * t + 6.1969)
            + 0.000005 * np.sin(606.9777 * t + 4.0212)
            + 0.000005 * np.sin(52.9691 * t + 0.4444)
            + 0.000002 * np.sin(21.3299 * t + 5.5431)
            + 0.000010 * t * np.sin(628.3076 * t + 4.2490))


def split_jd(jd):
    """Split a Julian Day into (whole, fraction): whole is a half-integer, the fraction is exact."""
    jd = np.asarray(jd, dtype=float)
    whole = np.floor(jd - 0.5) + 0.5
    return whole, jd - whole


def ut_to_tt_parts(jd_ut):
    """Terrestrial Time as (whole, fraction) from a Julian Day (UT), keeping full precision."""
    whole, frac = split_jd(jd_ut)
    return whole, frac + np.asarray(delta_t(jd_ut)) / SECONDS_PER_DAY


def tt_to_tdb(jd_tt):
    return np.asarray(jd_tt, dtype=float) + tdb_minus_tt(jd_tt) / SECONDS_PER_DAY
