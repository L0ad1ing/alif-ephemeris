"""Compare our time scales with SOFA (ERFA) and Skyfield. Development check, not shipped."""
import numpy as np
import erfa
from skyfield.api import load
from alif_ephemeris import timescales as T

ts = load.timescale()
w = 0
for jd in np.linspace(2400000, 2500000, 400):
    w = max(w, abs(T.tdb_minus_tt(jd) - erfa.dtdb(jd, 0.0, 0.0, 0.0, 0.0, 0.0)))
print("TDB-TT worst difference vs SOFA dtdb: %.2f microseconds" % (w * 1e6))
print("year      ours   skyfield   diff(s)")
worst = 0
for y in [-700, -300, 0, 500, 1000, 1300, 1500, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1920, 1940,
          1955, 1965, 1972, 1973.5, 1980, 2000, 2020, 2026.5, 2030, 2033, 2035, 2040, 2050, 2100, 2300, 3000]:
    yr = int(np.floor(y))
    jd = T.julian_day(yr, 1, 1) + (y - yr) * 365.25
    a = T.delta_t(jd)
    b = float(ts.tt_jd(jd + a / 86400).delta_t)
    worst = max(worst, abs(a - b))
    print("%7.1f %9.3f %9.3f %8.3f" % (y, a, b, a - b))
print("worst diff vs Skyfield delta-T: %.3f s" % worst)
print("uncertainty 1850:", T.delta_t_uncertainty(T.julian_day(1850, 1, 1)),
      " 2060:", T.delta_t_uncertainty(T.julian_day(2060, 1, 1)))
