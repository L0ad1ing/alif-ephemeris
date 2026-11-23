import os
import numpy as np, swisseph as swe
from alif_ephemeris import Ephemeris, positions as P, planets as PL, timescales as T, julian_day
swe.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
e = Ephemeris("data/raw/de440.bsp")
rng = np.random.default_rng(13)
d = lambda a, b: abs((a - b + 180) % 360 - 180) * 3600
for name, body, sid in (("Moon", PL.MOON, 1), ("Sun", PL.SUN, 0), ("Mars", PL.MARS, 4)):
    wl = wb = wd = 0
    for _ in range(25):
        jd = julian_day(1900, 1, 1) + rng.uniform(0, 365.25 * 125)
        lon, lat, alt = rng.uniform(-180, 180), rng.uniform(-70, 70), rng.uniform(0, 3000)
        swe.set_topo(lon, lat, alt)
        tt = float(T.ut_to_tt(jd)); w, f = T.split_jd(tt)
        lo, la, di = P.observe(e.spk, body, float(w), float(f), observer="site", site=(lon, lat, alt), jd_ut=jd)
        r = swe.calc_ut(jd, sid, swe.FLG_SWIEPH | swe.FLG_TOPOCTR)[0]
        wl = max(wl, d(np.degrees(lo), r[0])); wb = max(wb, abs(np.degrees(la) - r[1]) * 3600); wd = max(wd, abs(di - r[2]) / r[2])
    print("%-5s topocentric: longitude %.3f  latitude %.3f arcsec, distance relative error %.2e" % (name, wl, wb, wd))
