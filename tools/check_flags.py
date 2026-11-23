import os
import numpy as np, swisseph as swe
from alif_ephemeris import Ephemeris, positions as P, planets as PL, timescales as T, julian_day
swe.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
e = Ephemeris("data/raw/de440.bsp")
rng = np.random.default_rng(12)
dates = julian_day(1900, 1, 1) + rng.uniform(0, 365.25 * 125, 12)
bodies = [(PL.SUN, 0), (PL.MOON, 1), (PL.MERCURY, 2), (PL.MARS, 4), (PL.JUPITER, 5), (PL.SATURN, 6), (PL.PLUTO, 9)]
F = swe
cases = {
 "apparent": (F.FLG_SWIEPH, {}),
 "true position": (F.FLG_SWIEPH | F.FLG_TRUEPOS, dict(light_time=False, deflection=False, aberration=False)),
 "no aberration/deflection": (F.FLG_SWIEPH | F.FLG_NOABERR | F.FLG_NOGDEFL, dict(deflection=False, aberration=False)),
 "no nutation": (F.FLG_SWIEPH | F.FLG_NONUT, dict(nutation=False)),
 "J2000": (F.FLG_SWIEPH | F.FLG_J2000, dict(frame="j2000")),
 "equatorial": (F.FLG_SWIEPH | F.FLG_EQUATORIAL, dict(equatorial=True)),
 "equatorial J2000": (F.FLG_SWIEPH | F.FLG_EQUATORIAL | F.FLG_J2000, dict(equatorial=True, frame="j2000")),
 "heliocentric": (F.FLG_SWIEPH | F.FLG_HELCTR, dict(observer="sun")),
 "barycentric": (F.FLG_SWIEPH | F.FLG_BARYCTR, dict(observer="ssb")),
}
d = lambda a, b: abs((a - b + 180) % 360 - 180) * 3600
print("%-26s %s" % ("flags", "worst longitude / latitude difference in arcsec, per body"))
for label, (flags, kw) in cases.items():
    row = []
    for body, sid in bodies:
        wl = wb = 0
        for jd in dates:
            tt = float(T.ut_to_tt(jd)); w, f = T.split_jd(tt)
            if (label in ("heliocentric",) and body == PL.SUN) or (label == "barycentric" and False):
                continue
            lon, lat, dist = P.observe(e.spk, body, float(w), float(f), **kw)
            r = swe.calc(tt, sid, flags)[0]
            wl = max(wl, d(np.degrees(lon), r[0])); wb = max(wb, abs(np.degrees(lat) - r[1]) * 3600)
        row.append("%.3f/%.3f" % (wl, wb))
    print("%-26s %s" % (label, "  ".join(row)))
print("bodies:", [b[1] for b in bodies])
