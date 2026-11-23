import os
import numpy as np, swisseph as swe
from alif_ephemeris import Ephemeris, stars as S, julian_day, timescales as T
swe.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
e = Ephemeris("data/raw/de440s.bsp")
rng = np.random.default_rng(8)
dates = julian_day(1900, 1, 1) + rng.uniform(0, 365.25 * 150, 12)
print("%-10s %24s %24s" % ("star", "apparent lon/lat vs Swiss", "mean (precession only) vs Swiss"))
for nm in ["Spica", "Regulus", "Aldebaran", "Antares", "Algol", "Sirius", "Fomalhaut", "Revati", "Pollux", "Arcturus", "Vega", "Alcyone"]:
    st = S.find(nm)
    wl = wb = ml = 0
    for jd in dates:
        tt = float(T.ut_to_tt(jd)); w, f = T.split_jd(tt)
        lon, lat = S.apparent(e.spk, st, float(w), float(f))
        try:
            r = swe.fixstar(nm, tt, swe.FLG_SWIEPH)[0]
        except Exception as ex:
            print(nm, "swiss error", str(ex)[:50]); break
        wl = max(wl, abs((lon - r[0] + 180) % 360 - 180) * 3600); wb = max(wb, abs(lat - r[1]) * 3600)
        rm = swe.fixstar(nm, tt, swe.FLG_SWIEPH | swe.FLG_NOABERR | swe.FLG_NOGDEFL | swe.FLG_NONUT)[0]
        mlon, mlat = S.mean_ecliptic(st, tt)
        ml = max(ml, abs((mlon - rm[0] + 180) % 360 - 180) * 3600)
    print("%-10s lon %.4f lat %.4f arcsec      mean lon %.4f" % (nm, wl, wb, ml))
