"""Show that Spica at 180 degrees reproduces JHora's printed Lahiri (1888-2000 charts)."""
import numpy as np
from alif_ephemeris import ayanamsa as A, timescales as T
src = open("tools/jhora_ayanamsa_scan.py", encoding="utf-8").read().split("swe.set_sid_mode")[0]
ns = {}
exec(src, ns)
res = []
for name, jd, printed in ns["rows"]:
    tt = float(T.ut_to_tt(jd))
    res.append(((float(A.ayanamsa("Lahiri (JHora)", tt)) - printed) * 3600, jd))
d = np.array([r[0] for r in res]); jds = np.array([r[1] for r in res])
close = np.abs(d) < 0.05
print("%d charts; %d agree with Spica-at-180 to within 0.05 arcsec (worst of those %.4f, mean %.4f)" % (
    len(d), close.sum(), np.abs(d[close]).max(), d[close].mean()))
print("the other %d charts use a different ayanamsa; their differences range from %.0f to %.0f arcsec" % (
    (~close).sum(), np.abs(d[~close]).min(), np.abs(d[~close]).max()))
y = 2000 + (jds[close] - 2451545.0) / 365.25
print("agreeing charts span the years %.0f to %.0f" % (y.min(), y.max()))
