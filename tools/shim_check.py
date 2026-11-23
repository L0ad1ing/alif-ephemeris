"""Compare the Swiss-style adapter with real pyswisseph call by call (development check)."""
import os
import sys, numpy as np
import swisseph as real
import alif_ephemeris.swe_compat as ours
real.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
ours.set_ephe_path("data/raw")
rng = np.random.default_rng(31)
d = lambda a, b: abs((a - b + 180) % 360 - 180) * 3600
print("julday/revjul:", ours.julday(2026, 11, 22, 12.5) == real.julday(2026, 11, 22, 12.5), ours.revjul(2461000.3) == real.revjul(2461000.3))
w = {}
for _ in range(30):
    jd = real.julday(1900, 1, 1) + rng.uniform(0, 365.25 * 150)
    for body in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13):
        for flags in (real.FLG_SWIEPH | real.FLG_SPEED, real.FLG_SWIEPH | real.FLG_SPEED | real.FLG_TRUEPOS | real.FLG_NONUT,
                      real.FLG_SWIEPH | real.FLG_EQUATORIAL):
            a = ours.calc_ut(jd, body, flags)[0]; b = real.calc_ut(jd, body, flags)[0]
            key = (body, flags & ~real.FLG_SWIEPH)
            w[key] = max(w.get(key, 0), d(a[0], b[0]), abs(a[1] - b[1]) * 3600)
bad = {k: v for k, v in w.items() if v > 1.0}
print("calc_ut worst position error: %.3f arcsec; cases over 1 arcsec: %s" % (max(w.values()), {(k[0], k[1]): round(v, 2) for k, v in bad.items()}))
