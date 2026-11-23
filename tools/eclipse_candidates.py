import os
import numpy as np, swisseph as real
import alif_ephemeris.swe_compat as ours
from alif_ephemeris import eclipses as E, timescales as T
real.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", "")); ours.set_ephe_path("data/raw")
spk = ours._spk()
starts = [2430000.5 + 1200 * k for k in range(0, 20)]
ref = [real.sol_eclipse_when_glob(s, real.FLG_SWIEPH, 0, False) for s in starts]
dt = lambda t: float(T.delta_t(t)) / 86400
def run(flat, rm, re):
    E.EARTH_FLATTENING, E.R_MOON_SOLAR, E.R_EARTH_SOLAR = flat, rm, re
    b, e_, m = [], [], []
    for s, (kind, r) in zip(starts, ref):
        x = E.next_solar_eclipse(spk, s + dt(s), False)
        b.append((x.begin - dt(x.maximum) - r[2]) * 86400); e_.append((x.end - dt(x.maximum) - r[3]) * 86400); m.append((x.maximum - dt(x.maximum) - r[0]) * 86400)
    b, e_, m = map(np.array, (b, e_, m))
    return "begin mean %+.1f rms %.1f | end mean %+.1f rms %.1f | max rms %.1f worst %.1f" % (b.mean(), np.sqrt((b**2).mean()), e_.mean(), np.sqrt((e_**2).mean()), np.sqrt((m**2).mean()), np.abs(m).max())
f = 1 / 298.257223563
for name, args in (("ellipse, Moon 1737.4", (f, 1737.4, 6378.14)), ("ellipse, Moon 1738.5", (f, 1738.5, 6378.14)),
                   ("sphere,  Moon 1737.4", (0.0, 1737.4, 6378.14)), ("sphere,  Moon 1738.5", (0.0, 1738.5, 6378.14)),
                   ("sphere 6371, Moon 1737.4", (0.0, 1737.4, 6371.0)), ("sphere 6371, Moon 1738.5", (0.0, 1738.5, 6371.0))):
    print("%-26s %s" % (name, run(*args)))

print("--- grid over Earth radius (ellipse) and Moon radius: rms of begin and end together")
best = None
for re in (6340.0, 6355.0, 6365.0, 6371.0, 6378.14, 6390.0):
    row = []
    for rm in (1720.0, 1730.0, 1737.4, 1745.0, 1755.0):
        E.EARTH_FLATTENING, E.R_MOON_SOLAR, E.R_EARTH_SOLAR = 1 / 298.257223563, rm, re
        r = []
        for s, (kind, rr) in zip(starts[:10], ref[:10]):
            x = E.next_solar_eclipse(spk, s + dt(s), False)
            r += [(x.begin - dt(x.maximum) - rr[2]) * 86400, (x.end - dt(x.maximum) - rr[3]) * 86400]
        v = float(np.sqrt((np.array(r) ** 2).mean())); row.append(v)
        if best is None or v < best[0]: best = (v, re, rm)
    print("Earth %.1f: " % re + "  ".join("%.1f" % v for v in row), "  (Moon 1720 1730 1737.4 1745 1755)")
print("best", best)
