import os
import numpy as np, swisseph as real
import alif_ephemeris.swe_compat as ours
from alif_ephemeris import eclipses as E, timescales as T
real.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", "")); ours.set_ephe_path("data/raw")
spk = ours._spk()
place = (77.2167, 28.6667, 0.0)
dt = lambda t: float(T.delta_t(t)) / 86400
# every new moon with any overlap at this place, between 1980 and 2000, with the Sun's altitude at C1, max, C4
sw = []
t = 2444240.0
while t < 2451500.0:
    r = real.sol_eclipse_when_loc(t, place, real.FLG_SWIEPH, False); sw.append(r[1][0]); t = r[1][0] + 20
sw = np.array(sw)
print("reference lists %d eclipses for this place between 1980 and 2000" % len(sw))
E.next_local_solar_eclipse.__defaults__
# our candidate list (no visibility filter) via the internal pieces
from scipy.optimize import brentq
cands = []
for t_new in E._syzygies(spk, 2444240.0 + dt(2444240.0), 1, "new"):
    if t_new > 2451500.0: break
    ut = lambda tt: tt - dt(tt)
    tmax = E._minimize(lambda tt: E._local_sep(spk, tt, place, ut(tt))[0], t_new - 0.3, t_new + 0.3)
    sep, rs, rm = E._local_sep(spk, tmax, place, ut(tmax))
    if sep < rs + rm:
        g1 = lambda tt: E._local_sep(spk, tt, place, ut(tt))[0] - sum(E._local_sep(spk, tt, place, ut(tt))[1:])
        c1 = brentq(g1, tmax - 0.25, tmax, xtol=1e-9); c4 = brentq(g1, tmax, tmax + 0.25, xtol=1e-9)
        alts = [E._sun_altitude(spk, x, place, ut(x)) for x in (c1, tmax, c4)]
        cands.append((tmax - dt(tmax), alts, (rs + rm - sep) / (2 * rs)))
print("%-12s %-10s %-34s %s" % ("max (UT)", "in ref?", "Sun altitude at C1 / max / C4 (deg)", "magnitude"))
for tm, alts, mag in cands:
    inref = np.any(np.abs(sw - tm) < 0.01)
    print("%-12.4f %-10s %-34s %.3f" % (tm, "yes" if inref else "NO", " / ".join("%6.2f" % a for a in alts), mag))
