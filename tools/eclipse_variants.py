import os
import itertools, numpy as np, swisseph as real
import alif_ephemeris.swe_compat as ours
from alif_ephemeris import eclipses as E, planets, timescales as T
real.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", "")); ours.set_ephe_path("data/raw")
spk = ours._spk()
starts = (2432050.0, 2451545.0, 2459000.5)
ref_l = [real.lun_eclipse_when(s, real.FLG_SWIEPH, 0, False)[1] for s in starts]
ref_s = [real.sol_eclipse_when_glob(s, real.FLG_SWIEPH, 0, False)[1] for s in starts]
variants = {
 "sun true, moon true": (dict(light_time=False, deflection=False, aberration=False), dict(light_time=False, deflection=False, aberration=False)),
 "sun light-time, moon true": (dict(light_time=True, deflection=False, aberration=False), dict(light_time=False, deflection=False, aberration=False)),
 "sun apparent, moon apparent": (dict(), dict()),
 "sun light-time, moon light-time": (dict(light_time=True, deflection=False, aberration=False), dict(light_time=True, deflection=False, aberration=False)),
 "sun apparent, moon true": (dict(), dict(light_time=False, deflection=False, aberration=False)),
}
for name, (ks, km) in variants.items():
    E._sun_moon = lambda spk, tt, ks=ks, km=km: (E._geocentric(spk, planets.SUN, tt, **ks), E._geocentric(spk, planets.MOON, tt, **km))
    out = []
    for s, rl, rs_ in zip(starts, ref_l, ref_s):
        a = E.next_lunar_eclipse(spk, s + float(T.delta_t(s)) / 86400, False)
        b = E.next_solar_eclipse(spk, s + float(T.delta_t(s)) / 86400, False)
        dt = lambda t: float(T.delta_t(t)) / 86400
        out.append(((a.maximum - dt(a.maximum) - rl[0]) * 86400, (a.penumbral[0] - dt(a.maximum) - rl[6]) * 86400,
                    (a.penumbral[1] - dt(a.maximum) - rl[7]) * 86400, (b.maximum - dt(b.maximum) - rs_[0]) * 86400))
    print("%-34s lunar max/pen-begin/pen-end diff (s): %s | solar max diff: %s" % (
        name, " ".join("%.0f/%.0f/%.0f" % o[:3] for o in out), " ".join("%.0f" % o[3] for o in out)))
