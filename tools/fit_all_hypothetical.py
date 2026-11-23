import os
import sys, json, numpy as np, swisseph as swe
from scipy.optimize import least_squares
swe.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
J2000 = 2451545.0
exec(open("tools/fit_hypothetical.py").read().split("body = int(")[0].split("J2000 = 2451545.0")[1])
names = {40: "Cupido", 41: "Hades", 42: "Zeus", 43: "Kronos", 44: "Apollon", 45: "Admetos", 46: "Vulkanus", 47: "Poseidon",
         48: "Isis-Transpluto", 49: "Nibiru", 50: "Harrington", 51: "Neptune-Leverrier", 52: "Neptune-Adams",
         53: "Pluto-Lowell", 54: "Pluto-Pickering", 55: "Vulcan", 56: "White Moon", 57: "Proserpina", 58: "Waldemath"}
tt = J2000 + np.linspace(-36525, 36525, 120)
out = {}
for body, nm in names.items():
    for mode, helio in (("heliocentric", True), ("geocentric", False)):
        flags = swe.FLG_SWIEPH | swe.FLG_TRUEPOS | swe.FLG_NONUT | swe.FLG_J2000 | (swe.FLG_HELCTR if helio else 0)
        try:
            ref = np.array([swe.calc(t, body, flags)[0][:3] for t in tt])
        except Exception as ex:
            print(nm, mode, "unavailable:", str(ex)[:50]); continue
        a0 = float(np.median(ref[:, 2]))
        def resid(p):
            lon, lat, r = model(p, tt, J2000)
            return np.concatenate([((lon - ref[:, 0] + 180) % 360 - 180) * 3600, (lat - ref[:, 1]) * 3600, (r - ref[:, 2]) * 1e5])
        best = None
        for inc in (1.0, 5.0, 12.0, 30.0):
            for node in (0, 90, 180, 270):
                for M0 in (0, 120, 240):
                    try:
                        r = least_squares(resid, [a0, 0.02, inc, node, 0.0, M0], x_scale=[a0 * 0.1, 0.05, 5, 30, 30, 30], max_nfev=200)
                    except Exception:
                        continue
                    if best is None or r.cost < best.cost: best = r
        rms = float(np.sqrt((resid(best.x) ** 2).mean()))
        print("%-18s %-13s fit rms %.4f  a=%.5f e=%.5f i=%.4f node=%.4f argp=%.4f M0=%.4f" % (nm, mode, rms, *best.x))
        out.setdefault(nm, {})[mode] = dict(rms=rms, elements=[float(x) for x in best.x])
        if mode == "heliocentric" and rms < 0.01: break
json.dump(out, open("tools/_hypothetical_fit.json", "w"), indent=1)
