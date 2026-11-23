import os
import sys, numpy as np, swisseph as swe
from scipy.optimize import least_squares
swe.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
J2000 = 2451545.0
def kepler(e, M):
    E = M.copy()
    for _ in range(80): E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E
def model(p, tt):
    a, e, inc, node, argp, M0, n, dnode, dargp = p
    t = tt - J2000
    M = np.radians(M0) + np.radians(n) * t
    E = kepler(e, np.mod(M, 2 * np.pi))
    x, y = a * (np.cos(E) - e), a * np.sqrt(max(1 - e * e, 1e-9)) * np.sin(E)
    r = np.hypot(x, y); u = np.arctan2(y, x) + np.radians(argp) + np.radians(dargp) * t
    I, O = np.radians(inc), np.radians(node) + np.radians(dnode) * t
    X = r * (np.cos(O) * np.cos(u) - np.sin(O) * np.sin(u) * np.cos(I)); Y = r * (np.sin(O) * np.cos(u) + np.cos(O) * np.sin(u) * np.cos(I)); Z = r * np.sin(u) * np.sin(I)
    return np.degrees(np.arctan2(Y, X)) % 360, np.degrees(np.arcsin(np.clip(Z / r, -1, 1))), r
body = int(sys.argv[1]); mode = sys.argv[2]; span = float(sys.argv[3])
tt = J2000 + np.linspace(-span, span, 200)
flags = swe.FLG_SWIEPH | swe.FLG_TRUEPOS | swe.FLG_NONUT | (swe.FLG_J2000 if len(sys.argv) > 4 else 0) | (swe.FLG_HELCTR if mode == "helio" else 0)
ref = np.array([swe.calc(t, body, flags)[0][:3] for t in tt])
print("range of distance %.5f..%.5f; longitude advance over span %.1f deg" % (ref[:, 2].min(), ref[:, 2].max(), (np.unwrap(np.radians(ref[:, 0]))[-1] - np.unwrap(np.radians(ref[:, 0]))[0]) * 57.2958))
def resid(p):
    lon, lat, r = model(p, tt)
    return np.concatenate([((lon - ref[:, 0] + 180) % 360 - 180) * 3600, (lat - ref[:, 1]) * 3600, (r - ref[:, 2]) / max(ref[:, 2].mean(), 1e-9) * 1e5])
dl = np.unwrap(np.radians(ref[:, 0])); n0 = np.degrees((dl[-1] - dl[0]) / (tt[-1] - tt[0]))
best = None
for a0 in {float(np.median(ref[:, 2])), 1.0, 0.00257, 0.1}:
    for inc in (0.5, 5.0, 20.0):
        for node in (0, 120, 240):
            for M0 in (0, 90, 180, 270):
                for e0 in (0.05, 0.2):
                    try:
                        r = least_squares(resid, [a0, e0, inc, node, 0.0, M0, n0, 0.0, 0.0], x_scale=[max(a0, 0.01) * 0.1, 0.05, 5, 30, 30, 30, abs(n0) * 0.1 + 0.01, 0.01, 0.01], max_nfev=400, xtol=1e-14, ftol=1e-14, gtol=1e-14)
                    except Exception:
                        continue
                    if best is None or r.cost < best.cost: best = r
print("rms %.4f  a=%.9f e=%.9f i=%.9f node=%.9f argp=%.9f M0=%.9f n=%.11f dnode=%.11f dargp=%.11f" % ((np.sqrt((resid(best.x) ** 2).mean()),) + tuple(best.x)))
