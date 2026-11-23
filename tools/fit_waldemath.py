"""Fit Kepler elements (geocentric, mean J2000 ecliptic, epoch J2000) to the reference program's Waldemath dark moon."""
import os
import numpy as np, swisseph as r
from scipy.optimize import least_squares

r.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
FL = r.FLG_SWIEPH | r.FLG_TRUEPOS | r.FLG_NOABERR | r.FLG_NOGDEFL | r.FLG_J2000 | r.FLG_NONUT
K = 0.01720209895
J2000 = 2451545.0

def kepler(e, M):
    E = M + e * np.sin(M)
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E

def model(p, days):
    a, e, inc, node, argp, M0, n_extra, node_dot, argp_dot = p
    n = K / a ** 1.5 * (1 + n_extra)
    M = np.radians(M0) + n * days
    E = kepler(e, M)
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e * e) * np.sin(E)
    T = days / 36525.0
    w, O, i = np.radians(argp + argp_dot * T), np.radians(node + node_dot * T), np.radians(inc)
    X = (np.cos(O) * np.cos(w) - np.sin(O) * np.sin(w) * np.cos(i)) * x + (-np.cos(O) * np.sin(w) - np.sin(O) * np.cos(w) * np.cos(i)) * y
    Y = (np.sin(O) * np.cos(w) + np.cos(O) * np.sin(w) * np.cos(i)) * x + (-np.sin(O) * np.sin(w) + np.cos(O) * np.cos(w) * np.cos(i)) * y
    Z = np.sin(w) * np.sin(i) * x + np.cos(w) * np.sin(i) * y
    return np.stack([X, Y, Z], axis=-1)

def reference(days):
    out = []
    for d in days:
        x = r.calc(J2000 + d, 58, FL)[0]
        lon, lat, dist = np.radians(x[0]), np.radians(x[1]), x[2]
        out.append([dist * np.cos(lat) * np.cos(lon), dist * np.cos(lat) * np.sin(lon), dist * np.sin(lat)])
    return np.array(out)

def fit_span(q0, days, ref, free):
    base = np.array(q0, dtype=float)
    def pack(v):
        q = base.copy(); q[free] = v; return q
    lo = np.array([0.001, 0.0, 0.0, -1e5, -1e5, -1e5, -1.5, -1e5, -1e5])[free]
    hi = np.array([0.05, 0.6, 20.0, 1e5, 1e5, 1e5, 0.5, 1e5, 1e5])[free]
    x0 = np.clip(base[free], lo + 1e-12, hi - 1e-12)
    res = least_squares(lambda v: (model(pack(v), days) - ref).ravel() * 1e5, x0, bounds=(lo, hi), max_nfev=400, xtol=1e-14, ftol=1e-14)
    return pack(res.x), np.linalg.norm(model(pack(res.x), days) - ref, axis=1).max()

all_free = list(range(9))
q = [0.00684, 0.16, 2.5, 0.0, 0.0, 0.0, -0.99828, 0.0, 0.0]
spans = [(60, 1.0), (240, 2.0), (900, 3.0), (3000, 4.0), (12000, 5.0), (36500, 5.0)]
free_sets = [[0, 1, 2, 3, 4, 5, 6]] * 3 + [all_free] * 3
for (span, step), free in zip(spans, free_sets):
    days = np.arange(0, span, step)
    ref = reference(days)
    best = None
    for dq in [[0, 0, 0, nd, ad, md, 0, 0, 0] for nd in range(0, 360, 60) for ad in range(0, 360, 90) for md in range(0, 360, 90)] if span == 60 else ([0] * 9,):
        try:
            cand, err = fit_span(np.array(q) + np.array(dq, dtype=float), days, ref, free)
        except Exception:
            continue
        if best is None or err < best[1]:
            best = (cand, err)
    q, err = best
    print("span %6d days: max error %.3e AU" % (span, err), np.round(q, 8))
print("FINAL", list(q))
