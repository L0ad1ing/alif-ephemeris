"""Recover the Kepler elements of the hypothetical (Uranian and similar) bodies from reference heliocentric positions.

The bodies are defined by fixed orbital elements. We sample heliocentric longitude, latitude and distance and fit an
ellipse; if the fit leaves no residual, the model (and the elements) are right.
"""
import os
import sys, numpy as np, swisseph as swe
from scipy.optimize import least_squares
swe.set_ephe_path(os.environ.get("REFERENCE_EPHE_PATH", ""))
J2000 = 2451545.0
def kepler(e, M):
    E = M.copy()
    for _ in range(60): E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E
def model(p, tt, frame_epoch):
    a, e, inc, node, argp, M0 = p[:6]
    k = 0.01720209895                                   # Gaussian gravitational constant
    n = k / a ** 1.5
    M = np.radians(M0) + n * (tt - frame_epoch)
    E = kepler(e, M % (2 * np.pi))
    x, y = a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E)
    r = np.hypot(x, y); nu = np.arctan2(y, x)
    u = nu + np.radians(argp)
    I, O = np.radians(inc), np.radians(node)
    X = r * (np.cos(O) * np.cos(u) - np.sin(O) * np.sin(u) * np.cos(I))
    Y = r * (np.sin(O) * np.cos(u) + np.cos(O) * np.sin(u) * np.cos(I))
    Z = r * np.sin(u) * np.sin(I)
    return np.degrees(np.arctan2(Y, X)) % 360, np.degrees(np.arcsin(Z / r)), r
body = int(sys.argv[1]); epoch_guess = float(sys.argv[2]) if len(sys.argv) > 2 else J2000
tt = J2000 + np.linspace(-20000, 20000, 80)
flags = swe.FLG_SWIEPH | swe.FLG_HELCTR | swe.FLG_TRUEPOS | swe.FLG_NONUT | (swe.FLG_J2000 if len(sys.argv) > 3 else 0)
ref = np.array([swe.calc(t, body, flags)[0][:3] for t in tt])
print("reference sample:", ref[:2].round(4).tolist())
a0 = ref[:, 2].mean()
def resid(p):
    lon, lat, r = model(p, tt, epoch_guess)
    return np.concatenate([((lon - ref[:, 0] + 180) % 360 - 180) * 3600, (lat - ref[:, 1]) * 3600, (r - ref[:, 2]) * 1e5])
best = None
for inc in (1.0, 5.0, 10.0):
    for node in (0, 90, 180, 270):
        for argp in (0, 180):
            for M0 in (0, 120, 240):
                x0 = [a0, 0.02, inc, node, argp, M0]
                try:
                    r = least_squares(resid, x0, x_scale=[1, 0.05, 5, 30, 30, 30], max_nfev=300)
                except Exception:
                    continue
                if best is None or r.cost < best.cost: best = r
print("best fit elements a=%.5f e=%.5f i=%.4f node=%.4f argp=%.4f M0=%.4f at JD %.1f" % (*best.x, epoch_guess))
res = resid(best.x)
print("residual rms %.4f arcsec-equivalent (worst %.4f)" % (np.sqrt((res ** 2).mean()), np.abs(res).max()))
