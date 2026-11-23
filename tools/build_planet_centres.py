"""Build src/alif_ephemeris/data/planet_centres.npz.

The planet file we start from (DE440s) gives the barycentre of each planet-plus-moons system.
The planet's own centre is offset from it by up to a few thousand km (Pluto: about 2,100 km).
NASA's satellite kernels give that offset (body 599 relative to 5, and so on). Those files are
huge, so this tool reads only the records it needs over HTTP range requests, refits them into
compact Chebyshev pieces of our own, and records the measured error of the refit.

Output per planet: piece length (days), polynomial degree, start Julian Day (TDB), float32
coefficients (pieces, 3, degree+1) in km, and the worst measured error against NASA's data.

Run:  PYTHONPATH=src python tools/build_planet_centres.py
Needs network. Development tool; the .npz it writes is committed.
"""
import struct
import sys
import urllib.request

import numpy as np
from numpy.polynomial import chebyshev as C

from alif_ephemeris.spk import Segment, SPK, J2000_JD, SECONDS_PER_DAY

BASE = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/"

# Planet-centre offsets: body, centre, file, first and last year, closest distance to Earth in AU
JOBS = [
    (599, 5, "jup365.bsp", 1800.0, 2200.0, 3.9),
    (699, 6, "sat441.bsp", 1800.0, 2200.0, 7.9),
    (799, 7, "ura184_part-3.bsp", 1800.0, 2200.0, 17.2),
    (899, 8, "nep098_part-1.bsp", 1800.0, 2200.0, 28.7),
    (999, 9, "plu060.bsp", 1800.0, 2200.0, 28.5),
]
# Newer barycentre orbits for Uranus and Neptune (NASA's ura184 and nep098 solutions, the ones
# JPL Horizons uses). Stored as the small correction to the DE440s barycentre, which varies slowly.
DELTA_JOBS = [
    (7, 0, "ura184_part-3.bsp", 1850.0, 2150.0, 17.2),
    (8, 0, "nep098_part-1.bsp", 1850.0, 2150.0, 28.7),
]
DE440S = "data/raw/de440s.bsp"
TOLERANCE_MAS = 0.01          # worst allowed error from the refit, as an angle seen from Earth
AU_KM = 149597870.7
TEST_POINTS = 40


def fetch(url, start, length):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, start + length - 1)})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as exc:        # network hiccup: retry
            err = exc
    raise err


def segments(url):
    head = fetch(url, 0, 1024)
    rec = struct.unpack("<i", head[76:80])[0]
    out = []
    while rec:
        data = fetch(url, (rec - 1) * 1024, 1024)
        nxt, _p, nsum = struct.unpack("<ddd", data[:24])
        for i in range(int(nsum)):
            off = 24 + i * 40
            et0, et1 = struct.unpack("<dd", data[off:off + 16])
            target, center, frame, dtype, a0, a1 = struct.unpack("<6i", data[off + 16:off + 40])
            out.append(dict(target=target, center=center, et0=et0, et1=et1, a0=a0, a1=a1))
        rec = int(nxt)
    return out


def year_to_et(y):
    return (y - 2000.0) * 365.25 * SECONDS_PER_DAY


def load_pieces(url, body, centre, y0, y1):
    """Fetch the NASA type-2 records covering years y0..y1 as Segment objects."""
    segs = []
    for s in segments(url):
        if s["target"] != body or s["center"] != centre:
            continue
        if s["et1"] < year_to_et(y0) or s["et0"] > year_to_et(y1):
            continue
        trailer = np.frombuffer(fetch(url, (s["a1"] - 4) * 8, 32), dtype="<f8")
        init, intlen, rsize, n = trailer[0], trailer[1], int(trailer[2]), int(trailer[3])
        lo = max(s["et0"], year_to_et(y0))
        hi = min(s["et1"], year_to_et(y1))
        i0 = max(0, int((lo - init) // intlen) - 1)
        i1 = min(n - 1, int((hi - init) // intlen) + 1)
        raw = fetch(url, (s["a0"] - 1 + i0 * rsize) * 8, (i1 - i0 + 1) * rsize * 8)
        block = np.frombuffer(raw, dtype="<f8").reshape(-1, rsize)
        ncoef = (rsize - 2) // 3
        segs.append(Segment(body, centre, 1, 2, init + i0 * intlen, init + (i1 + 1) * intlen,
                            init + i0 * intlen, intlen, ncoef, block.shape[0],
                            block[:, 2:].reshape(-1, 3, ncoef).copy(), block[:, 0].copy(), block[:, 1].copy()))
        print("   fetched %s %d->%d: %d records, %.1f MB" % (url.split("/")[-1], body, centre,
                                                         block.shape[0], len(raw) / 1e6))
    segs.sort(key=lambda g: g.et_start)
    return segs


def truth(segs, et):
    """NASA's own value at times `et` (seconds past J2000), km."""
    et = np.atleast_1d(et)
    pos = np.full((3,) + et.shape, np.nan)
    for g in segs:
        m = (et >= g.et_start) & (et < g.et_end)
        if m.any():
            p, _ = SPK._cheb(g, et[m])
            pos[:, m] = p
    return pos


def fit(truthfn, et_start, et_end, length_days, degree):
    """Refit to pieces of `length_days`; return start et, coefficients (n,3,degree+1) float64."""
    L = length_days * SECONDS_PER_DAY
    n_pieces = int((et_end - et_start) // L)
    x = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))       # Chebyshev nodes
    mids = et_start + (np.arange(n_pieces) + 0.5) * L
    t = mids[:, None] + x[None, :] * (L / 2)
    y = truthfn(t.ravel()).reshape(3, n_pieces, degree + 1)
    V = C.chebvander(x, degree)
    coef = np.linalg.solve(V, np.transpose(y, (2, 0, 1)).reshape(degree + 1, -1)).reshape(degree + 1, 3, n_pieces)
    return np.transpose(coef, (2, 1, 0))


def evaluate(coef, et_start, length_days, et):
    L = length_days * SECONDS_PER_DAY
    i = np.clip(((et - et_start) // L).astype(int), 0, coef.shape[0] - 1)
    x = (et - (et_start + (i + 0.5) * L)) / (L / 2)
    out = np.empty((3,) + et.shape)
    for a in range(3):
        out[a] = np.array([C.chebval(xx, c) for xx, c in zip(x, coef[i, a].astype(np.float64))]) if et.size < 50 else \
            _vec_cheb(coef[i, a].astype(np.float64), x)
    return out


def _vec_cheb(c, x):
    """Clenshaw evaluation, vectorised: c has shape (n, degree+1), x shape (n,)."""
    b1 = np.zeros_like(x)
    b2 = np.zeros_like(x)
    for k in range(c.shape[1] - 1, 0, -1):
        b1, b2 = 2 * x * b1 - b2 + c[:, k], b1
    return x * b1 - b2 + c[:, 0]


def measure(truthfn, coef32, et_start, length_days, et_lo, et_hi):
    rng = np.random.default_rng(3)
    et_hi = min(et_hi, et_start + coef32.shape[0] * length_days * SECONDS_PER_DAY - 1.0)
    et = rng.uniform(et_lo, et_hi, 60000)
    ref = truthfn(et)
    got = evaluate(coef32, et_start, length_days, et)
    ok = ~np.isnan(ref[0])
    err = np.linalg.norm(ref[:, ok] - got[:, ok], axis=0)
    return float(err.max()), float(np.sqrt((err ** 2).mean()))


def truncate(coef, tol_km):
    """Zero the high-order coefficients that cannot matter (below a small fraction of the tolerance)."""
    out = coef.copy()
    out[np.abs(out) < tol_km * 0.01] = 0.0
    return out


def choose(truthfn, et_lo, et_hi, tol_km, lengths, degrees):
    import io
    best = None
    for length in lengths:
        for degree in degrees:
            coef = truncate(fit(truthfn, et_lo, et_hi, length, degree), tol_km)
            c32 = coef.astype(np.float32)
            worst, rms = measure(truthfn, c32, et_lo, length, et_lo, et_hi)
            if worst <= tol_km:
                buf = io.BytesIO()
                np.savez_compressed(buf, c=c32)
                size = buf.tell() / 1e6
                print("   piece %2d d, degree %2d: worst %.4f km  rms %.5f km  stored %.1f MB" % (length, degree, worst, rms, size))
                if best is None or size < best[0]:
                    best = (size, length, degree, c32, worst, rms)
                break
    return best


def store(out, key, best, et_lo):
    size, length, degree, c32, worst, rms = best
    print("   chosen: %d d pieces, degree %d, %.1f MB, worst %.4f km" % (length, degree, size, worst))
    out["b%d_coef" % key] = c32
    out["b%d_start_jd" % key] = np.array(J2000_JD + et_lo / SECONDS_PER_DAY)
    out["b%d_len_days" % key] = np.array(float(length))
    out["b%d_worst_km" % key] = np.array(worst)


def main():
    out = {}
    for body, centre, fname, y0, y1, min_au in JOBS:
        tol_km = TOLERANCE_MAS / 1000.0 / 206264.806247 * min_au * AU_KM
        print("planet centre", body, fname, "tolerance %.0f m" % (tol_km * 1000))
        segs = load_pieces(BASE + fname, body, centre, y0, y1)
        et_lo = max(segs[0].et_start, year_to_et(y0)) + 1.0
        et_hi = min(segs[-1].et_end, year_to_et(y1)) - 1.0
        best = choose(lambda et, s=segs: truth(s, et), et_lo, et_hi, tol_km, (24, 16, 12, 8, 6, 4), range(10, 27, 2))
        if best is None:
            sys.exit("no setting met the tolerance for body %d" % body)
        store(out, body, best, et_lo)
    spk440 = SPK(DE440S)
    for body, centre, fname, y0, y1, min_au in DELTA_JOBS:
        tol_km = TOLERANCE_MAS / 1000.0 / 206264.806247 * min_au * AU_KM
        print("barycentre correction", body, fname, "tolerance %.0f m" % (tol_km * 1000))
        segs = load_pieces(BASE + fname, body, centre, y0, y1)
        et_lo = max(segs[0].et_start, year_to_et(y0)) + 1.0
        et_hi = min(segs[-1].et_end, year_to_et(y1)) - 1.0

        def delta(et, s=segs, b=body):
            new = truth(s, et)
            old, _ = spk440.relative(b, 0, J2000_JD + np.asarray(et) / SECONDS_PER_DAY)
            return new - old
        mid = delta(np.linspace(et_lo, et_hi, 5))
        print("   correction size: %.0f km (min over samples) to %.0f km (max)" % (
            np.linalg.norm(mid, axis=0).min(), np.linalg.norm(mid, axis=0).max()))
        best = choose(delta, et_lo, et_hi, tol_km, (96, 64, 48, 32, 24, 16), range(6, 21, 2))
        if best is None:
            sys.exit("no setting met the tolerance for barycentre %d" % body)
        store(out, 1000 + body, best, et_lo)
    np.savez_compressed("src/alif_ephemeris/data/planet_centres.npz", **out)
    print("written")


if __name__ == "__main__":
    main()
