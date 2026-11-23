"""Build src/alif_ephemeris/data/smallbodies.npz: asteroid and centaur ephemerides from NASA JPL Horizons.

For each body we download barycentric ICRF positions and velocities every 2 days (1800 to 2200) and fit compact
Chebyshev pieces of 32 days with polynomial degree 15, using both positions and velocities. The error of the fit
is measured at times that were NOT in the fit (the odd days) and stored with the table.

Needs network. Raw downloads are cached in data/raw/smallbodies/. Development tool; the .npz is committed.

Run:  PYTHONPATH=src python tools/build_smallbodies.py
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from numpy.polynomial import chebyshev as C

BODIES = {1: "Ceres", 2: "Pallas", 3: "Juno", 4: "Vesta", 2060: "Chiron", 5145: "Pholus"}
START_YEAR, END_YEAR = 1800, 2200
PIECE_DAYS = 32.0
DEGREE = 15
CACHE = "data/raw/smallbodies"
AU_KM = 149597870.7


def horizons(command, start, stop, step):
    p = {"format": "json", "COMMAND": "'%s;'" % command, "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'",
         "EPHEM_TYPE": "'VECTORS'", "CENTER": "'500@0'", "START_TIME": "'%s'" % start, "STOP_TIME": "'%s'" % stop,
         "STEP_SIZE": "'%s'" % step, "OUT_UNITS": "'AU-D'", "REF_PLANE": "'FRAME'", "REF_SYSTEM": "'ICRF'",
         "VEC_TABLE": "'2'", "CSV_FORMAT": "'YES'", "VEC_LABELS": "'NO'", "TIME_TYPE": "'TDB'"}
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(p)
    for attempt in range(5):
        try:
            text = json.loads(urllib.request.urlopen(url, timeout=240).read())["result"]
            if "$$SOE" in text:
                return text
            raise RuntimeError(text[:300])
        except Exception as exc:
            err = exc
            time.sleep(2 + 3 * attempt)
    raise err


def parse(text):
    rows = []
    for line in text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
        parts = [x.strip() for x in line.split(",") if x.strip()]
        rows.append([float(parts[0])] + [float(x) for x in parts[2:8]])
    return np.array(rows)


def fetch(number, offset_day):
    """All rows 1800..2200 at a 2-day step starting `offset_day` days after 1800-01-01 (cached)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "%d_%d.npy" % (number, offset_day))
    if os.path.exists(path):
        return np.load(path)
    out = []
    for y in range(START_YEAR, END_YEAR, 20):
        start = "%d-01-01 %s" % (y, "00:00" if offset_day == 0 else "00:00")
        # an offset of one day is done by starting one day later on the same grid
        if offset_day:
            start = "%d-01-02" % y
        text = horizons(number, start, "%d-01-01" % (y + 20), "2 d")
        out.append(parse(text))
        print("   %s %d..%d: %d rows" % (BODIES[number], y, y + 20, len(out[-1])), flush=True)
    data = np.vstack(out)
    np.save(path, data)
    return data


def fit_pieces(data):
    t = data[:, 0]
    t0 = t[0]
    n_pieces = int((t[-1] - t0) // PIECE_DAYS)
    coef = np.zeros((n_pieces, 3, DEGREE + 1))
    half = PIECE_DAYS / 2.0
    worst_fit = 0.0
    for k in range(n_pieces):
        a, b = t0 + k * PIECE_DAYS, t0 + (k + 1) * PIECE_DAYS
        m = (t >= a - 1e-6) & (t <= b + 1e-6)
        x = (t[m] - (a + half)) / half
        V = C.chebvander(x, DEGREE)
        dV = np.stack([C.chebval(x, C.chebder(np.eye(DEGREE + 1)[j])) for j in range(DEGREE + 1)], axis=1)
        A = np.vstack([V, dV])
        for ax in range(3):
            rhs = np.concatenate([data[m, 1 + ax], half * data[m, 4 + ax]])
            c, *_ = np.linalg.lstsq(A, rhs, rcond=None)
            coef[k, ax] = c
            worst_fit = max(worst_fit, float(np.abs(A @ c - rhs).max()))
    return t0, coef, worst_fit


def evaluate(t0, coef, t):
    k = np.clip(((t - t0) // PIECE_DAYS).astype(int), 0, len(coef) - 1)
    x = (t - (t0 + (k + 0.5) * PIECE_DAYS)) / (PIECE_DAYS / 2.0)
    out = np.empty((len(t), 3))
    for ax in range(3):
        for i in range(len(t)):
            out[i, ax] = C.chebval(x[i], coef[k[i], ax])
    return out


def main():
    arrays = {}
    for number, name in BODIES.items():
        print("body", number, name, flush=True)
        data = fetch(number, 0)
        t0, coef, worst_fit = fit_pieces(data)
        check = fetch(number, 1)                         # the odd days, never used in the fit
        sel = np.arange(0, len(check), 7)               # a sample of them is enough
        pred = evaluate(t0, coef, check[sel, 0])
        err_au = np.linalg.norm(pred - check[sel, 1:4], axis=1)
        dist = np.linalg.norm(check[sel, 1:4], axis=1)
        print("   fit residual %.2e AU; error at unused times: worst %.3f km (%.6f arcsec at 1 AU)" % (
            worst_fit, err_au.max() * AU_KM, err_au.max() * 206264.806))
        arrays["n%d_coef" % number] = coef
        arrays["n%d_t0" % number] = np.array(t0)
        arrays["n%d_len" % number] = np.array(PIECE_DAYS)
        arrays["n%d_err_km" % number] = np.array(err_au.max() * AU_KM)
    np.savez_compressed("src/alif_ephemeris/data/smallbodies.npz", **arrays)
    print("written")


if __name__ == "__main__":
    main()
