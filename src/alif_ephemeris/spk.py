"""Reader for NASA/JPL SPK ephemeris files (DAF container, segment type 2).

Written from the public NAIF documentation ("DAF Required Reading" and
"SPK Required Reading"). Type 2 segments store Chebyshev polynomials for position;
velocity is the exact derivative of the same polynomial.

No global state. A SPK object reads the file once into memory (numpy memory map),
so any number of threads can call it at the same time.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

SECONDS_PER_DAY = 86400.0
J2000_JD = 2451545.0

_RECORD = 1024  # bytes in one DAF record


@dataclass(frozen=True)
class Segment:
    target: int
    center: int
    frame: int
    data_type: int
    et_start: float  # seconds past J2000 (TDB)
    et_end: float
    init: float      # start of first record, seconds past J2000
    intlen: float    # length of one record in seconds
    n_coef: int      # Chebyshev coefficients per axis
    n_records: int
    coeffs: np.ndarray  # shape (n_records, 3, n_coef), km
    mid: np.ndarray     # shape (n_records,), seconds
    radius: np.ndarray  # shape (n_records,), seconds


class SPK:
    """An open SPK file. Use `position(target, jd_tdb)` to get km and km/s."""

    def __init__(self, path: str):
        self.path = path
        raw = np.memmap(path, dtype="uint8", mode="r")
        self._raw = raw
        head = bytes(raw[:_RECORD])
        if not head.startswith(b"DAF/SPK"):
            raise ValueError("not an SPK file: " + path)
        endian = head[88:96].strip(b"\x00 ").decode("ascii", "ignore")
        if endian not in ("LTL-IEEE", ""):
            raise ValueError("only little-endian SPK files are supported, got " + endian)
        nd, ni = struct.unpack("<ii", head[8:16])
        if (nd, ni) != (2, 6):
            raise ValueError("unexpected DAF summary format")
        fward = struct.unpack("<i", head[76:80])[0]
        self._doubles = raw.view("<f8") if raw.size % 8 == 0 else None
        if self._doubles is None:
            raise ValueError("file size is not a multiple of 8 bytes")
        self.segments: dict[tuple[int, int], list[Segment]] = {}
        self._read_summaries(fward)

    # -- DAF structure -------------------------------------------------------------
    def _read_summaries(self, fward: int) -> None:
        rec = fward
        while rec:
            base = (rec - 1) * _RECORD // 8
            nxt, _prev, nsum = self._doubles[base:base + 3]
            nxt, nsum = int(nxt), int(nsum)
            for i in range(nsum):
                off = base + 3 + i * 5  # summary size = nd + (ni + 1) // 2 = 5 doubles
                et0, et1 = self._doubles[off], self._doubles[off + 1]
                ints = struct.unpack("<6i", self._doubles[off + 2:off + 5].tobytes())
                target, center, frame, dtype, a0, a1 = ints
                if dtype != 2:
                    continue  # planetary DE files only use type 2 for these bodies
                seg = self._load_type2(target, center, frame, dtype, float(et0), float(et1), a0, a1)
                self.segments.setdefault((target, center), []).append(seg)
            rec = nxt

    def _load_type2(self, target, center, frame, dtype, et0, et1, a0, a1) -> Segment:
        d = self._doubles
        init, intlen, rsize, n = d[a1 - 4:a1]  # last four doubles of the segment
        rsize, n = int(rsize), int(n)
        n_coef = (rsize - 2) // 3
        # views into the memory-mapped file: nothing is read until a date in that record is asked for, so the
        # multi-gigabyte DE441 files cost almost no memory
        block = np.asarray(d[a0 - 1:a0 - 1 + rsize * n]).reshape(n, rsize)
        mid = block[:, 0].copy()
        radius = block[:, 1].copy()
        coeffs = block[:, 2:].reshape(n, 3, n_coef)
        return Segment(target, center, frame, dtype, et0, et1, float(init), float(intlen),
                       n_coef, n, coeffs, mid, radius)

    # -- evaluation -----------------------------------------------------------------
    def _segment(self, target: int, center: int, et: float) -> Segment:
        for seg in self.segments.get((target, center), ()):
            if seg.et_start <= et <= seg.et_end:
                return seg
        raise ValueError(f"no data for body {target} (center {center}) at ET={et}")

    @staticmethod
    def _cheb(seg: Segment, et):
        """Position (km) and velocity (km/s) from a Chebyshev segment. `et` may be an array."""
        et = np.asarray(et, dtype=float)
        if et.ndim == 0:                                   # one time: skip the array machinery
            return SPK._cheb_scalar(seg, float(et))
        idx = np.clip(((et - seg.init) // seg.intlen).astype(int), 0, seg.n_records - 1)
        tau = (et - seg.mid[idx]) / seg.radius[idx]
        n = seg.n_coef
        T = np.empty((n,) + tau.shape)
        dT = np.empty_like(T)
        T[0] = 1.0
        dT[0] = 0.0
        if n > 1:
            T[1] = tau
            dT[1] = 1.0
        for k in range(2, n):
            T[k] = 2 * tau * T[k - 1] - T[k - 2]
            dT[k] = 2 * T[k - 1] + 2 * tau * dT[k - 1] - dT[k - 2]
        c = seg.coeffs[idx]  # (..., 3, n)
        pos = np.einsum("...ak,k...->a...", c, T)
        vel = np.einsum("...ak,k...->a...", c, dT) / seg.radius[idx]
        return pos, vel

    @staticmethod
    def _cheb_scalar(seg: Segment, et: float):
        i = int((et - seg.init) // seg.intlen)
        i = 0 if i < 0 else seg.n_records - 1 if i >= seg.n_records else i
        x = (et - seg.mid[i]) / seg.radius[i]
        n = seg.n_coef
        T = np.empty(n)
        dT = np.empty(n)
        T[0], dT[0] = 1.0, 0.0
        if n > 1:
            T[1], dT[1] = x, 1.0
        t0, t1, d0, d1 = 1.0, x, 0.0, 1.0
        for k in range(2, n):
            t2 = 2.0 * x * t1 - t0
            d2 = 2.0 * t1 + 2.0 * x * d1 - d0
            T[k], dT[k] = t2, d2
            t0, t1, d0, d1 = t1, t2, d1, d2
        c = seg.coeffs[i]                                   # (3, n)
        return c @ T, (c @ dT) / seg.radius[i]

    def relative(self, target: int, center: int, jd_tdb, frac=0.0):
        """Position and velocity of `target` relative to `center`, in km and km/s.

        Time is `jd_tdb + frac` days. Passing a whole-day number (a half-integer) in `jd_tdb` and the
        rest in `frac` keeps full precision: one float Julian Day can only resolve about 40 microseconds.
        Arrays may span several data segments; each is evaluated where it applies.
        """
        et = (np.asarray(jd_tdb, dtype=float) - J2000_JD) * SECONDS_PER_DAY             + np.asarray(frac, dtype=float) * SECONDS_PER_DAY
        et = np.broadcast_to(et, np.broadcast(et, np.asarray(frac)).shape)
        segs = self.segments.get((target, center), ())
        if not segs:
            raise ValueError(f"this file has no data for body {target} relative to {center}")
        if len(segs) == 1:
            seg = segs[0]
            if et.size and (float(et.min()) < seg.et_start or float(et.max()) > seg.et_end):
                raise ValueError(_range_message(target, center, et, segs))
            return self._cheb(seg, et)
        pos = np.full((3,) + et.shape, np.nan)
        vel = np.full((3,) + et.shape, np.nan)
        for seg in segs:
            m = (et >= seg.et_start) & (et <= seg.et_end) & np.isnan(pos[0])
            if m.any():
                p, v = self._cheb(seg, et[m])
                pos[:, m] = p
                vel[:, m] = v
        if np.isnan(pos[0]).any():
            raise ValueError(_range_message(target, center, et, segs))
        return pos, vel

    def coverage(self, target: int = 399, center: int = 3):
        """First and last Julian Day (TDB) this file covers for a body, as (start, end)."""
        segs = self.segments[(target, center)]
        lo = min(s.et_start for s in segs) / SECONDS_PER_DAY + J2000_JD
        hi = max(s.et_end for s in segs) / SECONDS_PER_DAY + J2000_JD
        return lo, hi

    def barycentric(self, body: int, jd_tdb, frac=0.0):
        """Position and velocity of a body relative to the solar-system barycentre (ICRF axes)."""
        chain = _CHAINS[body]
        pos = 0.0
        vel = 0.0
        for target, center in chain:
            p, v = self.relative(target, center, jd_tdb, frac)
            pos = pos + p
            vel = vel + v
        return pos, vel


def _range_message(target, center, et, segs):
    lo = min(sg.et_start for sg in segs) / SECONDS_PER_DAY + J2000_JD
    hi = max(sg.et_end for sg in segs) / SECONDS_PER_DAY + J2000_JD
    want_lo = float(np.min(et)) / SECONDS_PER_DAY + J2000_JD
    want_hi = float(np.max(et)) / SECONDS_PER_DAY + J2000_JD
    return (f"date outside the ephemeris file: it covers JD {lo:.1f} to {hi:.1f} for body {target} "
            f"(you asked for JD {want_lo:.1f} to {want_hi:.1f})")


# How each body is reached from the barycentre (NAIF ids). Earth and Moon go through
# the Earth-Moon barycentre (3).
_CHAINS = {
    0: [],
    1: [(1, 0)], 2: [(2, 0)], 3: [(3, 0)], 4: [(4, 0)], 5: [(5, 0)],
    6: [(6, 0)], 7: [(7, 0)], 8: [(8, 0)], 9: [(9, 0)],
    10: [(10, 0)],
    199: [(1, 0), (199, 1)], 299: [(2, 0), (299, 2)],
    301: [(3, 0), (301, 3)], 399: [(3, 0), (399, 3)],
}


class SPKSet:
    """Several SPK files used together. Each time is served by the first file that covers it, so list the file you
    trust most for modern dates first (DE440) and the long-range one after it (DE441, in its two parts)."""

    def __init__(self, paths):
        self.files = [SPK(p) for p in paths]
        if not self.files:
            raise ValueError("no ephemeris files given")
        self.path = paths[0]

    @property
    def segments(self):
        merged = {}
        for f in self.files:
            for key, segs in f.segments.items():
                merged.setdefault(key, []).extend(segs)
        return merged

    def _covering(self, target, center, t):
        for f in self.files:
            segs = f.segments.get((target, center), ())
            if any(s.et_start <= t <= s.et_end for s in segs):
                return f
        return None

    def relative(self, target: int, center: int, jd_tdb, frac=0.0):
        t = (np.asarray(jd_tdb, dtype=float) - J2000_JD) * SECONDS_PER_DAY + np.asarray(frac, dtype=float) * SECONDS_PER_DAY
        if t.ndim == 0:
            f = self._covering(target, center, float(t))
            if f is None:
                f = self.files[0]                                # raises the usual "date outside" error
            return f.relative(target, center, jd_tdb, frac)
        pos = np.full((3,) + t.shape, np.nan)
        vel = np.full((3,) + t.shape, np.nan)
        for f in self.files:
            segs = f.segments.get((target, center), ())
            if not segs:
                continue
            lo, hi = min(s.et_start for s in segs), max(s.et_end for s in segs)
            m = (t >= lo) & (t <= hi) & np.isnan(pos[0])
            if m.any():
                jd_m = np.broadcast_to(np.asarray(jd_tdb, dtype=float), t.shape)[m]
                fr_m = np.broadcast_to(np.asarray(frac, dtype=float), t.shape)[m]
                p, v = f.relative(target, center, jd_m, fr_m)
                pos[:, m], vel[:, m] = p, v
        if np.isnan(pos[0]).any():
            return self.files[0].relative(target, center, jd_tdb, frac)       # raises a clear error
        return pos, vel

    def barycentric(self, body: int, jd_tdb, frac=0.0):
        pos = 0.0
        vel = 0.0
        for target, center in _CHAINS[body]:
            p, v = self.relative(target, center, jd_tdb, frac)
            pos = pos + p
            vel = vel + v
        return pos, vel

    def coverage(self, target: int = 399, center: int = 3):
        los, his = zip(*(f.coverage(target, center) for f in self.files))
        return min(los), max(his)
