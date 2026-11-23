"""A compact way to ship JPL ephemeris data, and the way back to a normal SPK file.

A JPL DE file stores Chebyshev coefficients as 64-bit floats to the last digit, far more than astrology needs. `compress`
rounds every coefficient to a step (a quantum) and stores the whole numbers, LZMA-compressed: about one sixth of the size
of the original. `expand` turns that file back into an ordinary SPK file the library reads as usual.

Rounding error: each coefficient is off by at most half the quantum. Default quanta are 0.00002 km for the Earth and Moon
(a few tenths of a metre in position, under 0.001 arcsecond at the Moon) and 0.05 km for the Sun and planets (under
0.001 arcsecond at their closest). `check` measures the real error on random dates.
"""
from __future__ import annotations

import io
import json
import lzma
import os
import struct
import zipfile

import numpy as np

from .spk import SPK

REC = 1024
DEFAULT_QUANTA = {301: 0.00002, 399: 0.00002, 3: 0.00002}   # km; everything else 0.05 km
OTHER_QUANTUM = 0.05


def compress(spk_path, out_path, quanta=None, level=6):
    """Write a compact file from an SPK file. Returns the size in bytes."""
    quanta = {**DEFAULT_QUANTA, **(quanta or {})}
    spk = SPK(spk_path)
    manifest = []
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as z:
        i = 0
        for (target, center), segs in sorted(spk.segments.items()):
            for seg in segs:
                q = quanta.get(target, OTHER_QUANTUM)
                whole = np.rint(np.asarray(seg.coeffs) / q).astype("<i8")
                blob = lzma.compress(whole.tobytes(), preset=level)
                name = "seg%03d.xz" % i
                z.writestr(name, blob)
                manifest.append({"name": name, "target": target, "center": center, "frame": seg.frame, "init": seg.init,
                                 "intlen": seg.intlen, "n_records": seg.n_records, "n_coef": seg.n_coef, "quantum": q,
                                 "et_start": seg.et_start, "et_end": seg.et_end})
                i += 1
        z.writestr("manifest.json", json.dumps({"format": "alif-compact-1", "segments": manifest}))
    return os.path.getsize(out_path)


def expand(compact_path, out_path):
    """Write an ordinary SPK file from a compact file. Returns the size in bytes."""
    with zipfile.ZipFile(compact_path) as z:
        info = json.loads(z.read("manifest.json"))
        if info.get("format") != "alif-compact-1":
            raise ValueError("not an Alif compact ephemeris file")
        segs = info["segments"]
        blocks = []
        for m in segs:
            whole = np.frombuffer(lzma.decompress(z.read(m["name"])), dtype="<i8").reshape(m["n_records"], 3, m["n_coef"])
            coefs = whole.astype("<f8") * m["quantum"]
            rsize = 2 + 3 * m["n_coef"]
            block = np.empty((m["n_records"], rsize))
            block[:, 0] = m["init"] + m["intlen"] * (np.arange(m["n_records"]) + 0.5)
            block[:, 1] = m["intlen"] / 2.0
            block[:, 2:] = coefs.reshape(m["n_records"], 3 * m["n_coef"])
            blocks.append((m, rsize, block))
    if len(segs) > 25:
        raise ValueError("too many segments for one summary record")
    addr = 4 * (REC // 8) + 1
    summaries, bodies = [], []
    for m, rsize, block in blocks:
        a0 = addr
        a1 = addr + block.size + 4 - 1
        summaries.append(struct.pack("<2d6i", m["et_start"], m["et_end"], m["target"], m["center"], m["frame"], 2, a0, a1))
        bodies.append(np.concatenate([block.reshape(-1), [m["init"], m["intlen"], float(rsize), float(m["n_records"])]]))
        addr = a1 + 1
    head = bytearray(REC)
    head[0:8] = b"DAF/SPK "
    struct.pack_into("<ii", head, 8, 2, 6)
    note = b"Alif Ephemeris, expanded from a compact file"
    head[16:16 + len(note)] = note
    struct.pack_into("<iii", head, 76, 3, 3, addr)
    head[88:96] = b"LTL-IEEE"
    comment = bytearray(REC)
    text = b"Alif Ephemeris compact file expanded; coefficients rounded to a fixed step"
    comment[:len(text)] = text
    comment[len(text)] = 4
    summ = bytearray(REC)
    struct.pack_into("<3d", summ, 0, 0.0, 0.0, float(len(summaries)))
    for i, sm in enumerate(summaries):
        summ[24 + 40 * i:24 + 40 * (i + 1)] = sm
    tmp = out_path + ".part"
    with open(tmp, "wb") as f:
        f.write(head)
        f.write(comment)
        f.write(summ)
        f.write(bytearray(b" " * REC))
        for arr in bodies:
            f.write(np.asarray(arr, dtype="<f8").tobytes())
    os.replace(tmp, out_path)
    return os.path.getsize(out_path)


def check(original_path, expanded_path, samples=400, seed=1):
    """Worst difference in position (km) and in angle (arcseconds, seen from a distance of 0.3 AU) over random dates."""
    a, b = SPK(original_path), SPK(expanded_path)
    rng = np.random.default_rng(seed)
    worst_km = 0.0
    for (target, center), segs in sorted(a.segments.items()):
        for seg in segs:
            for et in rng.uniform(seg.et_start, seg.et_end, max(2, samples // max(1, len(a.segments)))):
                pa = SPK._cheb(a._segment(target, center, et), et)[0]
                pb = SPK._cheb(b._segment(target, center, et), et)[0]
                worst_km = max(worst_km, float(np.abs(pa - pb).max()))
    return worst_km, worst_km / (0.3 * 149597870.7) * 206264.8
