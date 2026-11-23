"""Cut a date range out of a JPL DE441 SPK file into a small SPK file of its own.

The records are copied exactly as they are (no refitting), so the positions in the new file are identical to DE441's.
Written from the public NAIF "DAF Required Reading" and "SPK Required Reading".

    python tools/make_range_pack.py data/raw/de441_part-1.bsp out.bsp JD_START JD_END
"""
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from alif_ephemeris.spk import SPK, J2000_JD, SECONDS_PER_DAY  # noqa: E402

REC = 1024


def build(src_path, out_path, jd0, jd1, note="Alif Ephemeris range pack, records copied from JPL DE441"):
    spk = SPK(src_path)
    et0 = (jd0 - J2000_JD) * SECONDS_PER_DAY
    et1 = (jd1 - J2000_JD) * SECONDS_PER_DAY
    segs = []
    for (target, center), lst in sorted(spk.segments.items()):
        for seg in lst:
            if seg.et_end < et0 or seg.et_start > et1:
                continue
            i0 = max(0, int(math.floor((max(et0, seg.et_start) - seg.init) / seg.intlen)))
            i1 = min(seg.n_records, int(math.ceil((min(et1, seg.et_end) - seg.init) / seg.intlen)))
            if i1 <= i0:
                continue
            rsize = 2 + 3 * seg.n_coef
            block = np.empty((i1 - i0, rsize))
            block[:, 0] = seg.mid[i0:i1]
            block[:, 1] = seg.radius[i0:i1]
            block[:, 2:] = np.asarray(seg.coeffs[i0:i1]).reshape(i1 - i0, 3 * seg.n_coef)
            init = seg.init + i0 * seg.intlen
            segs.append((target, center, seg.frame, init, init + (i1 - i0) * seg.intlen, seg.intlen, rsize, block))
    if not segs:
        raise SystemExit("no data in that range")
    if len(segs) > 25:
        raise SystemExit("too many segments for one summary record (%d)" % len(segs))
    addr = 4 * (REC // 8) + 1                               # data starts in record 5
    summaries, bodies = [], []
    for target, center, frame, s0, s1, intlen, rsize, block in segs:
        n = block.shape[0]
        a0 = addr
        a1 = addr + block.size + 4 - 1
        summaries.append(struct.pack("<2d6i", s0, s1, target, center, frame, 2, a0, a1))
        bodies.append(np.concatenate([block.reshape(-1), [s0, intlen, float(rsize), float(n)]]))
        addr = a1 + 1
    head = bytearray(REC)
    head[0:8] = b"DAF/SPK "
    struct.pack_into("<ii", head, 8, 2, 6)
    head[16:76] = note.encode("ascii")[:60].ljust(60)
    struct.pack_into("<iii", head, 76, 3, 3, addr)
    head[88:96] = b"LTL-IEEE"
    comment = bytearray(REC)
    text = (note + "; free for any use").encode("ascii")
    comment[:len(text)] = text
    comment[len(text)] = 4
    summ = bytearray(REC)
    struct.pack_into("<3d", summ, 0, 0.0, 0.0, float(len(segs)))
    for i, sm in enumerate(summaries):
        summ[24 + 40 * i:24 + 40 * (i + 1)] = sm
    names = bytearray(b" " * REC)
    with open(out_path, "wb") as f:
        f.write(head)
        f.write(comment)
        f.write(summ)
        f.write(names)
        for arr in bodies:
            f.write(np.asarray(arr, dtype="<f8").tobytes())
    return len(segs), os.path.getsize(out_path)


if __name__ == "__main__":
    n, size = build(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    print("%d segments, %.1f MB" % (n, size / 1e6))
