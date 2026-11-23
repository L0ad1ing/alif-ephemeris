"""List the segments (body, centre, dates) of remote SPK files by reading only their headers.

Uses HTTP range requests, so it downloads a few kilobytes per file. Development tool.
"""
import struct
import sys
import urllib.request

BASE = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/"


def fetch(url, start, length):
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, start + length - 1)})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def j2000_to_year(et):
    return 2000.0 + et / (86400.0 * 365.25)


def inspect(name):
    url = BASE + name
    head = fetch(url, 0, 1024)
    nd, ni = struct.unpack("<ii", head[8:16])
    fward = struct.unpack("<i", head[76:80])[0]
    rec = fward
    out = []
    while rec:
        data = fetch(url, (rec - 1) * 1024, 1024)
        nxt, _prev, nsum = struct.unpack("<ddd", data[:24])
        for i in range(int(nsum)):
            off = 24 + i * 40
            et0, et1 = struct.unpack("<dd", data[off:off + 16])
            target, center, frame, dtype, a0, a1 = struct.unpack("<6i", data[off + 16:off + 40])
            out.append((target, center, dtype, j2000_to_year(et0), j2000_to_year(et1), (a1 - a0 + 1) * 8 / 1e6))
        rec = int(nxt)
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(name)
        for seg in inspect(name):
            if seg[0] % 100 == 99 or seg[0] in (5, 6, 7, 8, 9):    # planet centres and barycentres only
                print("   body %d centre %d type %d  %.1f to %.1f  (%.0f MB)" % seg)
