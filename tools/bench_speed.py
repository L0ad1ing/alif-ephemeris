"""Speed of Alif Ephemeris against pyswisseph (used only as a stopwatch reference; it is never imported by src/).

    python tools/bench_speed.py [path to the folder with Swiss files]

Each row is the average time of one call in microseconds, warm (the fast windows already built).
"""
import os
import sys
import time

os.environ.setdefault("ALIF_EPHEMERIS_FILE", "data/raw/de440.bsp")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import alif_ephemeris.swe_compat as alif  # noqa: E402

try:
    import swisseph as swiss
    if len(sys.argv) > 1:
        swiss.set_ephe_path(sys.argv[1])
except ImportError:
    swiss = None

JD = 2461000.5
SITE = (77.2, 28.6, 200.0)
SPEED, SID, TOPO, EQU, TRUE = 256, 65536, 32768, 2048, 16


def bench(fn, n=3000):
    for i in range(n):          # one full pass first, so every fast window is already built (warm)
        fn(i)
    t = time.perf_counter()
    for i in range(n):
        fn(i)
    return (time.perf_counter() - t) / n * 1e6


def rows(lib):
    lib.set_topo(*SITE)
    lib.set_sid_mode(1)
    return [
        ("calc_ut Sun, speed", lambda i: lib.calc_ut(JD + i * 0.0173, 0, 2 | SPEED)),
        ("calc_ut Moon, speed", lambda i: lib.calc_ut(JD + i * 0.0173, 1, 2 | SPEED)),
        ("calc_ut Mars, speed, sidereal", lambda i: lib.calc_ut(JD + i * 0.0173, 4, 2 | SPEED | SID)),
        ("calc_ut Pluto", lambda i: lib.calc_ut(JD + i * 0.0173, 9, 2)),
        ("calc_ut Sun topocentric RA/Dec", lambda i: lib.calc_ut(JD + i * 0.00731, 0, 2 | TOPO | EQU | TRUE)),
        ("houses_ex Placidus", lambda i: lib.houses_ex(JD + i * 0.00731, 28.6, 77.2, b"P", 2)),
        ("houses_ex Placidus sidereal", lambda i: lib.houses_ex(JD + i * 0.00731, 28.6, 77.2, b"P", SID)),
        ("azalt", lambda i: lib.azalt(JD + i * 0.00731, 0, SITE, 1013.25, 15.0, (100.0 + i * 0.01, 1.0, 1.0))),
        ("sidtime", lambda i: lib.sidtime(JD + i * 0.01)),
        ("get_ayanamsa_ut", lambda i: lib.get_ayanamsa_ut(JD + i * 0.01)),
        ("fixstar_ut Spica", lambda i: lib.fixstar_ut("Spica", JD + i * 0.01, 2)),
        ("julday + revjul", lambda i: lib.revjul(lib.julday(2026, 11, i % 28 + 1, 3.5))),
        ("rise_trans Sun", lambda i: lib.rise_trans(JD + i * 0.1, 0, 1, SITE, 1013.25, 15.0), 300),
    ]


a_rows = rows(alif)
s_rows = rows(swiss) if swiss else None
print("%-34s %10s %10s %8s" % ("call", "Alif us", "Swiss us", "ratio"))
for k, row in enumerate(a_rows):
    name, fn = row[0], row[1]
    n = row[2] if len(row) > 2 else 3000
    a = bench(fn, n)
    if s_rows:
        s = bench(s_rows[k][1], n)
        print("%-34s %10.1f %10.1f %7.1fx" % (name, a, s, a / s))
    else:
        print("%-34s %10.1f" % (name, a))
