"""Build the compact ephemeris files (.alz) the installer ships: python tools/make_compact.py
Reads data/raw/de440.bsp and data/packs/alif_past.bsp, alif_future.bsp (tools/make_range_pack.py makes the packs)
and writes data/compact/*.alz, about a fifth of the size. `alif_ephemeris.compact.expand` turns them back into .bsp files."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from alif_ephemeris import compact  # noqa: E402

os.makedirs("data/compact", exist_ok=True)
for src, name in (("data/raw/de440.bsp", "de440.alz"), ("data/packs/alif_future.bsp", "alif_future.alz"),
                  ("data/packs/alif_past.bsp", "alif_past.alz")):
    if os.path.exists(src):
        size = compact.compress(src, os.path.join("data/compact", name))
        print("%-22s %6.1f MB -> %6.1f MB" % (src, os.path.getsize(src) / 1e6, size / 1e6))
