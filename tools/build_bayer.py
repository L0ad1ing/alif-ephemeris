"""Build src/alif_ephemeris/data/star_bayer.txt: Bayer codes (as astrology software writes them, e.g. alLeo, al-2Lib,
beGem) for the stars in the IAU Catalog of Star Names, with their Hipparcos numbers.

Input: data/raw/IAU-CSN.txt (see docs/DATA.md). Columns used: Bayer id (4th text column) and constellation.
"""
import re

GREEK = {"alf": "al", "bet": "be", "gam": "ga", "del": "de", "eps": "ep", "zet": "ze", "eta": "et", "tet": "th",
         "the": "th", "iot": "io", "kap": "ka", "lam": "la", "mu": "mu", "nu": "nu", "xi": "xi", "omi": "om",
         "pi": "pi", "rho": "rh", "sig": "si", "tau": "ta", "ups": "up", "phi": "ph", "chi": "ch", "psi": "ps",
         "ome": "om"}
out = {}
for line in open("data/raw/IAU-CSN.txt", encoding="utf-8"):
    if line.startswith("#") or line.startswith("$") or not line.strip():
        continue
    parts = line.split()
    # name(s), designation, bayer ascii, bayer greek, constellation ... the number of name words varies, so anchor on the HIP column
    m = re.search(r"\s(\d{3,6})\s+(\d+|_)\s+[\d.]+\s+[-\d.]+\s+\d{4}-\d\d-\d\d", line)
    if not m:
        continue
    hip = int(m.group(1))
    head = line[:m.start()].split()
    # find the Bayer token: letters like alf, bet, alf02, tet01
    token = None
    for i, t in enumerate(head):
        mm = re.fullmatch(r"([a-z]{2,3})(\d{2})?", t)
        if mm and mm.group(1) in GREEK and i + 2 < len(head) and re.fullmatch(r"[A-Z][A-Za-z]{2}", head[i + 2] if head[i + 1] != "_" else head[i + 1]):
            token = mm
            con = head[i + 2] if re.fullmatch(r"[A-Z][A-Za-z]{2}", head[i + 2]) else head[i + 1]
            break
    if token is None:
        continue
    num = token.group(2)
    code = GREEK[token.group(1)] + ("-%d" % int(num) if num and int(num) > 0 else "") + con
    out.setdefault(code, hip)
with open("src/alif_ephemeris/data/star_bayer.txt", "w", encoding="utf-8") as f:
    for code, hip in sorted(out.items()):
        f.write("%s %d\n" % (code, hip))
print(len(out), "Bayer codes; examples:", [(c, out[c]) for c in ("alLeo", "alVir", "al-2Lib", "beGem", "deCnc") if c in out])
