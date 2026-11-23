"""Print the Python registry lines for the fixed-point ayanamsas (value at J2000, mean).

The J2000 values are the numerical definition of each convention, read from the reference implementation
as numbers (facts), then checked against the published definitions in docs/AYANAMSAS.md.
"""
import swisseph as swe
J2000 = 2451545.0
skip = {17, 27, 28, 29, 30, 31, 32, 33, 35, 36, 39, 40, 255}      # star and galaxy based: built from the star catalogue
label = {0: "Fagan-Bradley", 1: "Lahiri", 2: "DeLuce", 3: "Raman", 4: "Usha-Shashi", 5: "Krishnamurti (KP)",
         6: "Djwhal Khul", 7: "Yukteshwar", 8: "JN Bhasin", 9: "Babylonian (Kugler 1)", 10: "Babylonian (Kugler 2)",
         11: "Babylonian (Kugler 3)", 12: "Babylonian (Huber)", 13: "Babylonian (Eta Piscium)",
         14: "Aldebaran 15 Taurus (fixed)", 15: "Hipparchos", 16: "Sassanian", 18: "J2000", 19: "J1900", 20: "B1950",
         21: "Suryasiddhanta", 22: "Suryasiddhanta (mean Sun)", 23: "Aryabhata", 24: "Aryabhata (mean Sun)",
         25: "Suryasiddhanta (Revati)", 26: "Suryasiddhanta (Citra)", 34: "Galactic (Mardyks)", 37: "Aryabhata 522",
         38: "Babylonian (Britton)", 41: "Galactic equator (Fiorenza)", 42: "Valens (Moon)", 43: "Lahiri 1940",
         44: "Lahiri (VP285)", 45: "Krishnamurti (VP291)", 46: "Lahiri (ICRC)"}
for mid in sorted(label):
    swe.set_sid_mode(mid, 0, 0)
    a0 = swe.get_ayanamsa_ex(J2000, swe.FLG_SWIEPH | swe.FLG_NONUT)[1]
    print('    ("%s", %.9f),' % (label[mid], a0))
