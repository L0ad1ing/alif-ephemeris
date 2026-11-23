"""Build src/alif_ephemeris/data/stars.npz from the ESA Hipparcos catalogue (via NASA HEASARC) and the
IAU Working Group on Star Names list.

Hipparcos: positions and proper motions at epoch J1991.25 (ICRS), all 9,831 stars brighter than V 6.6.
IAU names: the official proper names (Creative Commons Attribution, credited in NOTICE).

Run:  python tools/build_stars.py     (needs data/raw/hipparcos_bright.txt and data/raw/IAU-CSN.txt)
"""
import numpy as np

# 1. Hipparcos
hip, ra, dec, plx, pmra, pmdec, vmag = [], [], [], [], [], [], []
for line in open("data/raw/hipparcos_bright.txt", encoding="utf-8"):
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != 7 or not parts[0].isdigit():
        continue
    try:
        vals = [float(x) if x else np.nan for x in parts[1:]]
    except ValueError:
        continue
    hip.append(int(parts[0]))
    ra.append(vals[0]); dec.append(vals[1]); plx.append(vals[2]); pmra.append(vals[3]); pmdec.append(vals[4])
    vmag.append(vals[5])
hip = np.array(hip)
print(len(hip), "Hipparcos stars")

# 2. IAU names. Names and designations are left-aligned fixed columns; the numbers are right-aligned,
#    so the HIP number is taken with a pattern.
import re
lines = open("data/raw/IAU-CSN.txt", encoding="utf-8").read().splitlines()
tail = re.compile(r"\s(\d{1,6}|_)\s+(\d{1,6}|_)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d{4}-\d{2}-\d{2}")
names = {}
for l in lines:
    if not l or l[0] in "#$":
        continue
    m = tail.search(l)
    if not m or not m.group(1).isdigit():
        continue
    name_d = l[18:36].strip() or l[0:18].strip()
    designation = l[36:49].strip()
    con = l[l.find(designation) + len(designation):].split()[2] if designation else ""
    names[int(m.group(1))] = (name_d, designation, con)
print(len(names), "IAU names with a Hipparcos number")

name_arr = np.array([names.get(int(h), ("", "", ""))[0] for h in hip])
desig_arr = np.array([names.get(int(h), ("", "", ""))[1] for h in hip])
con_arr = np.array([names.get(int(h), ("", "", ""))[2] for h in hip])
np.savez_compressed("src/alif_ephemeris/data/stars.npz", hip=hip, ra=np.array(ra), dec=np.array(dec),
                    plx=np.array(plx), pmra=np.array(pmra), pmdec=np.array(pmdec), vmag=np.array(vmag),
                    name=name_arr, designation=desig_arr, constellation=con_arr)
print("named stars in file:", int((name_arr != "").sum()))
