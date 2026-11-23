"""Split Alif Astrology Software's test files into three batches (written to data/raw/batchN.txt)."""
import glob
import os

alif = os.environ.get("ALIF_SOFTWARE_DIR", "")
files = sorted(glob.glob(alif + "/tests/test_*.py"))
rel = ["tests/" + os.path.basename(f) for f in files]
third = (len(rel) + 2) // 3
for i in range(3):
    with open("data/raw/batch%d.txt" % (i + 1), "w") as fh:
        fh.write(" ".join(rel[i * third:(i + 1) * third]))
print(len(rel), "files;", third, "per batch")
