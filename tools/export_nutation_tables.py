"""One-off: export the IAU 2000A nutation tables (an international standard) into our own data file.

Source of the numbers: IAU 2000A series as published by the IAU/IERS and as shipped in
Skyfield (MIT licence, Brandon Rhodes). Credited in NOTICE. Run once; the .npz is committed.
"""
import numpy as np
import skyfield.nutationlib as n

np.savez_compressed(
    "src/alif_ephemeris/data/iau2000a.npz",
    nals=n.nals_t.astype(np.int8),
    napl=n.napl_t.astype(np.int8),
    ls_lon=n.lunisolar_longitude_coefficients,
    ls_obl=n.lunisolar_obliquity_coefficients,
    pl_lon=n.nutation_coefficients_longitude,
    pl_obl=n.nutation_coefficients_obliquity,
    fa=np.stack([n.fa0[:, 0], n.fa1[:, 0], n.fa2[:, 0], n.fa3[:, 0], n.fa4[:, 0]]),
    anomaly_coefficient=np.asarray(n.anomaly_coefficient, dtype=float).ravel(),
    anomaly_constant=np.asarray(n.anomaly_constant, dtype=float).ravel(),
)
print("written")
