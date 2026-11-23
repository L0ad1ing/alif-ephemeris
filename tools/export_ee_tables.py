"""One-off: export the IERS Conventions 2010 table 5.2e (complementary terms of the equation of the
equinoxes, an international standard) into our own data file. Credited in NOTICE."""
import numpy as np
import skyfield.nutationlib as n

np.savez_compressed(
    "src/alif_ephemeris/data/equinox_terms.npz",
    ke0=np.asarray(n.ke0_t, dtype=np.int8), ke1=np.asarray(n.ke1, dtype=np.int8).ravel(),
    se0_sin=np.asarray(n.se0_t_0), se0_cos=np.asarray(n.se0_t_1),
    se1_sin=np.array(float(n.se1_0)), se1_cos=np.array(float(n.se1_1)))
print("written", n.ke0_t.shape, np.asarray(n.ke1).shape)
