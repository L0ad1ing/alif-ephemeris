import os
import pytest

DE440S = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "de440s.bsp")


@pytest.fixture(scope="session")
def spk():
    from alif_ephemeris.spk import SPK
    if not os.path.exists(DE440S):
        pytest.skip("data/raw/de440s.bsp not downloaded (see docs/DATA.md)")
    return SPK(DE440S)
