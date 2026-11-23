import alif_ephemeris.swe_compat as c


def test_heliacal_sirius_rising_plausible():
    geo = (-77.0, 39.0, 0.0)
    out = c.heliacal_ut(2461000.5, geo, (1013.25, 15.0, 40.0, 0.0), (36.0, 1.0, 0, 0, 0, 0), "Sirius", 1, 0)
    assert 2461000.5 < out[0] <= out[1] <= out[2] < 2461000.5 + 400
    # Sirius rises heliacally in late summer at this latitude (northern hemisphere): JD within Aug-Sep window
    assert 2461240 < out[0] < 2461290


def test_csroundsec_never_reaches_thirty():
    assert c.csroundsec(10799950) == 10799900 and c.csroundsec(10) == 0 and c.csroundsec(10800030) == 10800000
    assert c.csroundsec(12345) == 12300
