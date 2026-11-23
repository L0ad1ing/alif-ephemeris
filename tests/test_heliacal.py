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


def test_new_sidereal_modes_have_sane_values():
    import alif_ephemeris.swe_compat as c
    jd = 2451545.0
    for mode, lo, hi in ((30, 22.4, 22.6), (31, 29.9, 30.1), (32, 29.9, 30.2), (33, 23.3, 23.5), (39, 25.1, 25.4)):
        c.set_sid_mode(mode)
        assert lo < c.get_ayanamsa_ut(jd) < hi
