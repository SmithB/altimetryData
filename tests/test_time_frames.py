import numpy as np

from altimetryData import timeref, itrf, frames


def test_delta_time_epoch_and_leap_seconds():
    t = np.array(["2018-01-01T00:00:00", "2017-01-01T00:00:00", "2016-12-31T23:59:59"], dtype="datetime64[ns]")
    dt = timeref.delta_time_from_utc_datetime(t)
    assert dt[0] == 0
    assert dt[1] == -365 * 86400
    assert dt[2] == -365 * 86400 - 2          # the 2016-12-31T23:59:60 leap second lies in between
    gps = timeref.utc_to_gps(timeref.utc_seconds_since_gps_epoch(np.array(["1998-05-20"], dtype="datetime64[ns]")))
    assert timeref.gps_minus_utc_at_gps(gps)[0] == 12
    assert timeref.gps_to_utc_datetime(gps)[0] == np.datetime64("1998-05-20", "ns")
    assert timeref.year_from_delta_time(-365.25 * 86400 * 9) == 2009.0


def test_itrf():
    lon, lat, h = np.array([-45.0, 120.0]), np.array([70.0, -75.0]), np.array([2000.0, 3000.0])
    same = itrf.to_itrf2020(lon, lat, h, 2010.0, "ITRF2020")
    assert np.array_equal(same[2], h)
    # TOPEX/Poseidon -> WGS84 ellipsoid is about -0.71 m (GLAS d_deltaEllip is 0.710 to 0.714 m)
    _, _, h2 = itrf.to_itrf2020(lon, lat, h, 2005.0, "ITRF2020", "TOPEX")
    assert np.all((h - h2 > 0.705) & (h - h2 < 0.716))
    # ITRF2014 -> ITRF2020 differences are a few mm
    _, _, h3 = itrf.to_itrf2020(lon, lat, h, 2015.0, "ITRF2014")
    assert np.all(np.abs(h3 - h) < 0.01)


def test_frames():
    assert frames.normalize_frame("161020_aa_l12_cfm_itrf08_28Mar17_combo") == "ITRF2008"
    assert frames.normalize_frame("ITRF2014") == "ITRF2014"
    assert frames.igs_frame("2012-05-02") == "ITRF2008"
    assert frames.igs_frame("2009-10-21") == "ITRF2005"
    assert frames.igs_frame("1998-05-20") == "ITRF96"
    data = {"header": "x\n./110412_aa_l12_cfm_itrf05_08jun11_6138\n"}
    assert frames.atm_frame(data, np.datetime64("2011-04-12")) == ("ITRF2005", "header")
    data = {"ancillary": {"reference_frame": np.array([b"ITRF2008"])}}
    assert frames.atm_frame(data, np.datetime64("2016-10-20")) == ("ITRF2008", "granule")


def test_lvis_frames_from_user_guides():
    from altimetryData import granule as gr
    from altimetryData.collections import COLLECTIONS
    lvis = [(c.short_name, c.version) for c in COLLECTIONS if c.sensor == "LVIS"]
    assert set(lvis) == set(gr.LVIS_FRAMES)          # every LVIS collection has a frame
    assert gr.LVIS_FRAMES[("ILVIS2", "1")][0] == "ITRF2000" and gr.LVIS_FRAMES[("ILVIS2", "2")][0] == "ITRF2008"
