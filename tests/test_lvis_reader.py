import numpy as np
import pytest

from altimetryData import cmr, timeref
from altimetryData.collections import COLLECTIONS
from altimetryData.readers import lvis

IS_HEADER = """# Level2 (v2.0.4) data set collected by NASA's Land, Vegetation and Ice Sensor (LVIS) Facility
#
# LFID SHOTNUMBER TIME LON_LOW LAT_LOW Z_LOW LON_MAXAMP LAT_MAXAMP Z_MAXAMP LON_HIGH LAT_HIGH Z_HIGH LON_LOW_ALT LAT_LOW_ALT Z_LOW_ALT AZIMUTH INCIDENTANGLE RANGE COMPLEXITY SENSITIVITY ENERGY1 ENERGY2 ENERGY3 CHANNEL
"""
IS_ROWS = [
    "1959688583  69756578  66833.44041  291.3666514   76.4195593  483.345  291.3666514   76.4195593  483.345  "
    "291.3666514   76.4195593  484.000 -999.0000000 -999.0000000 -999.000  77.81  3.339  1423.24 0.152 0.975    "
    "8576.0    1568.5    -999.0 1",
    "1959688583  69756579  86399.90000  291.3665968   76.4195572  483.216  291.3665968   76.4195572  483.216  "
    "291.3665968   76.4195572  483.216 -999.0000000 -999.0000000 -999.000  77.77  3.282  1423.70 0.205 0.975    "
    "9006.3    1596.0    -999.0 2",
    "1959688583  69756580      0.10000  291.3665422   76.4195550  483.078  291.3665422   76.4195550  483.078  "
    "291.3665422   76.4195550  483.078 -999.0000000 -999.0000000 -999.000  77.72  3.225  1424.15 0.205 0.974    "
    "8783.4    1459.4    -999.0 1",
]


def test_lvisf2_ice_surface_file(tmp_path):
    path = tmp_path / "LVISF2_IS_GL2022_0419_R2408_066833.TXT"
    path.write_text(IS_HEADER + "\n".join(IS_ROWS) + "\n")
    d = lvis.read(str(path))
    assert d["elevation"][0] == pytest.approx(483.345)              # h is Z_LOW
    assert d["z_high"][0] == pytest.approx(484.0)
    assert d["longitude"][0] == pytest.approx(291.3666514 - 360.0)
    assert d["latitude"][0] == pytest.approx(76.4195593)
    assert list(d["channel_zg"]) == [1, 2, 1]
    assert d["range"][1] == pytest.approx(1423.70)
    assert "z_top" not in d and "rh50" not in d                       # not in LVISF2
    utc = (np.datetime64("2022-04-19", "ns") - timeref.GPS_EPOCH) / np.timedelta64(1, "s") + 66833.44041
    assert d["gps_seconds"][0] == pytest.approx(timeref.utc_to_gps(np.array([utc]))[0])
    assert d["gps_seconds"][2] - d["gps_seconds"][1] == pytest.approx(0.2)   # past midnight: the next day


@pytest.mark.parametrize("name, date", [
    ("ILVIS2_GL2017_0825_R1801_049549.TXT", "2017-08-25"),
    ("ILVGH2_GL2013_1030_R1406_052640.TXT", "2013-10-30"),
    ("LVISF2_IS_GL2022_0419_R2408_066833.TXT", "2022-04-19"),
    ("LVISF2_IS_ARCSIX2024_0528_R2503_045800.TXT", "2024-05-28"),
])
def test_dates_from_names(name, date):
    assert lvis._date_from_name(name) == np.datetime64(date, "ns")


def _item(name):
    return {"meta": {"concept-id": "G1-NSIDC_CPRD"},
            "umm": {"GranuleUR": name, "TemporalExtent": {"RangeDateTime": {
                "BeginningDateTime": "2022-04-19T18:00:00Z", "EndingDateTime": "2022-04-19T18:01:00Z"}},
                    "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {"BoundingRectangles": [
                        {"NorthBoundingCoordinate": 77, "SouthBoundingCoordinate": 76,
                         "WestBoundingCoordinate": -69, "EastBoundingCoordinate": -68}]}}},
                    "RelatedUrls": [{"Type": "GET DATA", "URL": f"https://example.org/{name}"}]}}


def test_lvisf2_inventory_keeps_only_ice_surface_files():
    lvisf2 = next(c for c in COLLECTIONS if c.short_name == "LVISF2")
    names = ["LVISF2_IS_GL2022_0419_R2408_049849.TXT", "LVISF2_GL2022_0727_R2212_057999.TXT",
             "LVISF2_ABoVE2019_0712_R2003_076802.TXT", "LVISF2_IS_ARCSIX2024_0528_R2503_045800.TXT"]
    df = cmr.granule_rows(lvisf2, [_item(n) for n in names])
    assert df["granule"].tolist() == [names[0], names[3]]
    ilvis2 = next(c for c in COLLECTIONS if c.short_name == "ILVIS2")
    assert ilvis2.accepts("ILVIS2_GL2017_0825_R1801_049549.TXT")
