import numpy as np
import pytest

from altimetryData.readers import atm
from altimetryData.timeref import GPS_EPOCH


def _write_qi(path, hhmmss):
    """a minimal big-endian 12-word Qfit file: the record-size record, one header record, one shot"""
    rec = np.zeros((3, 12), dtype=">i4")
    rec[0, 0] = 48
    rec[1, 0] = -1
    rec[2, :4] = [1000, 70_000_000, 310_000_000, 1_500_000]       # rel_time, lat, lon (1e6 deg), elevation (mm)
    rec[2, 11] = hhmmss * 1000                                     # time_hhmmss, scale 1e3
    rec.tofile(path)


@pytest.mark.parametrize("name", ["BLATM1B_20050904_012629.qi", "BLATM1B_20050904_012629",
                                  "BLATM1B_20080802_134319.atm4cT3.F1", "BLATM1B_950530atm1_151023C.qi"])
def test_qi_names_with_and_without_extension(tmp_path, name):
    path = tmp_path / name
    _write_qi(path, 12629)
    d = atm.read_qi(str(path))
    date = "1995-05-30" if name.startswith("BLATM1B_95") else name[8:12] + "-" + name[12:14] + "-" + name[14:16]
    expected = (np.datetime64(date, "ns") - GPS_EPOCH) / np.timedelta64(1, "s") + 3600 + 26 * 60 + 29
    assert d["gps_seconds"][0] == pytest.approx(expected)
    assert d["latitude"][0] == pytest.approx(70.0) and d["longitude"][0] == pytest.approx(-50.0)


def test_qi_name_without_date_is_an_error(tmp_path):
    path = tmp_path / "not_an_atm_file"
    _write_qi(path, 12629)
    with pytest.raises(ValueError, match="date"):
        atm.read_qi(str(path))
