import numpy as np
import pytest

from altimetryData import s2

# (lon, lat, leaf id, level-18 id, level-7 id) from duckdb's geography extension (s2_cellfromlonlat and
# s2_cell_parent), which sliderule's atl06t builder uses
REFERENCE = [
    (-45.0, 70.0, 5687759339317276411, 5687759339301896192, 5687694485648048128),
    (-39.88, 66.2, 5676508118970304331, 5676508118956113920, 5676576224067977216),
    (0.0, 90.0, 5764607523034234881, 5764607523051012096, 5764677891778412544),
    (120.0, -75.0, 12560498960241077581, 12560498960243884032, 12560468991992135680),
    (-120.0, -85.0, 11909254419113706001, 11909254419113312256, 11909276633372033024),
    (179.999, -89.9, 12682128431163499957, 12682128431157084160, 12682066181931139072),
    (10.0, 59.5, 5062376882704185179, 5062376882708676608, 5062397824885325824),
    (-73.3, 77.4, 5634379090408706831, 5634379090396971008, 5634354977561378816),
]


def test_reference_ids():
    lon = np.array([r[0] for r in REFERENCE])
    lat = np.array([r[1] for r in REFERENCE])
    leaf, l18, l7 = (np.array([r[k] for r in REFERENCE], dtype=np.uint64) for k in (2, 3, 4))
    assert np.array_equal(s2.cell_ids(lon, lat), leaf)
    assert np.array_equal(s2.cell_ids(lon, lat, 18), l18)
    assert np.array_equal(s2.parent(s2.cell_ids(lon, lat), 7), l7)


def test_levels_and_ranges():
    ids = s2.cell_ids(np.array([-45.0, 120.0]), np.array([70.0, -75.0]), 18)
    assert all(s2.level(int(i)) == 18 for i in ids)
    lo, hi = s2.cell_range(s2.parent(ids, 7))
    assert np.all((ids >= lo) & (ids <= hi))


def test_matches_s2sphere():
    s2sphere = pytest.importorskip("s2sphere")
    rng = np.random.default_rng(3)
    lon = rng.uniform(-180, 180, 2000)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 2000)))
    ids = s2.cell_ids(lon, lat)
    for k in range(lon.size):
        ref = s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(lat[k], lon[k])).id()
        assert int(ids[k]) == ref
