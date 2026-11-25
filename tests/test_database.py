import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from altimetryData import database as db, read, reduce, s2, timeref


def _granule(seed, granule_id, n=40000):
    rng = np.random.default_rng(seed)
    # a ~30 km flight line near Jakobshavn
    s = np.sort(rng.uniform(0, 1, n))
    lat = 69.0 + 0.3 * s + rng.normal(0, 0.001, n)
    lon = -49.5 + 0.2 * s + rng.normal(0, 0.002, n)
    h = 1000 + 50 * s + rng.normal(0, 0.1, n)
    dt = timeref.delta_time_from_utc_datetime(np.array(["2011-04-12T13:00:00"], dtype="datetime64[ns]"))[0] + 600 * s
    native = {"longitude": lon, "latitude": lat, "h": h, "dh_frame": np.full(n, -0.002), "delta_time": dt,
              "source_index": np.arange(n, dtype=np.uint32), "granule_id": np.full(n, granule_id, np.uint32),
              "rcv_sigstr": rng.integers(100, 1000, n)}
    native["x"], native["y"] = db.project(lon, lat, "Arctic")
    return native


def test_build_and_stats(tmp_path):
    root = str(tmp_path)
    for gid in range(3):
        native = _granule(gid, gid)
        db.write_partitions(db.make_table(native, "ATM_wide", "native", "Arctic"), root, "ATM_wide", "native", "Arctic", gid)
        keep, stats = reduce.reduce_points(native["x"], native["y"], native["h"], 10, 50, 0.5)
        red = {k: v[keep] for k, v in native.items()}
        red.update(bias=stats["bias"], noise=stats["noise"], n_fit=np.nan_to_num(stats["N"]),
                   slope_x=stats["slope_x"], slope_y=stats["slope_y"])
        db.write_partitions(db.make_table(red, "ATM_wide", "reduced", "Arctic"), root, "ATM_wide", "reduced", "Arctic", gid)

    out = db.build_database_file(root, "ATM_wide", "native", "Arctic", 2011, target_bytes=200_000)
    pf = pq.ParquetFile(out)
    assert pf.metadata.num_rows == 120000
    assert pf.metadata.num_row_groups > 1
    ids = pq.read_table(out, columns=["cell_id"]).column("cell_id").to_numpy()
    assert np.all(ids[1:] >= ids[:-1])
    assert s2.level(int(ids[0])) == 18
    # no level-18 cell spans two row groups, and only the intended columns carry footer statistics
    last = None
    for rg in range(pf.num_row_groups):
        col = pf.metadata.row_group(rg)
        stats_cols = {col.column(i).path_in_schema for i in range(col.num_columns) if col.column(i).is_stats_set}
        assert stats_cols == {"cell_id", "delta_time", "geometry.x", "geometry.y", "granule_id"}
        encodings = {col.column(i).path_in_schema: col.column(i).encodings for i in range(col.num_columns)}
        assert "DELTA_BINARY_PACKED" in encodings["source_index"]
        assert "BYTE_STREAM_SPLIT" in encodings["geometry.x"] and "BYTE_STREAM_SPLIT" in encodings["h"]
        lo = col.column(0).statistics.min
        if last is not None:
            assert lo > last
        last = col.column(0).statistics.max
    meta = json.loads(pf.metadata.metadata[b"altimetry_data"])
    assert meta["sensor"] == "ATM_wide" and meta["year"] == 2011 and meta["crs"] == "EPSG:3413"
    geo = json.loads(pf.metadata.metadata[b"geo"])["columns"]["geometry"]
    assert geo["encoding"] == "point" and geo["crs"]["id"]["code"] == 3413
    assert not {"latitude", "longitude", "x", "y", "year"} & set(pf.schema_arrow.names)
    # descriptions: Arrow field metadata, altimetry_data columns, and GDAL's gdal:schema comments
    assert b"WGS84 ellipsoid" in pf.schema_arrow.field("h").metadata[b"description"]
    assert pf.schema_arrow.field("h").metadata[b"units"] == b"m"
    assert pf.schema_arrow.field("geometry").type.field("x").metadata[b"units"] == b"m"
    assert meta["columns"]["delta_time"]["units"] == "s" and "geometry.y" in meta["columns"]
    gdal = json.loads(pf.metadata.metadata[b"gdal:schema"])["columns"]
    assert "GPS seconds" in gdal["delta_time"]["comment"] and "geometry" not in gdal
    # provenance
    assert set(meta["code"]) >= {"repository", "commit", "dirty"}

    gpd = pytest.importorskip("geopandas")
    g = gpd.read_parquet(out)
    assert g.index.name == "cell_id" and g.crs.to_epsg() == 3413
    # cell_id is reproducible exactly from the stored geometry
    lon, lat = db.unproject(g.geometry.x.to_numpy(), g.geometry.y.to_numpy(), "Arctic")
    assert np.array_equal(s2.cell_ids(lon, lat, 18), g.index.to_numpy())
    # and the geometry matches the input positions to within the rounding quantum
    native = _granule(0, 0)
    x0, y0 = db.project(native["longitude"], native["latitude"], "Arctic")
    sel = g[g.granule_id == 0].sort_values("source_index")
    assert np.abs(sel.geometry.x.to_numpy() - x0).max() <= db.QUANTUM["x"] / 2

    st = db.build_stats(out, db.stats_path(root, "ATM_wide", "native", "Arctic", 2011))
    S = pq.read_table(st).to_pandas()
    assert set(S.level) == set(db.STAT_LEVELS)
    assert (S.groupby("level").row_count.sum() == 120000).all()
    T = pq.read_table(out, columns=["cell_id", "geometry", "h"])
    x, y = db.geometry_xy(T.column("geometry"))
    lon, lat = db.unproject(x, y, "Arctic")
    T = pd.DataFrame({"x": x, "lat": lat, "h": T.column("h").to_numpy()})
    ref = T.groupby(s2.parent(pq.read_table(out, columns=["cell_id"]).column("cell_id").to_numpy(), 12)).agg(
        ["min", "max", "mean"])
    s12 = S[S.level == 12].set_index("cell_id")
    assert np.allclose(s12["x_mean"], ref[("x", "mean")]) and np.allclose(s12["h_max"], ref[("h", "max")])
    assert np.allclose(s12["lat_min"], ref[("lat", "min")]) and np.allclose(s12["lat_mean"], ref[("lat", "mean")])

    red = db.build_database_file(root, "ATM_wide", "reduced", "Arctic", 2011)
    assert 0 < pq.read_metadata(red).num_rows < 120000


def test_granule_index_and_read(tmp_path):
    root = str(tmp_path)
    for gid in range(3):
        db.write_partitions(db.make_table(_granule(gid, gid), "ATM_wide", "native", "Arctic"), root, "ATM_wide",
                            "native", "Arctic", gid)
    out = db.build_database_file(root, "ATM_wide", "native", "Arctic", 2011, target_bytes=100_000)
    idx = pq.read_table(db.index_path(root, "ATM_wide", "native", "Arctic", 2011)).to_pandas()
    t = pq.read_table(out, columns=["granule_id", "source_index"])
    pf = pq.ParquetFile(out)
    # the index lists exactly the row groups (and row counts) holding each granule
    starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
    g_all = t.column("granule_id").to_numpy()
    for gid, rg, n in zip(idx.granule_id, idx.row_group, idx.n_rows):
        assert (g_all[starts[rg]:starts[rg + 1]] == gid).sum() == n
    assert idx.n_rows.sum() == t.num_rows
    assert db.write_granule_index(root, "ATM_wide", "native")
    pd.DataFrame({"granule_id": np.arange(3, dtype=np.uint32), "sensor": "ATM_wide",
                  "year": np.int16(2011)}).to_parquet(f"{root}/granules.parquet")
    for use_index in (True, False):
        got = read.read_granule(root, 1, columns=["source_index", "h"], use_index=use_index)
        assert list(got) == ["Arctic"]
        assert np.array_equal(np.sort(got["Arctic"].column("source_index").to_numpy()), np.arange(40000))


def test_read_box_matches_brute_force(tmp_path):
    root = str(tmp_path)
    for gid, sensor in [(0, "ATM_wide"), (1, "ATM_wide"), (2, "LVIS")]:
        native = _granule(gid, gid)
        if sensor == "LVIS":
            native.pop("rcv_sigstr")      # LVIS has no rcv_sigstr; its lfid/shotnumber stay null
        db.write_partitions(db.make_table(native, sensor, "reduced", "Arctic"), root, sensor, "reduced", "Arctic", gid)
    for sensor in ("ATM_wide", "LVIS"):
        f = db.build_database_file(root, sensor, "reduced", "Arctic", 2011, target_bytes=60_000)
        db.build_stats(f, db.stats_path(root, sensor, "reduced", "Arctic", 2011))
    x0, y0 = db.project(np.array([-49.4]), np.array([69.15]), "Arctic")
    bbox = (x0[0] - 3000, y0[0] - 3000, x0[0] + 3000, y0[0] + 3000)
    ref = {}
    for sensor in ("ATM_wide", "LVIS"):
        t = pq.read_table(db.data_path(root, sensor, "reduced", "Arctic", 2011))
        x, y = db.geometry_xy(t.column("geometry"))
        m = (x >= bbox[0]) & (x <= bbox[2]) & (y >= bbox[1]) & (y <= bbox[3])
        ref[sensor] = {"x": x[m], "h": t.column("h").to_numpy()[m], "source_index": t.column("source_index").to_numpy()[m]}
    assert 0 < ref["ATM_wide"]["x"].size < 80000 and ref["LVIS"]["x"].size > 0
    for strategy in ("stats", "footer", "dataset"):
        dd, info = read.read_box(root, ["ATM_wide", "LVIS"], "reduced", "Arctic", bbox=bbox, strategy=strategy, threads=3,
                                 as_pointCollection=False, return_info=True)
        assert info["rows_read"] < 80000 + 40000        # pruning skipped some row groups
        for k, sensor in enumerate(("ATM_wide", "LVIS")):
            sel = dd["sensor"] == k
            order = np.argsort(dd["source_index"][sel] + 1e6 * dd["granule_id"][sel])
            r_order = np.argsort(ref[sensor]["source_index"])
            assert sel.sum() == ref[sensor]["x"].size, strategy
            if sensor == "ATM_wide":
                assert np.array_equal(np.sort(dd["x"][sel]), np.sort(ref[sensor]["x"]))
            else:
                assert np.array_equal(dd["h"][sel][order], ref[sensor]["h"][r_order])
        assert dd["rcv_sigstr"].dtype == np.float64 and np.isnan(dd["rcv_sigstr"][dd["sensor"] == 1]).all()
        assert np.isnan(dd["lfid"]).all() and dd["h"].dtype == np.float32 and dd["cell_id"].dtype == np.uint64
    pc_ = pytest.importorskip("pointCollection")
    D = read.read_box(root, ["ATM_wide", "LVIS"], "reduced", "Arctic", bbox=bbox)
    assert D.size == sum(v["x"].size for v in ref.values()) and "bias" in D.fields
    # bds=[[xmin, xmax], [ymin, ymax]] gives the same points
    D2 = read.read_box(root, ["ATM_wide", "LVIS"], "reduced", "Arctic", bds=[[bbox[0], bbox[2]], [bbox[1], bbox[3]]])
    assert np.array_equal(np.sort(D2.x), np.sort(D.x))
    with pytest.raises(TypeError):
        read.read_box(root, ["ATM_wide"], "reduced", "Arctic", bbox=bbox, bds=[[0, 1], [0, 1]])
    with pytest.raises(TypeError):
        read.read_box(root, ["ATM_wide"], "reduced", "Arctic")
