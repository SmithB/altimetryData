"""
Readers for the database.

read_granule() returns every point of one raw granule.  A granule's points are spread over many row groups of
a data file (rows are sorted by S2 cell, not by granule), so it uses the granule index: granule_index.parquet
lists, for each granule, the (region, year, row group) triples that hold its points.  Only those row groups are
read, then filtered on granule_id.  Without the index, the granule_id footer statistics still let a reader skip
row groups whose granule_id range cannot contain the granule (see read_granule(..., use_index=False)).
"""

import glob
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from . import database as db


def granule_row_groups(root, sensor, resolution, granule_id):
    """{(region, year): [row groups]} holding the granule's points, from granule_index.parquet"""
    path = db.granule_index_path(root, sensor, resolution)
    t = pq.read_table(path, filters=[("granule_id", "=", int(granule_id))])
    out = {}
    for region, year, rg in zip(t.column("region").to_pylist(), t.column("year").to_pylist(),
                                t.column("row_group").to_pylist()):
        out.setdefault((region, int(year)), []).append(int(rg))
    return out


def read_granule(root, granule_id, resolution="native", columns=None, sensor=None, use_index=True):
    """
    every point of a granule as {region: pyarrow.Table} (the regions have different projections).  sensor is
    looked up in granules.parquet if not given.  use_index=False skips the granule index and relies on the
    granule_id footer statistics of the files of the granule's year(s)
    """
    granule_id = int(granule_id)
    if sensor is None or not use_index:
        g = pq.read_table(os.path.join(root, "granules.parquet"), filters=[("granule_id", "=", granule_id)],
                          columns=["sensor", "year"]).to_pylist()
        if not g:
            raise KeyError(f"granule_id {granule_id} is not in granules.parquet")
        sensor = sensor or g[0]["sensor"]
    if use_index:
        groups = granule_row_groups(root, sensor, resolution, granule_id)
    else:
        # the granule's start year, and the next one in case it runs past midnight on 31 December
        year = int(g[0]["year"])
        groups = {(r, y): None for r in ("GL", "AA") for y in (year, year + 1)}
    out = {}
    for (region, year), row_groups in sorted(groups.items()):
        path = db.data_path(root, sensor, resolution, region, year)
        if not os.path.isfile(path):
            continue
        if row_groups is not None:
            pf = pq.ParquetFile(path)
            table = pf.read_row_groups(row_groups, columns=None if columns is None else
                                       list(dict.fromkeys(["granule_id", *columns])))
            table = table.filter(pc.equal(table.column("granule_id"), pa.scalar(granule_id, pa.uint32())))
        else:
            table = ds.dataset(path).to_table(columns=columns, filter=pc.field("granule_id") == granule_id)
        if columns is not None:
            table = table.select(columns)
        if table.num_rows:
            out[region] = table if region not in out else pa.concat_tables([out[region], table])
    return out


# ---------------------------------------------------------------------------------------------------------------
# box reads
# ---------------------------------------------------------------------------------------------------------------

def data_files(root, sensor, resolution, region, years=None):
    """data files of one database, in year order"""
    files = sorted(glob.glob(os.path.join(root, sensor, resolution, "data", f"{sensor}_{resolution}_{region}_*.parquet")))
    if years is not None:
        files = [f for f in files if int(f.rsplit("_", 1)[1].split(".")[0]) in set(years)]
    return files


def _box_expression(bbox):
    x, y = pc.field("geometry", "x"), pc.field("geometry", "y")
    return (x >= bbox[0]) & (x <= bbox[2]) & (y >= bbox[1]) & (y <= bbox[3])


def _stats_row_groups(data_file, bbox, level=None):
    """row groups holding cells (at an S2 level of the stats file) whose points' x/y range meets the box"""
    stats = data_file.replace(os.sep + "data" + os.sep, os.sep + "stats" + os.sep).replace(".parquet", "_stats.parquet")
    pf = pq.ParquetFile(stats)
    levels = json.loads(pf.metadata.metadata[b"altimetry_data"])["levels"]
    # finest available level no finer than requested (default: the finest)
    use = max(v for v in levels if level is None or v <= level) if level is None or min(levels) <= level else min(levels)
    t = pf.read_row_group(levels.index(use), columns=["x_min", "x_max", "y_min", "y_max", "row_group_start",
                                                     "row_group_end"])
    keep = pc.and_(pc.and_(pc.greater_equal(t.column("x_max"), bbox[0]), pc.less_equal(t.column("x_min"), bbox[2])),
                   pc.and_(pc.greater_equal(t.column("y_max"), bbox[1]), pc.less_equal(t.column("y_min"), bbox[3])))
    t = t.filter(keep)
    rgs = set()
    for a, b in zip(t.column("row_group_start").to_numpy(), t.column("row_group_end").to_numpy()):
        rgs.update(range(int(a), int(b) + 1))
    return sorted(rgs)


def _read_file_box(path, bbox, columns, strategy, stats_level):
    """(table of the points in the box, row groups read, rows read) for one data file"""
    want = None if columns is None else list(dict.fromkeys(["geometry", *columns]))
    if strategy == "dataset":
        frag = next(ds.dataset(path, format="parquet").get_fragments())
        sub = frag.subset(_box_expression(bbox))
        n_rg = len(sub.row_groups)
        t = sub.to_table(columns=want, filter=_box_expression(bbox)) if n_rg else None
        rows = sum(rg.num_rows for rg in sub.row_groups)
        return t, n_rg, rows
    if strategy in ("stats", "footer"):
        pf = pq.ParquetFile(path)
        if strategy == "stats":
            rgs = _stats_row_groups(path, bbox, stats_level)
        else:
            frag = next(ds.dataset(path, format="parquet").get_fragments())
            rgs = [rg.id for rg in frag.subset(_box_expression(bbox)).row_groups]
        if not rgs:
            return None, 0, 0
        t = pf.read_row_groups(rgs, columns=want, use_threads=True)
        x, y = db.geometry_xy(t.column("geometry"))
        mask = (x >= bbox[0]) & (x <= bbox[2]) & (y >= bbox[1]) & (y <= bbox[3])
        return t.filter(pa.array(mask)), len(rgs), t.num_rows
    raise ValueError(f"unknown strategy {strategy}")


def read_box(root, sensors, resolution, region, bbox, columns=None, years=None, strategy="stats", threads=8,
             stats_level=None, as_pointCollection=True, return_info=False):
    """
    every point within bbox = (xmin, ymin, xmax, ymax) (projected coordinates of the region) from every year of
    the databases of `sensors`, as one pointCollection.data object (fields x, y, the requested columns, and
    sensor: index into `sensors`), or as {sensor: pyarrow.Table} if as_pointCollection is False.

    strategy: "stats" finds the row groups from the per-cell statistics files; "footer" from the row-group
    footer statistics of geometry.x/y; "dataset" lets pyarrow's dataset scanner prune and filter
    """
    from concurrent.futures import ThreadPoolExecutor
    if isinstance(sensors, str):
        sensors = [sensors]
    jobs = [(s, f) for s in sensors for f in data_files(root, s, resolution, region, years)]
    with ThreadPoolExecutor(max(1, threads)) as pool:
        results = list(pool.map(lambda job: _read_file_box(job[1], bbox, columns, strategy, stats_level), jobs))
    tables = {}
    info = {"files": len(jobs), "row_groups_read": 0, "rows_read": 0}
    for (sensor, _), (t, n_rg, rows) in zip(jobs, results):
        info["row_groups_read"] += n_rg
        info["rows_read"] += rows
        if t is not None and t.num_rows:
            tables.setdefault(sensor, []).append(t)
    tables = {s: pa.concat_tables(v) for s, v in tables.items()}
    out = to_pointCollection(tables, sensors, region) if as_pointCollection else tables
    return (out, info) if return_info else out


def _numpy(column):
    a = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if a.null_count:
        a = a.cast(pa.float64()) if not pa.types.is_floating(a.type) else a
        return a.to_numpy(zero_copy_only=False)
    return a.to_numpy(zero_copy_only=False)


def to_pointCollection(tables, sensors, region):
    """one pointCollection.data object from {sensor: table}: fields x, y, every column (NaN where a sensor lacks
    it), and sensor (index into `sensors`)"""
    import pointCollection as pc_
    fields, n = {}, {}
    for s, t in tables.items():
        n[s] = t.num_rows
        x, y = db.geometry_xy(t.column("geometry"))
        fields.setdefault("x", {})[s] = x
        fields.setdefault("y", {})[s] = y
        for name in t.column_names:
            if name != "geometry":
                fields.setdefault(name, {})[s] = _numpy(t.column(name))
    order = [s for s in sensors if s in tables]
    dd = {}
    for name, parts in fields.items():
        if len(parts) == len(order) and len({parts[s].dtype for s in order}) == 1:
            dd[name] = np.concatenate([parts[s] for s in order]) if order else np.zeros(0)
        else:
            dd[name] = np.concatenate([parts[s].astype(np.float64) if s in parts else np.full(n[s], np.nan)
                                       for s in order])
    dd["sensor"] = np.concatenate([np.full(n[s], sensors.index(s), dtype=np.int8) for s in order]) if order \
        else np.zeros(0, np.int8)
    if not order:
        dd = {"x": np.zeros(0), "y": np.zeros(0), "sensor": dd["sensor"]}
    from .collections import REGIONS
    return pc_.data(EPSG=REGIONS[region]["epsg"]).from_dict(dd)
