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
from .collections import REGIONS


def granule_row_groups(root, sensor, resolution, granule_id):
    """{(region, year): [row groups]} holding the granule's points, from granule_index.parquet"""
    path = db.granule_index_path(root, sensor, resolution)
    t = pq.read_table(path, filters=[("granule_id", "=", int(granule_id))])
    out = {}
    for region, year, rg in zip(t.column("region").to_pylist(), t.column("year").to_pylist(),
                                t.column("row_group").to_pylist()):
        out.setdefault((region, int(year)), []).append(int(rg))
    return out


def read_granule(root, granule_id, resolution="native", columns=None, sensor=None, use_index=True, threads=4):
    """
    every point of a granule as {region: pyarrow.Table} (the regions have different projections).  sensor is
    looked up in granules.parquet if not given.  With the granule index (default) only the row groups holding the
    granule are scanned; use_index=False relies on the granule_id footer statistics of the files of the
    granule's year(s).  The row groups are scanned and filtered on `threads` threads (default 4)
    """
    from concurrent.futures import ThreadPoolExecutor
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
        groups = {(r, y): None for r in REGIONS for y in (year, year + 1)}
    match = pc.field("granule_id") == granule_id
    want = None if columns is None else list(dict.fromkeys(["granule_id", *columns]))
    out = {}
    with ThreadPoolExecutor(max(1, threads)) as pool:
        for (region, year), row_groups in sorted(groups.items()):
            path = db.data_path(root, sensor, resolution, region, year)
            if not os.path.isfile(path):
                continue
            frag = next(ds.dataset(path, format="parquet").get_fragments())
            if row_groups is None:
                row_groups = [rg.id for rg in frag.subset(match).row_groups]
            if not row_groups:
                continue
            # contiguous slices of the row groups, one per thread, each scanned and filtered single-threaded
            slices = [s for s in np.array_split(np.asarray(row_groups), min(len(row_groups), max(1, threads)))
                      if s.size]
            tables = list(pool.map(lambda rgs: frag.subset(row_group_ids=[int(i) for i in rgs]).to_table(
                columns=want, filter=match, use_threads=False), slices))
            table = pa.concat_tables(tables)
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
                                                     "row_group_end"], use_threads=False)
    keep = pc.and_(pc.and_(pc.greater_equal(t.column("x_max"), bbox[0]), pc.less_equal(t.column("x_min"), bbox[2])),
                   pc.and_(pc.greater_equal(t.column("y_max"), bbox[1]), pc.less_equal(t.column("y_min"), bbox[3])))
    t = t.filter(keep)
    rgs = set()
    for a, b in zip(t.column("row_group_start").to_numpy(), t.column("row_group_end").to_numpy()):
        rgs.update(range(int(a), int(b) + 1))
    return sorted(rgs)


def _read_file_box(path, bbox, columns, strategy, stats_level):
    """
    (table, row groups read, rows read) for one data file.  "dataset" returns only the rows in the box; "stats"
    and "footer" return whole row groups, which _assemble filters while copying into the output arrays
    """
    want = None if columns is None else list(dict.fromkeys(["geometry", *columns]))
    if strategy == "dataset":
        frag = next(ds.dataset(path, format="parquet").get_fragments())
        sub = frag.subset(_box_expression(bbox))
        n_rg = len(sub.row_groups)
        t = sub.to_table(columns=want, filter=_box_expression(bbox)) if n_rg else None
        return t, n_rg, sum(rg.num_rows for rg in sub.row_groups)
    if strategy in ("stats", "footer"):
        if strategy == "stats":
            rgs = _stats_row_groups(path, bbox, stats_level)
        else:
            frag = next(ds.dataset(path, format="parquet").get_fragments())
            rgs = [rg.id for rg in frag.subset(_box_expression(bbox)).row_groups]
        if not rgs:
            return None, 0, 0
        # single-threaded decode: parallelism comes only from read_box's thread pool, so `threads` caps the cores
        t = pq.ParquetFile(path).read_row_groups(rgs, columns=want, use_threads=False)
        return t, len(rgs), t.num_rows
    raise ValueError(f"unknown strategy {strategy}")


def _chunk_index(geometry_chunk, bbox):
    """indices of the rows of one geometry chunk inside bbox"""
    x = geometry_chunk.field("x").to_numpy(zero_copy_only=False)
    y = geometry_chunk.field("y").to_numpy(zero_copy_only=False)
    return np.flatnonzero((x >= bbox[0]) & (x <= bbox[2]) & (y >= bbox[1]) & (y <= bbox[3]))


def _output_dtype(types_by_sensor, has_nulls, n_sensors):
    """
    numpy dtype of an output field: the common dtype if every sensor has it without nulls; for floats the widest
    float (missing values are NaN); otherwise (integers that need NaN) float64
    """
    dtypes = {np.dtype(t.to_pandas_dtype()) for t in types_by_sensor}
    if len(types_by_sensor) == n_sensors and len(dtypes) == 1 and not has_nulls:
        return dtypes.pop()
    if all(d.kind == "f" for d in dtypes):
        return np.result_type(*dtypes)
    return np.dtype(np.float64)


def _assemble(parts, bbox, sensors, threads):
    """
    {field: numpy array} of the points in bbox from [(sensor, table, filtered)], copying each selected value once,
    straight from the Arrow buffers into preallocated output arrays.  The copies run as one task per (chunk,
    field) on a thread pool, so a few large years do not leave most threads idle
    """
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max(1, threads))
    try:
        # rows to keep in each chunk (row group) of each table
        def index(part):
            sensor, t, filtered = part
            return [np.arange(len(c), dtype=np.int64) if filtered else _chunk_index(c, bbox)
                    for c in t.column("geometry").chunks]
        idx = list(pool.map(index, parts))
        present = [s for s in sensors if any(p[0] == s for p in parts)]
        # output fields and dtypes
        types, nulls = {}, set()
        for sensor in present:
            schema = next(p[1].schema for p in parts if p[0] == sensor)
            for f in schema:
                if f.name != "geometry":
                    types.setdefault(f.name, []).append(f.type)
        for sensor, t, _ in parts:
            nulls.update(n for n in t.column_names if n != "geometry" and t.column(n).null_count)
        total = sum(i.size for ii in idx for i in ii)
        out = {"x": np.empty(total), "y": np.empty(total)}
        for name, ts in types.items():
            out[name] = np.empty(total, dtype=_output_dtype(ts, name in nulls, len(present)))
        out["sensor"] = np.empty(total, dtype=np.int8)
        # one task per (chunk, field)
        tasks = []
        pos = 0
        for (sensor, t, _), ii in zip(parts, idx):
            names = set(t.column_names)
            geometry = t.column("geometry").chunks
            for c, i in enumerate(ii):
                if not i.size:
                    continue
                for name in out:
                    if name in ("x", "y"):
                        src = geometry[c].field(name)
                    elif name == "sensor":
                        src = sensors.index(sensor)
                    elif name in names:
                        src = t.column(name).chunk(c)
                    else:
                        src = None
                    tasks.append((name, src, i, pos))
                pos += i.size

        def run(task):
            name, src, i, start = task
            arr = out[name]
            if src is None:
                arr[start:start + i.size] = np.nan
            elif isinstance(src, int):
                arr[start:start + i.size] = src
            else:
                a = src.to_numpy(zero_copy_only=False)
                if a.dtype == arr.dtype:
                    # mode="clip" (the indices are all valid): with the default mode, take buffers the output
                    # and runs 4x slower on 4 threads
                    np.take(a, i, out=arr[start:start + i.size], mode="clip")
                else:
                    arr[start:start + i.size] = np.take(a, i, mode="clip")
        list(pool.map(run, tasks))
    finally:
        pool.shutdown()
    return out


def box_from_bds(bds):
    """(xmin, ymin, xmax, ymax) from bds = [[xmin, xmax], [ymin, ymax]]"""
    (xmin, xmax), (ymin, ymax) = bds
    return (float(xmin), float(ymin), float(xmax), float(ymax))


def read_box(root, sensors, resolution, region, *, bbox=None, bds=None, columns=None, years=None,
             strategy="stats", threads=4, stats_level=None, as_pointCollection=True, return_info=False):
    """
    every point within a box (projected coordinates of the region, m: EPSG:3413 for Arctic, EPSG:3031 for AA) from
    every year of the databases of `sensors`, as one pointCollection.data object with fields x, y, the requested
    columns (all by default; NaN where a sensor lacks a column), and sensor (index into `sensors`); or as a dict
    of numpy arrays if as_pointCollection is False.

    Give the box as exactly one of the keywords bbox = (xmin, ymin, xmax, ymax) or bds = [[xmin, xmax],
    [ymin, ymax]].

    strategy: "stats" (default) finds the row groups from the per-cell statistics files; "footer" from the
    row-group footer statistics of geometry.x/y; "dataset" lets pyarrow's dataset scanner prune and filter.
    threads caps the cores used (default 4): files are read and decoded on a pool of that size, with pyarrow's
    own threading off ("dataset" uses pyarrow's scanner threads instead)
    """
    from concurrent.futures import ThreadPoolExecutor
    if (bbox is None) == (bds is None):
        raise TypeError("read_box: give exactly one of bbox=(xmin, ymin, xmax, ymax) or bds=[[xmin, xmax], [ymin, ymax]]")
    bbox = box_from_bds(bds) if bds is not None else tuple(float(v) for v in bbox)
    if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
        raise ValueError(f"read_box: empty box {bbox} (min > max)")
    if isinstance(sensors, str):
        sensors = [sensors]
    jobs = [(s, f) for s in sensors for f in data_files(root, s, resolution, region, years)]
    with ThreadPoolExecutor(max(1, threads)) as pool:
        results = list(pool.map(lambda job: _read_file_box(job[1], bbox, columns, strategy, stats_level), jobs))
    info = {"files": len(jobs), "row_groups_read": sum(r[1] for r in results),
            "rows_read": sum(r[2] for r in results)}
    parts = [(sensor, t, strategy == "dataset") for (sensor, _), (t, _, _) in zip(jobs, results)
             if t is not None and t.num_rows]
    dd = _assemble(parts, bbox, sensors, threads)
    if as_pointCollection:
        import pointCollection as pc_
        from .collections import REGIONS
        dd = pc_.data(EPSG=REGIONS[region]["epsg"]).from_dict(dd)
    return (dd, info) if return_info else dd
