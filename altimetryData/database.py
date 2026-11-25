"""
Database layout, partition writing, and the optimized GeoParquet writer.

Layout (one file per sensor, resolution, region, and year):

    {root}/granules.parquet                                           granule lookup table
    {root}/{sensor}/{resolution}/data/{sensor}_{resolution}_{region}_{year}.parquet
    {root}/{sensor}/{resolution}/stats/{sensor}_{resolution}_{region}_{year}_stats.parquet
    {root}/{sensor}/{resolution}/manifest.json

Intermediate files live under a separate work root (default: a sibling of root named parquet_work):

    {work}/partitions/{sensor}/{resolution}/{region}/{year}/S{level-7 cell:016X}/{granule_id:08d}.parquet
    {work}/status/{sensor}/{granule_id:08d}.json                     per-granule processing record
    {work}/raw/{granule_id:08d}/{granule}                             transient downloads

Each granule is processed independently into partitions: one parquet file per S2 level-7 cell it touches.  A
database file is then built by walking the level-7 cells in cell-id order and sorting each cell's rows by
level-18 cell id.  Every level-18 id inside a level-7 cell lies between the ids of the cells before and after
it, so the output is globally sorted while only one level-7 cell is held in memory at a time.

Data files follow sliderule's atl06t format (README in sliderule/targets/atl06t) except that:
  * there is one file per sensor/region/year instead of one per level-7 cell;
  * the geometry is a GeoParquet native point (struct of x, y) in EPSG:3413 (GL) or EPSG:3031 (AA), and
    latitude and longitude are not stored (cell_id is computed from the inverse projection of the stored x, y);
  * footer statistics cover cell_id, delta_time, geometry.x, and geometry.y;
  * the time column is delta_time (GPS seconds since 2018-01-01, as in ICESat-2) rather than time_ns;
  * row groups target 8 MB but grow as needed to keep a file to at most MAX_ROW_GROUPS row groups.
"""

import glob
import json
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyproj

from . import provenance, s2
from . import schema as sc
from .collections import REGIONS
from .timeref import gps_to_utc_datetime, ATLAS_SDP_EPOCH_GPS

PARTITION_LEVEL = 7
# ICESat granules cross a whole ice sheet, so finer partitions would mean hundreds of thousands of tiny files
PARTITION_LEVELS = {"ICESat": 4}
DATA_LEVEL = 18
STAT_LEVELS = tuple(range(7, 15))
MIN_STAT_ROWS = 50             # finer stat levels are kept only while cells average at least this many rows
COMPRESSION_LEVEL = 9          # zstd level for the data files (partitions use the fast default)

# Stored coordinates and times are rounded to binary grids far finer than any measurement precision, so the
# low mantissa bits are zero and compress away.  Largest change: 0.5 mm in x or y, 0.03 ms in delta_time.
QUANTUM = {"x": 2.0 ** -10, "y": 2.0 ** -10, "delta_time": 2.0 ** -14}


def partition_level(sensor):
    return PARTITION_LEVELS.get(sensor, PARTITION_LEVEL)


def trim(values, name):
    """round to the binary grid QUANTUM[name]"""
    q = QUANTUM[name]
    return np.round(np.asarray(values, dtype=np.float64) / q) * q
ROW_GROUP_TARGET_BYTES = 8 * 1048576
MAX_ROW_GROUPS = 5000
ROW_BYTES_ESTIMATE = 60
# x, y are the EPSG:3413 / EPSG:3031 projections of ITRF2020 latitude and longitude (EPSG:9989); the files are
# labeled with the EPSG projected CRS, whose WGS84 datum agrees with ITRF2020 at the centimeter level
GEOGRAPHIC_FRAME = "ITRF2020 (EPSG:9989)"
FORMAT_VERSION = "3"
# optional parquet bloom filters (column path -> {"ndv", "fpp"}); see docs/database_format.md
BLOOM_FILTERS = {}
THREADS = 8                    # threads for the inverse projection
DEFAULT_ROOT = "/Volumes/ice1/ben/oib_database/parquet"


def default_work_root(root):
    return os.path.join(os.path.dirname(os.path.abspath(root)), os.path.basename(os.path.abspath(root)) + "_work")


# ---------------------------------------------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------------------------------------------

def data_path(root, sensor, resolution, region, year):
    return os.path.join(root, sensor, resolution, "data", f"{sensor}_{resolution}_{region}_{year}.parquet")


def stats_path(root, sensor, resolution, region, year):
    return os.path.join(root, sensor, resolution, "stats", f"{sensor}_{resolution}_{region}_{year}_stats.parquet")


def index_path(root, sensor, resolution, region, year):
    """per-file granule index: the row groups of the data file that hold each granule's points"""
    return os.path.join(root, sensor, resolution, "index", f"{sensor}_{resolution}_{region}_{year}_granules.parquet")


def granule_index_path(root, sensor, resolution):
    """granule index of every file of a database (built from the per-file indexes by write_granule_index)"""
    return os.path.join(root, sensor, resolution, "granule_index.parquet")


def partition_dir(root, sensor, resolution, region, year):
    return os.path.join(root, "partitions", sensor, resolution, region, str(year))


# ---------------------------------------------------------------------------------------------------------------
# building tables
# ---------------------------------------------------------------------------------------------------------------

def geometry_array(x, y):
    """GeoParquet native point geometry: struct<x: double, y: double>"""
    return pa.StructArray.from_arrays([pa.array(np.asarray(x, dtype=np.float64)),
                                       pa.array(np.asarray(y, dtype=np.float64))], fields=list(sc.GEOMETRY))


def geometry_xy(column):
    """x and y numpy arrays from a geometry column (ChunkedArray or StructArray)"""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    return column.field("x").to_numpy(zero_copy_only=False), column.field("y").to_numpy(zero_copy_only=False)


_PROJECTORS = {}


def _projector(region, inverse):
    key = (region, inverse)
    if key not in _PROJECTORS:
        src, dst = "EPSG:4326", f"EPSG:{REGIONS[region]['epsg']}"
        _PROJECTORS[key] = pyproj.Transformer.from_crs(*((dst, src) if inverse else (src, dst)), always_xy=True)
    return _PROJECTORS[key]


def project(lon, lat, region):
    x, y = _projector(region, False).transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def box_around(region, lon, lat, size):
    """(xmin, ymin, xmax, ymax), a size x size (m) box in the region's projection centered on lon, lat"""
    x, y = project(np.array([lon], dtype=float), np.array([lat], dtype=float), region)
    h = size / 2
    return (float(x[0]) - h, float(y[0]) - h, float(x[0]) + h, float(y[0]) + h)


def box_lonlat_bounds(region, bbox, n=41):
    """(lon_min, lat_min, lon_max, lat_max) enclosing a projected box (sampled on a grid; fine for boxes that
    do not contain a pole or cross the antimeridian)"""
    X, Y = np.meshgrid(np.linspace(bbox[0], bbox[2], n), np.linspace(bbox[1], bbox[3], n))
    lon, lat = unproject(X.ravel(), Y.ravel(), region, threads=1)
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def unproject(x, y, region, threads=THREADS):
    """longitude, latitude (degrees) of projected x, y; chunks run in threads (PROJ releases the GIL)"""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    lon, lat = np.empty_like(x), np.empty_like(y)
    chunk = 1 << 18
    starts = range(0, x.size, chunk)

    def run(i):
        # one Transformer per thread: pyproj transformers are not safe to share between threads
        tr = pyproj.Transformer.from_crs(f"EPSG:{REGIONS[region]['epsg']}", "EPSG:4326", always_xy=True)
        for j in starts[i::threads]:
            lon[j:j + chunk], lat[j:j + chunk] = tr.transform(x[j:j + chunk], y[j:j + chunk])

    if threads <= 1 or x.size <= chunk:
        lon[:], lat[:] = _projector(region, True).transform(x, y)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(run, range(threads)))
    return lon, lat


def make_table(columns, sensor, resolution, region):
    """
    arrow table in the database schema from a dict of numpy arrays holding longitude and latitude (ITRF2020) or
    projected x and y, plus h, delta_time, granule_id, and source_index.  The geometry (x, y rounded to QUANTUM)
    and cell_id (from the inverse projection of the rounded x, y, so it can be recomputed exactly from the stored
    geometry) are derived here; any other schema column that is missing is filled with NaN (floats)
    or the type's null
    """
    derived = dict(columns)
    if "x" not in derived:
        derived["x"], derived["y"] = project(columns["longitude"], columns["latitude"], region)
    x, y = trim(derived["x"], "x"), trim(derived["y"], "y")
    if "cell_id" not in derived:
        # single-threaded: granules are processed by many worker processes at once
        derived["cell_id"] = s2.cell_ids(*unproject(x, y, region, threads=1), DATA_LEVEL)
    derived["delta_time"] = trim(columns["delta_time"], "delta_time")
    n = x.size
    arrays = []
    schema = sc.arrow_schema(sensor, resolution)
    for field in schema:
        if field.name == "geometry":
            arrays.append(geometry_array(x, y))
        elif field.name in derived:
            arrays.append(pa.array(np.asarray(derived[field.name]).astype(field.type.to_pandas_dtype(), copy=False),
                                   type=field.type))
        elif pa.types.is_floating(field.type):
            arrays.append(pa.array(np.full(n, np.nan, dtype=field.type.to_pandas_dtype()), type=field.type))
        else:
            arrays.append(pa.nulls(n, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def calendar_year(delta_time):
    """UTC calendar year of each delta_time"""
    utc = gps_to_utc_datetime(np.asarray(delta_time, dtype=np.float64) + ATLAS_SDP_EPOCH_GPS)
    return utc.astype("datetime64[Y]").astype(np.int64) + 1970


def write_partitions(table, work_root, sensor, resolution, region, granule_id):
    """split one granule's table by calendar year and partition cell; returns the files written"""
    if table.num_rows == 0:
        return []
    table = table.sort_by("cell_id")
    cell7 = s2.parent(table.column("cell_id").to_numpy(), partition_level(sensor))
    years = calendar_year(table.column("delta_time").to_numpy())
    files = []
    for year in np.unique(years):
        in_year = years == year
        cells_year = cell7[in_year]
        sub_year = table.filter(pa.array(in_year))
        bounds = np.flatnonzero(np.diff(cells_year)) + 1
        starts = np.r_[0, bounds]
        stops = np.r_[bounds, cells_year.size]
        for start, stop in zip(starts, stops):
            out_dir = os.path.join(partition_dir(work_root, sensor, resolution, region, int(year)),
                                   f"S{int(cells_year[start]):016X}")
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"{int(granule_id):08d}.parquet")
            pq.write_table(sub_year.slice(start, stop - start), out_file, compression="zstd")
            files.append(out_file)
    return files


# ---------------------------------------------------------------------------------------------------------------
# optimized database file
# ---------------------------------------------------------------------------------------------------------------

def _geo_metadata(bbox, region):
    crs = pyproj.CRS.from_epsg(REGIONS[region]["epsg"]).to_json_dict()
    return {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "point", "geometry_types": ["Point"], "crs": crs,
                                 "bbox": [float(b) for b in bbox], "edges": "planar"}},
    }


def _pandas_metadata(schema):
    """pandas metadata that makes cell_id the index when the file is read with pandas or geopandas"""
    empty = pa.Table.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema).to_pandas()
    empty = empty.set_index("cell_id")
    return pa.Schema.from_pandas(empty, preserve_index=True).metadata[b"pandas"]


def _gdal_schema(sensor, resolution):
    """GDAL's gdal:schema metadata: per-column comments, shown by ogrinfo and QGIS"""
    columns = {}
    for name, info in sc.column_info(sensor, resolution).items():
        if name == "geometry" or name.startswith("geometry."):
            continue
        columns[name] = {"comment": info["description"] + (f" [{info['units']}]" if info["units"] else "")}
    return {"columns": columns}


def build_database_file(root, sensor, resolution, region, year, work_root=None, extra_metadata=None, verbose=False,
                        target_bytes=ROW_GROUP_TARGET_BYTES, max_row_groups=MAX_ROW_GROUPS, bloom_filters=None,
                        granule_commits=None):
    """
    assemble the partitions of one sensor/resolution/region/year into one optimized GeoParquet file, and write
    its granule index (index_path): one row per (granule_id, row_group) giving the rows of each granule in each
    row group.  granule_commits ({granule_id: commit}, from the processing status files) records which code
    commits processed the file's granules
    """
    bloom_filters = BLOOM_FILTERS if bloom_filters is None else bloom_filters
    pdir = partition_dir(work_root or root, sensor, resolution, region, year)
    cell_dirs = sorted(d for d in glob.glob(os.path.join(pdir, "S" + "?" * 16)) if os.path.isdir(d))
    if not cell_dirs:
        return None
    files_by_cell = [sorted(glob.glob(os.path.join(d, "*.parquet"))) for d in cell_dirs]
    total_rows = sum(pq.read_metadata(f).num_rows for files in files_by_cell for f in files)

    schema = sc.arrow_schema(sensor, resolution)
    order = [n for n in sc.FIRST_COLUMNS if n in schema.names] + [n for n in schema.names if n not in sc.FIRST_COLUMNS]
    schema = pa.schema([schema.field(n) for n in order])
    floats = [f.name for f in schema if pa.types.is_floating(f.type)]
    # byte-stream-split for floats (including the geometry's x and y), delta for steadily rising integers,
    # dictionary for the other integers
    encoding = {name: "BYTE_STREAM_SPLIT" for name in floats + ["geometry.x", "geometry.y"]}
    encoding.update({name: "DELTA_BINARY_PACKED" for name in sc.DELTA_COLUMNS if name in schema.names})
    dictionary = [n for n in schema.names if n not in encoding and n != "geometry"]
    metadata = {b"pandas": _pandas_metadata(schema)}
    schema = schema.with_metadata(metadata)

    out_file = data_path(root, sensor, resolution, region, year)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    tmp_file = out_file + ".tmp"
    bbox = [np.inf, np.inf, -np.inf, -np.inf]
    min_rows = int(np.ceil(total_rows / max_row_groups))
    rows_per_group = max(min_rows, target_bytes // ROW_BYTES_ESTIMATE)
    n_written = 0
    n_groups = 0
    pending = []
    pending_rows = 0
    granule_rows = []

    with pa.OSFile(tmp_file, "wb") as sink:
        with pq.ParquetWriter(sink, schema,
                              sorting_columns=[pq.SortingColumn(schema.get_field_index("cell_id"))],
                              compression="zstd", compression_level=COMPRESSION_LEVEL,
                              use_dictionary=dictionary, use_byte_stream_split=False, column_encoding=encoding,
                              write_statistics=sc.FOOTER_STATS,
                              **({"bloom_filter_options": bloom_filters} if bloom_filters else {})) as writer:

            def flush(final=False):
                nonlocal pending, pending_rows, rows_per_group, n_written, n_groups
                while pending_rows >= rows_per_group or (final and pending_rows > 0):
                    table = pa.concat_tables(pending)
                    ids = table.column("cell_id").to_numpy()
                    end = min(rows_per_group, table.num_rows)
                    # extend to the end of the last level-18 cell so no cell spans two row groups (a level-18
                    # cell never continues into the next level-7 cell, so the pending tail is always complete)
                    end = int(np.searchsorted(ids, ids[end - 1], side="right"))
                    before = sink.tell()
                    group = table.slice(0, end)
                    gid, count = np.unique(group.column("granule_id").to_numpy(), return_counts=True)
                    granule_rows.append((gid, np.full(gid.size, n_groups, np.int32), count.astype(np.int32)))
                    writer.write_table(group, row_group_size=end)
                    written = max(1, sink.tell() - before)
                    rows_per_group = max(min_rows, int(end * target_bytes / written))
                    n_written += end
                    n_groups += 1
                    rest = table.slice(end)
                    pending = [rest] if rest.num_rows else []
                    pending_rows = rest.num_rows

            for cell_dir, files in zip(cell_dirs, files_by_cell):
                table = pa.concat_tables([pq.read_table(f, schema=sc.arrow_schema(sensor, resolution)) for f in files])
                table = table.select(order).cast(schema)
                # stable sort by level-18 cell, then time, so repeat passes over a cell stay in time order
                idx = pc.sort_indices(table, sort_keys=[("cell_id", "ascending"), ("delta_time", "ascending")])
                table = table.take(idx)
                x, y = geometry_xy(table.column("geometry"))
                bbox = [min(bbox[0], x.min()), min(bbox[1], y.min()), max(bbox[2], x.max()), max(bbox[3], y.max())]
                pending.append(table)
                pending_rows += table.num_rows
                flush()
                if verbose:
                    print(f"{os.path.basename(cell_dir)}: {n_written}/{total_rows} rows, {n_groups} row groups")
            flush(final=True)

            processing = {}
            if granule_commits is not None:
                gids = np.unique(np.concatenate([g for g, _, _ in granule_rows])) if granule_rows else []
                processing = {"processing_commits": sorted({str(granule_commits.get(int(g))) for g in gids})}
            file_metadata = {
                "geo": json.dumps(_geo_metadata(bbox, region)),
                "altimetry_data": json.dumps({
                    "format_version": FORMAT_VERSION,
                    "sensor": sensor, "resolution": resolution, "region": region, "year": int(year),
                    "crs": f"EPSG:{REGIONS[region]['epsg']}",
                    "geographic_frame": GEOGRAPHIC_FRAME,
                    "geometry": "x, y: EPSG projection of ITRF2020 latitude and longitude; no lat/lon columns",
                    "vertical_reference": "WGS84 ellipsoid, ITRF2020",
                    "time_reference": "delta_time: GPS seconds since 2018-01-01T00:00:00 UTC",
                    "data_cell_level": DATA_LEVEL, "partition_cell_level": partition_level(sensor),
                    "quantum": QUANTUM,
                    "row_count": int(n_written), "row_groups": int(n_groups),
                    "granule_index": os.path.relpath(index_path(root, sensor, resolution, region, year),
                                                     os.path.dirname(out_file)),
                    "code": provenance.code_info(),
                    **processing,
                    **(extra_metadata or {}),
                    "columns": sc.column_info(sensor, resolution),
                }),
                "gdal:schema": json.dumps(_gdal_schema(sensor, resolution)),
            }
            writer.add_key_value_metadata(file_metadata)
    os.replace(tmp_file, out_file)
    write_file_index(granule_rows, index_path(root, sensor, resolution, region, year),
                     {"sensor": sensor, "resolution": resolution, "region": region, "year": int(year),
                      "data_file": os.path.relpath(out_file, os.path.join(root, sensor, resolution)),
                      "code": provenance.code_info()})
    return out_file


GRANULE_INDEX_COLUMNS = {
    "granule_id": ("granule (key of granules.parquet)", pa.uint32()),
    "region": ("region of the data file", pa.string()),
    "year": ("year of the data file", pa.int16()),
    "row_group": ("row group of the data file holding points of the granule", pa.int32()),
    "n_rows": ("number of the granule's points in that row group", pa.int32()),
}


def _index_table(columns):
    fields = [pa.field(n, t, metadata={"description": d}) for n, (d, t) in GRANULE_INDEX_COLUMNS.items()
              if n in columns]
    return pa.table({f.name: pa.array(columns[f.name], type=f.type) for f in fields},
                    schema=pa.schema(fields))


def write_file_index(granule_rows, out_file, metadata):
    """write a data file's granule index, sorted by granule_id then row_group"""
    if granule_rows:
        gid, rg, n = (np.concatenate(v) for v in zip(*granule_rows))
    else:
        gid, rg, n = np.zeros(0, np.uint32), np.zeros(0, np.int32), np.zeros(0, np.int32)
    order = np.lexsort((rg, gid))
    table = _index_table({"granule_id": gid[order], "row_group": rg[order], "n_rows": n[order]})
    table = table.replace_schema_metadata({"altimetry_data": json.dumps(metadata)})
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    pq.write_table(table, out_file + ".tmp", compression="zstd", write_statistics=["granule_id"])
    os.replace(out_file + ".tmp", out_file)
    return out_file


def write_granule_index(root, sensor, resolution):
    """
    combine the per-file granule indexes of a database into granule_index.parquet (granule_id, region, year,
    row_group, n_rows), sorted by granule_id with footer statistics on it, so one small read finds every row
    group of every file that holds a granule's points
    """
    parts = []
    for f in sorted(glob.glob(os.path.join(root, sensor, resolution, "index", "*_granules.parquet"))):
        t = pq.read_table(f)
        meta = json.loads(t.schema.metadata[b"altimetry_data"])
        parts.append(t.to_pandas().assign(region=meta["region"], year=meta["year"]))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True).sort_values(["granule_id", "year", "region", "row_group"])
    table = _index_table({k: df[k].to_numpy() for k in GRANULE_INDEX_COLUMNS})
    table = table.replace_schema_metadata({"altimetry_data": json.dumps(
        {"sensor": sensor, "resolution": resolution, "code": provenance.code_info(),
         "data_files": "{sensor}/{resolution}/data/{sensor}_{resolution}_{region}_{year}.parquet"})})
    out = granule_index_path(root, sensor, resolution)
    pq.write_table(table, out + ".tmp", compression="zstd", write_statistics=["granule_id"],
                   row_group_size=65536)
    os.replace(out + ".tmp", out)
    return out


# ---------------------------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------------------------

def build_stats(data_file, out_file, levels=STAT_LEVELS, min_rows=MIN_STAT_ROWS):
    """
    per-cell statistics at each S2 level in `levels`: row_count, row_group_start/end, and min, max, and mean
    of every numeric column, of the geometry's x and y (x_*, y_*), and of latitude and longitude from the inverse
    projection (lat_*, lon_*, as in atl06t).  One row group per level, sorted by cell_id; the level and cell_id
    columns carry footer statistics.  Levels finer than the first one whose cells average fewer than `min_rows`
    rows are left out (the coarsest level is always kept), so sparse data such as ICESat tracks do not get a
    stats file as large as the data.
    """
    pf = pq.ParquetFile(data_file)
    region = json.loads(pf.metadata.metadata[b"altimetry_data"])["region"]
    numeric = [f.name for f in pf.schema_arrow if f.name not in sc.NO_CELL_STATS and
               (pa.types.is_integer(f.type) or pa.types.is_floating(f.type))]
    columns = ["lat", "lon", "x", "y", *numeric]
    partials = {level: [] for level in levels}
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["cell_id", "geometry", *numeric])
        x, y = geometry_xy(table.column("geometry"))
        lon, lat = unproject(x, y, region)
        df = pd.concat([pd.DataFrame({"lat": lat, "lon": lon, "x": x, "y": y}),
                        table.select(numeric).to_pandas()], axis=1)
        ids = table.column("cell_id").to_numpy()
        for level in levels:
            grouped = df.groupby(s2.parent(ids, level), sort=False)
            size = grouped.size()
            partials[level].append({"sum": grouped.sum(min_count=1), "count": grouped.count(),
                                    "min": grouped.min(), "max": grouped.max(),
                                    "meta": pd.DataFrame({"row_count": size, "row_group": rg}, index=size.index)})
    out_tables = []
    for level in levels:
        if not partials[level]:
            continue
        parts = {key: pd.concat([p[key] for p in partials[level]]) for key in partials[level][0]}
        sums = parts["sum"].groupby(level=0, sort=True).sum(min_count=1)
        counts = parts["count"].groupby(level=0, sort=True).sum()
        mins = parts["min"].groupby(level=0, sort=True).min()
        maxs = parts["max"].groupby(level=0, sort=True).max()
        meta = parts["meta"].groupby(level=0, sort=True)
        cell_id = np.asarray(sums.index, dtype=np.uint64)
        stats = {"cell_id": cell_id, "level": np.full(cell_id.size, level, dtype=np.uint8)}
        for column in columns:
            name = column
            stats[f"{name}_min"] = mins[column].to_numpy()
            stats[f"{name}_max"] = maxs[column].to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                stats[f"{name}_mean"] = (sums[column] / counts[column].where(counts[column] > 0)).to_numpy()
        stats["row_count"] = meta["row_count"].sum().to_numpy().astype(np.int64)
        stats["row_group_start"] = meta["row_group"].min().to_numpy().astype(np.int32)
        stats["row_group_end"] = meta["row_group"].max().to_numpy().astype(np.int32)
        if out_tables and stats["row_count"].mean() < min_rows:
            break
        out_tables.append(pa.table(stats))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    meta = json.loads(pf.metadata.metadata[b"altimetry_data"])
    stats_meta = {
        "kind": "per-cell statistics",
        "data_file": os.path.basename(data_file),
        "sensor": meta["sensor"], "resolution": meta["resolution"], "region": region, "year": meta["year"],
        "levels": [int(t.column("level")[0].as_py()) for t in out_tables],
        "description": ("one row group per S2 level, sorted by cell_id.  For each cell: row_count, the data-file "
                        "row groups holding its rows (row_group_start..row_group_end), and {column}_min, _max, "
                        "_mean of each numeric column over the cell's rows; x and y come from the geometry, lat "
                        "and lon from its inverse projection (ITRF2020, degrees)"),
        "code": provenance.code_info(),
    }
    schema = out_tables[0].schema.with_metadata({"altimetry_data": json.dumps(stats_meta)})
    with pq.ParquetWriter(out_file + ".tmp", schema, compression="zstd",
                          write_statistics=["level", "cell_id"]) as writer:
        for t in out_tables:
            writer.write_table(t.cast(schema), row_group_size=max(1, t.num_rows))
    os.replace(out_file + ".tmp", out_file)
    return out_file
