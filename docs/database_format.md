# OIB / ICESat GeoParquet database format

The format follows sliderule's Tiled ATL06 product (`sliderule/targets/atl06t/README.md`) with the changes listed
under [Differences from atl06t](#differences-from-atl06t).

## Layout and naming

```
{root}/granules.parquet                                                    granule lookup table
{root}/{sensor}/{resolution}/data/{sensor}_{resolution}_{region}_{year}.parquet
{root}/{sensor}/{resolution}/stats/{sensor}_{resolution}_{region}_{year}_stats.parquet
{root}/{sensor}/{resolution}/index/{sensor}_{resolution}_{region}_{year}_granules.parquet   per-file granule index
{root}/{sensor}/{resolution}/granule_index.parquet                         all files' granule indexes
{root}/{sensor}/{resolution}/manifest.json
```

* `root` is `/Volumes/ice1/ben/oib_database/parquet`.
* `sensor`: `ATM_wide` (BLATM1B, ILATM1B), `ATM_narrow` (ILNSA1B), `ICESat` (GLAH12 r34), or `LVIS` (BLVIS2,
  ILVIS2, ILVGH2).
* `resolution`: `native` (every valid point) or `reduced` (block-median decimated points with local plane-fit
  statistics; ATM and LVIS only).
* `region`: `GL`, every point at or north of 59°N, projected to EPSG:3413; or `AA`, every point at or south of
  60°S, projected to EPSG:3031.  GL therefore also contains Arctic sea ice, Canadian Arctic, and Alaska flights.
* `year`: the UTC calendar year of the point.

Intermediate files (partitions, per-granule status, transient downloads) are kept under
`/Volumes/ice1/ben/oib_database/parquet_work`.

## Data files

Each data file is a GeoParquet 1.1 file:

* Rows are sorted by `cell_id`, the S2 level-18 cell (about 30 m) of the point.  Within a cell, rows are sorted
  by `delta_time`.  `cell_id` is the pandas index.
* Row groups target 8 MB.  They grow as needed so a file never has more than 5000 row groups, and no
  level-18 cell spans two row groups.
* Geometry: a GeoParquet 1.1 native point (`encoding: "point"`), stored as `geometry: struct<x: double, y:
  double>`.  The CRS is EPSG:3413 for GL and EPSG:3031 for AA.  x and y are those projections of the ITRF2020
  latitude and longitude.  The files carry the EPSG projected CRS as requested; its WGS84 datum agrees with
  ITRF2020 at the centimeter level.  Position is stored only once: there are no latitude, longitude, or
  separate x/y columns.
* Only `cell_id`, `delta_time`, `geometry.x`, `geometry.y`, and `granule_id` have footer (row-group)
  statistics.  A reader can therefore skip row groups by S2 cell range, time, a projected x/y box, or granule
  without reading the data.  For a lon/lat box, use the S2 cell ranges that cover it, or the box's projected
  bounds.
* The common columns are written first in each row group: `cell_id`, `geometry`, `delta_time`, `h`,
  `granule_id`, and, in reduced files, `bias`, `noise`, `n_fit`, `slope_x`, and `slope_y`.
* Compression: zstd level 9.  Floats, including `geometry.x` and `geometry.y`, use byte-stream-split encoding.
  `source_index` and `shotnumber` are delta-encoded: within a cell they rise steadily, so they cost under a byte
  per row, where dictionary encoding cost more than the raw integer.  The other integer columns are
  dictionary-encoded.
* x, y, and the times are rounded to binary grids far finer than any measurement precision, so their low
  mantissa bits compress away.  The largest change is 0.5 mm in x or y and 0.03 ms in `delta_time` (see
  `quantum` in the file metadata).
* `cell_id` is computed from the inverse projection of the stored (rounded) x and y, so it can be recomputed
  exactly from the geometry.
* File metadata key `altimetry_data` (JSON) records the sensor, region, year, CRS, frames, time reference, S2
  levels, row count, quantization, (for reduced files) the reduction parameters, the code provenance (below),
  the path of the file's granule index, and a description, units, and type for every column.

### Column descriptions

Parquet has no standard place for a column's description or units, and GeoParquet covers only the geometry
column.  Every data file therefore carries the descriptions (from `schema.DESCRIPTIONS`) in three places, each
seen by different readers:

| where | read by |
|:--|:--|
| Arrow field metadata (`description`, `units` on each field, and on `geometry.x` / `geometry.y`) | pyarrow: `pq.read_schema(f).field("h").metadata` |
| `columns` in the `altimetry_data` file metadata | any Parquet reader, e.g. duckdb `parquet_kv_metadata()` |
| GDAL's `gdal:schema` file metadata (`comment` per column) | GDAL/OGR and QGIS (`ogrinfo` shows `comment=...`) |

`manifest.json` holds the same table.  geopandas and pandas do not keep any of them.

### Provenance

Every data, stats, and index file records `code` in its `altimetry_data` metadata: the repository URL, the
commit, a link to the commit, the branch, and whether the working tree differed from the commit (`dirty`).  Data
files also list `processing_commits`, the commits that processed their granules (from the per-granule status
files; also the `code_commit` column of `granules.parquet`).  `process_granules.py` and `build_database.py`
refuse to run from uncommitted or unpushed code unless given `--allow_dirty`, in which case the commit is
recorded with `dirty: true` (and a `-dirty` suffix in `processing_commits`).

### Reading

```python
import geopandas as gpd                          # geopandas >= 1.0 reads native point encoding
g = gpd.read_parquet(f, bbox=(xmin, ymin, xmax, ymax))   # GeoDataFrame, cell_id index, CRS EPSG:3413/3031

import pyarrow.dataset as ds, pyarrow.compute as pc
x = pc.field("geometry", "x"); y = pc.field("geometry", "y")
t = ds.dataset(f).to_table(filter=(x > xmin) & (x < xmax) & (y > ymin) & (y < ymax))   # prunes row groups

from altimetryData import database as db
lon, lat = db.unproject(*db.geometry_xy(t.column("geometry")), "GL")   # ITRF2020 latitude and longitude
```

* duckdb: `SELECT geometry.x, geometry.y, h FROM 'f.parquet' WHERE geometry.x BETWEEN ...`.  With the spatial
  extension, use `ST_Point(geometry.x, geometry.y)` for a geometry value.
* GDAL/OGR (3.8 and later) read the files as Point layers with the right CRS, and `-spat` filters use the
  footer statistics.  OGR has no unsigned 64-bit type, so it shows `cell_id` as a double, which loses
  precision.

### Columns common to every database

| Column | Type | Description |
|:--|:--|:--|
| cell_id | uint64 | S2 level-18 cell id (index; footer stats) |
| geometry | struct<x, y: float64> | native point, EPSG:3413 (GL) or EPSG:3031 (AA), m (footer stats on x and y) |
| delta_time | float64 | GPS seconds since 2018-01-01T00:00:00 UTC, as in ICESat-2 (footer stats) |
| h | float32 | height above the WGS84 ellipsoid in ITRF2020, m |
| dh_frame | float32 | h minus the height in the source frame and ellipsoid (the correction applied) |
| granule_id | uint32 | row of `granules.parquet`, the raw source granule |
| source_index | uint32 | 0-based record number of the point in the raw granule (delta-encoded) |

`delta_time` counts GPS seconds, so differences are true elapsed time across leap seconds.  For a pre-2018 point it
is up to 13 s more negative than the UTC difference from 2018-01-01.

### ATM (`ATM_wide`, `ATM_narrow`)

Native: `azimuth`, `rcv_sigstr`, `xmt_sigstr`, `pitch`, `roll`, `gps_pdop`, `pulse_width`, as in the Qfit
granules.  `gps_pdop` and `pulse_width` are absent (NaN or null) for 10-word qi records.

Reduced: `bias`, `noise`, `n_fit`, `slope_x`, `slope_y` (see [Reduction](#reduction)), plus `rcv_sigstr` and
`azimuth` of the kept point.

### ICESat (`ICESat`, native only)

`h` is `d_elev + d_satElevCorr` (where the saturation correction is valid), moved from TOPEX/Poseidon and
ITRF2008 to WGS84 and ITRF2020.  The other columns are:

| Column | Source | Note |
|:--|:--|:--|
| sat_elev_corr | d_satElevCorr | already added to h |
| gmc | d_GmC | Gaussian-minus-centroid offset, not applied |
| ocean_tide | d_ocElv | as stored; 0 over grounded ice; not applied or removed |
| ice_svar, reflct_uc, tx_energy | d_IceSVar, d_reflctUC, d_TxNrg | |
| num_pk, gain, sat_ndx | i_numPk, i_gval_rcv, i_satNdx | |
| elev_use_flg, sigma_att_flg, sat_corr_flg, cloud_flg | quality flags | cloud_flg is elv_cloud_flg |
| campaign | ANCILLARY_DATA Campaign | 1-based index into 1A, 2A, 2B, 2C, 3A–3K, 2D, 2E, 2F (pointCollection.glah12 convention) |
| laser | | 1, 2, or 3 |
| track | Data_1HZ i_track | reference track |

No quality filtering is applied.  Cloud returns are present (for example, heights several hundred meters above
the surface near Summit); filter with the flags.

### LVIS (`LVIS`)

`h` is the center of the lowest mode: `ELEVATION_LOW` in ILVIS2 v1 and ILVGH2, `ZG` in ILVIS2 v2 and BLVIS2.  The
v2 header states that these are the same quantity.

Native: `z_high` (center of the highest mode, ELEVATION_HIGH / ZH), `z_top` (highest detected return, ZT; v2
only), and `z_centroid` (v1 only).  All three are corrected like `h`.  Also `rh50`, `range`, `incident_angle`,
`azimuth`, `complexity`, `channel_zg` (v2 only, where present), `lfid`, and `shotnumber`.  `lfid` is the LVIS
file id: digits 3–7 are the Modified Julian Date of the flight and the last three count the instrument's files.
`lfid` with `shotnumber` identifies a shot uniquely.

Reduced: fit statistics, plus `z_high`, `lfid`, and `shotnumber`.

## Reduction

The reduced products are made by `altimetryData.reduce`, a re-implementation of `pointCollection/scripts/reduce_ATM.py`
that selects identical points and reproduces the statistics to ~1e-10 m:

1. Block-median decimation on a `decimate_to` grid in projected coordinates.  Each bin keeps its median-height
   point; for even counts, the central point closer to the bin center wins.
2. For each kept point, a plane is fit to the kept points within `stats_radius`, with a weak slope damping and
   up to six rounds of three-sigma editing (sigma from the 16–84% spread, floored at `min_sigma`):
   * `bias` is the point minus the fit;
   * `noise` is the rms of the edited residuals;
   * `n_fit` is the number of points kept;
   * `slope_x` and `slope_y` are the plane slopes.

   Points with five or fewer neighbors within the radius get NaN.

| Sensor | decimate_to | stats_radius | min_sigma |
|:--|:--|:--|:--|
| ATM_wide, ATM_narrow | 10 m | 50 m | 0.5 m |
| LVIS | 20 m (≈ the 2007–2015 footprint diameter) | 50 m | 0.5 m |

## Reference frames and ellipsoids

Every point is moved from its source frame and ellipsoid to WGS84 heights in ITRF2020 at the epoch of the
measurement.  The chain is geodetic → ECEF → Helmert (IERS parameters from PROJ's `ITRF2020` init file) →
geodetic on WGS84.

The source frame of each granule is recorded in `granules.parquet` (`frame`, `frame_source`):

| Sensor | Source frame | frame_source |
|:--|:--|:--|
| ATM HDF5 (2013–2019) | `ancillary_data/reference_frame` in the granule | `granule` |
| ATM qi (1993–2012) | the `itrfNN` tag of the trajectory named in the processing header, e.g. `..._itrf05_...` | `header` |
| ICESat GLAH12 r34 | ITRF2008, TOPEX/Poseidon ellipsoid (TOPEX→WGS84 agrees with d_deltaEllip to 0.5 mm) | `product` |
| LVIS ILVIS2 v1 (2009-2015), ILVGH2 v1 (2013) | ITRF2000, WGS84 ellipsoid (NSIDC user guides, "Projection and Grid Description") | `user_guide` |
| LVIS ILVIS2 v2 (2017) | ITRF2008 (user guide Table 3: EPSG:5332) | `user_guide` |
| LVIS BLVIS2 (2007) | ITRF2000, assumed: no user guide, processed in 2008 with the ILVIS2 v1 system | `assumed (as ILVIS2 v1)` |
| any granule without the above | IGS frame on the date | `igs_default` |

## Granule lookup table (`granules.parquet`)

The table has one row per granule of every collection, keyed by `granule_id`.  The ids are frozen.  A
refreshed CMR listing keeps every existing id, numbers new granules after the largest existing id, and keeps
granules that have left CMR with `in_cmr = False`, so an id is never reused or renumbered.  It holds:

* the CMR metadata: short name, version, granule name, concept id, start and end time, bounding latitudes and
  longitudes, size, HTTPS URL, and S3 URL;
* `year` and `regions`;
* for processed granules: `frame`, `frame_source`, `ellipsoid` (ICESat), `transceiver` (ATM, e.g. T4),
  `campaign` (ICESat), `n_records`, `n_valid`, `n_native_{region}`, and `n_reduced_{region}`.

A point's raw record is `granules.parquet[granule_id]` → `https_url`, and record `source_index` in that granule.

## Reading one granule

Rows are sorted by S2 cell, so a granule's points are spread over many row groups of a data file (or of two
files, if its track crosses both regions or runs past the end of a year).  Two structures make reading one
granule cheap:

* **Granule index.**  Each data file has an index (`index/..._granules.parquet`) with one row per
  (`granule_id`, `row_group`) and the number of the granule's rows in that row group.  `granule_index.parquet`
  combines them for a whole database (adding `region` and `year`), sorted by `granule_id`, with footer
  statistics on it.  One small read gives exactly the row groups to read.
* **`granule_id` footer statistics.**  Readers that do not use the index (duckdb, a plain pyarrow filter) can
  still skip row groups whose `granule_id` range excludes the granule.  The range is coarse where flights
  overlap, so this reads more row groups than the index.

```python
from altimetryData import read
tables = read.read_granule(root, granule_id, resolution="native", columns=["geometry", "h", "source_index"])
# {"GL": pyarrow.Table, ...}: one table per region, since the regions have different projections
```

`read_granule` scans only the indexed row groups, filtering on `granule_id`, on at most `threads` (default 4)
threads.  Measured on full-resolution files, cold cache:

| granule | points | index | footer statistics only |
|:--|--:|--:|--:|
| ATM_wide 2009 (median size) | 1.1 M | 0.15-0.25 s (7 of 64 row groups) | 0.25-0.5 s (29) |
| ATM_wide 1993 (largest) | 10.4 M | 0.4-0.6 s (41 of 103) | 0.4 s (41) |
| ATM_narrow 2012 | 1.2 M | 0.2 s (4 of 1085) | 2.1-2.9 s (200) |
| LVIS 2007 | 4.5 M | 0.3-0.45 s (15 of 16) | 0.3 s (15) |
| ICESat 2005 (orbit across the ice sheet) | 0.3 M | 0.5 s (123 of 127) | 0.5 s (127) |

A short ATM or LVIS granule is found in a few row groups.  An ICESat granule crosses most of the ice sheet,
so reading it means scanning its year file.

## Reading a box

`altimetryData.read.read_box(root, sensors, resolution, region, bds=[[xmin, xmax], [ymin, ymax]], columns=None,
threads=4)` returns every point inside a projected box (EPSG:3413 for GL, EPSG:3031 for AA; or give
`bbox=(xmin, ymin, xmax, ymax)` instead of `bds`), from every year of one or more databases, as one `pointCollection.data` object.
Its fields are x, y, the requested columns (all by default), and `sensor`, the index into `sensors`; columns a
sensor lacks are NaN.  The reader works in three steps:

1. **Find the row groups.**  For each year file, read the finest level of its stats file and keep the cells whose
   points' x/y range meets the box; their `row_group_start`..`row_group_end` give the row groups to read.  This
   reads 13% fewer row groups than the `geometry.x/y` footer statistics, because a cell's range is tighter than
   a row group's.
2. **Read and decode them.**  Files are read on a pool of `threads` threads with pyarrow's own threading off, so
   `threads` caps the cores used.
3. **Copy the points in the box.**  Each selected value is copied once, straight from the Arrow buffers into
   preallocated output arrays, as one task per (row group, column).

Measured for a 40 x 40 km box at Jakobshavn: 42.5 M reduced points from 34 year files of ATM_wide,
ATM_narrow, and LVIS, 1993-2019, 4 threads.

| | first read (cold) | later reads (cached) |
|:--|--:|--:|
| all 19 fields | 5.3 s | 3.3 s |
| x, y, h, delta_time, bias, noise, n_fit, slope_x, slope_y, sensor | 5.2 s | 2.2 s |
| duckdb, 4 threads (numpy arrays only, for comparison) | 7.0 s | 3.3 s |

* About 1.65 rows are decoded per point returned.  Row groups of 4 MB or 2 MB cut that to 1.45 or 1.33 and
  save 10-15% of the cached time.  But production files would have 2-4x more row groups, and parsing their
  footers costs about 14 µs per row group, so the data files keep 8 MB row groups.  LZ4 instead of zstd
  decodes about 10% faster and makes files 9% larger, so zstd is kept.
* Jakobshavn is about the densest 40 km box in the database.  Read time scales with the number of points.

## Statistics files

These follow atl06t, with one stats file per data file.  Each S2 level is a separate row group, sorted by
`cell_id`, and `level` and `cell_id` carry footer statistics.

* Columns: `cell_id`, `level`, `row_count`, `row_group_start`, `row_group_end`, and `{column}_min`, `_max`, and
  `_mean` for every numeric column, including `x_*` and `y_*` (from the geometry), `lat_*` and `lon_*` (from
  the inverse projection), `h_*`, and `delta_time_*`.  `granule_id`, `source_index`, `lfid`, and
  `shotnumber` are left out.
* Levels run from 7 down to 14.  A finer level is added only while the cells of the previous level average at
  least 50 rows, so ICESat files stop at level 11 instead of carrying a nearly row-per-shot level 14.

## Differences from atl06t

1. One file per sensor/resolution/region/year instead of one per S2 level-7 cell.
2. The geometry is a native point in projected x and y (EPSG:3413 / EPSG:3031), not a WKB lon/lat point in
   EPSG:9989.  There are no latitude or longitude columns.
3. Footer statistics on `geometry.x` and `geometry.y` in addition to the cell id and time.
4. The time column is `delta_time` (GPS seconds since 2018-01-01, as in ICESat-2), in place of
   `time_ns`.
5. Row groups grow beyond 8 MB when needed to keep a file to at most 5000 row groups.
6. Coordinates and times are rounded to sub-millimeter / sub-millisecond binary grids for compression, and
   steadily rising integers are delta-encoded.
7. Statistics levels are chosen per file (above) and stored in one file with a `level` column.
