# altimetryData

This builds GeoParquet databases of airborne and satellite laser altimetry over Greenland and Antarctica: NASA
ATM (wide- and narrow-swath), ICESat/GLAS, and LVIS.  Heights are on the WGS84 ellipsoid in ITRF2020, so they
compare directly with ICESat-2 release 007 and later.

There is one file per sensor, resolution (native / reduced), region (GL / AA), and year, in the layout of
sliderule's atl06t product.  The geometry is a GeoParquet native point in EPSG:3413 (Greenland) or EPSG:3031
(Antarctica), and positions are stored only once (no latitude/longitude columns).  The full specification is in [docs/database_format.md](docs/database_format.md).
The database lives at `/Volumes/ice1/ben/oib_database/parquet`.

## Pipeline

```bash
pip install -e .            # numpy, numba, pandas, pyarrow, pyproj, h5py, requests (+ geopandas to read)

# 1. granule inventory from CMR (public; ~15 min, resumable).  Granule ids are frozen: rerunning (with --refresh
#    to re-list CMR) keeps every existing id and numbers new granules after the largest one
python scripts/build_inventory.py --out /Volumes/ice1/ben/oib_database/parquet_work/inventory/granules.parquet

# 2. download + process granules into partitions (resumable; Earthdata login from ~/.netrc or EARTHDATA_TOKEN)
python scripts/process_granules.py --granules .../inventory/granules.parquet --workers 96 [--sensors ATM_wide --years 2011]

# 3. build the data and stats files, manifests, and the lookup table
python scripts/build_database.py --granules .../inventory/granules.parquet --workers 8

# estimate volumes and times from the inventory
python scripts/estimate.py --granules .../inventory/granules.parquet --download_MBps 85 --cores 96
```

* `process_granules.py` downloads each granule into its own directory, reads it, and applies the frame and
  ellipsoid change.  It then splits the points by region, projects them, assigns S2 cells, and reduces them
  (ATM, LVIS).  It writes partitions, deletes the raw file, and records a status file.  Granules with a status
  file are skipped on rerun.  `--transport s3` reads the NSIDC bucket directly; it works only from AWS
  us-west-2 (e.g. NASA MAAP).
* Both scripts refuse to run from uncommitted or unpushed code (override with `--allow_dirty`).  Every file
  records the repository, commit, and commit URL that made it, and the commits that processed its granules
  (see Provenance in the format spec).
* Each data file's columns carry descriptions and units (Arrow field metadata, the `altimetry_data` metadata,
  and GDAL's `gdal:schema`, so they show in QGIS).
* `altimetryData.read.read_granule(root, granule_id)` reads every point of one raw granule through the granule
  index, which lists the row groups holding each granule.
* `build_database.py` builds only those sensor/years whose granules have all been processed, so a partial run
  cannot overwrite a year with a subset.  It deletes the partitions it has built from unless you pass
  `--keep_partitions`.
* `scripts/reduce_ATM.py` is a drop-in replacement for `pointCollection/scripts/reduce_ATM.py`: same arguments,
  same indexedH5 output, about 40× faster.

## Status

* Inventory: 104,342 granules from 10 collections, 2.46 TB, all in NSIDC Earthdata Cloud (us-west-2).
* **ICESat is complete** (format version 2): 637 granules and 191 M shots, 2003–2009, both regions.  That is
  6.2 GB of data (32.5 B/row) plus 0.9 GB of stats, built from 16.4 GB of raw GLAH12.  Download and
  processing took 2.5 min and the build took 3 min.
* ATM and LVIS have not been run yet (see estimates).

## Estimates (measured on this machine, 96 cores, 85 MB/s from NSIDC)

| | granules | raw | native pts | native parquet | reduced parquet | download | compute |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ATM_wide | 66,376 | 1.64 TB | 55 G | 0.86 TB | 0.21 TB | 5.4 h | 60 core-h |
| ATM_narrow | 33,025 | 0.37 TB | 19 G | 0.29 TB | 0.07 TB | 1.2 h | 19 core-h |
| LVIS | 4,047 | 0.43 TB | 2 G | 0.03 TB | 0.02 TB | 1.4 h | 3.5 core-h |
| ICESat | 637 | 0.02 TB | 0.2 G | 0.01 TB | — | 3 min (measured) | 0.4 core-h |
| **total** | **104,085** | **2.45 TB** | **76 G** | **1.18 TB** | **0.31 TB** | **8 h** | **82 core-h** |

* **Download is the bottleneck locally.** The link to NSIDC saturates at 85–95 MB/s regardless of the stream
  count (32 streams: 2.7 MB/s each; 64 streams: 1.5 MB/s each).  The ICESat run sustained 85–97 MB/s.
* **Processing** runs at 2.4–4.6 µs per point on one core, so 82 core-hours is under 1 h on 96 cores, fully
  hidden behind the download.
* **Building** costs about 2.6 µs/row (1.1 for the data file, 1.5 for the stats file), or 62 core-hours.  Files
  build in parallel, so the wall time is set by the largest file, ATM_wide GL 2017 (7 G points, about 110 GB):
  roughly 5 h.
* **Total locally: about 12–13 h.**  Peak disk is about 1.5 TB of output plus partitions for the years in
  flight; /Volumes/ice1 has 15 TB free.
* **Cloud (MAAP / us-west-2)** would cut the download to about 1 h at in-region S3 rates (assumed, not measured
  from here).  The 1.5 TB result would then have to come back to /Volumes/ice1, which takes about 5 h at the
  same 85 MB/s (plus AWS egress), so moving the processing saves a few hours at most unless the database's home
  is cloud object storage.

### File size

Every database is now smaller than its raw data: native ATM is 15.6 B/row, against 48 B/shot in qi and about
16 B/shot in gzip-compressed v2 HDF5.  How it got there (ATM native, B/row):

* 49.5: first version (WKB lon/lat point plus latitude, longitude, x, and y columns; float64 mantissas in full);
* 30.8: x, y, lat/lon, and times rounded to sub-mm / sub-ms binary grids, and zstd level 9;
* 15.9: position stored once as a native point in EPSG:3413/3031 (−10.6), and `source_index` delta-encoded
  instead of dictionary-encoded (−4.1);
* 15.6: the `year` column dropped (it is `2018 + delta_time / (365.25 × 86400)`).

Position (x and y) is now about 4.6 B/row, close to the information content of a point inside a 36 m cell at
0.5 mm resolution.

**S2 sort level (kept at 18).**  Rows are sorted by level-18 cell (about 36 m), then time.  Sorting by a
coarser cell keeps longer along-track runs in time order, so time, attitude, and `source_index` compress better.
Measured on single granules (B/row, with `cell_id` stored at the sort level):

| finest sort level | 18 (now) | 16 | 14 | 12 |
|:--|--:|--:|--:|--:|
| ATM 2016 | 15.7 | 13.4 | 12.6 | 12.3 |
| LVIS 2017 | 21.2 | 19.0 | 18.4 | 18.2 |
| ICESat 2005 | 31.5 | 31.1 | 28.8 | 25.2 |

## Validation done

* S2 ids match duckdb's geography extension (used by sliderule) for 2.2 M random and edge-case points; also
  covered in tests.
* The reduction selects the same points as `pointCollection.pt_blockmedian` and reproduces
  `reduce_ATM`/`calc_bias_and_R` statistics to ≤1e-10 on a real ILATM1B v2 granule.  The reduce step went from
  24.5 s to 0.57 s.
* The qi reader matches `pointCollection.ATM_Qfit.from_qi` for 1993, 1998, 2002, 2006, 2009, and 2011 files, at
  about 100× the speed (0.2 s vs 20 s per file).
* ITRF transforms match `pointCollection.tools.convert_ITRF` exactly for 2014→2020.  TOPEX→WGS84 matches GLAS
  `d_deltaEllip` to 0.5 mm.
* ICESat rows trace back through `granule_id`/`source_index` to a re-downloaded raw granule read directly with
  h5py: h to 0.12 mm (float32), x/y to 0.49 mm (the rounding grid), time to 0.03 ms.
* Data files read with geopandas (`bbox=` filter, cell_id index, CRS EPSG:3413/3031), pyarrow (filters on
  `geometry.x`/`geometry.y` prune row groups), duckdb, and GDAL (`-spat`).  All four return the same points
  for a test box.  `cell_id` recomputed from the stored geometry matches exactly.
* Run the tests with `pytest`.

## Things to verify / decisions made

* **ICESat frame:** GLAH12 r34 is taken to be ITRF2008 (product documentation); the granules carry no frame
  attribute.
* **LVIS frame:** the granules state none, so the IGS frame of the collection date is used: ITRF2005 for 2007,
  ITRF2008 for 2009–2015, ITRF2014 for 2017.  Recorded as `igs_default`; worth confirming with the LVIS team.
* **GLAS time:** `d_UTCTime_40` is treated as UTC without leap seconds, a ±1 s ambiguity after the 2005 and 2008
  leap seconds.
* **LVIS v2:** the three unnamed trailing columns are read as CHANNEL_ZT, CHANNEL_ZG, CHANNEL_RH.
* **BLATM1B:** all pre-IceBridge ATM is classed as `ATM_wide`.
* **Regions:** GL is everything ≥59°N, so it includes Arctic sea ice, the Canadian Arctic, and Alaska; AA is
  everything ≤60°S.
* **ICESat corrections:** the saturation correction is applied.  G−C (`gmc`) and the ocean tide are stored but
  not applied.

## Problems found in pointCollection (not changed here)

* `tools/calc_bias_and_R.py` uses `np.in1d` and `ATM_Qfit.from_qi` uses binary `np.fromstring`.  Both were
  removed in numpy 2.x, so `reduce_ATM.py` fails in the current `devel` env.
* `ATM_Qfit.from_qi` stores latitude, longitude, and time as float32 (about 0.5 m and 10 ms of error).
* `ATM_Qfit.from_h5` dates shots from the file-name time plus `rel_time`.  The file-name time is not the time of
  the first shot (28 s off for ILATM1B_20161020_185958), and `time_hhmmss` (GPS) is the documented tag.
* `tools/convert_ITRF.py` has the wrong sign on the 2008→2020 Y-rate and uses `t_epoch=2000` (IERS: 2015) for
  2005→2020, a mm-level error.
* `scripts/reduce_ATM.py`: `--out_name` with `--out_dir` joins the input file name instead of `out_name`.
