#! /usr/bin/env python
"""
Assemble processed partitions into the database.

    python scripts/build_database.py --granules granules.parquet [--root /Volumes/ice1/ben/oib_database/parquet]
        [--work_root ROOT_work] [--sensors ...] [--years ...] [--workers 8] [--keep_partitions]

For every sensor/resolution/region/year with partitions this writes the optimized data file and its statistics
file, then writes {root}/granules.parquet (the inventory plus the per-granule processing results: reference
frame and its source, point counts) and a manifest.json per sensor and resolution.
"""

import argparse
import glob
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from altimetryData import database as db
from altimetryData import granule as gr
from altimetryData import exclusions, provenance
from altimetryData import schema as sc
from altimetryData.collections import REGIONS


def build_one(root, work_root, sensor, resolution, region, year, keep_partitions, granule_commits):
    t0 = time.perf_counter()
    extra = {"reduction": gr.REDUCE_PARAMS.get(sensor)} if resolution == "reduced" else {}
    out = db.build_database_file(root, sensor, resolution, region, year, work_root=work_root, extra_metadata=extra,
                                 granule_commits=granule_commits)
    t1 = time.perf_counter()
    db.build_stats(out, db.stats_path(root, sensor, resolution, region, year))
    t2 = time.perf_counter()
    if not keep_partitions:
        shutil.rmtree(db.partition_dir(work_root, sensor, resolution, region, year))
    return {"file": out, "build_s": t1 - t0, "stats_s": t2 - t1, "bytes": os.path.getsize(out)}


def granule_table(work_root, granules):
    """
    inventory plus processing results from the status files, with status = ok, failed (after its retries; error
    holds the last line of the traceback), excluded (exclusion_reason says why), or not processed
    """
    rows = []
    for f in glob.glob(os.path.join(work_root, "status", "*", "*.json")):
        with open(f) as fh:
            s = json.load(fh)
        if f.endswith(".failed.json"):
            s = {"granule_id": s["granule_id"], "status": "failed", "attempts": s.get("attempts"),
                 "code_commit": s.get("code_commit"), "error": (s.get("error") or "").strip().splitlines()[-1:]}
            s["error"] = s["error"][0] if s["error"] else None
        rows.append({k: v for k, v in s.items() if k not in ("timings", "partition_files", "granule", "sensor")})
    table = granules.copy()
    if rows:
        status = pd.DataFrame(rows).drop_duplicates("granule_id", keep="first")
        table = table.merge(status, on="granule_id", how="left")
    if "status" not in table:
        table["status"] = None
    table["status"] = table["status"].fillna("not processed")
    table["exclusion_reason"] = table["granule"].map(exclusions.EXCLUDED)
    table.loc[table["exclusion_reason"].notna(), "status"] = "excluded"
    return table


def incomplete(work_root, granules):
    """
    {(sensor, year): number of in-region granules not yet processed}.  Granules that failed after their retries
    (a .failed.json status) and excluded granules do not count: their years are built without them
    """
    g = granules[(granules["regions"].str.len() > 0) & ~granules["granule"].isin(set(exclusions.EXCLUDED))]
    done = {(os.path.basename(os.path.dirname(f)), int(os.path.basename(f)[:8]))
            for f in glob.glob(os.path.join(work_root, "status", "*", "*.json"))}
    todo = [(s, int(y)) for s, y, i in zip(g["sensor"], g["year"], g["granule_id"]) if (s, int(i)) not in done]
    return pd.Series(todo, dtype=object).value_counts().to_dict() if todo else {}


def write_manifest(root, sensor, resolution, table):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(root, sensor, resolution, "data", "*.parquet")))
    processing_commits = set()
    for f in files:
        processing_commits.update(json.loads(pq.read_metadata(f).metadata[b"altimetry_data"]).get(
            "processing_commits", []))
    index = db.write_granule_index(root, sensor, resolution)
    stat_levels = {}
    for f in sorted(glob.glob(os.path.join(root, sensor, resolution, "stats", "*_stats.parquet"))):
        levels = pq.read_table(f, columns=["level"]).column("level").unique().to_pylist()
        stat_levels[os.path.basename(f)] = sorted(int(v) for v in levels)
    manifest = {
        "sensor": sensor, "resolution": resolution, "format_version": db.FORMAT_VERSION,
        "data_cell_level": db.DATA_LEVEL, "partition_cell_level": db.partition_level(sensor),
        "stat_levels": stat_levels, "min_stat_rows": db.MIN_STAT_ROWS, "quantum": db.QUANTUM,
        "geometry_encoding": "GeoParquet native point (struct x, y); no latitude/longitude columns",
        "crs": {code: f"EPSG:{r['epsg']}" for code, r in REGIONS.items()},
        "geographic_frame": db.GEOGRAPHIC_FRAME,
        "delta_encoded": sorted(sc.DELTA_COLUMNS),
        "vertical_reference": "WGS84 ellipsoid, ITRF2020",
        "time_reference": "delta_time: GPS seconds since 2018-01-01T00:00:00 UTC",
        "footer_statistics": sc.FOOTER_STATS,
        "reduction": gr.REDUCE_PARAMS.get(sensor) if resolution == "reduced" else None,
        "code": provenance.code_info(),
        "processing_commits": sorted(processing_commits),
        "granule_index": os.path.relpath(index, os.path.join(root, sensor, resolution)) if index else None,
        "columns": sc.column_info(sensor, resolution),
        "files": [os.path.relpath(f, os.path.join(root, sensor, resolution)) for f in files],
        # granules of this sensor left out of the files
        "failed_granules": [{"granule_id": int(r.granule_id), "granule": r.granule, "year": int(r.year),
                             "error": r.error if isinstance(getattr(r, "error", None), str) else None}
                            for r in table[(table["sensor"] == sensor) & (table["status"] == "failed")].itertuples()],
        "excluded_granules": [{"granule_id": int(r.granule_id), "granule": r.granule, "reason": r.exclusion_reason}
                              for r in table[(table["sensor"] == sensor) & (table["status"] == "excluded")].itertuples()],
    }
    with open(os.path.join(root, sensor, resolution, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--granules", required=True)
    parser.add_argument("--root", default=None, help="database root (default /Volumes/ice1/ben/oib_database/parquet)")
    parser.add_argument("--work_root", default=None, help="intermediate files (default: {root}_work)")
    parser.add_argument("--sensors", nargs="+")
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--resolutions", nargs="+", choices=["native", "reduced"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep_partitions", action="store_true")
    parser.add_argument("--allow_incomplete", action="store_true",
                        help="build sensor/years even if some of their granules have not been processed")
    parser.add_argument("--allow_dirty", action="store_true", help="build from uncommitted or unpushed code")
    args = parser.parse_args()
    code = provenance.require_committed(args.allow_dirty)
    print(f"code: {code['repository']} commit {code['commit']} (dirty: {code['dirty']})", flush=True)

    args.root = args.root or db.DEFAULT_ROOT
    work_root = args.work_root or db.default_work_root(args.root)
    granules = pd.read_parquet(args.granules)
    missing = incomplete(work_root, granules)
    jobs = []
    for d in sorted(glob.glob(os.path.join(work_root, "partitions", "*", "*", "*", "*"))):
        sensor, resolution, region, year = d.split(os.sep)[-4:]
        if args.sensors and sensor not in args.sensors:
            continue
        if args.years and int(year) not in args.years:
            continue
        if args.resolutions and resolution not in args.resolutions:
            continue
        if (sensor, int(year)) in missing and not args.allow_incomplete:
            print(f"skipping {sensor} {resolution} {region} {year}: {missing[(sensor, int(year))]} granules not yet "
                  f"processed (building now would leave them out and delete the partitions)", flush=True)
            continue
        jobs.append((sensor, resolution, region, int(year)))
    print(f"{len(jobs)} database files to build", flush=True)
    table = granule_table(work_root, granules)
    commits = {}
    if "code_commit" in table:
        done = table[table["code_commit"].notna()]
        commits = dict(zip(done["granule_id"].astype(int), done["code_commit"]))
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {}
        for job in jobs:
            sensor_commits = {int(i): commits[int(i)] for i in granules.loc[granules["sensor"] == job[0], "granule_id"]
                              if int(i) in commits}
            futures[pool.submit(build_one, args.root, work_root, *job, args.keep_partitions, sensor_commits)] = job
        for future, job in futures.items():
            r = future.result()
            print(f"{job}: {r['bytes'] / 1e6:.1f} MB, build {r['build_s']:.0f}s, stats {r['stats_s']:.0f}s", flush=True)

    table.to_parquet(os.path.join(args.root, "granules.parquet"))
    for sensor, resolutions in sc.RESOLUTIONS.items():
        for resolution in resolutions:
            if os.path.isdir(os.path.join(args.root, sensor, resolution, "data")):
                write_manifest(args.root, sensor, resolution, table)


if __name__ == "__main__":
    main()
