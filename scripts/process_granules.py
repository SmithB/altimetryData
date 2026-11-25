#! /usr/bin/env python
"""
Download and process granules into database partitions.

    python scripts/process_granules.py --granules granules.parquet \\
        [--root /Volumes/ice1/ben/oib_database/parquet] [--work_root ROOT_work] \\
        [--sensors ATM_wide] [--years 2011 2012] [--regions GL] [--workers 32] [--transport https|s3] \\
        [--keep_raw] [--limit 100]

Each granule is downloaded into its own directory under {work_root}/raw, processed, written to partitions under
{work_root}/partitions, and (unless --keep_raw) deleted.  A status file per granule
({work_root}/status/{sensor}/{granule_id:08d}.json) records the frame used, point counts, timings, and the
partition files; granules with a status file are skipped, so the script can be rerun to resume.  Use
--transport s3 only on compute in AWS us-west-2.
"""

import argparse
import json
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

_STATE = {}


def _init_worker(transport, threads):
    import numba
    numba.set_num_threads(threads)
    from altimetryData import download
    _STATE["fetch"] = download.fetcher(transport)
    _STATE["transport"] = transport


def _status_file(root, granule):
    return os.path.join(root, "status", granule["sensor"], f"{int(granule['granule_id']):08d}.json")


def run_one(granule, work_root, work_dir, keep_raw, local_dir, code_commit):
    from altimetryData import download, granule as gr
    status = {"granule_id": int(granule["granule_id"]), "granule": granule["granule"], "sensor": granule["sensor"],
              "code_commit": code_commit}
    t0 = time.perf_counter()
    path = None
    try:
        if local_dir:
            path = os.path.join(local_dir, granule["granule"])
            status["download_s"] = 0.0
        else:
            path = download.fetch_granule(_STATE["fetch"], granule, _STATE["transport"], work_dir)
            status["download_s"] = time.perf_counter() - t0
        status["bytes"] = os.path.getsize(path)
        info = gr.write_granule(path, granule, work_root)
        status.update(info)
        status["status"] = "ok"
    except Exception:
        status["status"] = "failed"
        status["error"] = traceback.format_exc()[-4000:]
    finally:
        if path and not keep_raw and not local_dir:
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)
    status["total_s"] = time.perf_counter() - t0
    out = _status_file(work_root, granule)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if status["status"] == "ok":
        with open(out, "w") as fh:
            json.dump(status, fh, default=str)
    else:
        with open(out.replace(".json", ".failed.json"), "w") as fh:
            json.dump(status, fh, default=str)
    return status


def in_areas(granules, areas):
    """granules whose CMR bounding box meets any of the areas [(region, lon, lat, size_km)]; granules without
    an extent in CMR (GLAH12) are kept"""
    from altimetryData import database as db
    keep = granules["lat_min"].isna().to_numpy().copy()
    for region, lon, lat, size_km in areas:
        lon0, lat0, lon1, lat1 = db.box_lonlat_bounds(region, db.box_around(region, float(lon), float(lat),
                                                                            float(size_km) * 1e3))
        lo, hi = granules["lon_min"].to_numpy(), granules["lon_max"].to_numpy()
        wraps = lo > hi       # crosses the antimeridian
        lon_ok = np.where(wraps, (hi >= lon0) | (lo <= lon1), (hi >= lon0) & (lo <= lon1))
        keep |= lon_ok & (granules["lat_max"].to_numpy() >= lat0) & (granules["lat_min"].to_numpy() <= lat1)
    return granules[keep]


def select(granules, sensors=None, years=None, regions=None, areas=None):
    g = granules
    if areas:
        g = in_areas(g, areas)
    if sensors:
        g = g[g["sensor"].isin(sensors)]
    if years:
        g = g[g["year"].isin(years)]
    if regions:
        g = g[g["regions"].apply(lambda r: any(code in r.split() for code in regions))]
    else:
        g = g[g["regions"].str.len() > 0]
    return g


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--granules", required=True)
    parser.add_argument("--root", default=None, help="database root (default /Volumes/ice1/ben/oib_database/parquet)")
    parser.add_argument("--work_root", default=None, help="intermediate files (default: {root}_work)")
    parser.add_argument("--sensors", nargs="+")
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--regions", nargs="+")
    parser.add_argument("--area", nargs=4, action="append", metavar=("REGION", "LON", "LAT", "SIZE_KM"),
                        help="only granules whose CMR bounding box meets a SIZE_KM box centered on LON, LAT "
                             "(repeatable)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--threads_per_worker", type=int, default=1)
    parser.add_argument("--transport", choices=["https", "s3"], default="https")
    parser.add_argument("--local_dir", default=None, help="read already-downloaded granules from this directory")
    parser.add_argument("--keep_raw", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow_dirty", action="store_true",
                        help="run from uncommitted or unpushed code (status files record the commit anyway)")
    args = parser.parse_args()

    from altimetryData import database as db, provenance
    code = provenance.require_committed(args.allow_dirty)
    print(f"code: {code['repository']} commit {code['commit']} (dirty: {code['dirty']})", flush=True)
    args.root = args.root or db.DEFAULT_ROOT
    work_root = args.work_root or db.default_work_root(args.root)
    work_dir = os.path.join(work_root, "raw")
    os.makedirs(work_dir, exist_ok=True)
    granules = select(pd.read_parquet(args.granules), args.sensors, args.years, args.regions, args.area)
    todo = [row for _, row in granules.iterrows() if not os.path.isfile(_status_file(work_root, row))]
    if args.limit:
        todo = todo[:args.limit]
    total_bytes = sum(r["size_bytes"] for r in todo)
    print(f"{len(todo)} granules to process ({total_bytes / 1e9:.1f} GB) of {len(granules)} selected", flush=True)

    t0 = time.perf_counter()
    done_bytes, n_ok, n_failed = 0, 0, 0
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(args.transport, args.threads_per_worker)) as pool:
        futures = [pool.submit(run_one, row.to_dict(), work_root, work_dir, args.keep_raw, args.local_dir,
                               code["commit"] + ("-dirty" if code["dirty"] else "")) for row in todo]
        for k, future in enumerate(as_completed(futures), start=1):
            s = future.result()
            elapsed = time.perf_counter() - t0
            if s["status"] == "ok":
                n_ok += 1
                done_bytes += s.get("bytes", 0)
            else:
                n_failed += 1
                print(f"FAILED {s['granule']}: {s['error'].splitlines()[-1]}", flush=True)
            if k % 10 == 0 or k == len(futures):
                print(f"[{k}/{len(futures)}] {n_ok} ok {n_failed} failed; {done_bytes / 1e6 / elapsed:.1f} MB/s, "
                      f"{k / elapsed * 3600:.0f} granules/h, elapsed {elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
