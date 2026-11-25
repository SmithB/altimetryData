#! /usr/bin/env python
"""
Download and process granules into database partitions.

    python scripts/process_granules.py --granules granules.parquet \\
        [--root /Volumes/ice1/ben/oib_database/parquet] [--work_root ROOT_work] \\
        [--sensors ATM_wide] [--years 2011 2012] [--regions Arctic] [--area REGION LON LAT SIZE_KM] \\
        [--max_downloads 6] [--max_MBps 40] [--retries 1] [--retry_failed] [--transport https|s3] \\
        [--keep_raw] [--limit 100]

Each granule is downloaded into its own directory under {work_root}/raw, processed, written to partitions under
{work_root}/partitions, and (unless --keep_raw) deleted.  A status file per granule
({work_root}/status/{sensor}/{granule_id:08d}.json) records the frame used, point counts, timings, and the
partition files; granules with a status file are skipped, so the script can be rerun to resume.  Use
--transport s3 only on compute in AWS us-west-2.

Downloads are limited so the run does not saturate the network: at most --max_downloads at a time, and at most
--max_MBps in total (a cap shared by every worker; 0 for none).  While the script runs, the cap can be changed by
writing a number (MB/s) to {work_root}/max_MBps; it is re-read every 10 s, and deleting the file restores
--max_MBps.

A granule that fails is retried --retries times at the end of the run, then recorded in
{sensor}/{granule_id:08d}.failed.json and skipped: later runs leave it alone unless --retry_failed is given, and
build_database.py builds its year without it and lists it in the manifest.  Granules in
altimetryData.exclusions are never processed.
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


def _init_worker(transport, threads, download_slots, next_time, lock, max_MBps, rate_file):
    import numba
    numba.set_num_threads(threads)
    from altimetryData import download
    _STATE["fetch"] = download.fetcher(transport)
    _STATE["transport"] = transport
    _STATE["slots"] = download_slots
    _STATE["limiter"] = download.RateLimiter(next_time, lock, max_MBps, rate_file)


def _status_file(root, granule):
    return os.path.join(root, "status", granule["sensor"], f"{int(granule['granule_id']):08d}.json")


def _failed_file(root, granule):
    return _status_file(root, granule).replace(".json", ".failed.json")


def _attempts(root, granule):
    try:
        with open(_failed_file(root, granule)) as fh:
            return int(json.load(fh).get("attempts", 1))
    except (OSError, ValueError):
        return 0


def run_one(granule, work_root, work_dir, keep_raw, local_dir, code_commit):
    from altimetryData import download, granule as gr
    status = {"granule_id": int(granule["granule_id"]), "granule": granule["granule"], "sensor": granule["sensor"],
              "code_commit": code_commit, "attempts": _attempts(work_root, granule) + 1}
    t0 = time.perf_counter()
    path = None
    try:
        if local_dir:
            path = os.path.join(local_dir, granule["granule"])
            status["download_s"] = 0.0
        else:
            with _STATE["slots"]:            # at most --max_downloads at once
                t_wait = time.perf_counter()
                path = download.fetch_granule(_STATE["fetch"], granule, _STATE["transport"], work_dir,
                                              limiter=_STATE["limiter"])
            status["download_wait_s"] = t_wait - t0
            status["download_s"] = time.perf_counter() - t_wait
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
        if os.path.isfile(_failed_file(work_root, granule)):
            os.remove(_failed_file(work_root, granule))
    else:
        with open(_failed_file(work_root, granule), "w") as fh:
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
    from altimetryData import exclusions
    g = granules[~granules["granule"].isin(set(exclusions.EXCLUDED))]
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
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes (default: 2 x max_downloads + 4, so processing overlaps downloads)")
    parser.add_argument("--max_downloads", type=int, default=6, help="simultaneous downloads (default 6)")
    parser.add_argument("--max_MBps", type=float, default=40.0,
                        help="total download rate cap, MB/s (default 40; 0 for none); {work_root}/max_MBps overrides")
    parser.add_argument("--retries", type=int, default=1, help="retries of failed granules at the end of the run")
    parser.add_argument("--retry_failed", action="store_true",
                        help="also retry granules that failed in earlier runs")
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
    todo = [row for _, row in granules.iterrows() if not os.path.isfile(_status_file(work_root, row))
            and (args.retry_failed or not os.path.isfile(_failed_file(work_root, row)))]
    if args.limit:
        todo = todo[:args.limit]
    total_bytes = sum(r["size_bytes"] for r in todo)
    print(f"{len(todo)} granules to process ({total_bytes / 1e9:.1f} GB) of {len(granules)} selected", flush=True)
    workers = args.workers or min(os.cpu_count(), 2 * args.max_downloads + 4)
    rate_file = os.path.join(work_root, "max_MBps")
    print(f"{workers} workers, at most {args.max_downloads} downloads at once, cap {args.max_MBps:g} MB/s "
          f"(override: echo N > {rate_file})", flush=True)

    import multiprocessing
    ctx = multiprocessing.get_context()
    slots, next_time, lock = ctx.BoundedSemaphore(args.max_downloads), ctx.Value("d", 0.0), ctx.Lock()
    commit = code["commit"] + ("-dirty" if code["dirty"] else "")
    t0 = time.perf_counter()
    done_bytes, n_ok, n_failed = 0, 0, 0
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker,
                             initargs=(args.transport, args.threads_per_worker, slots, next_time, lock,
                                       args.max_MBps, rate_file)) as pool:
        for attempt in range(1 + max(0, args.retries)):
            if attempt:
                todo = [row for row in todo if not os.path.isfile(_status_file(work_root, row))]
                if not todo:
                    break
                print(f"retry {attempt}: {len(todo)} failed granules", flush=True)
            futures = [pool.submit(run_one, row.to_dict(), work_root, work_dir, args.keep_raw, args.local_dir,
                                   commit) for row in todo]
            n_failed = 0
            for k, future in enumerate(as_completed(futures), start=1):
                s = future.result()
                elapsed = time.perf_counter() - t0
                if s["status"] == "ok":
                    n_ok += 1
                    done_bytes += s.get("bytes", 0)
                else:
                    n_failed += 1
                    print(f"FAILED {s['granule']} (attempt {s['attempts']}): {s['error'].splitlines()[-1]}",
                          flush=True)
                if k % 10 == 0 or k == len(futures):
                    print(f"[{k}/{len(futures)}] {n_ok} ok {n_failed} failed; {done_bytes / 1e6 / elapsed:.1f} MB/s, "
                          f"{k / elapsed * 3600:.0f} granules/h, elapsed {elapsed / 60:.1f} min", flush=True)
    if n_failed:
        print(f"{n_failed} granules failed after {1 + max(0, args.retries)} attempts; see "
              f"{work_root}/status/*/*.failed.json (rerun with --retry_failed to try them again)", flush=True)


if __name__ == "__main__":
    main()
