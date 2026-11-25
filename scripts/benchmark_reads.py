#! /usr/bin/env python
"""
Benchmark reading strategies.

    python scripts/benchmark_reads.py box --root ROOT --sensors ATM_wide ATM_narrow LVIS --resolution reduced \\
        --region GL --center X Y [--size 40000] [--strategies dataset footer stats] [--cold] [--repeat 3]
    python scripts/benchmark_reads.py granule --root ROOT --sensor ICESat --granules 99854 ... [--cold]

box: time to read every point in a size x size box (projected coordinates), for every year of each database,
into one pointCollection.data object.  granule: time to read every point of a granule.

--cold evicts the files from the page cache (posix_fadvise DONTNEED) before each run, so the timing includes
reading from disk; otherwise the second and later runs read from memory.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

from altimetryData import database as db, read


def evict(root):
    """drop the database files from the page cache (clean pages only; needs no privileges)"""
    for f in glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True):
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def time_box(args, strategy):
    x0, y0 = args.center
    h = args.size / 2
    bbox = (x0 - h, y0 - h, x0 + h, y0 + h)
    t0 = time.perf_counter()
    D, info = read.read_box(args.root, args.sensors, args.resolution, args.region, bbox, strategy=strategy,
                            columns=args.columns, threads=args.threads, stats_level=args.stats_level,
                            return_info=True)
    return time.perf_counter() - t0, D, info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["box", "granule"])
    parser.add_argument("--root", default=db.DEFAULT_ROOT)
    parser.add_argument("--sensors", nargs="+", default=["ATM_wide", "ATM_narrow", "LVIS"])
    parser.add_argument("--sensor", default=None)
    parser.add_argument("--resolution", default="reduced")
    parser.add_argument("--region", default="GL")
    parser.add_argument("--center", nargs=2, type=float)
    parser.add_argument("--size", type=float, default=40000.0)
    parser.add_argument("--columns", nargs="+", default=None)
    parser.add_argument("--strategies", nargs="+", default=["dataset", "footer", "stats"])
    parser.add_argument("--stats_level", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--granules", nargs="+", type=int)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    results = []
    if args.mode == "box":
        for strategy in args.strategies:
            for k in range(args.repeat):
                if args.cold:
                    evict(args.root)
                dt, D, info = time_box(args, strategy)
                r = {"strategy": strategy, "run": k, "seconds": round(dt, 3), "points": int(D.size),
                     **{key: info[key] for key in ("files", "row_groups_read", "rows_read")}}
                print(json.dumps(r), flush=True)
                results.append(r)
    else:
        for strategy in args.strategies:
            for gid in args.granules:
                for k in range(args.repeat):
                    if args.cold:
                        evict(args.root)
                    t0 = time.perf_counter()
                    tables = read.read_granule(args.root, gid, resolution=args.resolution, sensor=args.sensor,
                                               columns=args.columns, use_index=(strategy == "index"))
                    dt = time.perf_counter() - t0
                    r = {"strategy": strategy, "granule_id": gid, "run": k, "seconds": round(dt, 3),
                         "points": int(sum(t.num_rows for t in tables.values()))}
                    print(json.dumps(r), flush=True)
                    results.append(r)


if __name__ == "__main__":
    main()
