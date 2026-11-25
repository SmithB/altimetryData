#! /usr/bin/env python
"""
Build the granule lookup table (granules.parquet) from CMR.

    python scripts/build_inventory.py --out granules.parquet [--sensors ATM_wide ATM_narrow ICESat LVIS] [--refresh]

Collections are listed concurrently; each collection's table is cached next to the output so an interrupted
run resumes where it stopped (--refresh lists every collection from CMR again).

Granule ids are frozen: if --out already exists, every granule in it keeps its id, new granules are numbered
after the largest existing id, and granules no longer in CMR keep their rows (in_cmr = False).  The previous
table is kept as {out}.prev.  Delete --out only to renumber from scratch, which invalidates every database
file built so far.
"""

import argparse
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from altimetryData import cmr
from altimetryData.collections import COLLECTIONS


def list_one(collection, cache_dir):
    cache = os.path.join(cache_dir, f"{collection.short_name}_v{collection.version}.parquet")
    if os.path.isfile(cache):
        return pd.read_parquet(cache)
    t0 = time.time()
    session = requests.Session()
    concept_id = cmr.collection_concept_id(session, collection.short_name, collection.version)
    rows, headers = [], {}
    while True:
        t = time.time()
        response = cmr._get(session, f"{cmr.CMR_URL}/granules.umm_json",
                            {"collection_concept_id": concept_id, "page_size": cmr.PAGE_SIZE}, headers=headers)
        items = response.json()["items"]
        df = cmr.granule_rows(collection, items)
        rows.append(df)
        print(f"{collection.short_name} v{collection.version}: +{len(df)} ({sum(map(len, rows))} of "
              f"{response.headers.get('CMR-Hits')}) page {time.time() - t:.1f}s", flush=True)
        search_after = response.headers.get("CMR-Search-After")
        if not items or not search_after:
            break
        headers = {"CMR-Search-After": search_after}
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(cache)
    print(f"{collection.short_name} v{collection.version}: done, {len(df)} granules in {time.time() - t0:.0f}s",
          flush=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sensors", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh", action="store_true", help="ignore the per-collection cache and re-list CMR")
    args = parser.parse_args()
    collections = [c for c in COLLECTIONS if args.sensors is None or c.sensor in args.sensors]
    cache_dir = os.path.splitext(args.out)[0] + "_cache"
    os.makedirs(cache_dir, exist_ok=True)
    if args.refresh:
        for c in collections:
            cache = os.path.join(cache_dir, f"{c.short_name}_v{c.version}.parquet")
            if os.path.isfile(cache):
                os.remove(cache)
    with ThreadPoolExecutor(args.workers) as pool:
        tables = list(pool.map(lambda c: list_one(c, cache_dir), collections))
    previous = pd.read_parquet(args.out) if os.path.isfile(args.out) else None
    df = cmr.finalize_inventory(pd.concat(tables, ignore_index=True), previous)
    if previous is not None:
        kept = previous[["granule_id"]].merge(df[["granule_id"]], on="granule_id")
        assert len(kept) == len(previous), "granule ids must never be dropped"
        shutil.copy2(args.out, args.out + ".prev")
    tmp = args.out + ".tmp"
    df.to_parquet(tmp)
    os.replace(tmp, args.out)
    n_new = len(df) - (0 if previous is None else len(previous))
    print(f"wrote {len(df)} granules to {args.out} ({n_new} new, {int((~df['in_cmr']).sum())} no longer in CMR)")


if __name__ == "__main__":
    main()
