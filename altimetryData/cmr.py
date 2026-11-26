"""
Granule inventory from NASA's Common Metadata Repository (CMR).

CMR search is public, so no credentials are needed to build the inventory.  The inventory is the granule
lookup table that every row of the database points back to through its granule_id column.
"""

import time
import numpy as np
import pandas as pd
import requests

from .collections import COLLECTIONS, CMR_PROVIDER, REGIONS

CMR_URL = "https://cmr.earthdata.nasa.gov/search"
PAGE_SIZE = 2000  # CMR granule queries sorted by start_date time out; sort locally instead


def _get(session, url, params, headers=None, retries=5, timeout=120):
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def collection_concept_id(session, short_name, version):
    response = _get(session, f"{CMR_URL}/collections.json",
                    {"short_name": short_name, "version": version, "provider": CMR_PROVIDER, "page_size": 10})
    entries = response.json()["feed"]["entry"]
    if len(entries) != 1:
        raise RuntimeError(f"expected one collection for {short_name} v{version}, found {len(entries)}")
    return entries[0]["id"]


def _size_bytes(umm):
    total = 0.0
    for info in umm.get("DataGranule", {}).get("ArchiveAndDistributionInformation", []):
        name = info.get("Name", "")
        if name.endswith(".xml"):
            continue
        if "SizeInBytes" in info:
            total += info["SizeInBytes"]
        elif "Size" in info:
            scale = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}[info.get("SizeUnit", "MB")]
            total += info["Size"] * scale
    return total


def _lat_lon_points(umm):
    """every vertex in the granule's spatial extent, as (lat, lon) arrays"""
    geometry = umm.get("SpatialExtent", {}).get("HorizontalSpatialDomain", {}).get("Geometry", {})
    lats, lons = [], []
    for rect in geometry.get("BoundingRectangles", []):
        lats += [rect["NorthBoundingCoordinate"], rect["SouthBoundingCoordinate"]]
        lons += [rect["WestBoundingCoordinate"], rect["EastBoundingCoordinate"]]
    for poly in geometry.get("GPolygons", []):
        for point in poly["Boundary"]["Points"]:
            lats.append(point["Latitude"])
            lons.append(point["Longitude"])
    for line in geometry.get("Lines", []):
        for point in line["Points"]:
            lats.append(point["Latitude"])
            lons.append(point["Longitude"])
    for point in geometry.get("Points", []):
        lats.append(point["Latitude"])
        lons.append(point["Longitude"])
    return np.array(lats, dtype=float), np.array(lons, dtype=float)


def _urls(umm):
    https, s3 = None, None
    for url in umm.get("RelatedUrls", []):
        u = url.get("URL", "")
        if u.endswith(".xml") or u.endswith(".png") or u.endswith(".jpg"):
            continue
        if url.get("Type") == "GET DATA" and u.startswith("https") and https is None:
            https = u
        elif url.get("Type") == "GET DATA VIA DIRECT ACCESS" and u.startswith("s3://") and s3 is None:
            s3 = u
    return https, s3


def granule_regions(lat_min, lat_max):
    """space-separated region codes whose latitude band the granule's extent touches"""
    codes = [code for code, r in REGIONS.items() if lat_max >= r["lat_min"] and lat_min <= r["lat_max"]]
    return " ".join(codes)


def granule_rows(collection, items):
    """one row per CMR UMM-G item"""
    rows = []
    for item in items:
        umm = item["umm"]
        if not collection.accepts(umm["GranuleUR"]):
            continue
        lats, lons = _lat_lon_points(umm)
        temporal = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
        https, s3 = _urls(umm)
        rows.append({
            "sensor": collection.sensor,
            "short_name": collection.short_name,
            "version": collection.version,
            "format": collection.fmt,
            "instrument": collection.instrument,
            "granule": umm["GranuleUR"],
            "concept_id": item["meta"]["concept-id"],
            "start_time": temporal.get("BeginningDateTime"),
            "end_time": temporal.get("EndingDateTime"),
            "lat_min": float(np.min(lats)) if lats.size else np.nan,
            "lat_max": float(np.max(lats)) if lats.size else np.nan,
            "lon_min": float(np.min(lons)) if lons.size else np.nan,
            "lon_max": float(np.max(lons)) if lons.size else np.nan,
            "size_bytes": _size_bytes(umm),
            "https_url": https,
            "s3_url": s3,
        })
    return pd.DataFrame(rows)


def list_granules(collection, session=None, verbose=False):
    """one row per granule of a collection"""
    session = session or requests.Session()
    concept_id = collection_concept_id(session, collection.short_name, collection.version)
    tables, headers = [], {}
    while True:
        response = _get(session, f"{CMR_URL}/granules.umm_json",
                        {"collection_concept_id": concept_id, "page_size": PAGE_SIZE}, headers=headers)
        items = response.json()["items"]
        tables.append(granule_rows(collection, items))
        if verbose:
            print(f"{collection.short_name} v{collection.version}: {sum(map(len, tables))} granules", flush=True)
        search_after = response.headers.get("CMR-Search-After")
        if not items or not search_after:
            break
        headers = {"CMR-Search-After": search_after}
    return pd.concat(tables, ignore_index=True)


GRANULE_KEY = ["short_name", "version", "granule"]


def finalize_inventory(df, previous=None):
    """
    parse times, add year and regions, and assign granule_id.

    With no previous inventory, ids run 0..n-1 in (sensor, start_time, granule) order.  With one, ids are frozen:
    a granule (keyed by short_name, version, and granule name) keeps its previous id and its row is refreshed
    from the new listing; new granules get ids after the previous maximum, in (sensor, start_time, granule)
    order; granules that have left CMR keep their rows and ids with in_cmr = False, so ids are never reused and
    existing database rows always resolve.  Previous granules of sensors absent from this listing (a partial
    listing) are carried over unchanged.
    """
    df = df.copy()
    for col in ["start_time", "end_time"]:
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    df["year"] = df["start_time"].dt.year.astype("int16")
    # granules without a spatial extent in CMR (GLAH12 is described by orbit) may touch either region
    df["regions"] = [granule_regions(a, b) if np.isfinite(a) else " ".join(REGIONS)
                     for a, b in zip(df["lat_min"], df["lat_max"])]
    df["in_cmr"] = True
    df = df.sort_values(["sensor", "start_time", "granule"], kind="stable").reset_index(drop=True)
    dup = df.duplicated(GRANULE_KEY)
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicate granules in the listing, e.g. {df.loc[dup, 'granule'].iloc[0]}")
    if previous is None or len(previous) == 0:
        df.insert(0, "granule_id", np.arange(len(df), dtype=np.uint32))
        return df
    prev = previous.drop(columns=["in_cmr"], errors="ignore")
    ids = df[GRANULE_KEY].merge(prev[GRANULE_KEY + ["granule_id"]], on=GRANULE_KEY, how="left")["granule_id"]
    new = ids.isna().to_numpy()
    next_id = int(prev["granule_id"].max()) + 1
    ids = ids.to_numpy(dtype=np.float64, copy=True)
    ids[new] = np.arange(next_id, next_id + new.sum())
    df.insert(0, "granule_id", ids.astype(np.uint32))
    # previous granules missing from the listing: gone from CMR if their sensor was listed, otherwise carried over
    gone = prev[~prev["granule_id"].isin(df["granule_id"])].copy()
    was = previous["in_cmr"] if "in_cmr" in previous else pd.Series(True, index=previous.index)
    gone["in_cmr"] = ~gone["sensor"].isin(df["sensor"].unique()) & was.loc[gone.index].astype(bool)
    out = pd.concat([df, gone[[c for c in df.columns if c in gone.columns]]], ignore_index=True)
    out = out.sort_values("granule_id", kind="stable").reset_index(drop=True)
    out["in_cmr"] = out["in_cmr"].astype(bool)
    return out


def build_inventory(collections=COLLECTIONS, verbose=False):
    """granule lookup table for every collection (serial; scripts/build_inventory.py runs collections in parallel)"""
    session = requests.Session()
    df = pd.concat([list_granules(c, session=session, verbose=verbose) for c in collections], ignore_index=True)
    return finalize_inventory(df)
