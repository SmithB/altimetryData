"""
Process one raw granule into database tables.

process_granule() reads a granule, moves every point to WGS84 heights in ITRF2020, splits the points by region,
projects them, assigns S2 cells, and (for ATM and LVIS) reduces them; it returns arrow tables keyed by
(resolution, region) along with per-granule information for the lookup table.
"""

import time
import numpy as np

from . import itrf, frames, reduce, timeref
from . import database as db
from .collections import REGIONS
from .readers import atm as atm_reader
from .readers import glas as glas_reader
from .readers import lvis as lvis_reader

# reduction parameters for each sensor (decimation bin size, statistics radius, three-sigma floor), in m
REDUCE_PARAMS = {
    "ATM_wide": {"decimate_to": 10.0, "stats_radius": 50.0, "min_sigma": 0.5},
    "ATM_narrow": {"decimate_to": 10.0, "stats_radius": 50.0, "min_sigma": 0.5},
    # LVIS footprints are ~20 m (2007-2015) and ~10 m (2013 Global Hawk, 2017); bins match the larger footprint
    "LVIS": {"decimate_to": 20.0, "stats_radius": 50.0, "min_sigma": 0.5},
    "LVIS_GH": {"decimate_to": 20.0, "stats_radius": 50.0, "min_sigma": 0.5},
}

# frames fixed by product documentation (used when the granule itself does not say)
PRODUCT_FRAMES = {"GLAH12": ("ITRF2008", "TOPEX")}

# LVIS reference frames by (short_name, version), from the NSIDC user guides; the granules do not state a frame.
# ILVIS2 v1 and ILVGH2 v1: "International Terrestrial Reference Frame (ITRF 2000), WGS-84 Ellipsoid" (section
# 1.3.3 / 1.3.2).  ILVIS2 v2: datum ITRF 2008, EPSG:5332 (Table 3; GRS80 differs from WGS84 by 0.1 mm).
# BLVIS2 (2007) has no user guide; it was processed in 2008 with the same system as ILVIS2 v1, so ITRF2000 is
# assumed and recorded as such.
LVIS_FRAMES = {
    ("ILVIS2", "1"): ("ITRF2000", "user_guide"),
    ("ILVGH2", "1"): ("ITRF2000", "user_guide"),
    ("ILVIS2", "2"): ("ITRF2008", "user_guide"),
    ("BLVIS2", "1"): ("ITRF2000", "assumed (as ILVIS2 v1)"),
}


def _region_masks(lat):
    return {code: (lat >= r["lat_min"]) & (lat <= r["lat_max"]) for code, r in REGIONS.items()}


def _reduced_columns(native, keep, stats):
    cols = {k: v[keep] for k, v in native.items()}
    cols["bias"] = stats["bias"]
    cols["noise"] = stats["noise"]
    cols["n_fit"] = np.where(np.isfinite(stats["N"]), stats["N"], 0)
    cols["slope_x"] = stats["slope_x"]
    cols["slope_y"] = stats["slope_y"]
    return cols


def process_atm(path, granule, sensor, timings):
    t = time.perf_counter()
    data = atm_reader.read(path)
    timings["read"] = time.perf_counter() - t
    t = time.perf_counter()
    n = data["latitude"].size
    source_index = np.arange(n, dtype=np.uint32)
    valid = np.isfinite(data["latitude"]) & np.isfinite(data["longitude"]) & np.isfinite(data["elevation"]) & \
        (np.abs(data["elevation"]) < 1.0e4) & (np.abs(data["latitude"]) <= 90)
    delta_time = timeref.delta_time_from_gps(data["gps_seconds"])
    year = timeref.year_from_delta_time(delta_time)
    date = timeref.gps_to_utc_datetime(data["gps_seconds"][:1])[0] if n else np.datetime64("NaT")
    frame, frame_source = frames.atm_frame(data, date)
    lon, lat, h = itrf.to_itrf2020(data["longitude"][valid], data["latitude"][valid], data["elevation"][valid],
                                   year[valid], frame, "WGS84")
    timings["frame"] = time.perf_counter() - t
    native_all = {
        "longitude": lon, "latitude": lat, "h": h, "dh_frame": h - data["elevation"][valid],
        "delta_time": delta_time[valid], "source_index": source_index[valid],
        "granule_id": np.full(lon.size, granule["granule_id"], dtype=np.uint32),
    }
    for key in ("azimuth", "rcv_sigstr", "xmt_sigstr", "pitch", "roll", "gps_pdop", "pulse_width"):
        if key in data:
            native_all[key] = data[key][valid]
    info = {"frame": frame, "frame_source": frame_source, "n_records": int(n), "n_valid": int(valid.sum()),
            "transceiver": atm_reader.instrument_from_name(granule["granule"])}
    tables = _split_and_reduce(native_all, sensor, info, timings, reduce_sensor=True)
    return tables, info


def _split_and_reduce(native_all, sensor, info, timings, reduce_sensor):
    """split corrected points by region, project, and (optionally) reduce; returns {(resolution, region): table}"""
    tables = {}
    for region, mask in _region_masks(native_all["latitude"]).items():
        if not mask.any():
            continue
        t = time.perf_counter()
        native = {k: v[mask] for k, v in native_all.items()}
        native["x"], native["y"] = db.project(native["longitude"], native["latitude"], region)
        tables[("native", region)] = db.make_table(native, sensor, "native", region)
        timings["native_table"] = timings.get("native_table", 0) + time.perf_counter() - t
        info[f"n_native_{region}"] = int(mask.sum())
        if not reduce_sensor:
            continue
        t = time.perf_counter()
        keep, stats = reduce.reduce_points(native["x"], native["y"], native["h"].astype(np.float64),
                                           **REDUCE_PARAMS[sensor])
        timings["reduce"] = timings.get("reduce", 0) + time.perf_counter() - t
        t = time.perf_counter()
        tables[("reduced", region)] = db.make_table(_reduced_columns(native, keep, stats), sensor, "reduced", region)
        timings["reduced_table"] = timings.get("reduced_table", 0) + time.perf_counter() - t
        info[f"n_reduced_{region}"] = int(keep.size)
    return tables


def _correct(lon, lat, z, delta_time, frame, ellipsoid):
    lon2, lat2, h2 = itrf.to_itrf2020(lon, lat, z, timeref.year_from_delta_time(delta_time), frame, ellipsoid)
    return lon2, lat2, h2, h2 - z


def process_glas(path, granule, sensor, timings):
    t = time.perf_counter()
    data = glas_reader.read_glah12(path)
    timings["read"] = time.perf_counter() - t
    t = time.perf_counter()
    n = data["latitude"].size
    valid = np.isfinite(data["latitude"]) & np.isfinite(data["longitude"]) & np.isfinite(data["elevation"])
    # only shots in a region need the frame change
    in_region = np.zeros(n, dtype=bool)
    for mask in _region_masks(data["latitude"]).values():
        in_region |= mask
    valid &= in_region
    frame, ellipsoid = PRODUCT_FRAMES["GLAH12"]
    delta_time = timeref.delta_time_from_gps(data["gps_seconds"][valid])
    lon, lat, h, dh = _correct(data["longitude"][valid], data["latitude"][valid], data["elevation"][valid],
                               delta_time, frame, ellipsoid)
    timings["frame"] = time.perf_counter() - t
    native_all = {"longitude": lon, "latitude": lat, "h": h, "dh_frame": dh, "delta_time": delta_time,
                  "source_index": np.flatnonzero(valid).astype(np.uint32),
                  "granule_id": np.full(lon.size, granule["granule_id"], dtype=np.uint32)}
    for key in ("sat_elev_corr", "gmc", "ocean_tide", "ice_svar", "reflct_uc", "tx_energy", "num_pk", "gain",
                "elev_use_flg", "sigma_att_flg", "sat_corr_flg", "sat_ndx", "cloud_flg", "campaign", "laser",
                "track"):
        if key in data:
            native_all[key] = data[key][valid]
    info = {"frame": frame, "frame_source": "product", "ellipsoid": ellipsoid, "n_records": int(n),
            "n_valid": int(valid.sum()), "campaign": data["campaign_name"]}
    return _split_and_reduce(native_all, sensor, info, timings, reduce_sensor=False), info


def process_lvis(path, granule, sensor, timings):
    t = time.perf_counter()
    data = lvis_reader.read(path)
    timings["read"] = time.perf_counter() - t
    t = time.perf_counter()
    n = data["latitude"].size
    valid = np.isfinite(data["latitude"]) & np.isfinite(data["longitude"]) & np.isfinite(data["elevation"]) & \
        (np.abs(data["elevation"]) < 1.0e4)
    delta_time = timeref.delta_time_from_gps(data["gps_seconds"][valid])
    date = timeref.gps_to_utc_datetime(data["gps_seconds"][:1])[0] if n else np.datetime64("NaT")
    frame, frame_source = LVIS_FRAMES.get((granule.get("short_name"), str(granule.get("version"))),
                                          (frames.igs_frame(date), "igs_default"))
    lon, lat, h, dh = _correct(data["longitude"][valid], data["latitude"][valid], data["elevation"][valid],
                               delta_time, frame, "WGS84")
    timings["frame"] = time.perf_counter() - t
    native_all = {"longitude": lon, "latitude": lat, "h": h, "dh_frame": dh, "delta_time": delta_time,
                  "source_index": np.flatnonzero(valid).astype(np.uint32),
                  "granule_id": np.full(lon.size, granule["granule_id"], dtype=np.uint32)}
    for key in ("z_high", "z_top", "z_centroid"):
        if key in data:
            native_all[key] = np.asarray(data[key], dtype=np.float64)[valid] + dh
    for key in ("rh50", "range", "incident_angle", "azimuth", "complexity", "channel_zg", "lfid", "shotnumber"):
        if key in data:
            native_all[key] = np.asarray(data[key])[valid]
    info = {"frame": frame, "frame_source": frame_source, "n_records": int(n), "n_valid": int(valid.sum())}
    return _split_and_reduce(native_all, sensor, info, timings, reduce_sensor=True), info


PROCESSORS = {"ATM_wide": process_atm, "ATM_narrow": process_atm, "ICESat": process_glas, "LVIS": process_lvis,
              "LVIS_GH": process_lvis}


def process_granule(path, granule):
    """granule is a row (dict-like) of the granule lookup table"""
    timings = {}
    tables, info = PROCESSORS[granule["sensor"]](path, granule, granule["sensor"], timings)
    info["timings"] = timings
    return tables, info


def write_granule(path, granule, work_root):
    """process a granule and write its partitions; returns the info dict (with the partition files written)"""
    tables, info = process_granule(path, granule)
    t = time.perf_counter()
    files = []
    for (resolution, region), table in tables.items():
        files += db.write_partitions(table, work_root, granule["sensor"], resolution, region, granule["granule_id"])
    info["timings"]["write"] = time.perf_counter() - t
    info["partition_files"] = files
    return info
