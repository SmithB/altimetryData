"""
Readers for LVIS L2 granules.

Three formats:
  * ILVIS2 v1 and ILVGH2 text: LVIS_LFID SHOTNUMBER TIME and centroid / low / high longitude, latitude, elevation;
  * ILVIS2 v2 text: LFID SHOTNUMBER TIME GLON GLAT ZG HLON HLAT ZH TLON TLAT ZT RH10..RH100 AZIMUTH
    INCIDENTANGLE RANGE COMPLEXITY, followed by three columns the header does not name (taken here to be
    CHANNEL_ZT, CHANNEL_ZG, CHANNEL_RH, as in the LVIS-Facility L2 documentation);
  * BLVIS2 .lge binary: big-endian 52-byte records (lfid, shotnumber, time, glon, glat, zg, rh25, rh50, rh75, rh100).

The v2 header says the earlier "low" and "high" elevations are the v2 zg and zh (the centers of the lowest and
highest modes), so h is ELEVATION_LOW / ZG for every format.  TIME is UTC seconds of day on the date in the
file name; times that wrap past midnight are moved to the next day.
"""

import os
import re
import numpy as np
import pandas as pd

from .. import timeref

_TXT_NAME = re.compile(r"I?LV[A-Z]*2_[A-Z]{2}(\d{4})_(\d{2})(\d{2})_")
_LGE_NAME = re.compile(r"LVIS_[A-Z]{2}_(\d{4})(\d{2})(\d{2})_")
_V2_EXTRA = ["CHANNEL_ZT", "CHANNEL_ZG", "CHANNEL_RH"]

_V1_MAP = {"LVIS_LFID": "lfid", "SHOTNUMBER": "shotnumber", "TIME": "time", "LONGITUDE_LOW": "longitude",
           "LATITUDE_LOW": "latitude", "ELEVATION_LOW": "elevation", "ELEVATION_HIGH": "z_high",
           "ELEVATION_CENTROID": "z_centroid"}
_V2_MAP = {"LFID": "lfid", "SHOTNUMBER": "shotnumber", "TIME": "time", "GLON": "longitude", "GLAT": "latitude",
           "ZG": "elevation", "ZH": "z_high", "ZT": "z_top", "RH50": "rh50", "AZIMUTH": "azimuth",
           "INCIDENTANGLE": "incident_angle", "RANGE": "range", "COMPLEXITY": "complexity",
           "CHANNEL_ZG": "channel_zg"}

_LGE_DTYPE = np.dtype([("lfid", ">u4"), ("shotnumber", ">u4"), ("time", ">f8"), ("longitude", ">f8"),
                       ("latitude", ">f8"), ("elevation", ">f4"), ("rh25", ">f4"), ("rh50", ">f4"),
                       ("rh75", ">f4"), ("rh100", ">f4")])


def _date_from_name(name):
    m = _TXT_NAME.search(name) or _LGE_NAME.search(name)
    if not m:
        raise ValueError(f"no date in LVIS file name {name}")
    return np.datetime64(f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "ns")


def _column_names(filename):
    names = None
    with open(filename, "r", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            tokens = line[1:].split()
            if tokens and tokens[0] in ("LFID", "LVIS_LFID"):
                names = tokens
    if names is None:
        raise ValueError(f"{filename}: no column header")
    return names


def _read_txt(filename):
    names = _column_names(filename)
    first = pd.read_csv(filename, comment="#", sep=r"\s+", header=None, nrows=1)
    if first.shape[1] == len(names) + len(_V2_EXTRA):
        names = names + _V2_EXTRA
    df = pd.read_csv(filename, comment="#", sep=r"\s+", header=None, names=names, engine="c")
    mapping = _V2_MAP if "GLON" in names else _V1_MAP
    return {new: df[old].to_numpy() for old, new in mapping.items() if old in df}


def _read_lge(filename):
    rec = np.fromfile(filename, dtype=_LGE_DTYPE)
    return {name: rec[name].astype(rec.dtype[name].newbyteorder("=")) for name in rec.dtype.names}


def read(filename):
    data = _read_lge(filename) if filename.lower().endswith(".lge") else _read_txt(filename)
    for key in ("longitude", "latitude", "elevation"):
        data[key] = np.asarray(data[key], dtype=np.float64)
    data["longitude"] = np.mod(data["longitude"] + 180.0, 360.0) - 180.0
    sod = np.asarray(data.pop("time"), dtype=np.float64)
    if sod.size:
        sod = np.where(sod < sod[0] - 43200.0, sod + 86400.0, sod)
    day = _date_from_name(os.path.basename(filename))
    utc_seconds = (day - timeref.GPS_EPOCH) / np.timedelta64(1, "s") + sod
    data["gps_seconds"] = timeref.utc_to_gps(utc_seconds)
    return data
