"""
Readers for ATM Qfit L1B granules: binary .qi (BLATM1B, ILATM1B v1, ILNSA1B v1) and HDF5 (ILATM1B v2, ILNSA1B v2).

Both return a dict of numpy arrays with at least latitude, longitude (-180..180), elevation (WGS84 ellipsoid,
m), and gps_seconds (GPS seconds since 1980-01-06), plus the instrument fields of the format.

The qi reader follows pointCollection.ATM_Qfit.data.from_qi (same header parsing and scaling tables), but
decodes every record at once instead of in a python loop, keeps the GPS time instead of converting to UTC, and
unwraps the GPS time of day across midnight (from_qi would put the points after 00:00 on the previous day).
"""

import os
import re
import numpy as np
import h5py

from ..timeref import GPS_EPOCH

_QI_VARIABLES = {
    10: ["rel_time", "latitude", "longitude", "elevation", "xmt_sigstr", "rcv_sigstr", "azimuth", "pitch",
         "roll", "time_hhmmss"],
    12: ["rel_time", "latitude", "longitude", "elevation", "xmt_sigstr", "rcv_sigstr", "azimuth", "pitch",
         "roll", "gps_pdop", "pulse_width", "time_hhmmss"],
    14: ["rel_time", "latitude", "longitude", "elevation", "xmt_sigstr", "rcv_sigstr", "azimuth", "pitch",
         "roll", "passive_sig", "pass_foot_lat", "pass_foot_long", "pass_foot_synth_elev", "time_hhmmss"],
}
_QI_SCALES = {
    10: [1e3, 1e6, 1e6, 1e3, 1, 1, 1e3, 1e3, 1e3, 1e3],
    12: [1e3, 1e6, 1e6, 1e3, 1, 1, 1e3, 1e3, 1e3, 1.0e1, 1, 1e3],
    14: [1e3, 1e6, 1e6, 1e3, 1, 1, 1e3, 1e3, 1e3, 1, 1e6, 1e6, 1e3, 1e3],
}
_QI_INT_FIELDS = {"xmt_sigstr", "rcv_sigstr", "pulse_width", "passive_sig"}

# a few NSIDC granules have no .qi extension (e.g. BLATM1B_20050904_012629, BLATM1B_20080802_134319.atm4cT3.F1)
_QI_NAME = re.compile(r"(BLATM1B|ILATM1B|ILNSA1B)_((\d{4})|(\d{2}))(\d{2})(\d{2})(.*?)(\.qi)?$")
_H5_NAME = re.compile(r"(ILATM1B|ILNSA1B)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")


def _seconds_of_day(hhmmss):
    hhmmss = np.asarray(hhmmss, dtype=np.float64)
    hour = np.floor(hhmmss / 10000.0)
    minute = np.floor((hhmmss - hour * 10000.0) / 100.0)
    second = hhmmss - hour * 10000.0 - minute * 100.0
    return hour * 3600.0 + minute * 60.0 + second


def _unwrap_day(seconds_of_day):
    """add a day to times that wrapped past midnight relative to the start of the granule"""
    sod = seconds_of_day.copy()
    if sod.size:
        sod[sod < sod[0] - 43200.0] += 86400.0
    return sod


def instrument_from_name(name):
    """ATM transceiver tag (e.g. 'T3') from a granule name like ILATM1B_20110403_123456.ATM4BT4.qi"""
    m = re.search(r"\.ATM\d[A-Z]?(T\d)", name, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def read_qi(filename):
    with open(os.path.expanduser(filename), "rb") as fid:
        raw = fid.read()
    dtype = np.dtype(">i4")
    value = int(np.frombuffer(raw, dtype=dtype, count=1)[0])
    if value > 100:
        dtype = np.dtype("<i4")
        value = int(np.frombuffer(raw, dtype=dtype, count=1)[0])
    n_blocks = value // dtype.itemsize
    if n_blocks not in _QI_VARIABLES:
        raise ValueError(f"{filename}: unexpected number of variables {n_blocks}")
    rec_bytes = n_blocks * dtype.itemsize
    n_total = len(raw) // rec_bytes
    words = np.frombuffer(raw, dtype=dtype, count=n_total * n_blocks).reshape(n_total, n_blocks)
    # header records follow the first (record-size) record and have a negative first word
    first_data = 1
    while first_data < n_total and words[first_data, 0] < 0:
        first_data += 1
    header = b"".join(raw[k * rec_bytes + dtype.itemsize:(k + 1) * rec_bytes] for k in range(1, first_data))
    data = words[first_data:]
    out = {}
    for k, (name, scale) in enumerate(zip(_QI_VARIABLES[n_blocks], _QI_SCALES[n_blocks])):
        column = data[:, k]
        out[name] = column.astype(np.int32) if name in _QI_INT_FIELDS else (column / scale).astype(np.float64)

    m = _QI_NAME.match(os.path.basename(filename))
    if m is None:
        raise ValueError(f"{filename}: cannot get the date from the file name")
    year = int(m.group(2))
    if m.group(4):
        year = year + 1900 if year >= 90 else year + 2000
    month, day = int(m.group(5)), int(m.group(6))
    day_start = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", "ns")
    # time_hhmmss is GPS time of day on the date in the file name
    sod = _unwrap_day(_seconds_of_day(out["time_hhmmss"]))
    out["gps_seconds"] = (day_start - GPS_EPOCH) / np.timedelta64(1, "s") + sod
    out["longitude"] = np.mod(out["longitude"] + 180.0, 360.0) - 180.0
    out["header"] = header.replace(b"\x00", b"").decode("latin-1", errors="replace").rstrip()
    return out


H5_FIELDS = {
    "latitude": "latitude",
    "longitude": "longitude",
    "elevation": "elevation",
    "instrument_parameters/azimuth": "azimuth",
    "instrument_parameters/rel_time": "rel_time",
    "instrument_parameters/xmt_sigstr": "xmt_sigstr",
    "instrument_parameters/rcv_sigstr": "rcv_sigstr",
    "instrument_parameters/pitch": "pitch",
    "instrument_parameters/roll": "roll",
    "instrument_parameters/gps_pdop": "gps_pdop",
    "instrument_parameters/pulse_width": "pulse_width",
    "instrument_parameters/time_hhmmss": "time_hhmmss",
}


def read_h5(filename):
    out = {}
    with h5py.File(filename, "r") as h5:
        for path, name in H5_FIELDS.items():
            if path in h5:
                out[name] = h5[path][:]
        attrs = {}
        if "ancillary_data" in h5:
            for key, ds in h5["ancillary_data"].items():
                if isinstance(ds, h5py.Dataset) and ds.size < 1000:
                    attrs[key] = ds[()]
        out["ancillary"] = attrs
    m = _H5_NAME.search(os.path.basename(filename))
    y, mo, d = (int(m.group(k)) for k in (2, 3, 4))
    day_start = np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}", "ns")
    # time_hhmmss is GPS time of day on the date in the file name (see check_atm_h5_time in tests)
    sod = _unwrap_day(_seconds_of_day(out["time_hhmmss"]))
    out["gps_seconds"] = (day_start - GPS_EPOCH) / np.timedelta64(1, "s") + sod
    for key in ("latitude", "longitude", "elevation"):
        out[key] = out[key].astype(np.float64)
    out["longitude"] = np.mod(out["longitude"] + 180.0, 360.0) - 180.0
    return out


def read(filename):
    if filename.endswith(".h5"):
        return read_h5(filename)
    return read_qi(filename)
