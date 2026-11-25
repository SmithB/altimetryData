"""
Reader for GLAS/ICESat GLAH12 release 34 (HDF5).

Returns heights relative to the TOPEX/Poseidon ellipsoid as stored (d_elev, with the saturation correction added
where it is valid), GPS seconds, and the quality fields in schema.ICESAT_NATIVE.  Fill values become NaN (floats)
and the ellipsoid and frame change to WGS84/ITRF2020 is applied by the caller.

Notes from inspecting the granules:
  * d_UTCTime_40 is UTC seconds since 2000-01-01T12:00:00; it is treated as a UTC time without leap seconds
    (a +/-1 s ambiguity for shots after the 2005-12-31 and 2008-12-31 leap seconds).
  * d_deltaEllip (TOPEX minus WGS84 height, ~0.71 m) agrees with the PROJ ellipsoid change to 0.5 mm.
  * d_ocElv is 0 over grounded ice; it is stored but not applied or removed.
  * i_track is a 1 Hz field; each 1 Hz record holds 40 shots.
"""

import numpy as np
import h5py

from ..schema import ICESAT_CAMPAIGNS
from .. import timeref

FILL = 1.0e38
J2000_NOON_UTC = np.datetime64("2000-01-01T12:00:00", "ns")

FIELDS = {
    "latitude": "Data_40HZ/Geolocation/d_lat",
    "longitude": "Data_40HZ/Geolocation/d_lon",
    "elevation": "Data_40HZ/Elevation_Surfaces/d_elev",
    "utc_j2000": "Data_40HZ/Time/d_UTCTime_40",
    "sat_elev_corr": "Data_40HZ/Elevation_Corrections/d_satElevCorr",
    "gmc": "Data_40HZ/Elevation_Corrections/d_GmC",
    "ocean_tide": "Data_40HZ/Geophysical/d_ocElv",
    "delta_ellip": "Data_40HZ/Geophysical/d_deltaEllip",
    "ice_svar": "Data_40HZ/Elevation_Surfaces/d_IceSVar",
    "reflct_uc": "Data_40HZ/Reflectivity/d_reflctUC",
    "tx_energy": "Data_40HZ/Transmit_Energy/d_TxNrg",
    "num_pk": "Data_40HZ/Waveform/i_numPk",
    "gain": "Data_40HZ/Waveform/i_gval_rcv",
    "elev_use_flg": "Data_40HZ/Quality/elev_use_flg",
    "sigma_att_flg": "Data_40HZ/Quality/sigma_att_flg",
    "sat_corr_flg": "Data_40HZ/Quality/sat_corr_flg",
    "sat_ndx": "Data_40HZ/Quality/i_satNdx",
    "cloud_flg": "Data_40HZ/Elevation_Flags/elv_cloud_flg",
}


def campaign_laser(campaign):
    """(1-based campaign index as in pointCollection.glah12, laser number) for a campaign name like '3D'"""
    return ICESAT_CAMPAIGNS.index(campaign) + 1, int(campaign[0])


def read_glah12(filename):
    out = {}
    with h5py.File(filename, "r") as h5:
        for name, path in FIELDS.items():
            if path not in h5:
                continue
            values = h5[path][:]
            if values.dtype.kind == "f":
                values = values.astype(np.float64)
                values[np.abs(values) > FILL] = np.nan
            out[name] = values
        track = h5["Data_1HZ/Geolocation/i_track"][:]
        campaign = h5["ANCILLARY_DATA"].attrs["Campaign"]
        campaign = campaign.decode("ascii") if isinstance(campaign, bytes) else str(campaign)
    n = out["latitude"].size
    out["track"] = np.repeat(track, n // track.size) if n % track.size == 0 else np.full(n, -1)
    out["campaign_name"] = campaign
    index, laser = campaign_laser(campaign)
    out["campaign"] = np.full(n, index, dtype=np.int8)
    out["laser"] = np.full(n, laser, dtype=np.int8)
    out["longitude"] = np.mod(out["longitude"] + 180.0, 360.0) - 180.0
    sat = out["sat_elev_corr"]
    out["elevation_raw"] = out["elevation"]
    out["elevation"] = out["elevation"] + np.where(np.isfinite(sat), sat, 0.0)
    utc_seconds = out.pop("utc_j2000") + (J2000_NOON_UTC - timeref.GPS_EPOCH) / np.timedelta64(1, "s")
    out["gps_seconds"] = timeref.utc_to_gps(utc_seconds)
    return out
