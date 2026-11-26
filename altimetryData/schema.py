"""
Column schemas for each database (sensor x resolution).

Every database shares the leading columns below; the sensor-specific columns follow.  Types are fixed here so
every partition and every year of a database has an identical schema.

Position is stored once, as a GeoParquet native ("point") geometry: a struct of projected x and y in EPSG:3413
(Arctic) or EPSG:3031 (AA).  Latitude and longitude are not stored; they follow from x and y by the inverse
projection, and cell_id is computed from them.
"""

import pyarrow as pa

GEOMETRY = pa.struct([("x", pa.float64()), ("y", pa.float64())])

COMMON = [
    ("cell_id", pa.uint64()),        # S2 level-18 cell, the sort key and pandas index
    ("geometry", GEOMETRY),          # native point: x, y in EPSG:3413 (Arctic) or EPSG:3031 (AA), m
    ("delta_time", pa.float64()),    # GPS seconds since 2018-01-01T00:00:00 UTC, as in ICESat-2
    ("h", pa.float32()),             # WGS84 ellipsoid height in ITRF2020
    ("dh_frame", pa.float32()),      # h minus the height in the source frame and ellipsoid
    ("granule_id", pa.uint32()),     # row of the granule lookup table (granules.parquet)
    ("source_index", pa.uint32()),   # record number of the point in the source granule (0-based)
]

FIT_STATS = [
    ("bias", pa.float32()),          # point minus local plane fit (m)
    ("noise", pa.float32()),         # rms of edited plane-fit residuals (m)
    ("n_fit", pa.int16()),           # points kept by the edited fit
    ("slope_x", pa.float32()),       # plane-fit slope in x (m/m)
    ("slope_y", pa.float32()),
]

ATM_NATIVE = [
    ("azimuth", pa.float32()),
    ("rcv_sigstr", pa.int32()),
    ("xmt_sigstr", pa.int32()),
    ("pitch", pa.float32()),
    ("roll", pa.float32()),
    ("gps_pdop", pa.float32()),
    ("pulse_width", pa.int32()),
]

ATM_REDUCED = FIT_STATS + [
    ("rcv_sigstr", pa.int32()),
    ("azimuth", pa.float32()),
]

ICESAT_NATIVE = [
    ("sat_elev_corr", pa.float32()),  # d_satElevCorr, already added to h
    ("gmc", pa.float32()),            # d_GmC, Gaussian-minus-centroid offset, NOT applied
    ("ocean_tide", pa.float32()),     # d_ocElv as stored (0 over grounded ice), NOT applied or removed
    ("ice_svar", pa.float32()),       # d_IceSVar
    ("reflct_uc", pa.float32()),      # d_reflctUC
    ("tx_energy", pa.float32()),      # d_TxNrg
    ("num_pk", pa.int8()),            # i_numPk
    ("gain", pa.int16()),             # i_gval_rcv
    ("elev_use_flg", pa.int8()),
    ("sigma_att_flg", pa.int8()),
    ("sat_corr_flg", pa.int8()),
    ("sat_ndx", pa.int8()),           # i_satNdx
    ("cloud_flg", pa.int8()),         # elv_cloud_flg
    ("campaign", pa.int8()),          # 1-based index into ICESAT_CAMPAIGNS (pointCollection.glah12 convention)
    ("laser", pa.int8()),             # 1, 2, or 3
    ("track", pa.int16()),            # reference track (Data_1HZ i_track)
]

LVIS_NATIVE = [
    ("z_high", pa.float32()),         # center of the highest mode (ELEVATION_HIGH / ZH / Z_HIGH), corrected like h
    ("z_top", pa.float32()),          # highest detected return (ZT, v2 only), corrected like h
    ("z_centroid", pa.float32()),     # waveform centroid (ELEVATION_CENTROID, v1 only), corrected like h
    ("rh50", pa.float32()),           # v2 and .lge only
    ("range", pa.float32()),          # ILVIS2 v2 and LVISF2 (as are the next four)
    ("incident_angle", pa.float32()),
    ("azimuth", pa.float32()),
    ("complexity", pa.float32()),
    ("channel_zg", pa.int8()),
    ("lfid", pa.int64()),
    ("shotnumber", pa.int64()),
]

LVIS_REDUCED = FIT_STATS + [
    ("z_high", pa.float32()),
    ("lfid", pa.int64()),
    ("shotnumber", pa.int64()),
]

SCHEMAS = {
    ("ATM_wide", "native"): COMMON + ATM_NATIVE,
    ("ATM_wide", "reduced"): COMMON + ATM_REDUCED,
    ("ATM_narrow", "native"): COMMON + ATM_NATIVE,
    ("ATM_narrow", "reduced"): COMMON + ATM_REDUCED,
    ("ICESat", "native"): COMMON + ICESAT_NATIVE,
    ("LVIS", "native"): COMMON + LVIS_NATIVE,
    ("LVIS", "reduced"): COMMON + LVIS_REDUCED,
    ("LVIS_GH", "native"): COMMON + LVIS_NATIVE,
    ("LVIS_GH", "reduced"): COMMON + LVIS_REDUCED,
}

# column chunks written first in every row group, so reads of the common columns are contiguous
FIRST_COLUMNS = ["cell_id", "geometry", "delta_time", "h", "granule_id", "bias", "noise", "n_fit", "slope_x",
                 "slope_y"]

# columns (parquet column paths) with footer (row group) statistics; granule_id min/max lets a reader skip row
# groups that cannot hold a granule (the granule index gives the exact row groups)
FOOTER_STATS = ["cell_id", "delta_time", "geometry.x", "geometry.y", "granule_id"]

# integer columns that rise steadily within a cell: delta encoding stores them in well under a byte per row,
# where dictionary encoding of their nearly unique values costs more than the raw integer
DELTA_COLUMNS = {"source_index", "shotnumber"}

# columns left out of the per-cell statistics files
NO_CELL_STATS = {"cell_id", "geometry", "granule_id", "source_index", "shotnumber", "lfid"}

ICESAT_CAMPAIGNS = ["1A", "2A", "2B", "2C", "3A", "3B", "3C", "3D", "3E", "3F", "3G", "3H", "3I", "3J", "3K",
                    "2D", "2E", "2F"]

RESOLUTIONS = {"ATM_wide": ["native", "reduced"], "ATM_narrow": ["native", "reduced"], "ICESat": ["native"],
               "LVIS": ["native", "reduced"], "LVIS_GH": ["native", "reduced"]}

# Column descriptions and units, written into every data file (Arrow field metadata, the altimetry_data and
# gdal:schema file metadata) and into manifest.json.  "" means dimensionless or not applicable.
DESCRIPTIONS = {
    "cell_id": ("S2 cell id at level 18 (cells about 36 m across) of the point, computed from the inverse "
                "projection of the stored x and y; rows are sorted by it", ""),
    "geometry": ("point location: x and y in the file's CRS, EPSG:3413 (Arctic) or EPSG:3031 (AA), projected from "
                 "ITRF2020 latitude and longitude", "m"),
    "geometry.x": ("x in EPSG:3413 (Arctic) or EPSG:3031 (AA)", "m"),
    "geometry.y": ("y in EPSG:3413 (Arctic) or EPSG:3031 (AA)", "m"),
    "delta_time": ("time of the measurement: GPS seconds since 2018-01-01T00:00:00 UTC, as in ICESat-2", "s"),
    "h": ("height above the WGS84 ellipsoid in ITRF2020 at the epoch of the measurement", "m"),
    "dh_frame": ("h minus the height in the source reference frame and ellipsoid: the correction that was applied",
                 "m"),
    "granule_id": ("raw NSIDC granule the point came from: the granule_id key of granules.parquet", ""),
    "source_index": ("0-based record number of the point in its raw granule", ""),
    # reduced products
    "bias": ("h minus the local plane fit at the point (plane fit to the decimated points within the stats "
             "radius, with iterative three-sigma editing)", "m"),
    "noise": ("rms of the edited residuals of the local plane fit", "m"),
    "n_fit": ("number of points kept in the edited local plane fit (0: too few points for a fit)", ""),
    "slope_x": ("x slope of the local plane fit", "m/m"),
    "slope_y": ("y slope of the local plane fit", "m/m"),
    # ATM
    "azimuth": ("scan azimuth of the shot", "degrees"),
    "rcv_sigstr": ("received laser signal strength", "counts"),
    "xmt_sigstr": ("transmitted (start) pulse signal strength", "counts"),
    "pitch": ("aircraft pitch", "degrees"),
    "roll": ("aircraft roll", "degrees"),
    "gps_pdop": ("GPS position dilution of precision (12-word Qfit and HDF5 granules only)", ""),
    "pulse_width": ("received pulse width (12-word Qfit and HDF5 granules only)", "digitizer samples"),
    # ICESat GLAH12
    "sat_elev_corr": ("GLAH12 d_satElevCorr, saturation elevation correction; already added to h", "m"),
    "gmc": ("GLAH12 d_GmC, Gaussian-minus-centroid elevation offset; not applied", "m"),
    "ocean_tide": ("GLAH12 d_ocElv, ocean tide as stored (0 over grounded ice); not applied or removed", "m"),
    "ice_svar": ("GLAH12 d_IceSVar, ice-sheet waveform fit standard deviation", ""),
    "reflct_uc": ("GLAH12 d_reflctUC, surface reflectance not corrected for the atmosphere", ""),
    "tx_energy": ("GLAH12 d_TxNrg, transmitted pulse energy", "J"),
    "num_pk": ("GLAH12 i_numPk, number of peaks in the return waveform", ""),
    "gain": ("GLAH12 i_gval_rcv, receiver gain", "counts"),
    "elev_use_flg": ("GLAH12 elevation use flag (0: valid)", ""),
    "sigma_att_flg": ("GLAH12 attitude quality flag (0: good)", ""),
    "sat_corr_flg": ("GLAH12 saturation correction flag", ""),
    "sat_ndx": ("GLAH12 i_satNdx, waveform saturation index", ""),
    "cloud_flg": ("GLAH12 elv_cloud_flg, cloud contamination flag", ""),
    "campaign": ("ICESat campaign: 1-based index into 1A, 2A, 2B, 2C, 3A-3K, 2D, 2E, 2F", ""),
    "laser": ("GLAS laser (1, 2, or 3)", ""),
    "track": ("ICESat reference track (GLAH12 i_track)", ""),
    # LVIS
    "z_high": ("center of the highest waveform mode (ELEVATION_HIGH / ZH / Z_HIGH), corrected like h", "m"),
    "z_top": ("highest detected return (ZT; ILVIS2 v2 only), corrected like h", "m"),
    "z_centroid": ("waveform centroid (ELEVATION_CENTROID; ILVIS2 v1 and ILVGH2 only), corrected like h", "m"),
    "rh50": ("height above the lowest mode at which 50% of the waveform energy is reached (ILVIS2 v2 and BLVIS2)",
             "m"),
    "range": ("distance from the instrument to the lowest mode (ILVIS2 v2 and LVISF2)", "m"),
    "incident_angle": ("off-nadir incidence angle of the shot (ILVIS2 v2 and LVISF2)", "degrees"),
    "complexity": ("waveform complexity metric (ILVIS2 v2 and LVISF2)", ""),
    "channel_zg": ("LVIS channel used for the lowest mode (CHANNEL_ZG in ILVIS2 v2, CHANNEL in LVISF2)", ""),
    "lfid": ("LVIS file id: digits 3-7 are the Modified Julian Date, the last three count the instrument's files; "
             "lfid with shotnumber identifies a shot", ""),
    "shotnumber": ("LVIS shot number within the instrument file", ""),
}
SENSOR_DESCRIPTIONS = {
    "LVIS": {"azimuth": ("azimuth of the shot (ILVIS2 v2 and LVISF2)", "degrees")},
    "LVIS_GH": {"azimuth": ("azimuth of the shot (not in ILVGH2)", "degrees")},
}


def describe(sensor, name):
    """(description, units) of a column in a sensor's database"""
    return SENSOR_DESCRIPTIONS.get(sensor, {}).get(name) or DESCRIPTIONS[name]


def column_info(sensor, resolution):
    """{column: {"type", "description", "units"}} for a database, including the geometry's x and y"""
    info = {}
    for name, dtype in SCHEMAS[(sensor, resolution)]:
        names = [name] + ([f"{name}.x", f"{name}.y"] if name == "geometry" else [])
        for n in names:
            description, units = describe(sensor, n)
            t = "double" if n.startswith("geometry.") else str(dtype)
            info[n] = {"type": t, "description": description, "units": units}
    return info


def _field(sensor, name, dtype):
    description, units = describe(sensor, name)
    if name == "geometry":
        dtype = pa.struct([pa.field(c, pa.float64(), metadata=_meta(*describe(sensor, f"geometry.{c}")))
                           for c in ("x", "y")])
    return pa.field(name, dtype, metadata=_meta(description, units))


def _meta(description, units):
    return {"description": description, **({"units": units} if units else {})}


def arrow_schema(sensor, resolution):
    """arrow schema of a database, with a description (and units) in each field's metadata"""
    return pa.schema([_field(sensor, name, dtype) for name, dtype in SCHEMAS[(sensor, resolution)]])
