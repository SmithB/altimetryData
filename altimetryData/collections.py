"""
NSIDC collections that feed the database, grouped by sensor.

Every collection listed here is hosted in NASA Earthdata Cloud (provider NSIDC_CPRD, AWS us-west-2), so the
same granule can be fetched over HTTPS from anywhere or read directly from S3 by compute running in us-west-2.
"""

from dataclasses import dataclass

CMR_PROVIDER = "NSIDC_CPRD"


@dataclass(frozen=True)
class Collection:
    sensor: str        # ATM_wide, ATM_narrow, ICESat, or LVIS: one database family per sensor
    short_name: str
    version: str       # CMR version_id, as CMR spells it
    fmt: str           # qi, h5, or txt: which reader to use
    instrument: str    # recorded in the granule table and the rows (e.g. LVIS vs LVIS on the Global Hawk)


COLLECTIONS = [
    # ATM Qfit L1B.  Wide- and narrow-swath scanners are separate databases.
    Collection("ATM_wide", "BLATM1B", "1", "qi", "ATM"),       # pre-IceBridge, 1993-2008
    Collection("ATM_wide", "ILATM1B", "1", "qi", "ATM"),       # IceBridge, 2009-2012
    Collection("ATM_wide", "ILATM1B", "2", "h5", "ATM"),       # IceBridge, 2013-2019
    Collection("ATM_narrow", "ILNSA1B", "1", "qi", "ATM_NS"),  # IceBridge narrow swath, 2011-2012
    Collection("ATM_narrow", "ILNSA1B", "2", "h5", "ATM_NS"),  # IceBridge narrow swath, 2013-2019
    # ICESat/GLAS ice-sheet altimetry, release 34
    Collection("ICESat", "GLAH12", "034", "h5", "GLAS"),
    # LVIS L2 geolocated surface elevation
    Collection("LVIS", "BLVIS2", "1", "txt", "LVIS"),     # pre-IceBridge, Greenland 2007
    Collection("LVIS", "ILVIS2", "1", "txt", "LVIS"),     # IceBridge, 2009-2015
    Collection("LVIS", "ILVIS2", "2", "txt", "LVIS"),     # IceBridge, Greenland 2017
    Collection("LVIS", "ILVGH2", "1", "txt", "LVIS_GH"),  # LVIS on the Global Hawk, Greenland 2013
]

SENSORS = sorted({c.sensor for c in COLLECTIONS})


def collections_for(sensor):
    return [c for c in COLLECTIONS if c.sensor == sensor]


# Regions.  Points are assigned to a region by latitude; the projected CRS is used for the x and y columns.
# The limits keep every Greenland and Antarctic point while excluding transit and mid-latitude test flights.
REGIONS = {
    "GL": {"name": "Greenland", "epsg": 3413, "lat_min": 59.0, "lat_max": 90.0},
    "AA": {"name": "Antarctica", "epsg": 3031, "lat_min": -90.0, "lat_max": -60.0},
}


def region_for_lat(lat):
    """region code for a latitude (scalar), or None"""
    for code, region in REGIONS.items():
        if region["lat_min"] <= lat <= region["lat_max"]:
            return code
    return None
