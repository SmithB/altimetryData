"""
Reference-frame and ellipsoid corrections to WGS84 ellipsoid heights in ITRF2020.

ICESat-2 release 007 and later report heights relative to the WGS84 ellipsoid in ITRF2020.  Each source point is
moved from its native frame and ellipsoid to that frame at the epoch of the measurement:

    geodetic (source ellipsoid) -> ECEF -> Helmert(source ITRF -> ITRF2020, t) -> geodetic (WGS84)

The Helmert parameters come from the IERS tables that PROJ ships (the <ITRF2020> init file, which lists the
transformation from ITRF2020 to each earlier frame, so the step is inverted).  Using PROJ's tables avoids
hand-copied parameters; the hand-copied pipelines in pointCollection.tools.convert_ITRF differ from the IERS
values in two places (the sign of the 2008->2020 Y-translation rate, and the reference epoch of 2005->2020).
"""

from functools import lru_cache
import numpy as np
import pyproj

TARGET_FRAME = "ITRF2020"

# ellipsoids the source products use
ELLIPSOIDS = {
    "WGS84": "+ellps=WGS84",
    "TOPEX": "+a=6378136.3 +rf=298.257",   # GLAS/ICESat
}

# frames PROJ's ITRF2020 init file can transform from
SUPPORTED_FRAMES = ("ITRF2020", "ITRF2014", "ITRF2008", "ITRF2005", "ITRF2000", "ITRF97", "ITRF96", "ITRF94",
                    "ITRF93", "ITRF92", "ITRF91", "ITRF90", "ITRF89", "ITRF88")


def _pipeline(source_frame, ellipsoid):
    steps = ["+proj=pipeline",
             "+step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m",
             f"+step +proj=cart {ELLIPSOIDS[ellipsoid]}"]
    if source_frame != TARGET_FRAME:
        steps.append(f"+step +inv +init={TARGET_FRAME}:{source_frame}")
    steps += ["+step +inv +proj=cart +ellps=WGS84",
              "+step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m"]
    return " ".join(steps)


@lru_cache(maxsize=None)
def _transformer(source_frame, ellipsoid):
    if source_frame not in SUPPORTED_FRAMES:
        raise ValueError(f"unsupported reference frame {source_frame}")
    return pyproj.Transformer.from_pipeline(_pipeline(source_frame, ellipsoid))


def to_itrf2020(lon, lat, h, year, source_frame, ellipsoid="WGS84"):
    """
    longitude, latitude (degrees) and WGS84 ellipsoid height (m) in ITRF2020 for points measured at decimal
    year `year` in `source_frame` relative to `ellipsoid`
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    year = np.broadcast_to(np.asarray(year, dtype=np.float64), lon.shape)
    if source_frame == TARGET_FRAME and ellipsoid == "WGS84":
        return lon.copy(), lat.copy(), h.copy()
    lon2, lat2, h2, _ = _transformer(source_frame, ellipsoid).transform(lon, lat, h, year)
    return np.asarray(lon2), np.asarray(lat2), np.asarray(h2)


def decimal_year(time_ns):
    """decimal year for datetime64[ns] values (accurate to well under a day, plenty for Helmert rates)"""
    t = np.asarray(time_ns, dtype="datetime64[ns]")
    years = t.astype("datetime64[Y]")
    start = years.astype("datetime64[ns]")
    end = (years + 1).astype("datetime64[ns]")
    return years.astype(np.int64) + 1970 + (t - start) / (end - start)
