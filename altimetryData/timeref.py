"""
Time reference for the database.

delta_time is defined as in ICESat-2 products: GPS seconds since the ATLAS SDP epoch, 2018-01-01T00:00:00 UTC
(GPS second 1198800018).  Because it counts GPS seconds, the difference between two delta_time values is the
true elapsed time, including any leap seconds between them; for a pre-2018 measurement, delta_time is more
negative than the UTC difference from 2018-01-01 by the number of leap seconds inserted in between (up to 13).

year = 2018 + delta_time / (24 * 3600 * 365.25)
"""

import numpy as np

GPS_EPOCH = np.datetime64("1980-01-06T00:00:00", "ns")
ATLAS_SDP_EPOCH_GPS = 1198800018.0
SECONDS_PER_YEAR = 24 * 3600 * 365.25

# GPS seconds at which each leap second took effect (1981-07-01 ... 2017-01-01); GPS - UTC = 18 s since 2017
LEAPS_GPS = np.array([46828800, 78364801, 109900802, 173059203, 252028804, 315187205, 346723206, 393984007,
                      425520008, 457056009, 504489610, 551750411, 599184012, 820108813, 914803214, 1025136015,
                      1119744016, 1167264017], dtype=np.float64)
# the same instants as UTC seconds since the GPS epoch (k leap seconds had been inserted before leap k)
LEAPS_UTC = LEAPS_GPS - np.arange(LEAPS_GPS.size)


def gps_minus_utc_at_gps(gps_seconds):
    return np.searchsorted(LEAPS_GPS, np.asarray(gps_seconds, dtype=np.float64), side="right").astype(np.float64)


def gps_minus_utc_at_utc(utc_seconds):
    return np.searchsorted(LEAPS_UTC, np.asarray(utc_seconds, dtype=np.float64), side="right").astype(np.float64)


def utc_seconds_since_gps_epoch(t):
    """UTC seconds (no leap seconds counted) since 1980-01-06 for datetime64 values"""
    return (np.asarray(t, dtype="datetime64[ns]") - GPS_EPOCH) / np.timedelta64(1, "s")


def utc_to_gps(utc_seconds):
    """GPS seconds from UTC seconds since 1980-01-06 (leap seconds not counted)"""
    utc_seconds = np.asarray(utc_seconds, dtype=np.float64)
    return utc_seconds + gps_minus_utc_at_utc(utc_seconds)


def gps_to_utc_datetime(gps_seconds):
    gps_seconds = np.asarray(gps_seconds, dtype=np.float64)
    utc = gps_seconds - gps_minus_utc_at_gps(gps_seconds)
    return GPS_EPOCH + np.round(utc * 1e9).astype("timedelta64[ns]")


def delta_time_from_gps(gps_seconds):
    return np.asarray(gps_seconds, dtype=np.float64) - ATLAS_SDP_EPOCH_GPS


def delta_time_from_utc_datetime(t):
    return delta_time_from_gps(utc_to_gps(utc_seconds_since_gps_epoch(t)))


def year_from_delta_time(delta_time):
    return 2018.0 + np.asarray(delta_time, dtype=np.float64) / SECONDS_PER_YEAR
