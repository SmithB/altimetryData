#! /usr/bin/env python
"""
Estimate download volume, output size, and processing time from the granule inventory.

    python scripts/estimate.py --granules granules.parquet [--download_MBps 85] [--cores 96]

The rates below were measured on granules of each format: points per MB of raw
granule, single-core processing seconds per MB (read, frame change, projection, S2 cells, reduction, and
partition writing), the fraction of native points kept by the reduction, and parquet bytes per row
(see the estimates section of README.md).
"""

import argparse

import pandas as pd

# per (short_name, version): points per raw MB, single-core s per raw MB, reduced fraction
RATES = {
    ("BLATM1B", "1"): (20800, 0.060, 0.128),
    ("ILATM1B", "1"): (20800, 0.060, 0.128),
    ("ILNSA1B", "1"): (20800, 0.060, 0.128),
    ("ILATM1B", "2"): (62800, 0.150, 0.129),
    ("ILNSA1B", "2"): (62800, 0.150, 0.129),
    ("GLAH12", "034"): (12200, 0.054, None),
    ("BLVIS2", "1"): (19200, 0.073, 0.57),
    ("ILVIS2", "1"): (7500, 0.035, 0.57),
    ("ILVGH2", "1"): (7500, 0.019, 0.12),
    ("ILVIS2", "2"): (2800, 0.011, 0.08),
}
# parquet bytes per row (native, reduced) in format version 2 (native point x/y, no lat/lon, delta-coded
# source_index and shotnumber, mantissa trimming, zstd level 9), measured on single-granule files (ICESat: the
# full 2003-2009 database)
BYTES_PER_ROW = {"ATM_wide": (15.6, 30.4), "ATM_narrow": (15.6, 30.4), "ICESat": (32.5, 0.0), "LVIS": (15.0, 30.2),
                 "LVIS_GH": (15.0, 30.2)}
BUILD_US_PER_ROW = 1.08 + 1.50   # data file + stats file (stats use 8 threads for the inverse projection)
CONCURRENCY_SLOWDOWN = 1.5       # per-granule time with every core busy vs one isolated process


def estimate(granules):
    g = granules[granules["regions"].str.len() > 0].copy()
    key = list(zip(g["short_name"], g["version"]))
    g["MB"] = g["size_bytes"] / 1e6
    g["points"] = [mb * RATES[k][0] for mb, k in zip(g["MB"], key)]
    g["cpu_s"] = [mb * RATES[k][1] * CONCURRENCY_SLOWDOWN for mb, k in zip(g["MB"], key)]
    g["reduced_points"] = [p * (RATES[k][2] or 0) for p, k in zip(g["points"], key)]
    g["region"] = g["regions"].str.split().str[0]
    out = g.groupby(["sensor", "region", "year"]).agg(granules=("granule", "size"), raw_GB=("MB", lambda x: x.sum() / 1e3),
                                                     points_G=("points", lambda x: x.sum() / 1e9),
                                                     reduced_points_G=("reduced_points", lambda x: x.sum() / 1e9),
                                                     cpu_h=("cpu_s", lambda x: x.sum() / 3600))
    sensors = out.index.get_level_values("sensor")
    out["native_GB"] = out["points_G"] * [BYTES_PER_ROW[s][0] for s in sensors]
    out["reduced_GB"] = out["reduced_points_G"] * [BYTES_PER_ROW[s][1] for s in sensors]
    out["build_cpu_h"] = (out["points_G"] + out["reduced_points_G"]) * 1e9 * BUILD_US_PER_ROW / 1e6 / 3600
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--granules", required=True)
    parser.add_argument("--download_MBps", type=float, default=85.0, help="aggregate download rate")
    parser.add_argument("--cores", type=int, default=96)
    parser.add_argument("--by_year", action="store_true")
    args = parser.parse_args()
    est = estimate(pd.read_parquet(args.granules))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    if args.by_year:
        print(est.round(2).to_string())
    by_sensor = est.groupby(level="sensor").sum()
    by_sensor.loc["TOTAL"] = by_sensor.sum()
    by_sensor["download_h"] = by_sensor["raw_GB"] * 1e3 / args.download_MBps / 3600
    by_sensor["process_h"] = by_sensor["cpu_h"] / args.cores
    print(by_sensor.round(2).to_string())
    largest = est.sort_values("points_G", ascending=False).head(5)
    print("\nlargest files (sensor, region, year):")
    print(largest[["points_G", "native_GB", "build_cpu_h"]].round(2).to_string())
    print(f"\nbuild wall time is bounded by the largest file: ~{largest['build_cpu_h'].iloc[0]:.1f} h")


if __name__ == "__main__":
    main()
