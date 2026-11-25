"""
Readers for the database.

read_granule() returns every point of one raw granule.  A granule's points are spread over many row groups of
a data file (rows are sorted by S2 cell, not by granule), so it uses the granule index: granule_index.parquet
lists, for each granule, the (region, year, row group) triples that hold its points.  Only those row groups are
read, then filtered on granule_id.  Without the index, the granule_id footer statistics still let a reader skip
row groups whose granule_id range cannot contain the granule (see read_granule(..., use_index=False)).
"""

import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from . import database as db


def granule_row_groups(root, sensor, resolution, granule_id):
    """{(region, year): [row groups]} holding the granule's points, from granule_index.parquet"""
    path = db.granule_index_path(root, sensor, resolution)
    t = pq.read_table(path, filters=[("granule_id", "=", int(granule_id))])
    out = {}
    for region, year, rg in zip(t.column("region").to_pylist(), t.column("year").to_pylist(),
                                t.column("row_group").to_pylist()):
        out.setdefault((region, int(year)), []).append(int(rg))
    return out


def read_granule(root, granule_id, resolution="native", columns=None, sensor=None, use_index=True):
    """
    every point of a granule as {region: pyarrow.Table} (the regions have different projections).  sensor is
    looked up in granules.parquet if not given.  use_index=False skips the granule index and relies on the
    granule_id footer statistics of the files of the granule's year(s)
    """
    granule_id = int(granule_id)
    if sensor is None or not use_index:
        g = pq.read_table(os.path.join(root, "granules.parquet"), filters=[("granule_id", "=", granule_id)],
                          columns=["sensor", "year"]).to_pylist()
        if not g:
            raise KeyError(f"granule_id {granule_id} is not in granules.parquet")
        sensor = sensor or g[0]["sensor"]
    if use_index:
        groups = granule_row_groups(root, sensor, resolution, granule_id)
    else:
        # the granule's start year, and the next one in case it runs past midnight on 31 December
        year = int(g[0]["year"])
        groups = {(r, y): None for r in ("GL", "AA") for y in (year, year + 1)}
    out = {}
    for (region, year), row_groups in sorted(groups.items()):
        path = db.data_path(root, sensor, resolution, region, year)
        if not os.path.isfile(path):
            continue
        if row_groups is not None:
            pf = pq.ParquetFile(path)
            table = pf.read_row_groups(row_groups, columns=None if columns is None else
                                       list(dict.fromkeys(["granule_id", *columns])))
            table = table.filter(pc.equal(table.column("granule_id"), pa.scalar(granule_id, pa.uint32())))
        else:
            table = ds.dataset(path).to_table(columns=columns, filter=pc.field("granule_id") == granule_id)
        if columns is not None:
            table = table.select(columns)
        if table.num_rows:
            out[region] = table if region not in out else pa.concat_tables([out[region], table])
    return out
