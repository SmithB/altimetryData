import numpy as np
import pandas as pd

from altimetryData import cmr


def _listing(names, sensor="ATM_wide", short="ILATM1B"):
    return pd.DataFrame({
        "sensor": sensor, "short_name": short, "version": "2", "granule": names,
        "start_time": [f"2015-04-{i + 1:02d}T12:00:00Z" for i in range(len(names))],
        "end_time": [f"2015-04-{i + 1:02d}T12:10:00Z" for i in range(len(names))],
        "lat_min": 69.0, "lat_max": 70.0, "lon_min": -50.0, "lon_max": -49.0, "size_bytes": 1,
    })


def test_ids_frozen():
    first = cmr.finalize_inventory(_listing(["a", "b", "c"]))
    assert first["granule_id"].tolist() == [0, 1, 2] and first["in_cmr"].all()
    # b disappears, d appears (with an earlier start time than c), and a LVIS listing is added
    second_listing = pd.concat([_listing(["a", "d", "c"]), _listing(["l1"], sensor="LVIS", short="ILVIS2")])
    second = cmr.finalize_inventory(second_listing, first).set_index("granule")
    assert second.loc["a", "granule_id"] == 0 and second.loc["c", "granule_id"] == 2
    assert second.loc["b", "granule_id"] == 1 and not second.loc["b", "in_cmr"]
    assert set(second.loc[["d", "l1"], "granule_id"]) == {3, 4}
    assert second["granule_id"].is_unique
    # a partial listing (LVIS only) carries the ATM granules over unchanged
    third = cmr.finalize_inventory(_listing(["l1", "l2"], sensor="LVIS", short="ILVIS2"),
                                   second.reset_index()).set_index("granule")
    assert third.loc["a", "in_cmr"] and not third.loc["b", "in_cmr"] and third.loc["d", "in_cmr"]
    assert third.loc["l1", "granule_id"] == second.loc["l1", "granule_id"] and third.loc["l2", "granule_id"] == 5
    assert np.array_equal(third.sort_values("granule_id")["granule_id"].to_numpy(), np.arange(6))
