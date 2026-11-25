"""
Granules left out of the database, with the reason.

Each entry is a granule name (as in granules.parquet) and why it is excluded.  scripts/process_granules.py never
processes them, scripts/build_database.py does not wait for them, and granules.parquet marks them
status = "excluded" with the reason.
"""

EXCLUDED = {
    # exact duplicates: same start and end time and size in CMR, and byte-identical files (md5)
    "BLATM1B_950530atm1_151023C.qi": "byte-identical copy of BLATM1B_950530atm1_151023.qi",
    "BLATM1B_950530atm1_151406C.qi": "byte-identical copy of BLATM1B_950530atm1_151406.qi",
    # two processings of the same 4,241,431 shots (identical times and signal strengths; elevations differ by
    # 0.30 m).  Where the tracks meet the next file of the flight (BLATM1B_970521atm1_173042sr.qi), sr1200v
    # differs from it by -0.10 m and sr1500v by -0.39 m (571 shot pairs within 2 m), so sr1200v is kept.
    "BLATM1B_970521atm1_165945sr1500v.qi": "second processing of the shots in BLATM1B_970521atm1_165945sr1200v.qi; "
                                           "sr1200v agrees better with the adjacent file 173042sr",
}


def reason(granule_name):
    return EXCLUDED.get(granule_name)
