"""
Source reference frame of each granule.

Order of precedence:
  1. a reference frame stored in the granule (ATM HDF5 ancillary_data/reference_frame),
  2. the frame tag of the GPS trajectory named in the processing header (e.g. '110412_aa_l12_cfm_itrf05_...'),
  3. a fixed frame for the product (from its documentation),
  4. the frame of the IGS precise orbits in use on the collection date (GPS trajectories computed with IGS
     orbits inherit that frame).
The source used is recorded per granule (frame_source in the granule table) so defaulted granules can be found.
"""

import re
import numpy as np

# IGS reference frame realizations: (first day in use, ITRF it is aligned to)
_IGS_FRAMES = [
    ("1994-01-02", "ITRF92"),
    ("1995-01-01", "ITRF93"),
    ("1996-06-30", "ITRF94"),
    ("1998-03-01", "ITRF96"),
    ("1999-08-01", "ITRF97"),
    ("2000-06-04", "ITRF2000"),
    ("2006-11-05", "ITRF2005"),
    ("2011-04-17", "ITRF2008"),
    ("2017-01-29", "ITRF2014"),
    ("2022-11-27", "ITRF2020"),
]

_TAG_TO_FRAME = {"88": "ITRF88", "89": "ITRF89", "90": "ITRF90", "91": "ITRF91", "92": "ITRF92", "93": "ITRF93",
                 "94": "ITRF94", "96": "ITRF96", "97": "ITRF97", "00": "ITRF2000", "05": "ITRF2005",
                 "08": "ITRF2008", "14": "ITRF2014", "20": "ITRF2020"}


def igs_frame(date):
    """ITRF of the IGS orbits in use on a date (numpy datetime64 or ISO string)"""
    date = np.datetime64(date, "D")
    frame = "ITRF91"
    for start, name in _IGS_FRAMES:
        if date >= np.datetime64(start):
            frame = name
    return frame


def normalize_frame(text):
    """'ITRF2008', 'itrf08', 'ITRF 2005' -> 'ITRF2008', ...; None if not recognized"""
    m = re.search(r"itrf[\s_-]*(\d{4}|\d{2})", text, flags=re.IGNORECASE)
    if not m:
        return None
    tag = m.group(1)
    if len(tag) == 4:
        tag = tag[2:]
    return _TAG_TO_FRAME.get(tag)


def frame_from_header(header):
    """frame tag of the first trajectory/reference mentioned in a processing header"""
    for line in header.replace("\r", "\n").splitlines():
        frame = normalize_frame(line)
        if frame:
            return frame
    return None


def atm_frame(data, date):
    """(frame, source) for an ATM granule read by readers.atm"""
    ancillary = data.get("ancillary", {})
    if "reference_frame" in ancillary:
        value = ancillary["reference_frame"]
        value = value[0] if np.ndim(value) else value
        value = value.decode() if isinstance(value, bytes) else str(value)
        frame = normalize_frame(value)
        if frame:
            return frame, "granule"
    header = data.get("header", "")
    if not header and "header_text" in ancillary:
        value = ancillary["header_text"]
        value = value[0] if np.ndim(value) else value
        header = value.decode("latin-1", errors="replace") if isinstance(value, bytes) else str(value)
    frame = frame_from_header(header)
    if frame:
        return frame, "header"
    return igs_frame(date), "igs_default"
