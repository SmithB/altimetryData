"""
Vectorized S2 cell ids.

This follows the reference S2 geometry implementation (point -> face, u, v -> quadratic s, t -> i, j -> Hilbert
position via the 4-bit lookup table), so the ids match those that sliderule's atl06t builder computes with
duckdb's geography extension (s2_cellfromlonlat followed by s2_cell_parent).  tests/test_s2.py checks that.
"""

import numpy as np

MAX_LEVEL = 30
_LOOKUP_BITS = 4
_SWAP_MASK = 1
_INVERT_MASK = 2
_POS_TO_IJ = ((0, 1, 3, 2), (0, 2, 3, 1), (3, 2, 0, 1), (3, 1, 0, 2))
_POS_TO_ORIENTATION = (_SWAP_MASK, 0, 0, _INVERT_MASK | _SWAP_MASK)


def _build_lookup():
    lookup_pos = np.zeros(1 << (2 * _LOOKUP_BITS + 2), dtype=np.uint64)

    def init(level, i, j, orig_orientation, pos, orientation):
        if level == _LOOKUP_BITS:
            ij = (i << _LOOKUP_BITS) + j
            lookup_pos[(ij << 2) + orig_orientation] = (pos << 2) + orientation
            return
        level, i, j, pos = level + 1, i << 1, j << 1, pos << 2
        r = _POS_TO_IJ[orientation]
        for k in range(4):
            init(level, i + (r[k] >> 1), j + (r[k] & 1), orig_orientation, pos + k,
                 orientation ^ _POS_TO_ORIENTATION[k])

    for orientation in range(4):
        init(0, 0, 0, orientation, 0, orientation)
    return lookup_pos


_LOOKUP_POS = _build_lookup()


def _uv_to_st(u):
    with np.errstate(invalid="ignore"):  # np.where evaluates both branches
        return np.where(u >= 0, 0.5 * np.sqrt(1 + 3 * u), 1 - 0.5 * np.sqrt(1 - 3 * u))


def _st_to_ij(s):
    return np.clip(np.floor(s * (1 << MAX_LEVEL)), 0, (1 << MAX_LEVEL) - 1).astype(np.uint64)


def cell_ids(lon, lat, level=MAX_LEVEL):
    """S2 cell ids (uint64) at a level for longitude and latitude in degrees"""
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    cos_lat = np.cos(lat)
    x, y, z = cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    face = np.where(ax > ay, np.where(ax > az, 0, 2), np.where(ay > az, 1, 2))
    comp = np.choose(face, (x, y, z))
    face = np.where(comp < 0, face + 3, face).astype(np.int64)
    u = np.empty_like(x)
    v = np.empty_like(x)
    for f, (un, ud, vn, vd) in enumerate((
            (y, x, z, x), (-x, y, z, y), (-x, z, -y, z),
            (z, x, y, x), (z, y, -x, y), (-y, z, -x, z))):
        sel = face == f
        u[sel] = un[sel] / ud[sel]
        v[sel] = vn[sel] / vd[sel]
    i = _st_to_ij(_uv_to_st(u))
    j = _st_to_ij(_uv_to_st(v))

    n = face.astype(np.uint64) << np.uint64(60)
    bits = (face & _SWAP_MASK).astype(np.uint64)
    mask = np.uint64((1 << _LOOKUP_BITS) - 1)
    for k in range(7, -1, -1):
        shift = np.uint64(k * _LOOKUP_BITS)
        bits = bits + (((i >> shift) & mask) << np.uint64(_LOOKUP_BITS + 2))
        bits = bits + (((j >> shift) & mask) << np.uint64(2))
        bits = _LOOKUP_POS[bits.astype(np.intp)]
        n |= (bits >> np.uint64(2)) << np.uint64(k * 2 * _LOOKUP_BITS)
        bits &= np.uint64(_SWAP_MASK | _INVERT_MASK)
    ids = n * np.uint64(2) + np.uint64(1)
    return parent(ids, level) if level < MAX_LEVEL else ids


def lsb_for_level(level):
    return np.uint64(1) << np.uint64(2 * (MAX_LEVEL - level))


def parent(ids, level):
    """parent cell ids at a level (ids must be at that level or finer)"""
    lsb = lsb_for_level(level)
    ids = np.asarray(ids, dtype=np.uint64)
    return (ids & ~(lsb - np.uint64(1))) | lsb


def level(cell_id):
    """level of a single (python int) cell id"""
    cell_id = int(cell_id)
    lsb = cell_id & -cell_id
    return MAX_LEVEL - ((lsb.bit_length() - 1) >> 1)


def cell_range(ids):
    """[lo, hi] range of leaf cell ids covered by each cell"""
    ids = np.asarray(ids, dtype=np.uint64)
    ext = (ids & (~ids + np.uint64(1))) - np.uint64(1)
    return ids - ext, ids + ext
