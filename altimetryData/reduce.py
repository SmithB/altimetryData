"""
Fast point-cloud reduction: block-median decimation followed by local robust plane-fit statistics.

This is a re-implementation of pointCollection/scripts/reduce_ATM.py that produces the same output:

1. blockmedian_index() reproduces pointCollection.pt_blockmedian(..., break_ties=True, index_only=True), with
   the same sort key and the same tie-breaking, but with the per-bin python loop replaced by array operations.
2. local_stats() reproduces the per-point loop of reduce_ATM.reduce_data / pointCollection.calc_bias_and_R
   (KD-tree ball query, iterative three-sigma-edited plane fit with weak slope constraints, RDE spread) in a
   parallel numba kernel.  Neighbors come from a sorted grid hash with cell size equal to the search radius,
   so each point only examines the 3x3 cells around it.

The plane fit solves the regularized normal equations, (G'G + diag(1e-6, 1e-6, 0)) m = G'dz, which is the
same least-squares problem that the original solves with scipy.linalg.lstsq on [G; 0.001*I(2x3)].
"""

import numpy as np
import numba

STATS_FIELDS = ("bias", "noise", "N", "slope_x", "slope_y")


def blockmedian_index(x, y, z, delta, xy0=(0.0, 0.0)):
    """
    index of the median-z point in each delta x delta bin, breaking ties between the two central points of
    even-count bins in favor of the point closer to the bin center.  Bins come back in the same order as
    pointCollection.pt_blockmedian returns them.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(z)
    out_ind = np.flatnonzero(finite)
    if out_ind.size == 0:
        return np.zeros(0, dtype=np.int64)
    x, y, z = x[finite], y[finite], z[finite]
    # same composite sort key as pt_blockmedian, so ties sort identically
    yscale = np.maximum(1.1, np.ceil((np.nanmax(y) - np.nanmin(y)) / delta * 1.1))
    zscale = (np.nanmax(z) - np.nanmin(z)) * 1.1
    xr = np.floor((x - xy0[0]) / delta)
    yr = np.floor((y - xy0[1]) / delta)
    xyind = xr * yscale + (yr - np.min(yr))
    sorted_ind = np.argsort(xyind + (z - np.nanmin(z)) / zscale)
    xyind = xyind[sorted_ind]
    _, first = np.unique(xyind, return_index=True)
    count = np.diff(np.append(first, xyind.size))
    # odd counts: the central element; even counts: whichever central element is closer to the bin center
    lo = first + (count - 1) // 2
    even = (count % 2 == 0) & (count > 1)
    lo_e = lo[even]
    hi_e = lo_e + 1
    xs, ys = x[sorted_ind], y[sorted_ind]
    xc = xr[sorted_ind] * delta + xy0[0] + delta / 2
    yc = yr[sorted_ind] * delta + xy0[1] + delta / 2
    dr2_lo = (xs[lo_e] - xc[lo_e]) ** 2 + (ys[lo_e] - yc[lo_e]) ** 2
    dr2_hi = (xs[hi_e] - xc[hi_e]) ** 2 + (ys[hi_e] - yc[hi_e]) ** 2
    pick = lo.copy()
    pick[even] = np.where(dr2_lo < dr2_hi, lo_e, hi_e)
    return out_ind[sorted_ind[pick]]


@numba.njit(cache=True)
def _rde(r):
    """same as pointCollection.RDE for a finite vector: half the 16-84 percentile spread"""
    n = r.size
    if n < 2:
        return np.nan
    rs = np.sort(r)
    ind = np.arange(n) + 0.5
    lo = np.interp(0.16 * n, ind, rs)
    hi = np.interp(0.84 * n, ind, rs)
    return (hi - lo) / 2.0


@numba.njit(cache=True)
def _fit_point(dx, dy, dz, min_sigma, out):
    """iterative plane fit for one point; mirrors pointCollection.calc_bias_and_R"""
    n = dx.size
    ii = np.ones(n, dtype=np.bool_)
    m = np.zeros(3)
    r = np.zeros(n)
    count = 0
    n_ii = n
    while n_ii > 4:
        count += 1
        # normal equations for [G[ii]; 0.001*[1 0 0; 0 1 0]] m = [dz[ii]; 0; 0], G = [dx/1000, dy/1000, 1]
        a00 = 1e-6
        a11 = 1e-6
        a01 = 0.0
        a02 = 0.0
        a12 = 0.0
        a22 = 0.0
        b0 = 0.0
        b1 = 0.0
        b2 = 0.0
        for k in range(n):
            if ii[k]:
                g0 = dx[k] / 1000.0
                g1 = dy[k] / 1000.0
                a00 += g0 * g0
                a01 += g0 * g1
                a02 += g0
                a11 += g1 * g1
                a12 += g1
                a22 += 1.0
                b0 += g0 * dz[k]
                b1 += g1 * dz[k]
                b2 += dz[k]
        A = np.array([[a00, a01, a02], [a01, a11, a12], [a02, a12, a22]])
        b = np.array([b0, b1, b2])
        m = np.linalg.solve(A, b)
        for k in range(n):
            r[k] = dz[k] - (dx[k] / 1000.0 * m[0] + dy[k] / 1000.0 * m[1] + m[2])
        sigma_hat = _rde(r[ii])
        if count > 5:
            break
        threshold = 3.0 * max(sigma_hat, min_sigma) if np.isfinite(sigma_hat) else np.nan
        changed = False
        n_ii = 0
        for k in range(n):
            new = abs(r[k]) < threshold
            if new != ii[k]:
                changed = True
            ii[k] = new
            if new:
                n_ii += 1
        if not changed:
            break
    n_good = 0
    s = 0.0
    for k in range(n):
        if ii[k]:
            n_good += 1
            s += r[k]
    if n_good > 0:
        mean = s / n_good
        ss = 0.0
        for k in range(n):
            if ii[k]:
                ss += (r[k] - mean) ** 2
        noise = np.sqrt(ss / n_good)
    else:
        noise = np.nan
    out[0] = -m[2]
    out[1] = noise
    out[2] = n_good
    out[3] = m[0] / 1000.0
    out[4] = m[1] / 1000.0


@numba.njit(parallel=True, cache=True)
def _local_stats_kernel(x, y, z, order, keys, ncy, cx, cy, radius, min_sigma, min_points, out):
    n = x.size
    r2 = radius * radius
    for p in numba.prange(n):
        # candidate ranges: rows cx-1..cx+1, each spanning columns cy-1..cy+1, are contiguous in key order
        n_cand = 0
        starts = np.empty(3, dtype=np.int64)
        stops = np.empty(3, dtype=np.int64)
        for d in range(3):
            row = cx[p] + d - 1
            k_lo = row * ncy + cy[p] - 1
            k_hi = row * ncy + cy[p] + 1
            starts[d] = np.searchsorted(keys, k_lo, side="left")
            stops[d] = np.searchsorted(keys, k_hi, side="right")
            n_cand += stops[d] - starts[d]
        dx = np.empty(n_cand)
        dy = np.empty(n_cand)
        dz = np.empty(n_cand)
        n_ball = 0      # points within the radius, including p itself (what query_ball_point returns)
        n_nb = 0        # the same, excluding p
        for d in range(3):
            for s in range(starts[d], stops[d]):
                q = order[s]
                ddx = x[q] - x[p]
                ddy = y[q] - y[p]
                if ddx * ddx + ddy * ddy <= r2:
                    n_ball += 1
                    if q != p:
                        dx[n_nb] = ddx
                        dy[n_nb] = ddy
                        dz[n_nb] = z[q] - z[p]
                        n_nb += 1
        if n_ball <= min_points:
            continue
        _fit_point(dx[:n_nb], dy[:n_nb], dz[:n_nb], min_sigma, out[p])


def local_stats(x, y, z, radius=50.0, min_sigma=0.5, min_points=5):
    """
    for each point, a robust plane fit to its neighbors within `radius`:
    bias (point minus fit), noise (rms of edited residuals), N (points kept), slope_x, slope_y.
    Points with `min_points` or fewer neighbors within the radius (counting the point itself) get NaN.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    z = np.ascontiguousarray(z, dtype=np.float64)
    out = np.full((x.size, 5), np.nan)
    if x.size == 0:
        return {k: out[:, i] for i, k in enumerate(STATS_FIELDS)}
    # one-cell margin on every side so cx-1 and cy-1 are never negative
    cx = np.floor((x - x.min()) / radius).astype(np.int64) + 1
    cy = np.floor((y - y.min()) / radius).astype(np.int64) + 1
    ncy = int(cy.max()) + 2
    keys = cx * ncy + cy
    order = np.argsort(keys, kind="stable")
    _local_stats_kernel(x, y, z, order, keys[order], ncy, cx, cy, float(radius), float(min_sigma),
                        int(min_points), out)
    return {k: out[:, i] for i, k in enumerate(STATS_FIELDS)}


def reduce_points(x, y, z, decimate_to=10.0, stats_radius=50.0, min_sigma=0.5):
    """
    block-median decimate, then compute local statistics among the decimated points (as reduce_ATM does).
    Returns (index of each kept point in the input arrays, dict of statistics for the kept points).
    """
    if decimate_to > 0:
        keep = blockmedian_index(x, y, z, decimate_to)
    else:
        keep = np.flatnonzero(np.isfinite(z))
    stats = local_stats(np.asarray(x)[keep], np.asarray(y)[keep], np.asarray(z)[keep],
                        radius=stats_radius, min_sigma=min_sigma)
    return keep, stats
