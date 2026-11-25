import numpy as np
import pytest

from altimetryData import reduce as rd


def _swath(n=60000, seed=1):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 3000, n))
    th = rng.uniform(0, 2 * np.pi, n)
    x = t + 36 * np.cos(th) - 2.0e5
    y = 120 * np.sin(th) - 1.5e6
    z = 1000 + 0.02 * x - 0.01 * y + 0.3 * np.sin(x / 30) + rng.normal(0, 0.1, n)
    out = rng.random(n) < 0.03
    z[out] += rng.normal(0, 10, out.sum())
    return x, y, np.round(z, 3)


def test_plane_recovered():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 500, 20000)
    y = rng.uniform(0, 500, 20000)
    z = 100 + 0.05 * x - 0.02 * y
    stats = rd.local_stats(x, y, z, radius=50)
    ok = np.isfinite(stats["bias"])
    assert ok.mean() > 0.99
    # the weak slope damping of the original algorithm (0.001 per km) biases slopes by ~1e-6
    assert np.allclose(stats["slope_x"][ok], 0.05, atol=1e-5)
    assert np.allclose(stats["slope_y"][ok], -0.02, atol=1e-5)
    assert np.nanmax(np.abs(stats["bias"])) < 1e-3


def test_matches_pointcollection():
    pc = pytest.importorskip("pointCollection")
    if not hasattr(np, "in1d"):
        np.in1d = np.isin  # calc_bias_and_R predates numpy 2.4
    x, y, z = _swath()
    im_ref = pc.pt_blockmedian(x, y, z, 10, break_ties=True, index_only=True)
    im = rd.blockmedian_index(x, y, z, 10)
    assert np.array_equal(im, im_ref)
    sub = np.arange(0, im.size, 7)  # every 7th decimated point keeps the reference loop quick
    D = pc.data().from_dict({"x": x[im], "y": y[im], "z": z[im]})
    stats = rd.local_stats(D.x, D.y, D.z, 50, 0.5)
    from scipy.spatial import KDTree
    kdt = KDTree(np.c_[D.x, D.y])
    for i0 in sub:
        i1 = np.array(kdt.query_ball_point([D.x[i0], D.y[i0]], 50))
        if len(i1) <= 5:
            assert np.isnan(stats["bias"][i0])
            continue
        b, n, sl, N = pc.calc_bias_and_R(D, i0, i1, 50, min_sigma=0.5)
        assert stats["N"][i0] == N
        assert abs(stats["bias"][i0] - b) < 1e-8
        assert abs(stats["noise"][i0] - n) < 1e-8
        assert abs(stats["slope_x"][i0] - sl[0]) < 1e-10
        assert abs(stats["slope_y"][i0] - sl[1]) < 1e-10
