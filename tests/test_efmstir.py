"""Run with ``pytest``."""
import numpy as np
import pytest

import efmstir
from efmstir import lyapunov as L

FLOWS = [(efmstir.BlinkingVortex(0.8), 2.3, (0.31, 0.47)),
         (efmstir.TidalEddies(), 2.3 * 2 * np.pi, (0.31, 1.1))]


@pytest.mark.parametrize("flow,T,xy", FLOWS)
@pytest.mark.parametrize("sign", [1, -1])
def test_jacobian_matches_finite_differences(flow, T, xy, sign):
    x0, y0 = np.array([xy[0]]), np.array([xy[1]])
    t0, t1 = 0.2, 0.2 + sign * T
    _, _, M = L.advect(flow, x0, y0, t0, t1, jacobian=True)
    h = 1e-6
    xp, yp = L.advect(flow, x0 + h, y0, t0, t1)
    xm, ym = L.advect(flow, x0 - h, y0, t0, t1)
    xq, yq = L.advect(flow, x0, y0 + h, t0, t1)
    xr, yr = L.advect(flow, x0, y0 - h, t0, t1)
    fd = np.array([[(xp - xm)[0], (xq - xr)[0]], [(yp - ym)[0], (yq - yr)[0]]]) / (2 * h)
    assert np.allclose(M[0], fd, rtol=1e-6, atol=1e-6 * np.abs(fd).max())
    assert np.linalg.det(M[0]) == pytest.approx(1.0, abs=1e-6)     # area is conserved


@pytest.mark.parametrize("flow,T,xy", FLOWS)
def test_forward_then_backward_returns(flow, T, xy):
    x0, y0 = np.array([xy[0]]), np.array([xy[1]])
    x, y = L.advect(flow, x0, y0, 0.0, T)
    xb, yb = L.advect(flow, x, y, T, 0.0)
    tol = 1e-12 if isinstance(flow, efmstir.BlinkingVortex) else 1e-5   # exact map; RK4 with 64 steps per tide
    assert abs(xb - x0)[0] < tol and abs(yb - y0)[0] < tol


def test_blinking_vortex_period_map():
    """One period: a turn about (-1, 0), then about (+1, 0), through mu / r^2."""
    mu = 0.5
    z = 0.2 + 0.1j
    for c in (-1.0, 1.0):
        z = c + (z - c) * np.exp(1j * mu / abs(z - c) ** 2)
    x, y = efmstir.BlinkingVortex(mu).period_map(np.array(0.2), np.array(0.1))
    assert x == pytest.approx(z.real) and y == pytest.approx(z.imag)


def test_lyapunov_sea_and_island():
    """mu = 0.8: about 1.3 per period in the chaotic sea (as in the notes), zero on the island."""
    lam = L.lyapunov(efmstir.BlinkingVortex(0.8), np.array([0.0, 0.0]), np.array([0.0, 0.22]), 1000)[-1]
    assert 1.1 < lam[0] < 1.5
    assert lam[1] < 0.02


def test_tidal_flow_is_closed_and_incompressible():
    te = efmstir.TidalEddies()
    x = np.linspace(-3, 3, 7)
    for t in (0.0, 1.0, 2.5):
        assert np.allclose(te.velocity(x, 0 * x, t)[1], 0) and np.allclose(te.velocity(x, np.pi + 0 * x, t)[1], 0)
    G = te.gradient(x, x / 2)
    assert np.allclose(G[..., 0, 0] + G[..., 1, 1], 0)


def test_outline_refinement_and_length():
    x, y = L.outline(0.0, 0.0, 1.0, n=50)
    xr, yr = L.refine(x, y, 0.01)
    assert np.hypot(np.diff(xr), np.diff(yr)).max() <= 0.01 + 1e-12
    assert L.length(xr, yr) == pytest.approx(L.length(x, y))
    assert L.length(*L.outline(0, 0, 1.0, n=4000)) == pytest.approx(2 * np.pi, rel=1e-5)


def test_fsle_plateau_equals_lyapunov_exponent():
    flow = efmstir.BlinkingVortex(0.8)
    rng = np.random.default_rng(3)
    x0, y0 = rng.uniform(-0.7, -0.2, 300), rng.uniform(-0.3, 0.0, 300)
    delta, lam, n = L.fsle_curve(flow, x0, y0, delta0=1e-10, ratio=2.0, n_scales=30, t_max=200)
    plateau = np.median(lam[(delta > 1e-6) & (delta < 1e-2)])
    assert 1.0 < plateau < 1.5
