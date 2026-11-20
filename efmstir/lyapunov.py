"""Lyapunov exponents: the classical one, finite-time (FTLE) and finite-size (FSLE).

All functions take a flow from :mod:`efmstir.flows` (anything with
``step(x, y, t, dt, M=None)`` and ``period``).

- The **Lyapunov exponent** lambda is the long-time growth rate of an
  infinitesimal separation, delta(t) ~ delta_0 exp(lambda t)
  (Chapter 7 of the notes).  It is a single number for a whole chaotic
  region, reached only as t -> infinity.
- The **finite-time Lyapunov exponent** fixes the time T and measures
  how much an infinitesimal blob around each starting point has been
  stretched by then:  FTLE = ln(sigma_max) / |T|, with sigma_max the
  largest singular value of the Jacobian of the flow map.  It is a field
  over the starting points.  Its ridges are the Lagrangian coherent
  structures: lines of strongest repulsion (forward in time) or
  attraction (backward), along which the dye filaments are drawn.
- The **finite-size Lyapunov exponent** fixes the stretching and
  measures the time:  the time tau it takes two parcels to separate from
  delta to r * delta, and  FSLE(delta) = ln(r) / <tau>.  For small delta
  it tends to lambda; once delta approaches the size of the eddies it
  falls, because a finite separation can grow no faster than the flow
  can carry the parcels apart.  It is the relevant rate for a patch of
  finite size.
"""
import numpy as np


def advect(flow, x, y, t0, t1, jacobian=False):
    """Positions at t1 of parcels at (x, y) at t0, and the Jacobian of the flow map if asked."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    M = np.broadcast_to(np.eye(2), x.shape + (2, 2)).copy() if jacobian else None
    x, y, M = flow.step(x, y, t0, t1 - t0, M)
    return (x, y, M) if jacobian else (x, y)


def stretching(M):
    """Largest singular value of the 2x2 Jacobians M (shape ... x 2 x 2)."""
    C = np.swapaxes(M, -1, -2) @ M                     # Cauchy-Green tensor
    tr = C[..., 0, 0] + C[..., 1, 1]
    det = C[..., 0, 0] * C[..., 1, 1] - C[..., 0, 1] * C[..., 1, 0]
    return np.sqrt(0.5 * (tr + np.sqrt(np.maximum(tr * tr - 4 * det, 0.0))))


def ftle(flow, x, y, t0, T):
    """Finite-time Lyapunov exponent at the starting points (x, y), over the time T.

    T < 0 integrates backward in time: its ridges mark the lines that
    attract parcels, where filaments of dye released before t0 lie.
    """
    _, _, M = advect(flow, x, y, t0, t0 + T, jacobian=True)
    return np.log(stretching(M)) / abs(T)


def lyapunov(flow, x0, y0, n_periods, t0=0.0, seed=0):
    """Running estimate of the Lyapunov exponent along trajectories, per unit time.

    An infinitesimal separation vector is carried along with each parcel
    by the Jacobian and renormalised to unit length after every period,
    as with pairs of parcels reset to their initial distance
    (the method of the exercise in Chapter 7).  Returns the estimate after
    1, 2, ..., n_periods periods, with shape (n_periods,) + x0.shape.
    Multiply by ``flow.period`` for the exponent per period.
    """
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x0, float).copy(), np.asarray(y0, float).copy()
    ang = rng.uniform(0, 2 * np.pi, x.shape)
    w = np.stack([np.cos(ang), np.sin(ang)], axis=-1)[..., None]   # tangent vectors
    total = np.zeros(x.shape)
    out = np.empty((n_periods,) + x.shape)
    t = t0
    for n in range(n_periods):
        x, y, w = flow.step(x, y, t, flow.period, w)
        t += flow.period
        norm = np.sqrt(w[..., 0, 0] ** 2 + w[..., 1, 0] ** 2)
        total += np.log(norm)
        w = w / norm[..., None, None]
        out[n] = total / ((n + 1) * flow.period)
    return out


def fsle_field(flow, x, y, delta0, delta_f, t_max, dt, t0=0.0):
    """Finite-size Lyapunov exponent at each starting point (d'Ovidio et al. 2004).

    Each parcel gets four neighbours at the distance delta0; tau is the
    first time at which one of them is delta_f away, and the FSLE is
    ln(delta_f / delta0) / tau.  Where the separation never reaches
    delta_f within t_max, the FSLE is set to 0 (no stirring at that scale).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    off = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) * delta0
    X = np.concatenate([x[None]] + [x[None] + o[0] for o in off])
    Y = np.concatenate([y[None]] + [y[None] + o[1] for o in off])
    tau = np.full(x.shape, np.nan)
    t = t0
    while t - t0 < t_max - 1e-12:
        X, Y, _ = flow.step(X, Y, t, dt)
        t += dt
        d = np.sqrt((X[1:] - X[0]) ** 2 + (Y[1:] - Y[0]) ** 2).max(axis=0)
        new = np.isnan(tau) & (d >= delta_f)
        tau[new] = t - t0
        if not np.isnan(tau).any():
            break
    lam = np.log(delta_f / delta0) / tau
    return np.where(np.isnan(lam), 0.0, lam)


def fsle_curve(flow, x0, y0, delta0, ratio=2.0, n_scales=30, t_max=200.0, dt=None,
               t0=0.0, seed=0):
    """FSLE as a function of the separation (Aurell et al. 1997; Artale et al. 1997).

    Pairs of parcels start delta0 apart, in random directions, at the
    points (x0, y0).  For the scales delta_k = delta0 ratio^k, the time
    each pair takes to grow from delta_k to delta_{k+1} is recorded, and

        FSLE(delta_k) = ln(ratio) / <time from delta_k to delta_{k+1}>,

    averaged over the pairs that got there.  Returns (delta_k, FSLE, n_pairs).

    By default the separations are compared once per period, at the same
    phase of the flow, as in a Poincare section (dt = flow.period).  In the
    blinking vortex, comparing them more often lowers the FSLE well below
    the Lyapunov exponent: two parcels circling a vortex at slightly
    different radii lap one another, so the distance between them swings
    up and down within a period, and a threshold is often first crossed on
    one of those swings.
    """
    rng = np.random.default_rng(seed)
    dt = flow.period if dt is None else dt
    x0, y0 = np.ravel(x0).astype(float), np.ravel(y0).astype(float)
    ang = rng.uniform(0, 2 * np.pi, x0.size)
    xa, ya = x0, y0
    xb, yb = x0 + delta0 * np.cos(ang), y0 + delta0 * np.sin(ang)
    scales = delta0 * ratio ** np.arange(n_scales + 1)
    first = np.full((n_scales + 1, x0.size), np.nan)        # first time beyond each scale
    first[0] = 0.0
    t = t0
    while t - t0 < t_max - 1e-12:
        xa, ya, _ = flow.step(xa, ya, t, dt)
        xb, yb, _ = flow.step(xb, yb, t, dt)
        t += dt
        d = np.hypot(xb - xa, yb - ya)
        reached = (d[None] >= scales[:, None]) & np.isnan(first)
        first[reached] = t - t0
        if not np.isnan(first[-1]).any():
            break
    steps = np.diff(first, axis=0)                          # time from delta_k to delta_{k+1}
    n = np.sum(~np.isnan(steps), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(np.where(n[:, None] > 0, steps, 0.0), axis=1)
        lam = np.where(n > 0, np.log(ratio) / mean, np.nan)
    return scales[:-1], lam, n


def outline(x0, y0, radius, n=400):
    """Closed outline of a circular blob."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return x0 + radius * np.cos(th), y0 + radius * np.sin(th)


def refine(x, y, h):
    """Insert points along a closed outline wherever two neighbours are more than h apart."""
    dx = np.diff(np.r_[x, x[0]])
    dy = np.diff(np.r_[y, y[0]])
    k = np.maximum(1, np.ceil(np.hypot(dx, dy) / h).astype(int))
    if k.max() == 1:
        return x, y
    i = np.repeat(np.arange(len(x)), k)
    f = (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)) / k[i]
    return x[i] + f * dx[i], y[i] + f * dy[i]


def length(x, y):
    """Length of a closed outline."""
    return float(np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]])).sum())


def follow_outline(flow, x, y, t0, n_periods, h=0.01, substeps=16, max_points=2_000_000):
    """Carry the outline of a blob, a material line, over n_periods periods.

    The outline is refined as it stretches, so that filaments stay sharp.
    Returns a list of (x, y) after 0, 1, ..., n_periods periods; it stops
    early if the outline would need more than max_points points.
    """
    out = [(x.copy(), y.copy())]
    dt = flow.period / substeps
    t = t0
    for _ in range(n_periods):
        for _ in range(substeps):
            x, y, _ = flow.step(x, y, t, dt)
            t += dt
            x, y = refine(x, y, h)
            if len(x) > max_points:
                return out
        out.append((x.copy(), y.copy()))
    return out
