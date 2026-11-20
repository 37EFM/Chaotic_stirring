"""Two time-periodic two-dimensional flows that stir chaotically.

Each flow advances parcels, and optionally the Jacobian of the flow map
(the matrix of derivatives of the final position with respect to the
initial one), from a time t to t + dt.  Positions are numpy arrays x, y
of any shape; the Jacobian M has the shape x.shape + (2, 2).  The
functions in :mod:`efmstir.lyapunov` work with either flow.
"""
import numpy as np


class BlinkingVortex:
    """Aref's (1984) blinking vortex, as in Chapter 7 of the lecture notes.

    Two point vortices sit at x = -a and x = +a, here a = 1.  During the
    first half of each period only the left one is on, during the second
    half only the right one.  A parcel at distance r from the active
    vortex circles it, turning through the angle mu a^2 / r^2 in a half
    period, so

        mu = Gamma T / (4 pi a^2)

    is the angle through which a parcel at distance a turns while the
    vortex is on.  Time is in periods: T = 1.  Within a half period the
    rotation is uniform in time, so the motion is known exactly at any
    instant and no numerical integration is needed.
    """

    period = 1.0
    vortices = (-1.0, 1.0)

    def __init__(self, mu):
        self.mu = float(mu)

    def __repr__(self):
        return f"BlinkingVortex(mu={self.mu})"

    def active(self, t):
        """x position of the vortex that is on at time t."""
        return np.where((np.asarray(t) % 1.0) < 0.5, -1.0, 1.0)

    def _rotate(self, x, y, c, frac, M=None):
        """Turn about the vortex at (c, 0) for the fraction frac of a half period."""
        dx, dy = x - c, y
        r2 = np.maximum(dx * dx + dy * dy, 1e-12)
        phi = frac * self.mu / r2
        cs, sn = np.cos(phi), np.sin(phi)
        xn, yn = c + cs * dx - sn * dy, cs * dy + sn * dx
        if M is None:
            return xn, yn, None
        # derivative of R(phi) d with respect to d:  R(phi) [I + J d grad(phi)^T],
        # J the rotation by 90 degrees and grad(phi) = -2 phi d / r^2
        g = -2.0 * phi / r2
        a11 = 1.0 - dy * g * dx
        a12 = -dy * g * dy
        a21 = dx * g * dx
        a22 = 1.0 + dx * g * dy
        D = np.empty(x.shape + (2, 2))
        D[..., 0, 0] = cs * a11 - sn * a21
        D[..., 0, 1] = cs * a12 - sn * a22
        D[..., 1, 0] = sn * a11 + cs * a21
        D[..., 1, 1] = sn * a12 + cs * a22
        return xn, yn, D @ M

    def step(self, x, y, t, dt, M=None):
        """Advance from t to t + dt (dt may be negative), splitting at the switches."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        t_end, sgn = t + dt, np.sign(dt)
        while sgn * (t_end - t) > 1e-12:
            if sgn > 0:
                edge = (np.floor(t * 2 + 1e-9) + 1) / 2          # next switch
                t_next = min(edge, t_end)
                c = -1.0 if (t % 1.0) < 0.5 - 1e-9 else 1.0
            else:
                edge = (np.ceil(t * 2 - 1e-9) - 1) / 2           # previous switch
                t_next = max(edge, t_end)
                c = -1.0 if ((t - 1e-9) % 1.0) < 0.5 else 1.0
            frac = (t_next - t) / 0.5
            x, y, M = self._rotate(x, y, c, frac, M)
            t = t_next
        return x, y, M

    def period_map(self, x, y, n=1):
        """Positions after n full periods (the Poincare map), starting at t = 0."""
        for _ in range(n):
            x, y, _ = self.step(x, y, 0.0, 1.0)
        return x, y


class TidalEddies:
    """Kinematic model of tidal stirring (Ridderinkhof and Zimmerman 1992; Beerens et al. 1994).

    A uniform tidal current oscillating along x over a steady array of
    residual eddies, in dimensionless form (time in units of 1/omega, so
    that one tide lasts 2 pi; one eddy is a square of side pi):

        u = A cos t + nu sin x cos y ,     v = -nu cos x sin y .

    The residual stream function is nu sin x sin y, so neighbouring eddies
    turn in opposite directions and no fluid crosses y = 0 or y = pi.  The
    tidal excursion, from one slack water to the next, is 2A, and

        excursion / eddy size = 2A / pi ,   residual / tidal speed = nu / A .

    The defaults are those of the figure in Chapter 7 of the notes:
    excursion 0.8 eddy sizes, residual current half the tidal one.
    The equations are integrated with fourth-order Runge-Kutta, with
    ``steps`` steps per tide.
    """

    period = 2 * np.pi
    eddy = np.pi

    def __init__(self, A=1.25, nu=0.625, steps=64):
        self.A, self.nu, self.steps = float(A), float(nu), int(steps)

    @classmethod
    def from_ratios(cls, excursion=0.8, residual=0.5, steps=64):
        """Build from excursion/eddy size and residual/tidal speed."""
        A = excursion * np.pi / 2
        return cls(A=A, nu=residual * A, steps=steps)

    def __repr__(self):
        return (f"TidalEddies(A={self.A:.4g}, nu={self.nu:.4g}: excursion {2 * self.A / np.pi:.3g} "
                f"eddy sizes, residual/tidal {self.nu / self.A:.3g})")

    def velocity(self, x, y, t):
        return (self.A * np.cos(t) + self.nu * np.sin(x) * np.cos(y),
                -self.nu * np.cos(x) * np.sin(y))

    def streamfunction(self, x, y, t):
        """psi with u = d psi/dy, v = -d psi/dx."""
        return self.A * np.cos(t) * y + self.nu * np.sin(x) * np.sin(y)

    def gradient(self, x, y):
        """Velocity gradient [[du/dx, du/dy], [dv/dx, dv/dy]] (the tide is uniform)."""
        cc, ss = np.cos(x) * np.cos(y), np.sin(x) * np.sin(y)
        G = np.empty(np.shape(x) + (2, 2))
        G[..., 0, 0] = self.nu * cc
        G[..., 0, 1] = -self.nu * ss
        G[..., 1, 0] = self.nu * ss
        G[..., 1, 1] = -self.nu * cc
        return G

    def _rhs(self, x, y, t, M):
        u, v = self.velocity(x, y, t)
        return u, v, (None if M is None else self.gradient(x, y) @ M)

    def step(self, x, y, t, dt, M=None):
        """Advance from t to t + dt (dt may be negative) with RK4."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        n = max(1, int(np.ceil(abs(dt) / (self.period / self.steps) - 1e-9)))
        h = dt / n
        for _ in range(n):
            k1 = self._rhs(x, y, t, M)
            k2 = self._rhs(x + .5 * h * k1[0], y + .5 * h * k1[1], t + .5 * h,
                           None if M is None else M + .5 * h * k1[2])
            k3 = self._rhs(x + .5 * h * k2[0], y + .5 * h * k2[1], t + .5 * h,
                           None if M is None else M + .5 * h * k2[2])
            k4 = self._rhs(x + h * k3[0], y + h * k3[1], t + h,
                           None if M is None else M + h * k3[2])
            x = x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            y = y + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            if M is not None:
                M = M + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            t += h
        return x, y, M

    def period_map(self, x, y, n=1, t0=0.0):
        """Positions after n tides (the Poincare map at the phase t0)."""
        for k in range(n):
            x, y, _ = self.step(x, y, t0 + k * self.period, self.period)
        return x, y
