"""efmstir -- chaotic stirring in two time-periodic flows, for 37EFM.

    BlinkingVortex(mu)       Aref's blinking vortex (Chapter 7 of the notes)
    TidalEddies(A, nu)       a tidal current over an array of residual eddies
                             (Ridderinkhof and Zimmerman 1992; Beerens et al. 1994)

and, in :mod:`efmstir.lyapunov`, Poincare sections, material lines, and
the classical, finite-time and finite-size Lyapunov exponents.
"""
from .flows import BlinkingVortex, TidalEddies
from . import lyapunov
from .lyapunov import (advect, ftle, lyapunov as lyapunov_exponent, fsle_field, fsle_curve,
                       outline, follow_outline, length)

__version__ = "0.1.0"
