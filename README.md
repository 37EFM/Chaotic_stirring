# Chaotic_stirring — the blinking vortex and tidal stirring, with Lyapunov exponents

Course material for **37EFM Environmental Fluid Mechanics** (Eindhoven
University of Technology), with Chapter 7 of the lecture notes
(*Chaotic advection*). A smooth, laminar, time-periodic flow can stir
as effectively as turbulence. Two such flows are followed here, and the
stirring is measured in three ways: the Lyapunov exponent, the
finite-time Lyapunov exponent (FTLE) and the finite-size Lyapunov
exponent (FSLE).

| Notebook | What it does |
|---|---|
| `notebooks/01_blinking_vortex.ipynb` | Aref's blinking vortex: Poincaré sections for increasing $\mu$, blobs of dye in the chaotic sea and on an island, the Lyapunov exponent and how the chaotic region grows with $\mu$, FTLE fields and their convergence to the Lyapunov exponent, the backward FTLE as the skeleton along which dye is drawn out, and the FSLE as a function of the separation. |
| `notebooks/02_tidal_stirring.ipynb` | A uniform tidal current over a periodic array of residual eddies (Ridderinkhof and Zimmerman 1992; Beerens et al. 1994): the Poincaré section, the two blobs of the figure in the notes, where the stirring is strongest as the tidal excursion varies, FTLE and FSLE fields, and the FSLE at scales larger than the eddies, where chaotic stirring becomes an effective diffusion. |

Each notebook ends with questions. Both run in one or two minutes on a laptop.

## Installation

```bash
git clone https://github.com/37EFM/Chaotic_stirring.git
cd Chaotic_stirring
conda env create -f environment.yml
conda activate efmstir
jupyter lab notebooks
```

On Windows, run these in the *Anaconda Prompt*. Without git, download
the repository as a zip file from GitHub (*Code* > *Download ZIP*),
unpack it outside OneDrive, and run the last three lines in the
unpacked folder. Only numpy and matplotlib are needed, so
`pip install -e .` in any Python environment with JupyterLab works as
well, for instance in the environment of the
[tracer dispersion](https://github.com/37EFM/Tracer_dispersion)
exercises.

## The flows

**Blinking vortex** (`efmstir.BlinkingVortex(mu)`). Two point vortices at
$x = \pm a$ are switched on alternately, each for half a period $T$. A
parcel at distance $r$ from the active vortex turns about it through
$\mu a^2/r^2$ in a half period, with

$$\mu = \frac{\Gamma T}{4\pi a^2}.$$

Lengths are in units of $a$ and time in periods. The rotation is uniform
in time within each half period, so positions and the Jacobian of the
flow map are computed exactly, at any instant.

**Tidal current over residual eddies** (`efmstir.TidalEddies(A, nu)`).
In dimensionless form, with one tide lasting $2\pi$ and one eddy a
square of side $\pi$,

$$u = A\cos t + \nu \sin x \cos y, \qquad v = -\nu \cos x \sin y .$$

The tidal excursion is $2A$, so $2A/\pi$ is the excursion in eddy sizes
and $\nu/A$ the ratio of the residual to the tidal current. The defaults,
$A = 1.25$ and $\nu = 0.625$, are those of the figure in the notes. The
equations for the positions and for the Jacobian are integrated together
with fourth-order Runge-Kutta, 64 steps per tide by default.

## The measures

In `efmstir.lyapunov`:

- `lyapunov(flow, x0, y0, n)`: the **Lyapunov exponent**, from an
  infinitesimal separation carried along each trajectory and renormalised
  after every period. One number for a whole chaotic region, reached as
  $t \to \infty$.
- `ftle(flow, x, y, t0, tau)`: the **finite-time Lyapunov exponent**,
  $\ln\sigma_{\max}/|\tau|$, with $\sigma_{\max}$ the largest singular
  value of the Jacobian of the flow map over the time $\tau$. A field over
  the starting points; its ridges are the Lagrangian coherent structures.
  With $\tau < 0$, it gives the lines that attract the fluid, along which
  dye is drawn out.
- `fsle_curve(flow, x0, y0, delta0)`: the **finite-size Lyapunov
  exponent** $\lambda(\delta) = \ln r/\langle\tau(\delta)\rangle$, with
  $\tau(\delta)$ the time two parcels take to grow from $\delta$ to
  $r\delta$ apart. It equals the Lyapunov exponent at small separations
  and falls once the separation reaches the size of the flow structures.
  `fsle_field` maps it over the starting points.
- `outline`, `follow_outline`, `length`: the outline of a blob as a
  material line, refined as it stretches.

## Layout

```
efmstir/flows.py      BlinkingVortex, TidalEddies
efmstir/lyapunov.py   Lyapunov exponent, FTLE, FSLE, material lines
notebooks/            the two notebooks
tests/                pytest
```

The exercise on the blinking vortex in Chapter 7 asks students to write
the map and to estimate the Lyapunov exponent themselves; the first
notebook does both.

## References

- Aref, H. (1984). Stirring by chaotic advection. *Journal of Fluid Mechanics* 143, 1–21.
- Artale, V., Boffetta, G., Celani, A., Cencini, M. and Vulpiani, A. (1997). Dispersion of passive tracers in closed basins: beyond the diffusion coefficient. *Physics of Fluids* 9, 3162–3171.
- Aurell, E., Boffetta, G., Crisanti, A., Paladin, G. and Vulpiani, A. (1997). Predictability in the large: an extension of the concept of Lyapunov exponent. *Journal of Physics A* 30, 1–26.
- Beerens, S. P., Ridderinkhof, H. and Zimmerman, J. T. F. (1994). An analytical study of chaotic stirring in tidal areas. *Chaos, Solitons & Fractals* 4, 1011–1029.
- d'Ovidio, F., Fernández, V., Hernández-García, E. and López, C. (2004). Mixing structures in the Mediterranean Sea from finite-size Lyapunov exponents. *Geophysical Research Letters* 31, L17203.
- Ridderinkhof, H. and Zimmerman, J. T. F. (1992). Chaotic stirring in a tidal system. *Science* 258, 1107–1111.
- Shadden, S. C., Lekien, F. and Marsden, J. E. (2005). Definition and properties of Lagrangian coherent structures from finite-time Lyapunov exponents in two-dimensional aperiodic flows. *Physica D* 212, 271–304.
