# Estimator backends

`torchvinecopulib` uses the words `marginal_backend` and `bicop_backend` deliberately:
they select estimator backends, not a catalog of named parametric copula families.

For bivariate copulas, the public `BiCop` object stores a numerical approximation to a continuous
pair-copula on a regular grid, together with cumulative buffers used for query-time interpolation.
The current torch-native default is `bicop_backend="beta"`, selected from the synthetic benchmark
suite described in [Systems: benchmark interpretation](./benchmarks.md).

The 2D bicop surface now has two roles:

- stable production defaults: `beta`, `ttpi`, `grid_reflect`, `grid_probit`
- research-facing experimental backends: `ttcv`, `tll1`, `tll2`, `tll1nn`, `tll2nn`, `beta_qt`, `spline_pen`

All of them still end by materializing the same `pdf_grid` / `cdf_grid` / `hfunc_*_grid`
representation, so query-time code stays backend-agnostic.

## Common bivariate representation

For a continuous pair-copula density $c(u_1, u_2)$ on $(0, 1)^2$, `BiCop` stores:

- `pdf_grid`: a regular-grid approximation to $c(u_1, u_2)$,
- `cdf_grid`: a regular-grid approximation to

  $$
  C(u_1, u_2) = \int_0^{u_1} \int_0^{u_2} c(s, t)\, dt\, ds,
  $$

- `hfunc_l_grid`: a regular-grid approximation to

  $$
  h_l(u_1, u_2) = \int_0^{u_2} c(u_1, t)\, dt
  = P(U_2 \le u_2 \mid U_1 = u_1),
  $$

- `hfunc_r_grid`: a regular-grid approximation to

  $$
  h_r(u_1, u_2) = \int_0^{u_1} c(s, u_2)\, ds
  = P(U_1 \le u_1 \mid U_2 = u_2).
  $$

These cumulative buffers are built from the density grid with trapezoidal integration so the
discrete representation respects the expected copula boundary conditions.

## Canonical aligned `bw` semantics

For the `kdecopula`-aligned bicop names, `torchvinecopulib` now treats `bandwidth` as the canonical
upstream `bw` object and `mult` as the canonical bandwidth multiplier.

- `ttpi` / `ttcv`: `bandwidth="auto"` or a length-4 vector `(h, rho, theta1, theta2)`. `mult`
  multiplies only `h`.
- `tll1` / `tll2`: `bandwidth="auto"` or a `2x2` bandwidth matrix. `mult` multiplies the selected
  matrix.
- `tll1nn` / `tll2nn`: `bandwidth="auto"` or a mapping with `B`, `alpha`, `kappa`. `mult`
  multiplies `alpha`.
- `beta`: `bandwidth="auto"` or a positive scalar. `mult` multiplies the scalar `bw`.

## `bicop_backend="grid_reflect"`

`grid_reflect` is the simplest direct torch-native pair-copula backend on the unit square.

It fits a nonparametric copula density directly on the unit square:

1. observations in $(0, 1)^2$ are placed onto a regular grid by bilinear binning,
2. the grid is mirrored across all four boundaries,
3. the mirrored histogram is smoothed with a separable Gaussian kernel,
4. the center block is cropped back to the canonical copula domain,
5. the result is normalized so the discrete row and column marginals are approximately uniform.

At a high level, the estimator is

$$
\tilde c(u_1, u_2)
\propto
\left(K_h * H_{\text{reflect}}\right)(u_1, u_2),
$$

where $H_{\text{reflect}}$ is the reflected binned histogram and $K_h$ is a Gaussian smoothing
kernel with bandwidth $h$.

The final normalization step enforces the copula constraint numerically:

$$
\int_0^1 c(u_1, u_2)\, du_2 \approx 1,
\qquad
\int_0^1 c(u_1, u_2)\, du_1 \approx 1.
$$

In code, this is the path implemented by `fit_grid_reflect_bicop()`.

## `bicop_backend="grid_probit"`

`grid_probit` is also torch-native, but it estimates the density in Gaussianized coordinates.

First transform the copula observations with the probit map

$$
z_j = \Phi^{-1}(u_j),
$$

where $\Phi$ is the standard normal CDF. A 2D KDE is then fitted on a rectangular grid in
$z$-space, producing an approximation to a joint density $f_Z(z_1, z_2)$.

That density is mapped back to copula scale using the Jacobian of the probit transform:

$$
c(u_1, u_2)
=
\frac{f_Z(z_1, z_2)}
{\phi(z_1)\phi(z_2)},
\qquad
z_j = \Phi^{-1}(u_j),
$$

where $\phi$ is the standard normal PDF.

The resulting copula density grid is then normalized in the same way as `grid_reflect`, and the
same `cdf_grid`, `hfunc_l_grid`, and `hfunc_r_grid` buffers are built from it.

This backend is often a good fit when the dependence structure is smoother in latent Gaussian
coordinates than in the raw unit-square geometry.

## `bicop_backend="ttcv"` and `bicop_backend="ttpi"`

These torch-native backends implement the tapered transformation estimator of Wen and Wu.

They use the `kdecopula` parameterization

$$
\mathrm{bw} = (h, \rho, \theta_1, \theta_2),
$$

where:

- $h$ is the scalar smoothing parameter,
- $\rho$ is the latent Gaussian correlation used by the kernel,
- $\theta_1$ and $\theta_2$ are tapering parameters that damp the back-transform near tails and
  corners.

At evaluation points $(u, v)$ with latent coordinates $(s, t) = (\Phi^{-1}(u), \Phi^{-1}(v))$, the
estimator has the form

$$
\hat c(u, v)
=
\frac{\exp\{-\theta_1(s^2+t^2)-\theta_2 st\}}
{h^2 \eta \, \phi(s)\phi(t)}
\frac{1}{n}
\sum_{i=1}^n
\frac{\exp\left\{-\frac{q_\rho((s-S_i)/h, (t-T_i)/h)}{2(1-\rho^2)}\right\}}
{2\pi \sqrt{1-\rho^2}},
$$

where $(S_i, T_i) = (\Phi^{-1}(U_i), \Phi^{-1}(V_i))$, $\eta$ is the normalization term from Wen
and Wu, and the tapering term stabilizes the transformed estimator near the unit-square corners.

Both backends evaluate that density directly on the regular copula grid and then reuse the existing
copula renormalization and buffer-construction path.

- `ttpi` uses the plug-in selector for `(h, \rho, \theta_1, \theta_2)` and remains the stronger
  tapered-transformation production option when you want a selector-backed transformed estimator.
- `ttcv` uses the profile cross-validation selector and is kept as the research-facing companion.

For both methods, `mult` scales only `h`, matching the upstream `kdecopula` role of `mult`.

Compared with plain `grid_probit`, these backends explicitly target the main failure mode of
probit-style estimators: the back-transform Jacobian can become too aggressive in tails and corners.

## `bicop_backend="tll1"` and `bicop_backend="tll2"`

These torch-native backends now use the canonical fixed-bandwidth transformation local-likelihood
surface from the `kdecopula` line.

As with `grid_probit`, observations are first mapped to latent Gaussian coordinates
$z_j = \Phi^{-1}(u_j)$. Instead of smoothing the latent density with a plain KDE, the estimator fits
a local weighted log-likelihood model around each latent grid point:

$$
\log f_Z(\xi) \approx
\beta_0
+
\beta_1^\top(\xi-z)
+
\tfrac{1}{2}(\xi-z)^\top B_2(\xi-z).
$$

- `tll1` keeps the local model to log-linear order.
- `tll2` keeps the log-quadratic term.

The canonical `bandwidth` object is a `2x2` matrix, either supplied directly or selected
automatically by the same role as `bw_tll` in `kdecopula`; `mult` then scales that matrix.

In both cases the fitted latent density is mapped back to copula scale with the same Jacobian
correction used by `grid_probit`, then renormalized into a legal copula grid. The fit-time solver
is torch-native and the query-time runtime is still the shared grid materialization path.

## `bicop_backend="tll1nn"` and `bicop_backend="tll2nn"`

These experimental backends extend `tll1` and `tll2` with location-dependent bandwidth scaling in
latent Gaussian space.

The canonical `bandwidth` state follows the upstream nearest-neighbor object shape
`{B, alpha, kappa}`. The implementation starts from the pilot matrix `B` and modulates it with a
nearest-neighbor scale field derived from `alpha` and `kappa`:

$$
H(z) = s(z)^2 H_0,
$$

where $H_0$ is the pilot bandwidth matrix and $s(z)$ is estimated from local neighbor distances.

This keeps the public `bw` object close to `kdecopula`, while still materializing the estimator
back into the common torch grid runtime.

## `bicop_backend="beta"`

`beta` is the simplest strong unit-square production baseline among the current torch-native
pair-copula backends.

It uses a product beta-kernel estimator directly on the unit square:

$$
\hat c(u, v)
=
\frac{1}{n}
\sum_{i=1}^n
\mathrm{Beta}\!\left(u; \tfrac{U_i}{h}+1, \tfrac{1-U_i}{h}+1\right)
\mathrm{Beta}\!\left(v; \tfrac{V_i}{h}+1, \tfrac{1-V_i}{h}+1\right).
$$

For the aligned interface, the canonical `bandwidth` is a positive scalar chosen automatically in
the role of `bw_beta`; `mult` scales that scalar after selection.

Because the kernels live natively on $[0,1]$, this path is usually better behaved near corners and
tail-heavy boundary regions than plain reflected Gaussian smoothing. It remains a strong baseline and
usually the fastest robust alternative when you want a simpler estimator than the tapered
transformation default.

## `bicop_backend="beta_qt"`

`beta_qt` is an experimental beta-quantile-transformation backend.

It first applies a beta-shaped quantile transform to soften the unit-square boundary geometry,
performs the density fit in the transformed coordinates, and maps the result back with the inverse
Jacobian. This path is inspired by the beta quantile transformation literature for copula density
estimation and is primarily aimed at extreme tail and corner scenarios.

Compared with plain `beta`, it is more specialized and more sensitive to transform hyperparameters,
so it is currently documented as research-grade rather than a production default.

## `bicop_backend="spline_pen"`

`spline_pen` is an experimental tensor-product penalized-spline backend motivated by Dou et al.
(2024).

It bins pseudo-observations onto the unit square, evaluates a tensor-product B-spline basis, and
fits a smooth coefficient surface under a quadratic penalty. The implementation then enforces
nonnegativity numerically before the usual copula renormalization step.

This backend is intentionally documented as a Dou-inspired approximation, not a literal reproduction
of the paper's SCAD-penalized modified-EM algorithm.

## Fidelity audit against `kdecopula`

The current upstream reference point for named nonparametric bicop methods is
`kdecopula` 0.9.3. Its public method table is:

- `T`
- `TLL1`
- `TLL2`
- `TLL1nn`
- `TLL2nn`
- `TTPI`
- `TTCV`
- `MR`
- `beta`
- `bern`

The table below records how `torchvinecopulib` currently relates to those names.

| official method | status in `torchvinecopulib` | fidelity status | note |
| --- | --- | --- | --- |
| `T` | not implemented | not applicable | `grid_probit` is only a conceptual cousin; it is not the `kdecopula` `T` method. |
| `TLL1` | `tll1` | aligned torch-native port | canonical `bw` is now a `2x2` matrix and the fit uses a direct local weighted likelihood in latent Gaussian space; query-time still uses the shared grid runtime. |
| `TLL2` | `tll2` | aligned torch-native port | same alignment target as `TLL1`, with the log-quadratic basis and shared grid materialization. |
| `TLL1nn` | `tll1nn` | partially aligned torch-native port | canonical `bw` now matches the upstream `{B, alpha, kappa}` object, but the adaptive field and selector remain torch-native approximations. |
| `TLL2nn` | `tll2nn` | partially aligned torch-native port | same canonical bandwidth structure as upstream, with a torch-native adaptive fit and shared query runtime. |
| `TTPI` | `ttpi` | closest alignment target | uses the same public parameterization `(h, rho, theta1, theta2)` and the same selector role, but still integrates into the shared torch grid runtime. |
| `TTCV` | `ttcv` | closest alignment target | same public parameterization and profile-CV selector role; still a torch-integrated port rather than a line-by-line reproduction. |
| `MR` | not implemented | not applicable | `grid_reflect` is not a synonym for the official mirror-reflection estimator. |
| `beta` | `beta` | aligned torch-native port | same beta-kernel family with canonical scalar `bw` plus `mult`; legacy non-scalar inputs are only temporary compatibility shims. |
| `bern` | not implemented | not applicable | `spline_pen` is not the Bernstein copula estimator and must not be described that way. |

Current naming policy in this repository:

- keep `ttpi`, `ttcv`, `tll1`, `tll2`, `tll1nn`, `tll2nn`, and `beta` as the aligned names whose
  canonical `bw` shapes and `mult` semantics should track current `kdecopula`,
- keep `ttpi`, `ttcv`, and `beta` as the closest end-to-end alignment targets today,
- keep `tll1`, `tll2`, `tll1nn`, and `tll2nn` documented as torch-native aligned ports rather than
  line-by-line reproductions of the R object model,
- treat `grid_reflect`, `grid_probit`, `beta_qt`, and `spline_pen` as repository-native backend names, not as claims of upstream method identity.

## Locked fidelity fixtures

The repository now carries a small locked fixture set under `tests/fixtures/aligned_bicop/` for the
aligned methods `ttcv`, `ttpi`, `tll1`, `tll2`, `tll1nn`, `tll2nn`, and `beta`.

Those fixtures record:

- the canonical `bw` object after automatic selection,
- density values at fixed evaluation points,
- CDF values at fixed evaluation points.

The checked-in fixture file can be regenerated with
`scripts/generate_kdecopula_fixtures.py`. On machines without R, the script bootstraps a locked
torch-native regression fixture; on machines with `Rscript` plus `kdecopula`, it can regenerate the
same schema from the external reference implementation.

### What this means for `spline_pen`

`spline_pen` is the backend with the largest fidelity gap.

The current implementation:

1. bins pseudo-observations onto a regular unit-square grid,
2. applies mirrored Gaussian smoothing,
3. fits a tensor-product B-spline log-surface with a quadratic roughness penalty,
4. pushes the fitted log-surface through `softplus`,
5. renormalizes the resulting grid into a legal copula object.

That is a reasonable torch-native engineering estimator, but it is not the same optimization problem
as the recent penalized B-spline copula literature. In particular, the current implementation does
not reproduce the modified EM plus SCAD-penalized pseudo-likelihood formulation from Dou et al.

So the short answer is:

- yes, `spline_pen` could be made substantially closer to the latest spline-copula literature,
- but that would be a new backend implementation, not a small tweak to the current one,
- and until that reimplementation exists, `spline_pen` should stay explicitly labeled as an
  approximation.

## `bicop_backend="tll_ref"`

`tll_ref` is the CPU reference backend. It does not use the torch-native KDE code path.

Instead, it delegates to `pyvinecopulib` and fits its nonparametric transformation local
likelihood copula family:

- `nonparametric_method="constant"`
- `nonparametric_method="linear"`
- `nonparametric_method="quadratic"`

After fitting the reference bicop model, `torchvinecopulib` evaluates the fitted copula density on
the same regular unit-square grid and constructs the same cumulative buffers as the native
backends.

This backend is best understood as a reference or oracle comparison path rather than the default
production path.

Official reference implementation:

- [`pyvinecopulib.BicopFamily`](https://vinecopulib.github.io/pyvinecopulib/_generate/pyvinecopulib.BicopFamily.html),
  which defines `tll` as the transformation local-likelihood nonparametric copula family.
- [`pyvinecopulib.FitControlsBicop`](https://vinecopulib.github.io/pyvinecopulib/_generate/pyvinecopulib.FitControlsBicop.__init__.html),
  which exposes `nonparametric_method="constant" | "linear" | "quadratic"` for TLL fitting.

Background literature:

- [Geenens, Charpentier, and Paindaveine (2017), *Probit Transformation for Nonparametric Kernel Estimation of the Copula Density*](https://doi.org/10.3150/15-BEJ798).
  This is the main methodological reference behind probit-transformation plus local-likelihood
  copula density estimation.
- [Nagler (2018), *kdecopula: An R Package for the Kernel Estimation of Bivariate Copula Densities*](https://doi.org/10.18637/jss.v084.i07).
  This is a practical implementation reference for fast interpolation, renormalization, and TLL-style
  nonparametric copula workflows.
- [Nagler, Schellhase, and Czado, `kdecopula` reference manual](https://tnagler.github.io/kdecopula/reference/kdecop.html).
  This is the closest practical reference for the `TLL1` / `TLL2` / `TLL1nn` / `TLL2nn` /
  `TTPI` / `TTCV` naming and workflow.
- [Omelka, Gijbels, and Veraverbeke (2009), *Improved kernel estimation of copulas: weak
  convergence and goodness-of-fit testing*](https://doi.org/10.1016/j.csda.2008.11.023).
  This is a standard reference point for beta-kernel copula density estimation on the unit square.
- [Geenens, Poncet, and Veraverbeke (2021), *A beta quantile transformation for copula density
  estimation*](https://www.mdpi.com/2227-7390/9/10/1078).
  This motivates the `beta_qt` family of transformed beta-kernel estimators.
- [Dou et al. (2024), *Nonparametric Copula Density Estimation using Penalized B-Splines*](https://arxiv.org/abs/2402.07569).
  This is the motivating reference for the `spline_pen` backend, even though the implementation
  here is a simpler engineering approximation rather than a faithful SCAD/EM reproduction.

## Independence inside a vine

`BiCop` by itself does not perform family selection among Gaussian, Clayton, Gumbel, Frank, and so
on. The native and reference backends above all fit nonparametric pair-copula estimators.

Inside `VineCop.fit()`, an edge may still become an independence copula if truncation decides not to
fit a nontrivial pair-copula on that edge. In that case the corresponding `BiCop` remains an
independence module and query-time calls reduce to the analytic independence formulas.

## Practical selection guide

For continuous data, a good default workflow is:

1. start with `marginal_backend="grid"` and `bicop_backend="beta"` for the current benchmark-backed
   all-torch default,
2. try `bicop_backend="ttpi"` when you want the stronger tapered-transformation estimator and are
   willing to pay for selector work at fit time,
3. try `bicop_backend="grid_probit"` when the dependence appears smooth in latent Gaussian
   coordinates and you want a simpler transformed-KDE baseline,
4. try `bicop_backend="ttcv"` when you specifically want the profile-CV tapered-transformation
   variant,
5. use `bicop_backend="tll_ref"` when you want a CPU reference baseline or an oracle-style
   comparison against `pyvinecopulib`.

More concretely:

- `beta`: current default; best first choice when you want the strongest benchmark-backed simple
  unit-square estimator.
- `ttpi`: strongest tapered-transformation option when you are willing to pay for selector work at
  fit time.
- `grid_reflect`: best when you want the most direct reflected-kernel estimator on the unit square.
- `grid_probit`: best when the copula density is easier to smooth in latent Gaussian coordinates;
  this is often worth trying first for smooth continuous dependence with strong edge concentration.
- `ttcv`: profile-CV tapered-transformation variant; worth trying when `ttpi` is close but you want
  a more directly cross-validated selector.
- `tll1` / `tll2`: best when you want a torch-native local-likelihood fit rather than a plain KDE.
- `tll1nn` / `tll2nn`: adaptive-bandwidth research variants for irregular or spatially
  heterogeneous dependence surfaces.
- `beta_qt`: research option for aggressive tail and corner behavior.
- `spline_pen`: research option when you want a smooth spline surface rather than a kernel estimator.
- `tll_ref`: best for comparison, validation, regression tests, and research baselines; it is not the
  lightweight production default because it depends on `pyvinecopulib` and runs on the CPU reference
  path.

## Further bicop candidates

The current bicop surface already covers reflected KDE, probit-transformed KDE, transformation local
likelihood, adaptive local likelihood, tapered transformation estimators, beta-kernel estimators,
beta-quantile transforms, and a penalized-spline approximation. That is a large fraction of the
most natural continuous 2D nonparametric pair-copula estimators for the current runtime, but it is
not exhaustive.

Useful external references:

- [Wen and Wu (2018), *Transformation-Kernel Estimation of Copula Densities*](https://doi.org/10.1080/07350015.2018.1469999)
- [Nagler, Schellhase, and Czado, `kdecopula` reference manual](https://tnagler.github.io/kdecopula/reference/kdecop.html)

## What is fast and good in this runtime?

For the current `BiCop -> pdf_grid/cdf_grid/hfunc/hinv` design, the fastest useful answer is not a
single universal estimator family. Different synthetic geometries still favor different backends.
Within the current repository:

- `ttpi` is the strongest benchmark-backed default.
- `beta` is the simplest strong alternative when you want faster or lighter fit-time behavior.
- `grid_reflect` and `grid_probit` remain strong simple baselines.
- `ttcv`, `tll1`, `tll2`, `tll1nn`, and `tll2nn` are the research-facing choices when you are
  willing to spend more fit-time for a more specialized fit.

That is why the package keeps a benchmark-backed default instead of claiming that one nonparametric
pair-copula estimator is always best.

## What is not a natural fit for this runtime?

Some model families can certainly be made to work, but they are not the most natural next additions
for the current runtime contract.

### Gaussian-process or flow-based density models

These can be powerful density estimators, but they do not align naturally with the present
`BiCop` runtime representation. The current query path assumes that fit-time work ends by
materializing a regular `pdf_grid` and the cumulative buffers derived from it. A Gaussian-process or
normalizing-flow estimator is usually most valuable when queries are made directly against the model,
not after projecting it down to a fixed grid.

That means there are only two realistic integration strategies:

- pretabulate the learned model back onto a grid, which discards part of the benefit of using the
  richer model in the first place,
- or rewrite the runtime around model-specific `cdf`, `hfunc`, and inverse-conditional operations,
  which is a much larger architectural change.

For that reason, Gaussian-process and flow-style estimators are better understood as possible future
product lines, not as the most natural next bicop backends for the current grid runtime.

### Mixed or discrete copula estimators

These are also not impossible, but they solve a different problem. The current `BiCop` API assumes a
continuous pair-copula with smooth conditional CDFs and numerically stable inverse conditional
functions. Mixed or discrete copula models need different likelihoods, different pseudo-observation
semantics, and different sampling/query semantics.

So they are not just "another estimator family" that can be dropped into the current continuous
runtime. They would require a broader redesign of the input contract and the meaning of
`cdf_grid/hfunc/hinv`, not only a new fitting routine.

## Marginal backends

The 1D marginal side now has two torch-native roles:

- `marginal_backend="grid"`: default torch-native production path; this is the public
  `GridKDE1D` class.
- `marginal_backend="lp"`: torch-native continuous subset of the `kde1d` local-polynomial KDE.

This split is deliberate: the names describe estimator families, not just bandwidth options.

### `marginal_backend="grid"` and `GridKDE1D`

`GridKDE1D` is the default torch-native 1D marginal backend.

It fits a nonparametric density on a regular grid:

1. choose a support interval $[x_{\min}, x_{\max}]$ and a regular grid `grid_x`,
2. linearly bin the raw observations onto that grid,
3. smooth the binned counts with a Gaussian kernel,
4. normalize the result into a 1D density,
5. precompute interpolation slopes for `cdf()`, `ppf()`, and `pdf()`.

At a high level, the density construction is

$$
\hat f(x_j)
=
\frac{(K_h * H)_j}{n\, \Delta x},
$$

where $H$ is the linearly binned histogram, $K_h$ is a Gaussian kernel, and $\Delta x$ is the grid
step.

The discrete CDF is then formed by trapezoidal mass accumulation:

$$
\hat F(x_j)
=
\int_{x_{\min}}^{x_j} \hat f(s)\, ds.
$$

For this backend:

- the default bandwidth rule is ISJ,
- if ISJ fails to bracket a stable root, the implementation falls back to Silverman,
- a custom scalar bandwidth can also be passed explicitly.

This remains the simplest all-torch marginal path and the default choice when
`VineCop.fit(..., is_cop_scale=False)` needs to Gaussianize raw observations column by column.

### `marginal_backend="lp"`

`lp` is the torch-native local-polynomial KDE backend. It is the continuous-only torch port of the
core `kde1d` methodology used behind `pyvinecopulib.Kde1d`.

It supports the same three continuous local-likelihood degrees:

- `degree=0`: log-constant,
- `degree=1`: log-linear,
- `degree=2`: log-quadratic.

The estimator first works in transformed coordinates. Let $T(x)$ denote the boundary transform:

- no boundary: $T(x)=x$,
- left boundary: $T(x)=\log(x-x_{\min}+\varepsilon)$,
- right boundary: $T(x)=\log(x_{\max}-x+\varepsilon)$,
- two boundaries: a probit transform to the real line.

Write $z=T(x)$ and fit the local-polynomial model on a regular grid in $z$-space. As in `kde1d`,
the derivative estimates needed for local likelihood are built with binned Gaussian-derivative KDE
operators on an FFT grid. The final density on the original scale is obtained by the inverse
Jacobian correction

$$
\hat f_X(x)
=
\hat f_Z(T(x)) \left| T'(x) \right|.
$$

Bandwidth selection follows the `kde1d` plug-in idea for local-likelihood KDE. In this backend:

- `bandwidth="plugin"` is the default,
- `bandwidth="silverman"` remains available,
- `bandwidth="isj"` is accepted as a torch-native fallback rule,
- a custom scalar bandwidth can also be passed.

Current scope is intentionally narrower than the full `kde1d` package:

- continuous variables only,
- no discrete jittering path,
- no zero-inflated hurdle path,
- no weighted fitting interface yet.

Primary sources:

- [tnagler/kde1d repository](https://github.com/tnagler/kde1d)
- [kde1d C++ source tree](https://github.com/tnagler/kde1d/tree/main/inst/include/kde1d)
- [Geenens (2014), *Probit transformation for kernel density estimation on the unit interval*](https://arxiv.org/abs/1303.4121)
- [Geenens and Wang (2018), *Local-likelihood transformation kernel density estimation for positive random variables*](https://arxiv.org/abs/1602.04862)
- [Sheather and Jones (1991), bandwidth selection for kernel density estimation](https://doi.org/10.1111/j.2517-6161.1991.tb01857.x)

## Marginal selection guide

For continuous data, a good practical order is:

1. start with `marginal_backend="grid"` when you want the simplest and most conservative all-torch
   path,
2. try `marginal_backend="lp"` when boundary behavior matters or when you specifically want a
   torch-native approximation to the `kde1d` local-polynomial workflow.

More concretely:

- `grid`: best default for general continuous raw data and the lowest-maintenance all-torch path.
- `lp`: best when support boundaries are important and you want the torch-native analogue of
  `kde1d`'s continuous local-polynomial estimator.
