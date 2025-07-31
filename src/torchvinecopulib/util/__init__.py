"""
torchvinecopulib.util
----------------------
Utility routines for copula‐based dependence measures, 1D KDE CDF/PPF, and root-finding via the Interpolate Truncate and Project (ITP) method.

Decorators
-----------
* torch.compile() for solve_ITP()
* torch.no_grad() for solve_ITP(), kendall_tau(), mutual_info(), ferreira_tail_dep_coeff(), chatterjee_xi()

References
-----------
- O’Brien, T. A., Kashinath, K., Cavanaugh, N. R., Collins, W. D., & O’Brien, J. P. (2016). A fast and objective multidimensional kernel density estimation method: fastKDE. Computational Statistics & Data Analysis, 101, 148-160.
- O’Brien, T. A., Collins, W. D., Rauscher, S. A., & Ringler, T. D. (2014). Reducing the computational cost of the ECF using a nuFFT: A fast and objective probability density estimation method. Computational Statistics & Data Analysis, 79, 222-234.
- Purkayastha, S., & Song, P. X. K. (2024). fastMI: A fast and consistent copula-based nonparametric estimator of mutual information. Journal of Multivariate Analysis, 201, 105270.
- Ferreira, M. S. (2013). Nonparametric estimation of the tail-dependence coefficient.
- Chatterjee, S. (2021). A new coefficient of correlation. Journal of the American Statistical Association, 116(536), 2009-2022.
- Lin, Z., & Han, F. (2023). On boosting the power of Chatterjee’s rank correlation. Biometrika, 110(2), 283-299.
- Oliveira, I. F., & Takahashi, R. H. (2020). An enhancement of the bisection method average performance preserving minmax optimality. ACM Transactions on Mathematical Software (TOMS), 47(1), 1-24.
"""

import enum
from pprint import pformat

import fastkde
import torch
from scipy.stats import kendalltau

_EPS = 1e-10


@torch.no_grad()
def kendall_tau(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Kendall's tau correlation coefficient and p-value. Moves inputs to CPU and delegates
    to SciPy’s ``kendalltau``.

    Args:
        x (torch.Tensor): shape (n, 1)
        y (torch.Tensor): shape (n, 1)
    Returns:
        torch.Tensor: Kendall's tau correlation coefficient and p-value
    """
    return torch.as_tensor(
        kendalltau(x.view(-1).cpu(), y.view(-1).cpu()),
        dtype=x.dtype,
        device=x.device,
    )


@torch.no_grad()
def mutual_info(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Estimate mutual information using ``fastKDE``. Moves inputs to CPU and delegates to
    ``fastKDE.pdf``.

    - O’Brien, T. A., Kashinath, K., Cavanaugh, N. R., Collins, W. D., & O’Brien, J. P. (2016). A fast and objective multidimensional kernel density estimation method: fastKDE. Computational Statistics & Data Analysis, 101, 148-160.
    - O’Brien, T. A., Collins, W. D., Rauscher, S. A., & Ringler, T. D. (2014). Reducing the computational cost of the ECF using a nuFFT: A fast and objective probability density estimation method. Computational Statistics & Data Analysis, 79, 222-234.
    - Purkayastha, S., & Song, P. X. K. (2024). fastMI: A fast and consistent copula-based nonparametric estimator of mutual information. Journal of Multivariate Analysis, 201, 105270.

    Args:
        x (torch.Tensor): shape (n, 1)
        y (torch.Tensor): shape (n, 1)
    Returns:
        torch.Tensor: Estimated mutual information
    """
    x = x.clamp(_EPS, 1.0 - _EPS).view(-1).cpu()
    y = y.clamp(_EPS, 1.0 - _EPS).view(-1).cpu()
    joint = torch.as_tensor(fastkde.pdf(x, y).values, dtype=x.dtype, device=x.device)
    margin_x = torch.as_tensor(fastkde.pdf(x).values, dtype=x.dtype, device=x.device)
    margin_y = torch.as_tensor(fastkde.pdf(y).values, dtype=x.dtype, device=x.device)
    return (
        joint[joint > 0.0].log().mean()
        - margin_x[margin_x > 0.0].log().mean()
        - margin_y[margin_y > 0.0].log().mean()
    )


@torch.no_grad()
def ferreira_tail_dep_coeff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Estimate tail dependence coefficient (λ), modifed from Ferreira's method, symmetric for
    (x,y), (y,1-x), (1-x,1-y), (1-y,x), (y,x), (1-x,y), (1-y,1-x), (x,1-y).

    - Ferreira, M. S. (2013). Nonparametric estimation of the tail-dependence coefficient.

    Args:
        x (torch.Tensor): shape (n, 1)
        y (torch.Tensor): shape (n, 1)
    Returns:
        torch.Tensor: Estimated tail dependence coefficient
    """
    return (
        3.0
        - (
            1.0
            - torch.stack([torch.max(x, y), torch.max(1.0 - x, y)], dim=1)
            .mean(dim=0)
            .clamp(0.5, 0.6666666666666666)
            .min()
        ).reciprocal()
    )


@torch.no_grad()
def chatterjee_xi(x: torch.Tensor, y: torch.Tensor, M: int = 1) -> torch.Tensor:
    """Estimate Chatterjee's rank correlation coefficient (ξ)

    - Chatterjee, S. (2021). A new coefficient of correlation. Journal of the American Statistical Association, 116(536), 2009-2022.
    - Lin, Z., & Han, F. (2023). On boosting the power of Chatterjee’s rank correlation. Biometrika, 110(2), 283-299.

    Args:
        x (torch.Tensor): shape (n, 1)
        y (torch.Tensor): shape (n, 1)
        M (int, optional): num of nearest-neighbor. Defaults to 1.
    Returns:
        torch.Tensor: Estimated Chatterjee's rank correlation coefficient
    """
    # * ranks of x in the order of (ranks of y)
    # * ranks of y in the order of (ranks of x)
    xrank, yrank = (
        x.argsort(dim=0).argsort(dim=0) + 1,
        y.argsort(dim=0).argsort(dim=0) + 1,
    )
    xrank, yrank = xrank[yrank.argsort(dim=0)], yrank[xrank.argsort(dim=0)]

    # * the inner sum inside the numerator term, ∑min(Ri, Rjm(i))
    # * max for symmetry as following Remark 1 in Chatterjee (2021)
    def xy_sum(m: int) -> tuple:
        return (
            (torch.min(xrank[:-m], xrank[m:])).sum() + xrank[-m:].sum(),
            (torch.min(yrank[:-m], yrank[m:])).sum() + yrank[-m:].sum(),
        )

    # * whole eq. 3 in Lin and Han (2023)
    n = x.shape[0]
    return -2.0 + 24.0 * (
        torch.as_tensor([xy_sum(m) for m in range(1, M + 1)], device=x.device, dtype=x.dtype)
        .sum(dim=0)
        .max()
    ) / (M * (1.0 + n) * (1.0 + M + 4.0 * n))


class ENUM_FUNC_BIDEP(enum.Enum):
    """
    Enum wrapper for bivariate dependence functions.
    """

    chatterjee_xi = enum.member(chatterjee_xi)
    ferreira_tail_dep_coeff = enum.member(ferreira_tail_dep_coeff)
    kendall_tau = enum.member(kendall_tau)
    mutual_info = enum.member(mutual_info)

    def __call__(self, x: torch.Tensor, y: torch.Tensor, **kw):
        return self.value(x, y, **kw)


class kdeCDFPPF1D(torch.nn.Module):
    _EPS = _EPS

    def __init__(
        self,
        x: torch.Tensor,
        num_step_grid: int = None,
        x_min: float = None,
        x_max: float = None,
        pad: float = 0.1,
    ):
        """1D KDE CDF/PPF using ``fastKDE`` + Simpson's rule. Given a sample ``x``, fits a kernel
        density estimate via ``fastKDE`` on a grid of size ``num_step_grid`` (power of two plus
        one).  Precomputes PDF, CDF, and their finite‐difference slopes for fast interpolation.

        - O’Brien, T. A., Kashinath, K., Cavanaugh, N. R., Collins, W. D., & O’Brien, J. P. (2016). A fast and objective multidimensional kernel density estimation method: fastKDE. Computational Statistics & Data Analysis, 101, 148-160.
        - O’Brien, T. A., Collins, W. D., Rauscher, S. A., & Ringler, T. D. (2014). Reducing the computational cost of the ECF using a nuFFT: A fast and objective probability density estimation method. Computational Statistics & Data Analysis, 79, 222-234.

        Args:
            x (torch.Tensor): input sample to fit the KDE.
            num_step_grid (int, optional): number of grid points for the KDE, should be power of 2 plus 1. Defaults to None.
            x_min (float, optional): minimum value of the grid. Defaults to x.min() - pad.
            x_max (float, optional): maximum value of the grid. Defaults to x.max() + pad.
            pad (float, optional): padding to extend beyond the min/max when ``x_min``/``x_max`` is None. Defaults to 1.0.
        """
        super().__init__()
        self.num_obs = x.shape[0]
        self.x_min = x_min if x_min is not None else x.min().item() - pad
        self.x_max = x_max if x_max is not None else x.max().item() + pad
        # * power of 2 plus 1
        if num_step_grid is None:
            num_step_grid = int(2 ** torch.log2(torch.tensor(x.numel())).ceil().item()) + 1
        self.num_step_grid = num_step_grid
        # * fastkde
        res = fastkde.pdf(x.view(-1).cpu().numpy(), num_points=num_step_grid)
        xs = torch.from_numpy(res.var0.values).to(dtype=torch.float64)
        pdfs = torch.from_numpy(res.values).to(dtype=torch.float64).clamp_min(self._EPS)
        N = pdfs.shape[0]
        ws = torch.ones(N, dtype=torch.float64)
        ws[1:-1:2] = 4
        ws[2:-1:2] = 2
        h = xs[1] - xs[0]
        cdf = torch.cumsum(pdfs * ws, dim=0) * (h / 3)
        cdf = cdf / cdf[-1]
        slope_fwd = (cdf[1:] - cdf[:-1]) / h
        slope_inv = h / (cdf[1:] - cdf[:-1])
        slope_pdf = (pdfs[1:] - pdfs[:-1]) / h
        self.register_buffer("grid_x", xs)
        self.register_buffer("grid_pdf", pdfs)
        self.register_buffer("grid_cdf", cdf)
        self.register_buffer("slope_fwd", slope_fwd)
        self.register_buffer("slope_inv", slope_inv)
        self.register_buffer("slope_pdf", slope_pdf)
        self.h = h
        # ! device agnostic
        self.register_buffer("_dd", torch.tensor([], dtype=torch.float64))
        self.negloglik = -self.log_pdf(x).mean()

    @property
    def device(self):
        return self._dd.device

    @property
    def dtype(self):
        return self._dd.dtype

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the CDF of the fitted KDE at ``x``.

        Args:
            x (torch.Tensor): Points at which to evaluate the CDF.
        Returns:
            torch.Tensor: CDF values at ``x``, clamped to [0, 1].
        """
        # ! device agnostic
        x = x.to(device=self.device, dtype=self.dtype)
        x_clamped = x.clamp(self.x_min, self.x_max)
        idx = torch.searchsorted(self.grid_x, x_clamped, right=False)
        idx = idx.clamp(1, self.grid_cdf.numel() - 1)
        y = (self.grid_cdf[idx - 1]) + (self.slope_fwd[idx - 1]) * (
            x_clamped - self.grid_x[idx - 1]
        )
        y = torch.where(x < self.x_min, torch.zeros_like(y), y)
        y = torch.where(x > self.x_max, torch.ones_like(y), y)
        return y.clamp(0.0, 1.0)

    def ppf(self, q: torch.Tensor) -> torch.Tensor:
        """Compute the PPF (quantile function) of the fitted KDE at ``q``.

        Args:
            q (torch.Tensor): Quantiles at which to evaluate the PPF.
        Returns:
            torch.Tensor: PPF values at ``q``, clamped to [x_min, x_max].
        """
        # ! device agnostic
        q = q.to(device=self.device, dtype=self.dtype)
        q_clamped = q.clamp(0.0, 1.0)
        idx = torch.searchsorted(self.grid_cdf, q_clamped, right=False)
        idx = idx.clamp(1, self.grid_cdf.numel() - 1)
        x = (self.grid_x[idx - 1]) + (self.slope_inv[idx - 1]) * (
            q_clamped - self.grid_cdf[idx - 1]
        )
        x = torch.where(q < 0.0, torch.full_like(x, self.x_min), x)
        x = torch.where(q > 1.0, torch.full_like(x, self.x_max), x)
        return x.clamp(self.x_min, self.x_max)

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the PDF of the fitted KDE at ``x``.

        Args:
            x (torch.Tensor): Points at which to evaluate the PDF.
        Returns:
            torch.Tensor: PDF values at ``x``, clamped to [0, ∞).
        """
        # ! device agnostic
        x = x.to(device=self.device, dtype=self.dtype)
        x_clamped = x.clamp(self.x_min, self.x_max)
        idx = torch.searchsorted(self.grid_x, x_clamped, right=False)
        idx = idx.clamp(1, self.grid_pdf.numel() - 1)
        f = self.grid_pdf[idx - 1] + (self.slope_pdf[idx - 1]) * (x_clamped - self.grid_x[idx - 1])
        f = torch.where((x < self.x_min) | (x > self.x_max), torch.zeros_like(f), f)
        return f.clamp_min(0.0)

    def log_pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log PDF of the fitted KDE at ``x``.

        Args:
            x (torch.Tensor): Points at which to evaluate the log PDF.
        Returns:
            torch.Tensor: Log PDF values at ``x``, guaranteed to be finite.
        """
        return self.pdf(x).log().nan_to_num(posinf=0.0, neginf=-13.815510557964274)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average negative log-likelihood of the fitted KDE at ``x``.

        Args:
            x (torch.Tensor): Points at which to evaluate the negative log-likelihood.
        Returns:
            torch.Tensor: Negative log-likelihood values at ``x``, averaged over the batch.
        """
        return -self.log_pdf(x).mean()

    def __str__(self):
        """String representation of the ``kdeCDFPPF1D`` object.

        Returns:
            str: String representation of the ``kdeCDFPPF1D`` object.
        """
        header = self.__class__.__name__
        params = {
            "num_obs": int(self.num_obs),
            "negloglik": float(self.negloglik.round(decimals=4)),
            "x_min": float(round(self.x_min, 4)),
            "x_max": float(round(self.x_max, 4)),
            "num_step_grid": int(self.num_step_grid),
            "dtype": self.dtype,
            "device": self.device,
        }
        params_str = pformat(params, sort_dicts=False, underscore_numbers=True)
        return f"{header}\n{params_str[1:-1]}\n\n"


# @torch.compile
@torch.no_grad()
def solve_ITP(
    fun: callable,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    epsilon: float = _EPS,
    num_iter_max: int = 31,
    k_1: float = 0.2,
) -> torch.Tensor:
    """Root-finding for ``fun`` via the Interpolate Truncate and Project (ITP) method within
    [``x_a``, ``x_b``], with guaranteed average performance strictly better than the bisection
    method under any continuous distribution.

    - Oliveira, I. F., & Takahashi, R. H. (2020). An enhancement of the bisection method average performance preserving minmax optimality. ACM Transactions on Mathematical Software (TOMS), 47(1), 1-24.
        https://en.wikipedia.org/wiki/ITP_method
        https://docs.rs/kurbo/latest/kurbo/common/fn.solve_itp.html

    Args:
        fun (callable): function to find the root of.
        x_a (torch.Tensor): lower bound of the interval to search.
        x_b (torch.Tensor): upper bound of the interval to search.
        epsilon (float, optional): convergence tolerance. Defaults to _EPS.
        num_iter_max (int, optional): maximum number of iterations. Defaults to 31.
        k_1 (float, optional): scaling factor for the truncation step. Defaults to 0.2.
    Returns:
        torch.Tensor: approximated root of the function `fun` in the interval [x_a, x_b].
    """
    y_a, y_b = fun(x_a), fun(x_b)
    # * corner cases
    x_a = torch.where(condition=y_b.abs() < epsilon, input=x_b - epsilon * num_iter_max, other=x_a)
    x_b = torch.where(condition=y_a.abs() < epsilon, input=x_a + epsilon * num_iter_max, other=x_b)
    y_a, y_b, x_wid = fun(x_a), fun(x_b), x_b - x_a
    eps_2 = torch.as_tensor(epsilon * 2.0, device=x_a.device, dtype=x_a.dtype)
    eps_scale = epsilon * 2.0 ** (
        (x_wid / epsilon).max().clamp_min(1.0).log2().ceil().clamp_min(1.0).int()
    )
    x_half = torch.empty_like(x_wid)
    rho = torch.empty_like(x_wid)
    sigma = torch.empty_like(x_wid)
    delta = torch.empty_like(x_wid)
    for _ in range(num_iter_max):
        if (x_wid < eps_2).all():
            break
        # * update parameters
        x_half.copy_(0.5 * (x_a + x_b))
        rho.copy_(eps_scale - 0.5 * x_wid)
        # * interpolation
        x_f = (y_b * x_a - y_a * x_b) / (y_b - y_a)
        sigma.copy_(x_half - x_f)
        # ! here k2 = 2 hardwired for efficiency.
        delta.copy_(k_1 * x_wid.square())
        # * truncation
        x_t = torch.where(
            condition=delta <= sigma.abs(),
            input=x_f + torch.copysign(delta, sigma),
            other=x_half,
        )
        # * projection
        x_itp = torch.where(
            condition=rho >= (x_t - x_half).abs(),
            input=x_t,
            other=x_half - torch.copysign(rho, sigma),
        )
        # * update interval
        y_itp = fun(x_itp)
        idx = y_itp > 0.0
        x_b[idx], y_b[idx] = x_itp[idx], y_itp[idx]
        idx = ~idx
        x_a[idx], y_a[idx] = x_itp[idx], y_itp[idx]
        x_wid = x_b - x_a
        eps_scale *= 0.5
    return 0.5 * (x_a + x_b)
