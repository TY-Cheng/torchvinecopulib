"""
* torch.compile() for solve_ITP()
* torch.no_grad() for solve_ITP()
"""

import enum

import fastkde
import torch
from scipy.stats import kendalltau
from torch.special import ndtr, ndtri

_EPS = 1e-7


def kendall_tau(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Kendall's tau, x,y are both of shape (n, 1)"""
    return torch.as_tensor(
        kendalltau(x.ravel().cpu(), y.ravel().cpu()),
        dtype=x.dtype,
        device=x.device,
    )


def mutual_info(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """mutual information, x,y are both of shape (n, 1)

    O’Brien, T. A., Kashinath, K., Cavanaugh, N. R., Collins, W. D. & O’Brien, J. P.
    A fast and objective multidimensional kernel density estimation method: fastKDE.
    Comput. Stat. Data Anal. 101, 148–160 (2016).
    http://dx.doi.org/10.1016/j.csda.2016.02.014

    O’Brien, T. A., Collins, W. D., Rauscher, S. A. & Ringler, T. D.
    Reducing the computational cost of the ECF using a nuFFT: A fast and objective probability density estimation method.
    Comput. Stat. Data Anal. 79, 222–234 (2014).
    http://dx.doi.org/10.1016/j.csda.2014.06.002

    Purkayastha, S., & Song, P. X. K. (2024).
    fastMI: A fast and consistent copula-based nonparametric estimator of mutual information.
    Journal of Multivariate Analysis, 201, 105270.
    """
    x = ndtri(x.clamp(_EPS, 1.0 - _EPS)).ravel().cpu()
    y = ndtri(y.clamp(_EPS, 1.0 - _EPS)).ravel().cpu()
    joint = torch.as_tensor(fastkde.pdf(x, y).values, dtype=x.dtype, device=x.device)
    margin_x = torch.as_tensor(fastkde.pdf(x).values, dtype=x.dtype, device=x.device)
    margin_y = torch.as_tensor(fastkde.pdf(y).values, dtype=x.dtype, device=x.device)
    return (
        joint[joint > 0.0].log().mean()
        - margin_x[margin_x > 0.0].log().mean()
        - margin_y[margin_y > 0.0].log().mean()
    )


def ferreira_tail_dep_coeff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """pairwise tail dependence coefficient (λ) estimator, max of rotation 0, 90, 180, 270
    x and y are both of shape (n, 1) inside (0, 1)
    symmetric for (x,y), (y,1-x), (1-x,1-y), (1-y,x), (y,x), (1-x,y), (1-y,1-x), (x,1-y)

    Ferreira, M.S., 2013. Nonparametric estimation of the tail-dependence coefficient;
    """
    x1 = 1.0 - x
    y1 = 1.0 - y
    return (
        3.0
        - (
            1.0
            - torch.as_tensor(
                [
                    torch.max(x, y).mean(),
                    torch.max(x1, y).mean(),
                    torch.max(x1, y1).mean(),
                    torch.max(x, y1).mean(),
                ]
            )
            .clamp(0.5, 0.6666666666666666)
            .min()
        ).reciprocal()
    )


def chatterjee_xi(x: torch.Tensor, y: torch.Tensor, M: int = 1) -> torch.Tensor:
    """revised Chatterjee's rank correlation coefficient (ξ), taken max to be symmetric

    Chatterjee, S., 2021. A new coefficient of correlation.
    Journal of the American Statistical Association, 116(536), pp.2009-2022.

    Lin, Z. and Han, F., 2023. On boosting the power of Chatterjee’s rank correlation. Biometrika, 110(2), pp.283-299.
    "a large negative value of ξ has only one possible interpretation: the data does not resemble an iid sample."

    :param x: obs of shape (n,1)
    :type x: torch.Tensor
    :param y: obs of shape (n,1)
    :type y: torch.Tensor
    :param M: num of right nearest neighbors
    :type M: int
    :return: revised Chatterjee's rank correlation coefficient (ξ), taken max to be symmetric for (X,Y) and (Y,X)
    :rtype: float
    """
    # ranks of x in the order of (ranks of y)
    # ranks of y in the order of (ranks of x)
    xrank, yrank = (
        x.argsort(dim=0).argsort(dim=0) + 1,
        y.argsort(dim=0).argsort(dim=0) + 1,
    )
    xrank, yrank = xrank[yrank.argsort(dim=0)], yrank[xrank.argsort(dim=0)]

    # the inner sum inside the numerator term, ∑min(Ri, Rjm(i))
    # max for symmetry as following Remark 1 in Chatterjee (2021)
    def xy_sum(m: int) -> tuple:
        return (
            (torch.min(xrank[:-m], xrank[m:])).sum() + xrank[-m:].sum(),
            (torch.min(yrank[:-m], yrank[m:])).sum() + yrank[-m:].sum(),
        )

    # whole eq. 3 in Lin and Han (2023)
    n = x.shape[0]
    return -2.0 + 24.0 * (
        torch.as_tensor([xy_sum(m) for m in range(1, M + 1)]).sum(dim=0).max()
    ) / (M * (1.0 + n) * (1.0 + M + 4.0 * n))


class ENUM_FUNC_BIDEP(enum.Enum):
    """an enum class for bivariate dependence measures"""

    chatterjee_xi = enum.member(chatterjee_xi)
    ferreira_tail_dep_coeff = enum.member(ferreira_tail_dep_coeff)
    kendall_tau = enum.member(kendall_tau)
    mutual_info = enum.member(mutual_info)

    def __call__(self, x: torch.Tensor, y: torch.Tensor, **kw):
        return self.value(x, y, **kw)


def make_cdf_ppf_kernel(vec_obs: torch.Tensor, bandwidth: float | None = None):
    obs = vec_obs.flatten()
    obs = obs[torch.isfinite(obs)]
    n = obs.numel()
    device = obs.device
    dtype = obs.dtype
    if bandwidth is None:
        bandwidth = obs.std(unbiased=False) * 1.06 * n ** (-0.25)
    obs, _ = torch.sort(obs)

    def cdf(x):
        return (
            ndtr(
                (
                    x.to(device=device, dtype=dtype).flatten().unsqueeze(-1)
                    - obs.unsqueeze(0)
                )
                / bandwidth
            )
            .mean(dim=1)
            .reshape(x.shape)
        )

    F = cdf(obs)
    F, _ = torch.cummax(F, dim=0)

    def ppf(u):
        u0 = u.to(device=device, dtype=dtype).flatten().clamp(0.0, 1.0)
        out = torch.empty_like(u0, device=device, dtype=dtype)
        # below minimum
        mask_lo = u0 <= F[0]
        out[mask_lo] = obs[0]
        # above maximum
        mask_hi = u0 >= F[-1]
        out[mask_hi] = F[-1]

        # in between
        mask_mid = ~(mask_lo | mask_hi)
        if mask_mid.any():
            u_mid = u0[mask_mid]
            idx = torch.searchsorted(F, u_mid)
            idx = idx.clamp(1, n - 1)
            lo, hi = idx - 1, idx
            F_lo, F_hi = F[lo], F[hi]
            x_lo, x_hi = obs[lo], obs[hi]
            denom = F_hi - F_lo
            t = (u_mid - F_lo) / torch.where(denom == 0, torch.ones_like(denom), denom)
            t = t.clamp(0.0, 1.0)

            out[mask_mid] = x_lo + t * (x_hi - x_lo)

        return out.reshape(u.shape)

    return cdf, ppf


@torch.compile
@torch.no_grad()
def solve_ITP(
    fun: callable,  # ! (N,1) -> (N,1)
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    epsilon: float = 1e-7,
    num_iter_max: int = 31,
    k_1: float = 0.2,
) -> torch.Tensor:
    y_a, y_b = fun(x_a), fun(x_b)
    # * corner cases
    x_a = torch.where(
        condition=y_b.abs() < epsilon, input=x_b - epsilon * num_iter_max, other=x_a
    )
    x_b = torch.where(
        condition=y_a.abs() < epsilon, input=x_a + epsilon * num_iter_max, other=x_b
    )
    y_a, y_b, x_wid = fun(x_a), fun(x_b), x_b - x_a
    #
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
