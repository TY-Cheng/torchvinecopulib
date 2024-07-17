import math
from enum import Enum
from functools import partial
from itertools import product

import torch
from scipy.special import stdtr, stdtrit
from scipy.stats import kendalltau

__all__ = [
    "ENUM_FUNC_BIDEP",
    "cdf_func_kernel",
    "chatterjee_xi",
    "debye1",
    "ferreira_tail_dep_coeff",
    "kendall_tau",
    "mutual_info",
    "ref_count_hfunc",
    "solve_ITP",
    "wasserstein_dist_ind",
]

_CDF_MIN, _CDF_MAX = 0.0, 1.0
_NU_MIN, _NU_MAX = 1.001, 49.999
_RHO_MIN, _RHO_MAX = -0.99, 0.99
_TAU_MIN, _TAU_MAX = -0.999, 0.999


# def kendall_tau(
#     x: torch.Tensor,
#     y: torch.Tensor,
#     tau_min: float = _TAU_MIN,
#     tau_max: float = _TAU_MAX,
# ) -> float:
#     """https://gist.github.com/ili3p/f2b38b898f6eab0d87ec248ea39fde94
#     x,y are both of shape (n, 1)
#     """
#     n = x.shape[0]
#     n *= n - 1
#     return (
#         ((x.T - x).sign() * (y.T - y).sign()).sum().div(n).clamp(min=tau_min, max=tau_max).item()
#     )


def kendall_tau(
    x: torch.Tensor,
    y: torch.Tensor,
    tau_min: float = _TAU_MIN,
    tau_max: float = _TAU_MAX,
) -> float:
    """x,y are both of shape (n, 1)"""
    res = kendalltau(x.cpu().ravel(), y.cpu().ravel())
    return (max(min(res[0], tau_max), tau_min), res[1])


def mutual_info(x: torch.Tensor, y: torch.Tensor, is_sklearn: bool = True) -> float:
    """mutual information, need scikit-learn or fastkde installed.
    x,y are both of shape (n, 1)

    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html

    Purkayastha, S., & Song, P. X. K. (2024).
    fastMI: A fast and consistent copula-based nonparametric estimator of mutual information.
    Journal of Multivariate Analysis, 201, 105270.
    """
    if is_sklearn:
        from sklearn.feature_selection import mutual_info_regression

        return mutual_info_regression(
            X=x.cpu(),
            # ! notice the shape of y
            y=y.cpu().ravel(),
            discrete_features=False,
            n_neighbors=3,
            copy=True,
            random_state=None,
        )[0]
    else:
        # Purkayastha, S., & Song, P. X. K. (2024).
        from fastkde.fastKDE import pdf_at_points
        from numpy import log

        x_, y_ = x.ravel(), y.ravel()
        joint = pdf_at_points(var1=x_, var2=y_)
        joint = joint[joint > 0]
        margin_x = pdf_at_points(var1=x_)
        margin_x = margin_x[margin_x > 0]
        margin_y = pdf_at_points(var1=y_)
        margin_y = margin_y[margin_y > 0]
        return (log(joint).mean() - log(margin_x).mean() - log(margin_y).mean()).item()


def ferreira_tail_dep_coeff(x: torch.Tensor, y: torch.Tensor) -> float:
    """pairwise tail dependence coefficient (λ) estimator, max of rotation 0, 90, 180, 270
    x and y are both of shape (n, 1) inside (0, 1)
    symmetric for (x,y), (y,1-x), (1-x,1-y), (1-y,x), (y,x), (1-x,y), (1-y,1-x), (x,1-y)
    Ferreira, M.S., 2013. Nonparametric estimation of the tail-dependence coefficient;
    """
    x1 = 1 - x
    y1 = 1 - y
    return (
        3
        - (
            1
            - torch.as_tensor(
                [
                    torch.max(x, y).mean(),
                    torch.max(y, x1).mean(),
                    torch.max(x1, y1).mean(),
                    torch.max(y1, x).mean(),
                ]
            )
            .clamp(0.5, 0.6666666666666666)
            .min()
        ).reciprocal()
    ).item()


def chatterjee_xi(x: torch.Tensor, y: torch.Tensor, M: int = 1) -> float:
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
        torch.as_tensor([xy_sum(m) for m in range(1, M + 1)]).sum(dim=0).max().item()
    ) / (M * (1.0 + n) * (1.0 + M + 4.0 * n))


def wasserstein_dist_ind(
    x: torch.Tensor,
    y: torch.Tensor,
    p: int = 2,
    reg: float = 0.1,
    num_step: int = 50,
    seed: int = 0,
) -> float:
    """Wasserstein distance from bicop obs to indep bicop, by ot.sinkhorn2 (averaged for each observation).
        Need pot installed.

    :param x: copula obs of shape (n,1)
    :type x: torch.Tensor
    :param y: copula obs of shape (n,1)
    :type y: torch.Tensor
    :param p: p-norm to calculate distance between each vector pair, defaults to 2
    :type p: int, optional
    :param reg: regularization strength, defaults to 0.1
    :type reg: float, optional
    :param num_step: number of steps in the independent bivariate copula grid, defaults to 50
    :type num_step: int, optional
    :param seed: random seed for torch.manual_seed(), defaults to 0
    :type seed: int, optional
    :return: Wasserstein distance from bicop obs to indep bicop
    :rtype: float
    """
    from ot import sinkhorn2

    torch.manual_seed(seed=seed)
    return sinkhorn2(
        a=torch.ones_like(x) / x.shape[0],
        b=torch.ones(size=(num_step**2, 1), device=x.device, dtype=x.dtype) / num_step**2,
        M=torch.cdist(
            x1=torch.hstack([x, y]),
            x2=torch.as_tensor(
                (
                    *product(
                        torch.linspace(0, 1, num_step),
                        torch.linspace(0, 1, num_step),
                    ),
                ),
                device=x.device,
                dtype=x.dtype,
            ),
            p=p,
        ),
        reg=reg,
    ).item()


class ENUM_FUNC_BIDEP(Enum):
    """an enum class for bivariate dependence measures"""

    chatterjee_xi = partial(chatterjee_xi)
    ferreira_tail_dep_coeff = partial(ferreira_tail_dep_coeff)
    kendall_tau = partial(kendall_tau)
    mutual_info = partial(mutual_info)
    wasserstein_dist_ind = partial(wasserstein_dist_ind)


def cdf_func_kernel(obs: torch.Tensor, is_scott: bool = True) -> callable:
    """kernel density estimation (KDE) function of the cumulative distribution function (CDF)

    :param obs: observations, of shape (n,1)
    :type obs: torch.Tensor
    :param is_scott: whether to use Scott's rule for bandwidth, defaults to True
    :type is_scott: bool, optional
    :return: a CDF function by KDE
    :rtype: callable
    """
    if is_scott:
        # * bandwidth by Scott 1992
        band_width = (
            torch.nanquantile(input=obs, q=0.75) - torch.nanquantile(input=obs, q=0.25)
        ) / 1.349
        band_width = obs.std().clamp_max(band_width) * 1.059 * len(obs) ** (-0.2)
    else:
        band_width = obs.std() * 0.6973425390765554 * len(obs) ** -0.1111111111111111

    def func_cdf(q: torch.Tensor) -> torch.Tensor:
        return torch.special.ndtr((q - obs.T) / band_width).nanmean(dim=1)

    return func_cdf


# * Gaussian distribution CDF (p), PPF (q), density (d)
pnorm = torch.special.ndtr
qnorm = torch.special.ndtri


def pbvnorm(obs: torch.Tensor, rho: float) -> torch.Tensor:
    """CDF of (standard) bivariate Gaussian distribution
    modified from http://www.math.wsu.edu/faculty/genz/software/matlab/bvnl.m
    https://keisan.casio.com/exec/system/1280624821

    :param obs: bivariate Gaussian observations, of shape (num_obs, 2)
    :type obs: torch.Tensor
    :param rho: the rho parameter, also a Pearson's corr coef
    :type rho: float
    :return: cumulative distribution function (CDF) of shape (num_obs, 1), given the observation
    :rtype: torch.Tensor
    """
    rho = min(max(rho, _RHO_MIN), _RHO_MAX)
    # Gauss Legendre points and weights, n = 100
    w = torch.as_tensor(
        data=(
            0.007968192496166605,
            0.01846646831109096,
            0.02878470788332337,
            0.03879919256962705,
            0.04840267283059405,
            0.057493156217619065,
            0.06597422988218049,
            0.0737559747377052,
            0.08075589522942021,
            0.08689978720108298,
            0.09212252223778612,
            0.09636873717464425,
            0.09959342058679527,
            0.1017623897484055,
            0.10285265289355884,
            0.007968192496166605,
            0.01846646831109096,
            0.02878470788332337,
            0.03879919256962705,
            0.04840267283059405,
            0.057493156217619065,
            0.06597422988218049,
            0.0737559747377052,
            0.08075589522942021,
            0.08689978720108298,
            0.09212252223778612,
            0.09636873717464425,
            0.09959342058679527,
            0.1017623897484055,
            0.10285265289355884,
        ),
        device=obs.device,
        dtype=obs.dtype,
    ).reshape(-1, 1)
    x = torch.as_tensor(
        data=(
            0.003106515925350495,
            0.016331876720252825,
            0.03997813503169245,
            0.07379995257072569,
            0.11743946420794726,
            0.17043423761723164,
            0.23222256789517381,
            0.30214950520668415,
            0.3794738170107571,
            0.46337585185798014,
            0.5529662304619108,
            0.6472952744691218,
            0.7453630738321102,
            0.8461300863914165,
            0.9485281574446823,
            1.9968934840746495,
            1.983668123279747,
            1.9600218649683074,
            1.9262000474292744,
            1.8825605357920527,
            1.8295657623827684,
            1.7677774321048263,
            1.697850494793316,
            1.6205261829892428,
            1.5366241481420198,
            1.447033769538089,
            1.3527047255308782,
            1.2546369261678898,
            1.1538699136085835,
            1.0514718425553178,
        ),
        device=obs.device,
        dtype=obs.dtype,
    ).reshape(1, -1)
    #
    h, k = obs[:, [0]], obs[:, [1]]
    asr = math.asin(rho) / 2.0
    sn = (x * asr).sin()
    return (
        (
            (
                (
                    (h.square() + k.square()) / -2.0
                    # ! broadcast: x,sn are rows; w,h,k are columns
                    + sn * h * k
                )
                / (1.0 - sn.square())
            ).exp()
            @ w
            * (asr / 6.283185307179586)
            + (pnorm(h) * pnorm(k))
        )
        .nan_to_num()
        .clamp(min=_CDF_MIN, max=_CDF_MAX)
    )


# # * Student's t distribution CDF (p), PPF (q), density (d)
# def inc_beta_reg(vec: torch.Tensor, a: float, b: float) -> torch.Tensor:
#     # * regularized incomplete beta integral, with a = 0.5, b = nu / 2, vec in [0,1]
#     res = torch.empty_like(vec)
#     if (idx := vec > ((a + 1.0) / (2.0 + a + b))).any():
#         res[idx] = 1.0 - inc_beta_reg(vec=1.0 - vec[idx], a=b, b=a)
#     idx = ~idx
#     x = vec[idx]
#     qab = a + b
#     qap = a + 1.0
#     qam = a - 1.0
#     c = torch.ones_like(x)
#     d = 1.0 / (1.0 - qab * x / qap)
#     f = d.clone()
#     front = (
#         -math.lgamma(a)
#         - math.lgamma(b)
#         + math.lgamma(a + b)
#         - math.log(a)
#         + x.log() * a
#         + x.neg().log1p() * b
#     ).exp()
#     for i in range(1, 200):
#         i2 = i * 2.0
#         ai2 = a + i2
#         nmrt = (i * (b - i) / ((qam + i2) * ai2)) * x
#         d = (d * nmrt + 1.0).reciprocal()
#         c = nmrt / c + 1.0
#         f *= c * d
#         nmrt = (-(a + i) * (qab + i) / ((qap + i2) * ai2)) * x
#         d = (d * nmrt + 1.0).reciprocal()
#         c = nmrt / c + 1.0
#         cd = c * d
#         f *= cd
#         if ((1.0 - cd).abs() < 1e-6).all():
#             break
#     res[idx] = front * f
#     return res.nan_to_num().clamp(min=_CDF_MIN, max=_CDF_MAX)


# def inc_beta_reg_inv(vec: torch.Tensor, a: float, b: float) -> torch.Tensor:
#     a1, b1 = a - 1.0, b - 1.0
#     if a > 1.0 and b > 1.0:
#         tmp = vec.clone()
#         idx = vec >= 0.5
#         tmp[idx] = 1.0 - tmp[idx]
#         t = (-2.0 * tmp.log()).sqrt()
#         x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
#         idx = ~idx
#         x[idx] = -x[idx]
#         al = (x.square() - 3.0) / 6.0
#         h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0))
#         w = (x * math.sqrt(al + h) / h) - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (
#             al + 5.0 / 6.0 - 2.0 / (3.0 * h)
#         )
#         x = a / (a + b * (2.0 * w).exp())
#     else:
#         x = vec.clone()
#         t = math.exp(a * math.log(a / (a + b))) / a
#         w = t + math.exp(b * math.log(b / (a + b))) / b
#         idx = vec < t / w
#         x[idx] = (a * w * x[idx]).pow(1.0 / a)
#         idx = ~idx
#         x[idx] = 1.0 - (b * w * (1.0 - x[idx])).pow(1.0 / b)
#     afac = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
#     for _ in range(10):
#         t = (a1 * x.log() + b1 * x.neg().log1p() + afac).exp()
#         u = (inc_beta_reg(vec=x, a=a, b=b) - vec) / t
#         t = u / (1.0 - 0.5 * (u * (a1 / x - b1 / (1.0 - x))).clamp_max(max=1.0))
#         idx = (x > 0.0) & (x < 1.0)
#         x[idx] -= t[idx]
#         idx = x < 0.0
#         x[idx] = 0.5 * (x[idx] + t[idx])
#         idx = x > 1.0
#         x[idx] = 0.5 * (x[idx] + t[idx] + 1.0)
#         x = x.nan_to_num().clamp(min=_CDF_MIN, max=_CDF_MAX)
#     return x


# def pt(vec: torch.Tensor, nu: float) -> torch.Tensor:
#     if nu == 1:
#         res = vec.atan() / 3.141592653589793 + 0.5
#     elif nu == 2:
#         res = (vec / (vec.square() + 2.0).sqrt() + 1.0) / 2.0
#     elif nu == 3:
#         res = (
#             (vec / 1.7320508075688772).atan() / 3.141592653589793
#             + 0.5
#             + 0.5513288954217921 * vec / (vec.square() + 3.0)
#         )
#     elif nu == 4:
#         _ = vec.square()
#         res = (vec * (_ + 6.0) / (_ + 4.0).pow(1.5) + 1.0) / 2.0
#     elif nu == 5:
#         res = (
#             (vec / 2.23606797749979).atan() / 3.141592653589793
#             + (
#                 (vec.square() * 3.0 + 25.0)
#                 * 0.23725418113905902
#                 * vec
#                 / (vec.square() + 5.0).square()
#             )
#             + 0.5
#         )

#     elif nu == 6:
#         res = vec.square()
#         res = ((res.square() * 2.0 + res * 30.0 + 135.0) * vec / (res + 6.0).pow(2.5)) / 4.0 + 0.5

#     else:
#         nu = max(min(_NU_MAX, nu), _NU_MIN)
#         res = inc_beta_reg(vec=nu / (vec.square() + nu), a=nu / 2.0, b=0.5)
#         idx = vec > 0.0
#         res[idx] = 2.0 - res[idx]
#         res *= 0.5
#     return res.nan_to_num().clamp(min=_CDF_MIN, max=_CDF_MAX)


# def qt(vec: torch.Tensor, nu: float) -> torch.Tensor:
#     vec2 = vec * 2.0
#     nu2 = nu / 2.0
#     res = torch.empty_like(input=vec, device=vec.device)
#     idx = vec < 0.5
#     res[idx] = (inc_beta_reg_inv(vec=vec2[idx], a=nu2, b=0.5).reciprocal() - 1.0).sqrt().neg()
#     idx = ~idx
#     res[idx] = (inc_beta_reg_inv(vec=2.0 - vec2[idx], a=nu2, b=0.5).reciprocal() - 1.0).sqrt()
#     res[vec == 0.5] = 0.0
#     res *= math.sqrt(nu)
#     return res.nan_to_num()


def pt_scipy(vec: torch.Tensor, nu: float) -> torch.Tensor:
    if vec.device.type == "cpu":
        return stdtr(nu, vec)
    else:
        return stdtr(nu, vec.cpu()).cuda()


def qt_scipy(vec: torch.Tensor, nu: float) -> torch.Tensor:
    if vec.device.type == "cpu":
        return stdtrit(nu, vec)
    else:
        return stdtrit(nu, vec.cpu()).cuda()


pt = pt_scipy
qt = qt_scipy


def l_dt(obs: torch.Tensor, nu: float) -> torch.Tensor:
    """log density of Student's t distribution

    :param obs: Student's t observations
    :type obs: torch.Tensor
    :param nu: degrees of freedom
    :type nu: float
    :return: log density, given the observations
    :rtype: torch.Tensor
    """
    nu2 = nu / 2.0
    return ((obs.square() / nu).log1p() * (-nu2 - 0.5)) + (
        math.lgamma(nu2 + 0.5) - math.lgamma(nu2) - 0.5723649429247001 - 0.5 * math.log(nu)
    )


def pbvt(
    obs: torch.Tensor,
    rho: float,
    nu: float,
) -> torch.Tensor:
    """CDF of (standard) bivariate Student's t distribution
    modified from http://www.math.wsu.edu/faculty/genz/software/matlab/bvtl.m

    :param obs: bivariate Student's t observations, of shape (num_obs, 2)
    :type obs: torch.Tensor
    :param rho: the rho parameter
    :type rho: float
    :param nu: the nu parameter, degrees of freedom
    :type nu: float
    :return: cumulative distribution function (CDF) of shape (num_obs, 1), given the observations
    :rtype: torch.Tensor
    """
    h = obs[:, [0]]
    k = obs[:, [1]]
    rho: torch.Tensor = torch.as_tensor(
        data=rho,
        dtype=obs.dtype,
        device=obs.device,
    ).clamp(
        min=_RHO_MIN,
        max=_RHO_MAX,
    )
    nu: torch.Tensor = torch.as_tensor(
        data=nu,
        dtype=obs.dtype,
        device=obs.device,
    ).clamp(
        min=_NU_MIN,
        max=_NU_MAX,
    )
    #
    if nu < 1:
        bvt = pbvnorm(obs=obs, rho=rho)
    else:
        xnhk, xnkh = torch.zeros_like(h), torch.zeros_like(h)
        ors = 1.0 - rho**2
        snu = nu.sqrt()
        hhh, kkk = h - rho * k, k - rho * h
        hs, ks = hhh.sign(), kkk.sign()
        idx = hhh.abs() + ors > 0.0
        xnhk[idx] = hhh[idx].square() / (hhh[idx].square() + ors * (nu + k[idx].square()))
        xnkh[idx] = kkk[idx].square() / (kkk[idx].square() + ors * (nu + h[idx].square()))
        # gmph as hhh, gmpk as kkk
        if nu.ceil() % 2 == 0:
            bvt = torch.full_like(
                input=h,
                fill_value=torch.atan2(ors.sqrt(), -rho) / 6.283185307179586,
            )
            hhh = h / (16.0 * (h.square() + nu)).sqrt()
            kkk = k / (16.0 * (k.square() + nu)).sqrt()
            btnchk = 0.6366197723675814 * torch.atan2(xnhk.sqrt(), (1.0 - xnhk).sqrt())
            btpdhk = 0.6366197723675814 * (xnhk * (1.0 - xnhk)).sqrt()
            btnckh = 0.6366197723675814 * torch.atan2(xnkh.sqrt(), (1.0 - xnkh).sqrt())
            btpdkh = 0.6366197723675814 * (xnkh * (1.0 - xnkh)).sqrt()
            for j in torch.arange(start=1.0, end=nu / 2.0 + 1e-5, step=1):
                bvt += hhh * (btnckh * ks + 1.0) + kkk * (btnchk * hs + 1.0)
                hhh *= (j - 0.5) / (j * (1.0 + h.square() / nu))
                kkk *= (j - 0.5) / (j * (1.0 + k.square() / nu))
                btnchk += btpdhk
                btpdhk *= j * 2.0 * (1.0 - xnhk) / (2.0 * j + 1.0)
                btnckh += btpdkh
                btpdkh *= j * 2.0 * (1.0 - xnkh) / (2.0 * j + 1.0)
        else:
            qhrk = (h.square() + k.square() - 2.0 * rho * h * k + nu * ors).sqrt()
            hkrn = h * k + rho * nu
            hkn = h * k - nu
            bvt = (
                torch.atan2(
                    -snu * (hkn * qhrk + (h + k) * hkrn),
                    hkn * hkrn - nu * (h + k) * qhrk,
                )
                / 6.283185307179586
            )
            idx = bvt < -1e-8
            bvt[idx] += 1.0
            hhh = h / (6.283185307179586 * snu * (1.0 + h.square() / nu))
            kkk = k / (6.283185307179586 * snu * (1.0 + k.square() / nu))
            btnchk, btnckh = xnhk.sqrt(), xnkh.sqrt()
            btpdhk, btpdkh = btnchk * 1.0, btnckh * 1.0
            if (nu - 1.0) / 2.0 >= 1.0:
                for j in torch.arange(start=1, end=(nu - 1.0) / 2.0 + 1e-5, step=1):
                    bvt += hhh * (1.0 + ks * btnckh) + kkk * (1.0 + hs * btnchk)
                    hhh *= j / ((j + 0.5) * (1.0 + h.square() / nu))
                    kkk *= j / ((j + 0.5) * (1.0 + k.square() / nu))
                    btpdhk *= (j - 0.5) * (1.0 - xnhk) / j
                    btnchk += btpdhk
                    btpdkh *= (j - 0.5) * (1.0 - xnkh) / j
                    btnckh += btpdkh
    return bvt


def debye1(x: float) -> float:
    """computes the Debye function of order 1.

    https://github.com/openturns/openturns/blob/b5797d7e4a71c71faf86df51f26ad0d8d551ad08/lib/src/Base/Func/SpecFunc/Debye.cxx

    :param x: upper limit of the integral
    :type x: float
    :return: Debye function of order 1; 0 if x<=0
    :rtype: float
    """
    if x < 0:
        return 0
    elif x >= 3:
        # ! scalar loop faster than short np.array
        k_max = [0, 0, 0, 13, 10, 8, 7, 6, 5, 5, 4, 4, 4, 3][max(min(int(x), 13), 0)]
        res = 1.64493406684822643647241516665
        for k in range(1, 1 + k_max):
            xk = x * k
            res -= math.exp(-xk) * (1.0 / xk + 1.0 / xk**2) * x**2
        res /= x
    else:
        koeff = (
            0.0,
            1.289868133696452872944830333292,
            1.646464674222763830320073930823e-01,
            3.468612396889827942903585958184e-02,
            8.154712395888678757370477017305e-03,
            1.989150255636170674291917800638e-03,
            4.921731066160965972759960954793e-04,
            1.224962701174096585170902102707e-04,
            3.056451881730374346514297527344e-05,
            7.634586529999679712923289243879e-06,
            1.907924067745592226304077366899e-06,
            4.769010054554659800072963735060e-07,
            1.192163781025189592248804158716e-07,
            2.980310965673008246931701326140e-08,
            7.450668049576914109638408036805e-09,
            1.862654864839336365743529470042e-09,
            4.656623667353010984002911951881e-10,
            1.164154417580540177848737197821e-10,
            2.910384378208396847185926449064e-11,
            7.275959094757302380474472711747e-12,
            1.818989568052777856506623677390e-12,
            4.547473691649305030453643155957e-13,
            1.136868397525517121855436593505e-13,
            2.842170965606321353966861428348e-14,
            7.105427382674227346596939068119e-15,
            1.776356842186163180619218277278e-15,
            4.440892101596083967998640188409e-16,
            1.110223024969096248744747318102e-16,
            2.775557561945046552567818981300e-17,
            6.938893904331845249488542992219e-18,
            1.734723476023986745668411013469e-18,
            4.336808689994439570027820336642e-19,
            1.084202172491329082183740080878e-19,
            2.710505431220232916297046799365e-20,
            6.776263578041593636171406200902e-21,
            1.694065894509399669649398521836e-21,
            4.235164736272389463688418879636e-22,
            1.058791184067974064762782460584e-22,
            2.646977960169798160618902050189e-23,
            6.617444900424343177893912768629e-24,
            1.654361225106068880734221123349e-24,
            4.135903062765153408791935838694e-25,
            1.033975765691286264082026643327e-25,
            2.584939414228213340076225223666e-26,
            6.462348535570530772269628236053e-27,
            1.615587133892632406631747637268e-27,
            4.038967834731580698317525293132e-28,
            1.009741958682895139216954234507e-28,
            2.524354896707237808750799932127e-29,
            6.310887241768094478219682436680e-30,
            1.577721810442023614704107565240e-30,
            3.944304526105059031370476640000e-31,
            9.860761315262647572437533499000e-32,
            2.465190328815661892443976898000e-32,
            6.162975822039154730370601500000e-33,
            1.540743955509788682510501190000e-33,
            3.851859888774471706184973900000e-34,
            9.629649721936179265360991000000e-35,
            2.407412430484044816328953000000e-35,
            6.018531076210112040809600000000e-36,
            1.504632769052528010200750000000e-36,
            3.761581922631320025497600000000e-37,
            9.403954806578300063715000000000e-38,
            2.350988701644575015901000000000e-38,
            5.877471754111437539470000000000e-39,
            1.469367938527859384580000000000e-39,
            3.673419846319648458500000000000e-40,
            9.183549615799121117000000000000e-41,
            2.295887403949780249000000000000e-41,
            5.739718509874450320000000000000e-42,
            1.434929627468612270000000000000e-42,
        )
        x2pi = x * 0.159154943091895335768883763373  # 1/(2pi)
        res, k = 0.0, 1
        while k <= 69:
            res_0 = res
            res += (koeff[k] + 2.0) * x2pi ** (k * 2.0) / (k * 2.0 + 1.0)
            k += 1
            res -= (koeff[k] + 2.0) * x2pi ** (k * 2.0) / (k * 2.0 + 1.0)
            if abs(res - res_0) < 1e-7:
                break
            else:
                k += 1
        res += 1.0 - x / 4.0
    return res


def solve_ITP(
    f: callable,
    a: float,
    b: float,
    eps_2: float = 1e-9,
    n_0: int = 1,
    k_1: float = 0.2,
    k_2: float = 2.0,
    j_max: int = 31,
) -> float:
    """Solve an arbitrary function for a zero-crossing.

    Oliveira, I.F. and Takahashi, R.H., 2020.
    An enhancement of the bisection method average performance preserving minmax optimality.
    ACM Transactions on Mathematical Software (TOMS), 47(1), pp.1-24.

    https://docs.rs/kurbo/0.8.1/kurbo/common/fn.solve_itp.html

    https://en.wikipedia.org/wiki/ITP_method

    ! It is assumed that f(a) < 0 and f(b) > 0, otherwise unexpected results may occur.

    The ITP method has tuning parameters. This implementation hardwires k2 to 2.0,
    both because it avoids an expensive floating point exponentiation,
    and because this value has been tested to work well with curve fitting problems.

    The n0 parameter controls the relative impact of the bisection and secant components.
    When it is 0, the number of iterations is guaranteed to be no more than the number required by bisection
    (thus, this method is strictly superior to bisection). However, when the function is smooth,
    a value of 1 gives the secant method more of a chance to engage, so the average number of iterations is likely lower,
    though there can be one more iteration than bisection in the worst case.

    The k1 parameter is harder to characterize, and interested users are referred to the paper,
    as well as encouraged to do empirical testing. To match the the paper, a value of 0.2 / (b - a) is suggested,
    and this is confirmed to give good results. When the function is monotonic,
    the returned result is guaranteed to be within epsilon of the zero crossing.
    """
    y_a, y_b, x_wid = f(a), f(b), b - a
    n_max = n_0 + math.ceil(math.log2(x_wid / eps_2))
    for j in range(1, j_max + 1):
        # Calculating Parameters
        if x_wid < eps_2:
            break
        x_half = (a + b) / 2.0
        rho = eps_2 * 2.0 ** (n_max - j) - x_wid / 2.0
        delta = k_1 * x_wid**k_2
        # Interpolation
        x_f = (y_b * a - y_a * b) / (y_b - y_a)
        # Truncation
        tmp = x_half - x_f
        sign = tmp / abs(tmp)
        x_t = x_half if delta > abs(tmp) else x_f + sign * delta
        # Projection
        x_ITP = x_t if (rho >= abs(x_t - x_half)) else x_half - sign * rho
        # Updating interval
        y_ITP = f(x_ITP)
        if y_ITP > 0.0:
            b, y_b = x_ITP, y_ITP
        elif y_ITP < 0.0:
            a, y_a = x_ITP, y_ITP
        else:
            return x_ITP
        x_wid = b - a

    return (a + b) / 2.0


def ref_count_hfunc(
    dct_tree: dict,
    tpl_first_vs: tuple[tuple[int, frozenset]] = tuple(),
    tpl_sim: tuple[int] = tuple(),
) -> tuple[dict, tuple[int], int]:
    """reference counting for each data vertex during cond-sim workflow,
    for garbage collection (memory release) and source vertices selection;
    1. when len(tpl_sim) < num_dim: vertices not in tpl_sim are set on the top lv, vertices in tpl_sim are set at deepest lvs
    2. when len(tpl_sim) == num_dim: check dct_first_vs to move vertices up

    :param dct_tree: dct_tree inside a DataVineCop object, of the form {lv: {(v_l, v_r, s): bidep_func}}
    :type dct_tree: dict
    :param tpl_first_vs: tuple of vertices (explicitly arranged in conditioned - conditioning set)
        that are taken as known at the beginning of a simulation workflow,
        only used when len(tpl_sim)==num_dim, defaults to tuple()
    :type tpl_first_vs: tuple[tuple[int, frozenset]], optional
    :param tpl_sim: tuple of vertices in a full simulation workflow,
        gives flexibility to experienced users, defaults to tuple()
    :type tpl_sim: tuple[int], optional
    :return: number of visits for each vertex; tuple of source vertices in this simulation workflow
        from shallowest to deepest; number of hfunc calls
    :rtype: tuple[dict, tuple[int], int]
    """

    def _visit(v_down: int, s_down: frozenset, is_hinv: bool = False):
        """recursively visits vertices and update their reference counts"""
        # * locate the bicop on upper level that generates this pseudo obs
        lv_up = len(s_down) - 1
        s_bcp = {v_down} | s_down
        for (v_l, v_r, s_up), _ in dct_tree[lv_up].items():
            if ({v_l, v_r} | s_up) == s_bcp:
                break
        # * check missing parents
        if is_hinv:
            l_visit = [(v_l if v_down == v_r else v_r, s_up)]
        else:
            l_visit = [(v_l, s_up), (v_r, s_up)]
        for v, s in l_visit:
            # can be tensor or None
            if dct_ref_count.get((v, s), None) is None:
                _visit(v_down=v, s_down=s, is_hinv=False)
                # ! call hfunc only when parent missing
                num_hfunc[0] += 1
        # * increment reference counts for all three vertices
        l_visit = [(v_l, s_up), (v_r, s_up), (v_down, s_down)]
        for v, s in l_visit:
            dct_ref_count[v, s] = dct_ref_count.get((v, s), 0) + 1
        if is_hinv:
            return v_down, s_up

    if not len(tpl_sim):
        raise ValueError("tpl_sim must be a non-empty tuple")
    num_dim = max(dct_tree) + 2
    if len(tpl_sim) < num_dim:
        # ! when len(tpl_sim) < num_dim:
        # ! vertices not in tpl_sim are set on the top lv, vertices in tpl_sim are set at deepest lv
        s_cond = frozenset(range(num_dim)) - frozenset(tpl_sim)
        tpl_sim = tuple(
            (v, s_cond | frozenset(tpl_sim[idx + 1 :])) for idx, v in enumerate(tpl_sim)
        ) + tuple((v, frozenset()) for v in s_cond)
    else:
        # ! when len(tpl_sim) == num_dim:
        # ! check dct_first_vs to move vertices up
        dct_first_vs = {v_s_cond[0]: v_s_cond for v_s_cond in tpl_first_vs}
        tpl_sim = tuple(
            dct_first_vs.get(v, (v, frozenset(tpl_sim[(idx + 1) :])))
            for idx, v in enumerate(tpl_sim)
        )
    tpl_sim = tpl_sim[::-1]
    # ! count in initial sim (pseudo obs that are given at the beginning of each sim path)
    dct_ref_count = {_: 1 for _ in tpl_sim}
    # * mutable
    num_hfunc = [0]
    # * process each source vertex from the shallowest to the deepest
    for v, s in tpl_sim:
        while s:
            v, s = _visit(v_down=v, s_down=s, is_hinv=True)

    return dct_ref_count, tpl_sim, num_hfunc[0]
