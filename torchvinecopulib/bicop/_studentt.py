from math import ceil, floor, lgamma, log, log1p

import torch
from scipy.optimize import minimize

from ..util import (
    _NU_MAX,
    _NU_MIN,
    _RHO_MAX,
    _RHO_MIN,
    l_dt,
    pbvt,
    pt,
    qt,
    kendall_tau,
)
from ._elliptical import BiCopElliptical


class StudentT(BiCopElliptical):
    # Joe 2014 page 181
    # ! notice all `qt` are from scipy and cannot autograd
    # rho, nu
    _PAR_MIN, _PAR_MAX = (_RHO_MIN, _NU_MIN), (_RHO_MAX, _NU_MAX)

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        # * use pbvt for integer nu; otherwise interpolate linearly between floor(nu) and ceil(nu)
        rho, nu = par
        nu_low = floor(nu)
        if nu == nu_low:
            return pbvt(obs=qt(vec=obs, nu=nu), rho=rho, nu=nu)
        else:
            nu_high = ceil(nu)
            weight = (nu - nu_low) / (nu_high - nu_low)
            return (
                pbvt(obs=qt(vec=obs, nu=nu_low), rho=rho, nu=nu_low)
                .mul_(1.0 - weight)
                .add_(
                    pbvt(obs=qt(vec=obs, nu=nu_high), rho=rho, nu=nu_high), alpha=weight
                )
            )

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        rho, nu = par
        x, y = qt(obs[:, [0]], nu=nu), qt(obs[:, [1]], nu=nu)
        return pt(
            vec=(y.sub_(x, alpha=rho)).div_(
                x.square().add_(nu).div_(nu + 1.0).mul_(1.0 - rho**2).sqrt_()
            ),
            nu=nu + 1.0,
        )

    @staticmethod
    def hinv1_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0)"""
        rho, nu = par
        x, y = qt(obs[:, [0]], nu=nu), qt(obs[:, [1]], nu=nu + 1.0)
        return pt(
            vec=y.mul_(
                x.square().add_(nu).div_(nu + 1.0).mul_(1.0 - rho**2).sqrt_()
            ).add_(x, alpha=rho),
            nu=nu,
        )

    @staticmethod
    def l_pdf_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        rho, nu = max(min(par[0], _RHO_MAX), _RHO_MIN), max(
            min(par[1], _NU_MAX), _NU_MIN
        )
        nu2 = nu / 2.0
        x, y = qt(obs[:, [0]], nu=nu), qt(obs[:, [1]], nu=nu)
        return (
            x.square()
            .add_(y.square())
            .sub_(x * y, alpha=2.0 * rho)
            .div_(nu * (1 - rho**2))
            .log1p_()
            .mul_(-nu2 - 1.0)
            .sub_(l_dt(x, nu=nu))
            .sub_(l_dt(y, nu=nu))
            .add_(
                -0.5 * log1p(-(rho**2))
                - log(nu)
                - 1.1447298858494002
                + lgamma(nu2 + 1)
                - lgamma(nu2)
            )
        )

    @classmethod
    def par2tau_0(cls, par: tuple) -> float:
        return cls.rho2tau_0(rho=par[0])

    @classmethod
    def tau2par(
        cls,
        tau: float = None,
        obs: torch.Tensor = None,
        mtd_opt: str = "COBYLA",
        **kwargs
    ) -> tuple:
        if tau is None:
            tau = kendall_tau(x=obs[:, 0], y=obs[:, 1])

        rho = cls.tau2rho_0(tau=tau)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # * scipy minimize, Powell, Nelder-Mead, COBYLA
        def fun_nll(vec):
            return -StudentT.l_pdf_0(obs=obs, par=(rho, vec[0])).sum().item()

        nu = minimize(
            fun=fun_nll,
            x0=(2.0,),
            bounds=((_NU_MIN, _NU_MAX),),
            method=mtd_opt,
        ).x[0]
        return (rho, nu)
