import torch
from scipy.optimize import minimize

from ..util import _NU_MAX, _NU_MIN, _RHO_MAX, _RHO_MIN, kendall_tau, l_dt, pbvt, pt, qt
from ._elliptical import BiCopElliptical


class StudentT(BiCopElliptical):
    # Joe 2014 page 181
    # ! exchangeability
    # TODO notice all `qt` are from scipy and cannot autograd
    # rho, nu
    _PAR_MIN, _PAR_MAX = (
        (_RHO_MIN, _NU_MIN),
        (_RHO_MAX, _NU_MAX),
    )

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        # * use pbvt for integer nu; otherwise interpolate linearly between floor(nu) and ceil(nu)
        rho, nu = par[0], par[1]
        nu_low = nu.floor()
        if nu.isclose(nu_low):
            return pbvt(obs=qt(vec=obs, nu=nu), rho=rho, nu=nu)
        else:
            nu_high = nu.ceil()
            weight = (nu - nu_low) / (nu_high - nu_low)
            return (
                pbvt(obs=qt(vec=obs, nu=nu_low), rho=rho, nu=nu_low) * (1.0 - weight)
                + pbvt(obs=qt(vec=obs, nu=nu_high), rho=rho, nu=nu_high) * weight
            )

    @staticmethod
    def hfunc_l_0(obs: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        rho, nu = par[0], par[1]
        x, y = qt(obs[:, [0]], nu=nu), qt(obs[:, [1]], nu=nu)
        return pt(
            vec=(
                (y - rho * x) / ((1.0 - rho**2) * (nu + x.square()) / (nu + 1.0)).sqrt()
            ),
            nu=nu + 1.0,
        )

    @staticmethod
    def hinv_l_0(obs: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0)"""
        rho, nu = par[0], par[1]
        x, y = qt(obs[:, [0]], nu=nu), qt(obs[:, [1]], nu=nu + 1.0)
        return pt(
            vec=(
                x * rho + y * ((nu + x.square()) * (1.0 - rho**2) / (nu + 1.0)).sqrt()
            ),
            nu=nu,
        )

    @staticmethod
    def l_pdf_0(obs: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        rho = torch.clamp(par[0], min=_RHO_MIN, max=_RHO_MAX)
        nu = torch.clamp(par[1], min=_NU_MIN, max=_NU_MAX)
        nu2 = nu / 2.0
        x, y = qt(obs[:, [0]], nu=nu), qt(obs[:, [1]], nu=nu)
        return (
            -0.5 * (-(rho**2)).log1p()
            - nu.log()
            - 1.1447298858494002
            + (nu2 + 1).lgamma()
            - nu2.lgamma()
            - (nu2 + 1)
            * (
                (x.square() + y.square() - 2.0 * rho * x * y) / nu / (1 - rho**2)
            ).log1p()
            - l_dt(x, nu=nu)
            - l_dt(y, nu=nu)
        )

    @classmethod
    def par2tau_0(cls, par: torch.Tensor) -> torch.Tensor:
        return cls.rho2tau_0(rho=par[0])

    @classmethod
    def tau2par(
        cls,
        obs: torch.Tensor,
        tau: torch.Tensor = None,
        mtd_opt: str = "L-BFGS-B",
    ) -> torch.Tensor:
        """quasi MLE for StudentT nu; rho from Kendall's tau"""
        if tau is None:
            tau, _ = kendall_tau(x=obs[:, [0]], y=obs[:, [1]])
        rho = cls.tau2rho_0(tau=tau)
        nu = minimize(
            fun=lambda nu: StudentT.l_pdf_0(
                obs=obs,
                par=torch.tensor(
                    [rho, nu[0]],
                    dtype=obs.dtype,
                    device=obs.device,
                ),
            )
            .nan_to_num()
            .sum()
            .neg(),
            x0=(2.1,),
            bounds=((_NU_MIN, _NU_MAX),),
            method=mtd_opt,
        ).x[0]
        return torch.tensor(
            [
                rho,
                nu,
            ],
            dtype=obs.dtype,
            device=obs.device,
        )
