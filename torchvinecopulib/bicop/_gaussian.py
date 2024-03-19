from math import log1p, sqrt

import torch

from ..util import _RHO_MAX, _RHO_MIN, pbvnorm, pnorm, qnorm
from ._elliptical import BiCopElliptical


class Gaussian(BiCopElliptical):
    # Joe 2014 page 163
    _PAR_MIN, _PAR_MAX = (-0.9999,), (0.9999,)

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return pbvnorm(obs=qnorm(obs), rho=par[0])

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        rho = par[0]
        return pnorm((qnorm(obs[:, [1]]) - rho * qnorm(obs[:, [0]])) / sqrt(1.0 - rho**2))

    @staticmethod
    def hinv1_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0)"""
        rho = par[0]
        return pnorm(qnorm(obs[:, [1]]) * sqrt(1.0 - rho**2) + rho * qnorm(obs[:, [0]]))

    @staticmethod
    def l_pdf_0(
        obs: torch.Tensor,
        par: tuple[float],
    ) -> torch.Tensor:
        # https://math.stackexchange.com/questions/3918915/derivation-of-bivariate-gaussian-copula-density
        rho = max(min(par[0], _RHO_MAX), _RHO_MIN)
        rho2 = rho**2
        x, y = qnorm(obs[:, [0]]), qnorm(obs[:, [1]])
        return -0.5 * log1p(-rho2) - rho / (2.0 - 2.0 * rho2) * (
            (x.square() + y.square()) * rho - 2.0 * x * y
        )

    @classmethod
    def par2tau_0(cls, par: tuple[float]) -> torch.Tensor:
        return cls.rho2tau_0(rho=par[0])

    @classmethod
    def tau2par(cls, tau: float, **kwargs) -> torch.Tensor:
        return (cls.tau2rho_0(tau=tau),)
