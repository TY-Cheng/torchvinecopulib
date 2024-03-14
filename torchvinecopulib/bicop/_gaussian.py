from math import log1p, sqrt

import torch

from ..util import _RHO_MAX, _RHO_MIN, pbvnorm, pnorm, qnorm
from ._elliptical import BiCopElliptical


class Gaussian(BiCopElliptical):
    # Joe 2014 page 163
    _PAR_MIN, _PAR_MAX = (-0.9999,), (0.9999,)

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        return pbvnorm(obs=qnorm(obs), rho=par[0])

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""

        rho = par[0]
        return pnorm(
            qnorm(obs[:, [1]])
            .sub_(qnorm(obs[:, [0]]), alpha=rho)
            .div_(sqrt(1.0 - rho**2))
        )

    @staticmethod
    def hinv1_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0)"""
        rho = par[0]
        return pnorm(
            qnorm(obs[:, [1]])
            .mul_(sqrt(1.0 - rho**2))
            .add_(qnorm(obs[:, [0]]), alpha=rho)
        )

    @staticmethod
    def l_pdf_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        # https://math.stackexchange.com/questions/3918915/derivation-of-bivariate-gaussian-copula-density
        rho = max(min(par[0], _RHO_MAX), _RHO_MIN)
        rho2 = rho**2
        x, y = qnorm(obs[:, [0]]), qnorm(obs[:, [1]])

        return (
            (x.square().add_(y.square()))
            .mul_(rho)
            .sub_(x * y, alpha=2.0)
            .div_(2.0 * (rho2 - 1.0))
            .mul_(rho)
            .add_(-0.5 * log1p(-rho2))
        )

    @classmethod
    def par2tau_0(cls, par: tuple) -> torch.Tensor:
        return cls.rho2tau_0(rho=par[0])

    @classmethod
    def tau2par(cls, tau: float, **kwargs) -> torch.Tensor:
        return (cls.tau2rho_0(tau=tau),)
