from math import log1p

import torch

from ._archimedean import BiCopArchimedean


class Clayton(BiCopArchimedean):
    # Joe 2014 page 168 4.6.1 Bivariate Mardia-Takahasi-Clayton-Cook-Johnson
    # * suggest torch.float64 for |par|<61, torch.float32 for |par|<15
    _PAR_MIN, _PAR_MAX = (1e-4,), (61.0,)

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        delta = par[0]
        l_0, l_1 = obs[:, [0]].log(), obs[:, [1]].log()
        return (
            l_0.mul_(-delta)
            .expm1_()
            .add_(l_1.mul_(-delta).expm1_())
            .log1p_()
            .mul_(-1 / delta)
            .exp_()
        )

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        delta = par[0]
        return (obs[:, [0]].pow(delta).mul_(obs[:, [1]].pow(-delta) - 1.0) + 1.0).pow_(
            -(1 / delta) - 1.0
        )

    @staticmethod
    def hinv1_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0)"""
        delta = par[0]
        return (
            (obs[:, [1]])
            .pow(-delta / (delta + 1.0))
            .sub_(1)
            .mul_(obs[:, [0]].pow(-delta))
            .add_(1.0)
            .pow_(-1 / delta)
        )

    @staticmethod
    def l_pdf_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        delta = max(min(par[0], Clayton._PAR_MAX[0]), Clayton._PAR_MIN[0])
        l_0, l_1 = obs[:, [0]].log(), obs[:, [1]].log()
        return (
            l_0.mul(-delta)
            .expm1_()
            .add_(l_1.mul(-delta).expm1_())
            .log1p_()
            .mul_(-1.0 / delta - 2.0)
            .add_(l_0.add(l_1).mul_(-delta - 1.0))
            .add_(log1p(delta))
        )

    @staticmethod
    def par2tau_0(par: tuple) -> float:
        delta = par[0]
        return delta / (delta + 2.0)

    @staticmethod
    def tau2par_0(tau: float, **kwargs) -> tuple:
        # ! par δ > 0 (not -1!)
        t_a = abs(tau)
        return ((2.0 * t_a / (1.0 - t_a)),)
