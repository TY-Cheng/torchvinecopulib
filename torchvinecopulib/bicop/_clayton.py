from math import log1p

import torch

from ._archimedean import BiCopArchimedean


class Clayton(BiCopArchimedean):
    # Joe 2014 page 168 4.6.1 Bivariate Mardia-Takahasi-Clayton-Cook-Johnson
    # * suggest torch.float64 for |par|<61, torch.float32 for |par|<15
    _PAR_MIN, _PAR_MAX = (1e-4,), (61.0,)

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        delta = par[0]

        return (
            ((-delta * obs[:, [0]].log()).expm1() + (-delta * obs[:, [1]].log()).expm1()).log1p()
            * (-1.0 / delta)
        ).exp()

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        delta = par[0]
        return (obs[:, [0]].pow(delta) * (obs[:, [1]].pow(-delta) - 1.0) + 1.0).pow(
            -(1.0 / delta) - 1.0
        )

    @staticmethod
    def hinv1_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0)"""
        delta = par[0]
        return (
            ((obs[:, [1]]).pow(-delta / (delta + 1.0)) - 1.0) * obs[:, [0]].pow(-delta) + 1.0
        ).pow(-1.0 / delta)

    @staticmethod
    def l_pdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        delta = max(min(par[0], Clayton._PAR_MAX[0]), Clayton._PAR_MIN[0])
        l_0, l_1 = obs[:, [0]].log(), obs[:, [1]].log()
        return (
            log1p(delta)
            - (delta + 1.0) * (l_0 + l_1)
            - (1.0 / delta + 2.0) * ((-delta * l_0).expm1() + (-delta * l_1).expm1()).log1p()
        )

    @staticmethod
    def par2tau_0(par: tuple[float]) -> float:
        delta = par[0]
        return delta / (delta + 2.0)

    @staticmethod
    def tau2par_0(tau: float, **kwargs) -> tuple:
        # ! par δ > 0 (not -1!)
        t_a = abs(tau)
        return ((2.0 * t_a / (1.0 - t_a)),)
