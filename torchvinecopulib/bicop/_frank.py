from math import exp, expm1

import torch

from ..util import debye1, solve_ITP
from ._archimedean import BiCopArchimedean


def _g(vec: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return (-vec * delta).expm1()


class Frank(BiCopArchimedean):
    # Joe 2014 page 165
    # https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.FrankCopula.html
    # ! exchangeability
    # * suggest torch.float64 for |par|<35, torch.float32 for |par|<13
    # delta
    _PAR_MIN, _PAR_MAX = torch.tensor([-35.0]), torch.tensor([35.0])

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        delta = par[0]
        return (
            -(
                _g(vec=obs[:, [0]], delta=delta)
                * _g(vec=obs[:, [1]], delta=delta)
                / expm1(-delta)
            ).log1p()
            / delta
        )

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        delta = par[0]
        g_y = _g(vec=obs[:, [1]], delta=delta)
        g_x_g_y = _g(vec=obs[:, [0]], delta=delta) * g_y
        return (g_x_g_y + g_y) / (g_x_g_y + expm1(-delta))

    @staticmethod
    def hinv1_0(obs: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0)"""
        delta = par[0]
        x = obs[:, [0]]
        return (
            _g(
                vec=torch.as_tensor(data=1.0, dtype=obs.dtype, device=obs.device),
                delta=delta,
            )
            / ((-delta * x).exp() / obs[:, [1]] - _g(vec=x, delta=delta))
        ).log1p() / (-delta)

    @classmethod
    def l_pdf_0(cls, obs: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        return cls.pdf_0(obs=obs, par=par).log()

    @staticmethod
    def par2tau_0(par:  torch.Tensor) -> float:
        delta = par[0]
        tmp = abs(delta)
        if tmp < 1e-5:
            return 0.0
        else:
            tau = (debye1(tmp) - 1.0) / tmp * 4.0 + 1.0
            return tau if (delta > 0) else -tau

    @staticmethod
    def pdf_0(
        obs: torch.Tensor,
        par: torch.Tensor,
    ) -> torch.Tensor:
        delta = par[0].clamp(Frank._PAR_MIN[0], Frank._PAR_MAX[0])
        x, y = obs[:, [0]], obs[:, [1]]
        return (
            delta
            * expm1(delta)
            * (delta * (x + y + 1)).exp()
            / (
                (delta * (x + y)).exp()
                - (delta * (y + 1)).exp()
                - (delta * (x + 1)).exp()
                + exp(delta)
            ).square()
        )

    @staticmethod
    def tau2par(tau: float, **kwargs) -> torch.Tensor:
        tau_a = abs(tau)

        delta = solve_ITP(
            fun=lambda delta: Frank.par2tau_0(torch.tensor([delta])) - tau_a,
            x_a=Frank._PAR_MIN[0] + 1e-6,
            x_b=Frank._PAR_MAX[0] - 1e-5,
            epsilon=1e-6,
            k_1=0.1,
        )

        return torch.tensor([delta if tau > 0 else -delta])