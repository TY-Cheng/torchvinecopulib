import torch

from ._archimedean import BiCopArchimedean
from ._extremevalue import BiCopExtremeValue


class Gumbel(BiCopArchimedean, BiCopExtremeValue):
    # Joe 2014 page 172
    # https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.GumbelCopula.html
    # ! exchangeability
    # * suggest torch.float64 for |par|<88, torch.float32 for |par|<12
    # delta
    _PAR_MIN, _PAR_MAX = (1.000001,), (88.0,)

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        delta = par[0]
        return (
            (obs[:, [0]].log().neg().pow(delta) + obs[:, [1]].log().neg().pow(delta))
            .pow(1.0 / delta)
            .neg()
            .exp()
        )

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        delta = par[0]
        x = obs[:, [0]].log().neg()
        y = obs[:, [1]].log().neg()
        return (
            (x.pow(delta) + y.pow(delta)).pow(1.0 / delta).neg().exp()
            * (1.0 + (y / x).pow(delta)).pow(1.0 / delta - 1.0)
            / obs[:, [0]]
        )

    @staticmethod
    def hinv1_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0)"""
        # Newton, using log version f
        delta = par[0]
        delta_1 = delta - 1.0
        x = obs[:, [0]].log().neg()
        tmp = -x - delta_1 * x.log() + obs[:, [1]].log()
        z = x.clone()
        for _ in range(31):
            z *= -(tmp + (z.log() - 1.0) * delta_1) / (z + delta_1)
            z.clamp_min_(min=x)

        return (z.pow(delta) - x.pow(delta)).pow(1.0 / delta).neg().exp()

    @classmethod
    def l_pdf_0(cls, obs: torch.Tensor, par: tuple) -> torch.Tensor:
        return cls.pdf_0(obs=obs, par=par).log()

    @staticmethod
    def par2tau_0(par: tuple) -> float:
        return 1.0 - 1.0 / par[0]

    @staticmethod
    def pdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        delta = max(min(par[0], Gumbel._PAR_MAX[0]), Gumbel._PAR_MIN[0])
        x = obs[:, [0]].log().neg()
        y = obs[:, [1]].log().neg()
        tmp = (x.pow(delta) + y.pow(delta)).pow(1.0 / delta)
        return (
            tmp.neg().exp()
            * (tmp + delta - 1.0)
            * tmp.pow(1.0 - 2.0 * delta)
            * (x * y).pow(delta - 1.0)
            / obs[:, [0]]
            / obs[:, [1]]
        )

    @staticmethod
    def tau2par(tau: float, **kwargs) -> tuple:
        return (1.0 / (1.0 - abs(tau)),)
