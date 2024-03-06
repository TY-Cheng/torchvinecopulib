import torch

from ._archimedean import BiCopArchimedean


class Gumbel(BiCopArchimedean):
    # Joe 2014 page 172
    # * suggest torch.float64 for |par|<88, torch.float32 for |par|<12
    _PAR_MIN, _PAR_MAX = (1.000001,), (88.0,)

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        delta = par[0]
        return (
            (
                obs[:, [0]].log().negative().pow(delta)
                + obs[:, [1]].log().negative().pow(delta)
            )
            .pow(1 / delta)
            .negative()
            .exp()
        )

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        delta = par[0]
        x = obs[:, [0]].log().negative()
        y = obs[:, [1]].log().negative()
        return (
            obs[:, [0]].reciprocal()
            * (x.pow(delta) + y.pow(delta)).pow(1 / delta).negative().exp()
            * (1.0 + (y / x).pow(delta)).pow(1 / delta - 1.0)
        )

    @staticmethod
    def hinv1_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0)"""
        # Newton, using log version f
        delta = par[0]
        delta_1 = delta - 1.0
        x = obs[:, [0]].log().negative()
        tmp = (x + delta_1 * x.log() - obs[:, [1]].log()).negative()
        z = x.clone()
        for _ in range(31):
            z *= (tmp + (z.log() - 1.0) * delta_1) / (z + delta_1)
            z.negative_()
            z.clamp_min_(min=x)

        return (z.pow(delta) - x.pow(delta)).pow(1 / delta).negative().exp()

    @classmethod
    def l_pdf_0(cls, obs: torch.Tensor, par: tuple) -> torch.Tensor:
        return cls.pdf_0(obs=obs, par=par).log()

    @staticmethod
    def par2tau_0(par: tuple) -> float:
        return 1.0 - 1 / par[0]

    @staticmethod
    def pdf_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        delta = max(min(par[0], Gumbel._PAR_MAX[0]), Gumbel._PAR_MIN[0])
        x = obs[:, [0]].log().negative()
        y = obs[:, [1]].log().negative()
        tmp = (x.pow(delta) + y.pow(delta)).pow(1.0 / delta)
        return (
            tmp.negative().exp()
            * (tmp + delta - 1.0)
            * tmp.pow(1.0 - 2.0 * delta)
            * (x * y).pow(delta - 1.0)
            / obs[:, [0]]
            / obs[:, [1]]
        )

    @staticmethod
    def tau2par(tau: float, **kwargs) -> tuple:
        return (1 / (1 - abs(tau)),)
