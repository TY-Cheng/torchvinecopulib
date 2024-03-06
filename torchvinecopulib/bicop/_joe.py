import torch
from scipy.special import digamma

from ..util import solve_ITP, _CDF_MAX, _CDF_MIN
from ._archimedean import BiCopArchimedean


class Joe(BiCopArchimedean):
    # Joe 2014 page 170
    # * suggest torch.float64 for |par|<88, torch.float32 for |par|<7
    _PAR_MIN, _PAR_MAX = (1.000001,), (88.0,)

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        delta = par[0]
        x = (obs[:, [0]].negative().log1p() * delta).exp()
        y = (obs[:, [1]].negative().log1p() * delta).exp()
        return 1.0 - (x + y - x * y).pow(1 / delta)

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        delta = par[0]
        x = (1 - obs[:, [0]]).pow(delta)
        y = (1 - obs[:, [1]]).pow(delta)
        return (1.0 + y / x - y).pow(-1.0 + 1 / delta) * (1.0 - y)

    @staticmethod
    def hinv1_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0)"""
        # Newton, using x=(1-u)**delta, y=(1-v)**delta
        delta = max(min(par[0], Joe._PAR_MAX[0]), Joe._PAR_MIN[0])
        x = obs[:, [0]].negative().log1p()
        p = obs[:, [1]]
        delta_1m = 1.0 - delta
        delta_frac = delta_1m / delta
        # initial y as from initial v
        y = (1.0 + (-1.0 + (1.0 - p).pow(delta_frac)) * (x * delta_1m).exp()).pow(
            1 / delta_frac
        )
        x = (x * delta).exp()
        delta_frac = 1 / delta
        for _ in range(23):
            xy1 = x * (y - 1.0)
            x1y1delta = ((x.reciprocal() - 1.0) * y + 1.0).pow(delta_frac)
            y -= (
                delta
                * (xy1 - y)
                * (x1y1delta.reciprocal())
                * (p * (-xy1 + y) + xy1 * x1y1delta)
            ) / ((x - 1.0) * xy1 - delta * x)
            y.clamp_(min=_CDF_MIN, max=_CDF_MAX)
        return y.pow(delta_frac).negative() + 1.0

    @classmethod
    def l_pdf_0(cls, obs: torch.Tensor, par: tuple) -> torch.Tensor:
        return cls.pdf_0(obs=obs, par=par).log()

    @staticmethod
    def par2tau_0(par: tuple) -> float:
        delta = par[0]
        if delta == 2:
            # 1- PolyGamma[1, 2] = 2 - pi**2 / 6
            return 0.3550659331517736
        else:
            return 1.0 + 2.0 / (2.0 - delta) * (
                0.42278433509846713 - digamma(2.0 / delta + 1.0)
            )

    @staticmethod
    def pdf_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        delta = par[0]
        x = obs[:, [0]].negative().log1p()
        y = obs[:, [1]].negative().log1p()
        tmp = (x * delta).exp() + (y * delta).exp() - ((x + y) * delta).exp()
        return (
            tmp.pow(-2.0 + 1.0 / delta)
            * ((x + y) * (delta - 1.0)).exp()
            * (delta - 1.0 + tmp)
        )

    @staticmethod
    def tau2par(tau: float, **kwargs) -> tuple:
        tau_a = abs(tau)

        def f(delta: float) -> float:
            return Joe.par2tau_0(par=(delta,)) - tau_a

        return (solve_ITP(f=f, a=0.99, b=20600.0, eps_2=1e-6, k_1=0.1),)
