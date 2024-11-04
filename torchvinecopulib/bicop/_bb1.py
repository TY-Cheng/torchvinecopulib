import torch
from scipy.optimize import fsolve

from ._archimedean import BiCopArchimedean
from ..util import solve_ITP


class BB1(BiCopArchimedean):
    # Joe 2014 page 190
    # * two par: theta, delta
    _PAR_MIN, _PAR_MAX = (1e-6, 1.0 + 1e-6), (7, 7)

    @staticmethod
    def generator(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return (vec[:, [0]].pow(-par[0]) - 1.0).pow(par[1])

    @staticmethod
    def generator_inv(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return (vec.pow(1.0 / par[1]) + 1.0).pow(-1.0 / par[0])

    @staticmethod
    def generator_derivative(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        return (-delta * theta * vec.pow(-1.0 - theta)) * (vec.pow(-theta) - 1).pow(delta - 1)

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        return (
            (
                (obs[:, [0]].pow(-theta) - 1.0).pow(delta)
                + (obs[:, [1]].pow(-theta) - 1.0).pow(delta)
            ).pow(1.0 / delta)
            + 1.0
        ).pow(-1.0 / theta)

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        theta, delta = par
        detla_1 = 1 / delta
        u, v = obs[:, [0]], obs[:, [1]]
        x, y = (u.pow(-theta) - 1).pow(delta), (v.pow(-theta) - 1).pow(delta)
        x_y = x + y
        return (
            (x_y.pow(detla_1) + 1).pow(-1 / theta - 1)
            * x_y.pow(detla_1 - 1)
            * x.pow(1 - detla_1)
            * u.pow(-theta - 1)
        )

    @staticmethod
    def par2tau_0(par: tuple[float]) -> torch.Tensor:
        return 1 - 2 / (par[1] * (par[0] + 2))

    @staticmethod
    def pdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        delta_1 = 1 / delta
        u, v = obs[:, [0]], obs[:, [1]]
        s, p = (u.pow(-theta) - 1).pow(delta), (v.pow(-theta) - 1).pow(delta)
        s, p = s + p, s * p
        s_1_over_delta = s.pow(delta_1)
        return (
            (1 + s_1_over_delta).pow(-(1 / theta + 2))
            * s.pow(delta_1 - 2)
            * (theta * (delta - 1) + (theta * delta + 1) * s_1_over_delta)
            * p.pow(1 - delta_1)
            * (u * v).pow(-theta - 1)
        )

    @classmethod
    def l_pdf_0(cls, obs: torch.Tensor, par: tuple) -> torch.Tensor:
        return cls.pdf_0(obs=obs, par=par).log()

    @staticmethod
    def tau2par_0(tau: float, **kwargs) -> tuple[float]:
        # TODO
        # par = [0, 0]
        # par[1] = 1.5
        # par[0] = float((2 /par[1]*(1 - tau)) - 2)

        def func(par):
            return [
                (1 - tau - 2 / (par[1] * (par[0] + 2))),
                par[0] - par[1] + 2 == 0
                and par[0] > 0
                and par[0] < 7
                and par[1] > 1
                and par[1] < 7,
            ]

        root = fsolve(func, [2.1, 0.1])
        return root
