import torch
from scipy.optimize import minimize

from ..util import _CDF_MIN, kendall_tau
from ._archimedean import BiCopArchimedean


class BB1(BiCopArchimedean):
    # Joe 2014 page 190
    # * two par: theta, delta
    _PAR_MIN, _PAR_MAX = (1e-6, 1.000001), (7.0, 7.0)

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
        detla_1d = 1 / delta
        u, v = obs[:, [0]], obs[:, [1]]
        x, y = (u.pow(-theta) - 1).pow(delta), (v.pow(-theta) - 1).pow(delta)
        x_y = x + y
        return (
            (x_y.pow(detla_1d) + 1).pow(-1 / theta - 1)
            * x_y.pow(detla_1d - 1)
            * x.pow(1 - detla_1d)
            * u.pow(-theta - 1)
        )

    @staticmethod
    def hinv1_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0)"""
        # * Newton-Raphson for inverse of hfunc1_0,
        # * using x=(u**-theta-1)**delta, y=(v**-theta-1)**delta
        theta, delta = par
        theta_1d, delta_1d, theta_delta = 1 / theta, 1 / delta, theta * delta
        u, p = obs[:, [0]], obs[:, [1]]
        x = (u.pow(-theta) - 1).pow(delta)
        x_delta_1p = x.pow(delta_1d) + 1
        # * initial y, from p and x
        y = (x_delta_1p.pow(-theta_1d - 1) * p * x.pow(delta_1d - 1)).pow(
            1 / (1 + theta_delta)
        ) * x_delta_1p.pow(theta_1d)
        y *= x * (1 + x.pow(-delta_1d)) / p
        y -= x
        fix = p * x.pow(delta_1d - 1) * u.pow(theta + 1)
        for _ in range(23):
            x_y_delta_1d = (x + y).pow(delta_1d)
            y -= (
                x_y_delta_1d.pow(2 * delta - 1)
                * (x_y_delta_1d + 1).pow(2 + theta_1d)
                * (fix - x_y_delta_1d.pow(1 - delta) * (1 + x_y_delta_1d).pow(-theta_1d - 1))
                * theta_delta
                / (theta_delta - theta + x_y_delta_1d * (1 + theta_delta))
            )
            # ! y ∈ (0, +∞)
            y.clamp_(min=_CDF_MIN)
        # * y to v
        return (y.pow(delta_1d) + 1).pow(-theta_1d)

    @staticmethod
    def par2tau_0(par: tuple[float]) -> float:
        return 1 - 2 / (par[1] * (par[0] + 2))

    @staticmethod
    def pdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        delta_1d = 1 / delta
        u, v = obs[:, [0]], obs[:, [1]]
        s, p = (u.pow(-theta) - 1).pow(delta), (v.pow(-theta) - 1).pow(delta)
        s, p = s + p, s * p
        s_1_over_delta = s.pow(delta_1d)
        return (
            (1 + s_1_over_delta).pow(-(1 / theta + 2))
            * s.pow(delta_1d - 2)
            * (theta * (delta - 1) + (theta * delta + 1) * s_1_over_delta)
            * p.pow(1 - delta_1d)
            * (u * v).pow(-theta - 1)
        )

    @classmethod
    def l_pdf_0(cls, obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return cls.pdf_0(obs=obs, par=par).log()

    @classmethod
    def tau2par(
        cls, tau: float = None, obs: torch.Tensor = None, mtd_opt: str = "L-BFGS-B", **kwargs
    ) -> tuple[float]:
        """quasi MLE for BB1 theta delta; using Kendall's tau as a constraint"""
        if tau is None:
            tau, _ = kendall_tau(x=obs[:, [0]], y=obs[:, [1]])
        theta = minimize(
            fun=lambda theta: BB1.l_pdf_0(
                obs=obs, par=(theta.item(), 2 / (theta.item() + 2) / (1 - tau))
            )
            .nan_to_num_()
            .sum()
            .neg()
            .item(),
            x0=(0.1,),
            bounds=((BB1._PAR_MIN[0], BB1._PAR_MAX[0]),),
            method=mtd_opt,
        ).x.item()
        return (theta, 2 / (theta + 2) / (1 - tau))
