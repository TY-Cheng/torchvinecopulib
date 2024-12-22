from abc import abstractmethod

import torch
from ._abc import BiCopAbstract


class BiCopExtremeValue(BiCopAbstract):
    @staticmethod
    @abstractmethod
    def pickands_dep_func(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """
        Pickands dependence function, A(⋅): [0, 1] → [1/2 , 1]
            is convex and satisfies max{1−t, t} ≤ A(t) ≤ 1 for all t ∈ [0, 1].
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pickands_dep_func_derivative(
        vec: torch.Tensor, par: tuple[float]
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pickands_dep_func_derivative_2(
        vec: torch.Tensor, par: tuple[float]
    ) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def cdf_0(cls, obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        # * Czado 2019, page 49
        tmp = obs[:, [0]] * obs[:, [1]]
        return tmp.pow(
            exponent=cls.pickands_dep_func(vec=obs[:, [1]].log() / tmp.log(), par=par)
        )

    @classmethod
    def hfunc1_0(cls, obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        tmp = obs[:, [1]].log()
        tmp /= tmp + obs[:, [0]].log()
        tmp = (
            cls.cdf_0(obs=obs, par=par)
            / obs[:, [0]]
            * (
                cls.pickands_dep_func(vec=tmp, par=par)
                - tmp * cls.pickands_dep_func_derivative(vec=tmp, par=par)
            )
        )
        return torch.where(tmp.isnan(), obs[:, [1]], tmp)

    @classmethod
    def hfunc2_0(cls, obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """second h function, Prob(V0<=v0 | V1=v1)"""
        tmp = obs[:, [1]].log()
        tmp /= tmp + obs[:, [0]].log()
        tmp = (
            cls.cdf_0(obs=obs, par=par)
            / obs[:, [1]]
            * (
                cls.pickands_dep_func(tmp, par)
                + (1 - tmp) * cls.pickands_dep_func_derivative(tmp, par)
            )
        )
        return torch.where(tmp.isnan(), obs[:, [0]], tmp)

    @classmethod
    def pdf_0(cls, obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        tmp = obs.log()
        lsum = tmp.sum(dim=1, keepdim=True)
        tmp = tmp[:, [1]] / lsum
        pickands = cls.pickands_dep_func(vec=tmp, par=par)
        pickands1d = cls.pickands_dep_func_derivative(vec=tmp, par=par)
        return (
            cls.cdf_0(obs=obs, par=par)
            / obs.prod(dim=1, keepdim=True)
            * (
                pickands.square()
                + (1 - 2 * tmp) * pickands * pickands1d
                - (1 - tmp)
                * tmp
                * (
                    pickands1d.square()
                    # NOTE
                    + cls.pickands_dep_func_derivative_2(vec=tmp, par=par) / lsum
                )
            )
        )

    @classmethod
    def l_pdf_0(cls, obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return cls.pdf_0(obs=obs, par=par).log()

    @classmethod
    def par2tau_0(cls, par: tuple[float], num_step: float = 1000) -> float:
        """
        Kendall's tau for bivariate Extreme Value copula,
            numerical integration using Simpson's rule.
            Werner Hurlimann (2003)
        """
        vec_x = torch.linspace(0.0, 1.0, int(num_step))[1:-1].reshape(-1, 1)
        # ! number of intervals is even for Simpson's rule
        if len(vec_x) % 2 == 1:
            vec_x = vec_x[:-1]
        vec_y = cls.pickands_dep_func_derivative(
            vec=vec_x, par=par
        ) / cls.pickands_dep_func(vec=vec_x, par=par)
        vec_y = (1 + (1 - vec_x) * vec_y) * (1 - vec_x * vec_y)
        return (
            1
            - (
                (vec_x[1] - vec_x[0])
                * (
                    vec_y[0]
                    + 4 * vec_y[1::2].sum()
                    + 2 * vec_y[2:-1:2].sum()
                    + vec_y[-1]
                )
                / 3
            )
        ).item()
