from abc import abstractmethod

import torch

from ._abc import BiCopAbstract


class BiCopArchimedean(BiCopAbstract):
    # Joe 2014 page 91

    @classmethod
    def cdf_0(cls, obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return cls.generator_inv(
            cls.generator(obs[:, [0]], par) + cls.generator(obs[:, [1]], par), par
        )

    @staticmethod
    @abstractmethod
    def generator(vec: torch.Tensor, par: tuple[float]):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generator_derivative(vec: torch.Tensor, par: tuple[float]):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generator_inv(vec: torch.Tensor, par: tuple[float]):
        raise NotImplementedError

    @classmethod
    def hfunc1_0(cls, obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        tmp = cls.generator_inv(
            cls.generator(obs[:, [0]], par) + cls.generator(obs[:, [1]], par), par
        )
        tmp = cls.generator_derivative(obs[:, [0]], par) / cls.generator_derivative(tmp, par)
        tmp = torch.min(tmp, torch.tensor(1.0))
        idx = tmp.isnan()
        tmp[idx] = obs[:, [1]][idx]
        return tmp

    @classmethod
    def hinv1_0(cls, obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """first h inverse function, Q(V1=v1 | V0=v0)"""
        return cls.hinv1_num(obs, par)

    @classmethod
    def par2tau_0(cls, par: tuple[float], num_step: float = 1000) -> float:
        """
        Kendall's tau for bivariate Archimedean copula,
            numerical integration using Simpson's rule
        """
        vec_x = torch.linspace(0.0, 1.0, int(num_step))[1:-1]
        # ! number of intervals is even for Simpson's rule
        if len(vec_x) % 2 == 1:
            vec_x = vec_x[:-1]
        vec_y = cls.generator(vec=vec_x.reshape(-1, 1), par=par) / cls.generator_derivative(
            vec=vec_x.reshape(-1, 1), par=par
        )
        return (
            (
                ((vec_x[1] - vec_x[0]) / 3)
                * (vec_y[0] + 4 * vec_y[1::2].sum() + 2 * vec_y[2:-1:2].sum() + vec_y[-1])
            )
            * 4
            + 1
        ).item()
