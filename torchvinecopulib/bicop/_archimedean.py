from abc import abstractmethod

import torch

from ._abc import BiCopAbstract


class BiCopArchimedean(BiCopAbstract):
    # Joe 2014 page 91

    @classmethod
    def cdf_0(
        cls,
        obs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return cls.generator_inv(
            cls.generator(obs[:, [0]], **kwargs) + cls.generator(obs[:, [1]], **kwargs),
            **kwargs,
        )

    @staticmethod
    @abstractmethod
    def generator(vec: torch.Tensor):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generator_derivative(vec: torch.Tensor):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generator_inv(vec: torch.Tensor):
        raise NotImplementedError

    @classmethod
    def hfunc1_0(
        cls,
        obs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        tmp = cls.generator_inv(
            cls.generator(obs[:, [0]], **kwargs) + cls.generator(obs[:, [1]], **kwargs),
            **kwargs,
        )
        tmp = cls.generator_derivative(
            obs[:, [0]], **kwargs
        ) / cls.generator_derivative(
            tmp,
            **kwargs,
        )
        idx = tmp.isnan()
        tmp[idx] = obs[:, [1]][idx]
        return tmp
