import torch

from ..util import (
    _PICKANDS_DERIV_2_MIN,
    _PICKANDS_DERIV_MAX,
    _PICKANDS_DERIV_MIN,
    _PICKANDS_MAX,
    _PICKANDS_MIN,
)
from ._extremevalue import BiCopExtremeValue


class Tawn1(BiCopExtremeValue):
    # * Czado 2019 page 52
    # ! theta, psi2
    _PAR_MIN, _PAR_MAX = (1.000001, 1e-6), (10.0, 1.0)

    @staticmethod
    def pickands_dep_func(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, psi = par
        return (
            (1 - psi) * vec
            + ((1 - vec).pow(theta) + (psi * vec).pow(theta)).pow(1 / theta)
            # ! A[t] ∈ (0.5, 1)
        ).clamp(min=_PICKANDS_MIN, max=_PICKANDS_MAX)

    @staticmethod
    def pickands_dep_func_derivative(
        vec: torch.Tensor, par: tuple[float]
    ) -> torch.Tensor:
        theta, psi = par
        t_1 = 1 - vec
        t_1_theta = t_1.pow(theta)
        t_psi_theta = (psi * vec).pow(theta)
        return (
            1
            - psi
            + (-t_1_theta / t_1 + t_psi_theta / vec)
            * (t_1_theta + t_psi_theta).pow(-1 + 1 / theta)
            # ! A'[t] ∈ (-1, +1)
        ).clamp(min=_PICKANDS_DERIV_MIN, max=_PICKANDS_DERIV_MAX)

    @staticmethod
    def pickands_dep_func_derivative_2(
        vec: torch.Tensor, par: tuple[float]
    ) -> torch.Tensor:
        theta, psi = par
        t_1 = 1 - vec
        t_1_theta = t_1.pow(theta)
        return (
            (
                vec.pow(theta - 2)
                * (theta - 1)
                * psi**theta
                * t_1_theta
                * (t_1_theta + (psi * vec).pow(theta)).pow(-2 + 1 / theta)
            )
            / t_1.square()
            # ! A''[t] ∈ (0, +∞), A[t] convex
        ).clamp_min(min=_PICKANDS_DERIV_2_MIN)
