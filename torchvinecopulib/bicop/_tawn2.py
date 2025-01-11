import torch

from ..util import (
    _PICKANDS_DERIV_2_MIN,
    _PICKANDS_DERIV_MAX,
    _PICKANDS_DERIV_MIN,
    _PICKANDS_MAX,
    _PICKANDS_MIN,
)
from ._extremevalue import BiCopExtremeValue


class Tawn2(BiCopExtremeValue):
    # * Czado 2019 page 52
    # ! theta, psi1
    _PAR_MIN, _PAR_MAX = (torch.tensor([1.000001, 1e-6]), torch.tensor([10.0, 1.0]))

    @staticmethod
    def pickands_dep_func(vec: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        theta, psi = par[0], par[1]
        t_1 = 1 - vec
        return (
            (1 - psi) * t_1 + (vec.pow(theta) + (psi * t_1).pow(theta)).pow(1 / theta)
            # ! A[t] ∈ (0.5, 1)
        ).clamp(min=_PICKANDS_MIN, max=_PICKANDS_MAX)

    @staticmethod
    def pickands_dep_func_derivative(
        vec: torch.Tensor, par: torch.Tensor
    ) -> torch.Tensor:
        theta, psi = par[0], par[1]
        t_1 = 1 - vec
        t_theta = vec.pow(theta)
        t_1_psi_theta = (psi * t_1).pow(theta)
        return (
            -1
            + psi
            + (t_theta / vec - t_1_psi_theta / t_1)
            * (t_theta + t_1_psi_theta).pow(-1 + 1 / theta)
            # ! A'[t] ∈ (-1, +1)
        ).clamp(min=_PICKANDS_DERIV_MIN, max=_PICKANDS_DERIV_MAX)

    @staticmethod
    def pickands_dep_func_derivative_2(
        vec: torch.Tensor, par: torch.Tensor
    ) -> torch.Tensor:
        theta, psi = par[0], par[1]
        t_1 = 1 - vec
        t_theta = vec.pow(theta)
        t_1_psi_theta = (psi * t_1).pow(theta)
        return (
            (
                t_theta
                / vec.square()
                * (theta - 1)
                * t_1_psi_theta
                * (t_theta + t_1_psi_theta).pow(-2 + 1 / theta)
            )
            / t_1.square()
            # ! A''[t] ∈ (0, +∞), A[t] convex
        ).clamp_min(min=_PICKANDS_DERIV_2_MIN)
