import torch

from ..util import (
    _PICKANDS_DERIV_2_MIN,
    _PICKANDS_DERIV_MAX,
    _PICKANDS_DERIV_MIN,
    _PICKANDS_MAX,
    _PICKANDS_MIN,
)
from ._extremevalue import BiCopExtremeValue


class BB5(BiCopExtremeValue):
    # * Joe 2014 page 199
    # theta, delta
    _PAR_MIN, _PAR_MAX = (1.000001, 1e-6), (3.0, 3.0)

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        delta_neg = -delta
        x_theta = obs.log().neg().pow(theta)
        x_theta, y_theta = x_theta[:, [0]], x_theta[:, [1]]
        return (
            (
                x_theta
                + y_theta
                - (x_theta.pow(delta_neg) + y_theta.pow(delta_neg)).pow(1 / delta_neg)
            )
            .pow(1 / theta)
            .neg()
        ).exp()

    @staticmethod
    def pickands_dep_func(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        delta_neg = -delta
        t_theta, t_1_theta = vec.pow(theta), (1 - vec).pow(theta)
        return (
            (
                t_theta
                + t_1_theta
                - (t_1_theta.pow(delta_neg) + t_theta.pow(delta_neg)).pow(1 / delta_neg)
            )
            .pow(1 / theta)
            # ! A[t] ∈ (0.5, 1)
            .clamp_(min=_PICKANDS_MIN, max=_PICKANDS_MAX)
        )

    @staticmethod
    def pickands_dep_func_derivative(
        vec: torch.Tensor, par: tuple[float]
    ) -> torch.Tensor:
        theta, delta = par
        delta_neg = -delta
        t_theta, t_1_theta = vec.pow(theta), (1 - vec).pow(theta)
        t_theta_delta_neg, t_1_theta_delta_neg = (
            t_theta.pow(delta_neg),
            t_1_theta.pow(delta_neg),
        )
        inv_sum = (t_theta_delta_neg + t_1_theta_delta_neg).pow(1 / delta_neg)
        return (
            (t_1_theta + t_theta - inv_sum).pow(-1 + 1 / theta)
            * (
                t_1_theta / (vec - 1)
                + t_theta / vec
                + inv_sum.pow(delta + 1)
                * (t_1_theta_delta_neg / (1 - vec) - t_theta_delta_neg / vec)
            )
            # ! A'[t] ∈ (-1, +1)
        ).clamp_(min=_PICKANDS_DERIV_MIN, max=_PICKANDS_DERIV_MAX)

    @staticmethod
    def pickands_dep_func_derivative_2(
        vec: torch.Tensor, par: tuple[float]
    ) -> torch.Tensor:
        theta, delta = par
        theta_1 = theta - 1
        delta_frac = 1 / delta
        theta_delta_1 = delta * theta + 1
        t_theta, t_1_theta = vec.pow(theta), (1 - vec).pow(theta)
        sum_1 = t_1_theta.pow(-delta) + t_theta.pow(-delta)
        sum_2 = t_1_theta + t_theta - sum_1.pow(-delta_frac)
        sum_3 = (1 - vec).pow(-theta_delta_1) - vec.pow(-theta_delta_1)
        return (
            sum_2.pow(-2 + 1 / theta)
            * (
                (
                    -(1 - vec).pow(theta_1)
                    + vec.pow(theta_1)
                    + sum_3 / sum_1.pow(1 + delta_frac)
                ).square()
                * (-theta_1)
                + sum_2
                * (
                    (1 - vec).pow(-1 + theta_1) * theta_1
                    + vec.pow(-1 + theta_1) * theta_1
                    - sum_1.pow(-2 - delta_frac) * sum_3.square() * (1 + delta) * theta
                    + (
                        (
                            (1 - vec).pow(-1 - theta_delta_1)
                            + vec.pow(-1 - theta_delta_1)
                        )
                        * theta_delta_1
                    )
                    / sum_1.pow(1 + delta_frac)
                )
            )
            # ! A''[t] ∈ (0, +∞), A[t] convex
        ).clamp_min_(min=_PICKANDS_DERIV_2_MIN)
