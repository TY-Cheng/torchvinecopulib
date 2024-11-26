import torch

from ._archimedean import BiCopArchimedean


class BB7(BiCopArchimedean):
    # Joe 2014 page 202
    # ! exchangeability
    # theta, delta
    _PAR_MIN, _PAR_MAX = (1.000001, 0.000001), (6.0, 25.0)

    @staticmethod
    def generator(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return (1.0 - (1.0 - vec).pow(par[0])).pow(-par[1]) - 1.0

    @staticmethod
    def generator_inv(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return 1 - (1.0 - (1.0 + vec).pow(-1 / par[1])).pow(1 / par[0])

    @staticmethod
    def generator_derivative(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        vec_1 = 1.0 - vec
        return (
            -(1.0 - vec_1.pow(theta)).pow(-1.0 - delta) * vec_1.pow(-1.0 + theta) * theta * delta
        )

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        x, y = (1.0 - (1.0 - obs[:, [0]]).pow(theta)).pow(-delta) - 1, (
            1.0 - (1.0 - obs[:, [1]]).pow(theta)
        ).pow(-delta) - 1.0
        return 1.0 - (1.0 - (x + y + 1.0).pow(-1 / delta)).pow(1.0 / theta)

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        theta, delta = par
        u_bar = 1.0 - obs[:, [0]]
        x, y = (1.0 - u_bar.pow(theta)).pow(-delta) - 1, (
            1.0 - (1.0 - obs[:, [1]]).pow(theta)
        ).pow(-delta) - 1.0
        x_y_1 = x + y + 1.0
        return (
            (1.0 - x_y_1.pow(-1.0 / delta)).pow(1.0 / theta - 1.0)
            * x_y_1.pow(-1.0 / delta - 1.0)
            * (x + 1.0).pow(1.0 + 1.0 / delta)
            * u_bar.pow(theta - 1.0)
        )

    @staticmethod
    def pdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        theta_rec, delta_rec = 1.0 / theta, 1.0 / delta
        u_bar, v_bar = 1.0 - obs[:, [0]], 1.0 - obs[:, [1]]
        x, y = (1.0 - u_bar.pow(theta)).pow(-delta) - 1.0, (1.0 - v_bar.pow(theta)).pow(
            -delta
        ) - 1.0
        x_y_1 = x + y + 1.0
        x_y_1_delta_rec = x_y_1.pow(delta_rec)
        return (
            ((1 + x) * (1 + y)).pow(1 + delta_rec)
            * x_y_1_delta_rec.pow(-theta_rec)
            * (u_bar * v_bar).pow(theta - 1)
            * (x_y_1_delta_rec - 1).pow(theta_rec - 2)
            * (-1 + (x_y_1_delta_rec * (1 + delta) - delta) * theta)
            / x_y_1.square()
        )

    @classmethod
    def l_pdf_0(cls, obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return cls.pdf_0(obs=obs, par=par).log()
