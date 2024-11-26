import torch

from ._archimedean import BiCopArchimedean


class BB8(BiCopArchimedean):
    # Joe 2014 page 204
    # ! exchangeability
    # theta, delta
    _PAR_MIN, _PAR_MAX = (1.000001, 0.000001), (8.0, 0.99999999)

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        return (
            1
            - (
                1
                - (1 - (1 - delta * obs[:, [0]]).pow(theta))
                * (1 - (1 - delta * obs[:, [1]]).pow(theta))
                / (1 - (1 - delta) ** theta)
            ).pow(1 / theta)
        ) / delta

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        theta, delta = par
        theta_rec_1 = 1 / theta - 1
        eta_rec = 1 / (1 - (1 - delta) ** theta)
        x, y = 1 - (1 - delta * obs[:, [0]]).pow(theta), 1 - (1 - delta * obs[:, [1]]).pow(theta)
        return eta_rec * y * (1 - eta_rec * x * y).pow(theta_rec_1) / ((1 - x).pow(theta_rec_1))

    @staticmethod
    def pdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        theta_1 = theta - 1
        eta_rec = 1 / (1 - (1 - delta) ** theta)
        u_delta_1, v_delta_1 = 1 - delta * obs[:, [0]], 1 - delta * obs[:, [1]]
        eta_rec_x_y = eta_rec * (1 - u_delta_1.pow(theta)) * (1 - v_delta_1.pow(theta))
        return (
            eta_rec
            * delta
            * (1 - eta_rec_x_y).clamp_min(1e-8).pow(1 / theta - 2)
            * (theta - eta_rec_x_y)
            * u_delta_1.pow(theta_1)
            * v_delta_1.pow(theta_1)
        )

    @classmethod
    def l_pdf_0(cls, obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return cls.pdf_0(obs=obs, par=par).log()
