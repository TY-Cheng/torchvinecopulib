import torch

from ._archimedean import BiCopArchimedean


class BB8(BiCopArchimedean):
    # Joe 2014 page 204
    # ! exchangeability
    # theta, delta
    _PAR_MIN, _PAR_MAX = (1.000001, 1e-4), (8.0, 0.99999999)

    @staticmethod
    def generator(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        return (
            (torch.as_tensor(1 - delta).pow(theta) - 1) / ((1 - delta * vec).pow(theta) - 1)
        ).log()

    @staticmethod
    def generator_inv(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        return (
            1 - (1 + vec.exp().reciprocal() * ((1 - delta) ** theta - 1)).pow(1 / theta)
        ) / delta

    @staticmethod
    def generator_derivative(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        tmp = 1 - delta * vec
        tmp_pow = tmp.pow(theta)
        return theta * delta * tmp_pow / tmp / (tmp_pow - 1)

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
    def par2tau_0(par: tuple[float], num_step: float = 5000) -> float:
        """
        Kendall's tau for bivariate Archimedean copula, numerical integration
        """
        theta, delta = par
        vec_x = torch.linspace(0.0, 1.0, int(num_step))[1:-1].reshape(-1, 1)
        # ! number of intervals is even for Simpson's rule
        if len(vec_x) % 2 == 1:
            vec_x = vec_x[:-1]
        tmp = 1 - delta * vec_x
        tmp_pow = tmp.pow(theta)
        vec_y = (tmp * (1 - tmp_pow) * ((tmp_pow - 1) / ((1 - delta) ** theta - 1)).log()) / (
            tmp_pow * theta * delta
        )
        return (
            (vec_x[1] - vec_x[0])
            * (vec_y[0] + 4 * vec_y[1::2].sum() + 2 * vec_y[2:-1:2].sum() + vec_y[-1])
            / 3
            * 4
            + 1
        ).item()

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
