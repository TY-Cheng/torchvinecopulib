import torch

from ._archimedean import BiCopArchimedean


class BB7(BiCopArchimedean):
    # Joe 2014 page 202
    # ! exchangeability
    # theta, delta
    _PAR_MIN, _PAR_MAX = torch.tensor([1.000001, 0.000001]), torch.tensor([6.0, 25.0])
    # ! l_pdf_0
    _EPS = 1e-7

    @staticmethod
    def generator(vec: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        return (1.0 - (1.0 - vec).pow(par[0])).pow(-par[1]) - 1.0

    @staticmethod
    def generator_inv(vec: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        return 1 - (1.0 - (1.0 + vec).pow(-1 / par[1])).pow(1 / par[0])

    @staticmethod
    def generator_derivative(vec: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        theta, delta = par[0], par[1]
        vec_1 = 1.0 - vec
        return (
            -(1.0 - vec_1.pow(theta)).pow(-1.0 - delta)
            * vec_1.pow(-1.0 + theta)
            * theta
            * delta
        )

    @staticmethod
    def par2tau_0(par: torch.Tensor, num_step: float = 5000) -> float:
        """
        Kendall's tau for bivariate Archimedean copula, numerical integration
        """
        theta, delta = par[0], par[1]
        vec_x = torch.linspace(0.0, 1.0, int(num_step))[1:-1].reshape(-1, 1)
        # ! number of intervals is even for Simpson's rule
        if len(vec_x) % 2 == 1:
            vec_x = vec_x[:-1]
        vec_y = 1 - vec_x
        tmp = vec_y.pow(theta)
        vec_y = (1 - (1 - tmp).pow(delta)) * (-1 + tmp) * vec_y / (tmp * theta * delta)
        return (
            (vec_x[1] - vec_x[0])
            * (vec_y[0] + 4 * vec_y[1::2].sum() + 2 * vec_y[2:-1:2].sum() + vec_y[-1])
            / 3
            * 4
            + 1
        ).item()

    @staticmethod
    def l_pdf_0(obs: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        theta, delta = par[0], par[1]
        theta_delta = theta * delta
        theta_rec, delta_rec = 1.0 / theta, 1.0 / delta
        u_bar, v_bar = (
            (1.0 - obs[:, [0]]).clamp(min=BB7._EPS, max=1 - BB7._EPS),
            (1.0 - obs[:, [1]]).clamp(min=BB7._EPS, max=1 - BB7._EPS),
        )
        x, y = (
            (1.0 - u_bar.pow(theta)).pow(-delta) - 1.0,
            (1.0 - v_bar.pow(theta)).pow(-delta) - 1.0,
        )
        x_y_1 = x + y + 1.0
        x_y_1_neg_delta_rec = x_y_1.pow(-delta_rec)
        return (
            (1.0 + delta_rec) * (x.log1p() + y.log1p())
            + x_y_1_neg_delta_rec.log()
            + (theta - 1.0) * (u_bar.log() + v_bar.log())
            + (theta_rec - 2.0) * (1.0 - x_y_1_neg_delta_rec).log()
            + (theta + theta_delta - x_y_1_neg_delta_rec * (1.0 + theta_delta)).log()
            - 2.0 * x_y_1.log()
        )
