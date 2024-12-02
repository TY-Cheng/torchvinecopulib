import torch

from ._archimedean import BiCopArchimedean


class BB6(BiCopArchimedean):
    # Joe 2014 page 200
    # ! exchangeability
    # theta, delta
    _PAR_MIN, _PAR_MAX = (1.000001, 1.000001), (6.0, 8.0)

    @staticmethod
    def generator(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return (1 - vec).pow(par[0]).neg().log1p().neg().pow(par[1])

    @staticmethod
    def generator_inv(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return vec.pow(1.0 / par[1]).neg().expm1().neg().pow(1.0 / par[0]).neg() + 1.0

    @staticmethod
    def generator_derivative(vec: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        vec_1 = 1.0 - vec
        return (
            vec_1.pow(theta).neg().log1p().neg().pow(delta - 1.0)
            * vec_1.pow(theta - 1.0)
            / (vec_1.pow(theta) - 1.0)
            * theta
            * delta
        )

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        theta, delta = par
        x_delta, y_delta = BB6.generator(obs[:, [0]], par=par), BB6.generator(obs[:, [1]], par=par)
        return (x_delta + y_delta).pow(1.0 / delta).neg().expm1().neg().pow(
            1.0 / theta
        ).neg() + 1.0

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        theta, delta = par
        theta_rec, delta_rec = 1.0 / theta, 1.0 / delta
        x, y = (1.0 - obs[:, [0]]).pow(theta).neg().log1p().neg(), (1.0 - obs[:, [1]]).pow(
            theta
        ).neg().log1p().neg()
        x_delta, y_delta = x.pow(delta), y.pow(delta)
        x_exp = x.exp()
        pow_sum = x_delta + y_delta
        w = pow_sum.pow(delta_rec).neg().exp()
        return (
            (1.0 - w).pow(theta_rec - 1.0)
            * w
            * pow_sum.pow(delta_rec - 1.0)
            * x.pow(delta - 1.0)
            * x_exp
            * (1.0 - x_exp.reciprocal()).pow(1.0 - theta_rec)
        )

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
        vec_y = 1 - vec_x
        tmp = vec_y.pow(theta)
        vec_y = (1 - tmp) * vec_y * (1 - tmp).log() / (tmp * theta * delta)
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
        theta_rec, delta_rec = 1.0 / theta, 1.0 / delta
        delta_1 = delta - 1.0
        u_bar, v_bar = (1.0 - obs[:, [0]]).clamp(1e-10, 1 - 1e-10), (1.0 - obs[:, [1]]).clamp(
            1e-10, 1 - 1e-10
        )
        x, y = u_bar.pow(theta).neg().log1p().neg(), v_bar.pow(theta).neg().log1p().neg()
        pow_sum = (x.pow(delta) + y.pow(delta)).clamp_min(1e-10)
        pow_sum_delta = pow_sum.pow(delta_rec)
        w = pow_sum_delta.neg().exp()
        w_1 = 1.0 - w
        return (
            w_1.pow(theta_rec - 2.0)
            * w
            * pow_sum.pow(delta_rec - 2.0)
            * ((theta - w) * pow_sum_delta + theta * delta_1 * w_1)
            * (x * y).pow(delta_1)
            * (1 - u_bar.pow(theta)).reciprocal()
            * (1 - v_bar.pow(theta)).reciprocal()
            * (u_bar * v_bar).pow(theta - 1.0)
        )

    @classmethod
    def l_pdf_0(cls, obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        return cls.pdf_0(obs=obs, par=par).log()
