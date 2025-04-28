import torch

from ._archimedean import BiCopArchimedean


class Clayton(BiCopArchimedean):
    # Joe 2014 page 168 4.6.1 Bivariate Mardia-Takahasi-Clayton-Cook-Johnson
    # https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.ClaytonCopula.html
    # ! exchangeability
    # * suggest torch.float64 for |par|<61, torch.float32 for |par|<15
    # delta
    _PAR_MIN, _PAR_MAX = (1e-4,), (61.0,)

    @staticmethod
    def cdf_0(obs: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        delta = par[0]
        return (
            (
                (-delta * obs[:, [0]].log()).expm1()
                + (-delta * obs[:, [1]].log()).expm1()
            ).log1p()
            * (-1.0 / delta)
        ).exp()

    @staticmethod
    def hfunc_l_0(obs: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        delta = par[0]
        return (obs[:, [0]].pow(delta) * (obs[:, [1]].pow(-delta) - 1.0) + 1.0).pow(
            -(1.0 / delta) - 1.0
        )

    @staticmethod
    def hinv_l_0(obs: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0)"""
        delta = par[0]
        return (
            ((obs[:, [1]]).pow(-delta / (delta + 1.0)) - 1.0) * obs[:, [0]].pow(-delta)
            + 1.0
        ).pow(-1.0 / delta)

    @staticmethod
    def l_pdf_0(obs: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
        delta = par[0].clamp(min=Clayton._PAR_MIN[0], max=Clayton._PAR_MAX[0])
        l_0, l_1 = obs[:, [0]].log(), obs[:, [1]].log()
        return (
            delta.log1p()
            - (delta + 1.0) * (l_0 + l_1)
            - (1.0 / delta + 2.0)
            * ((-delta * l_0).expm1() + (-delta * l_1).expm1()).log1p()
        )

    @staticmethod
    def par2tau_0(par: torch.Tensor) -> torch.Tensor:
        delta = par[0]
        return delta / (delta + 2.0)

    @staticmethod
    def tau2par(tau: torch.Tensor) -> torch.Tensor:
        # ! par δ > 0 (not -1!)
        t_a = tau.abs()
        return torch.tensor(
            [
                2.0 * t_a / (1.0 - t_a),
            ],
            dtype=tau.dtype,
            device=tau.device,
        )
