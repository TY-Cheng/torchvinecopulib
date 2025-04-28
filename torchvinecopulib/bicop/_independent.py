import torch

from ._abc import BiCopAbstract


class Independent(BiCopAbstract):
    # https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.IndependentCopula.html
    # ! exchangeability
    # no parameter

    @staticmethod
    def cdf_0(obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return obs[:, [0]] * obs[:, [1]]

    @staticmethod
    def pdf_0(obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.ones((obs.shape[0], 1), device=obs.device, dtype=obs.dtype)

    @staticmethod
    def hfunc_l_0(obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        return obs[:, [1]]

    hinv_l_0 = hfunc_l_0

    @staticmethod
    def l_pdf_0(obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros(size=(obs.shape[0], 1), device=obs.device, dtype=obs.dtype)

    @staticmethod
    def par2tau_0(par: torch.Tensor) -> torch.Tensor:
        return torch.tensor(data=0.0, dtype=par.dtype, device=par.device)

    @staticmethod
    def sim(
        num_sim: int,
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        **kwargs,
    ) -> torch.Tensor:
        # inverse Rosenblatt transform
        # * p1p2~Unif, hfunc_l(v1|v0)=p, hinv_l(p|v0)=v1
        torch.manual_seed(seed=seed)
        return torch.rand(size=(num_sim, 2), device=device, dtype=dtype)

    @staticmethod
    def tau2par(tau: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.tensor(data=[], dtype=tau.dtype, device=tau.device)
