import torch

from ._abc import BiCopAbstract


class Independent(BiCopAbstract):
    _PAR_MIN, _PAR_MAX = tuple(), tuple()

    @staticmethod
    def cdf_0(obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return obs[:, [0]] * obs[:, [1]]

    @staticmethod
    def pdf_0(obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.ones((obs.shape[0], 1), device=obs.device, dtype=obs.dtype)

    @staticmethod
    def hfunc1_0(obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """first h function, Prob(V2<=v2 | V1=v1)"""
        return obs[:, [1]]

    hinv1_0 = hfunc1_0

    @staticmethod
    def l_pdf_0(obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros((obs.shape[0], 1), device=obs.device, dtype=obs.dtype)

    @staticmethod
    def par2tau_0(**kwargs) -> float:
        return 0.0

    @staticmethod
    def sim(
        num_sim: int,
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        **kwargs,
    ) -> torch.Tensor:
        # inverse Rosenblatt transform
        # * p1p2~Unif, hfunc1(v2|v1)=p2, hinv1(p2|v1)=v2
        torch.manual_seed(seed=seed)
        return torch.rand(size=(num_sim, 2), device=device, dtype=dtype)

    @staticmethod
    def tau2par_0(**kwargs) -> tuple:
        return tuple()
