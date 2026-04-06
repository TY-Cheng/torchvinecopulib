import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
EPS = 1e-8


def gaussian_copula(num_obs: int = 1024, rho: float = 0.7, dim: int = 2) -> torch.Tensor:
    cov = torch.full((dim, dim), rho, dtype=DTYPE)
    cov.fill_diagonal_(1.0)
    chol = torch.linalg.cholesky(cov)
    z = torch.randn(num_obs, dim, dtype=DTYPE) @ chol.T
    return torch.special.ndtr(z)


def correlated_raw(num_obs: int = 1024, rho: float = 0.7, dim: int = 4) -> torch.Tensor:
    cov = torch.full((dim, dim), rho, dtype=DTYPE)
    cov.fill_diagonal_(1.0)
    chol = torch.linalg.cholesky(cov)
    return torch.randn(num_obs, dim, dtype=DTYPE) @ chol.T
