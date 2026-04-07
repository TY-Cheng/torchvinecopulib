import os

import pytest
import torch
from hypothesis import settings

import torchvinecopulib as tvc

from . import DEVICE, DTYPE, correlated_raw, gaussian_copula


settings.register_profile(
    "tvc",
    deadline=None,
    max_examples=int(os.getenv("TVC_HYPOTHESIS_MAX_EXAMPLES", "25")),
)
settings.load_profile("tvc")


@pytest.fixture(autouse=True)
def reset_torch_seed():
    torch.manual_seed(0)


@pytest.fixture()
def cpu_generator() -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    return gen


@pytest.fixture()
def independent_copula_obs(cpu_generator: torch.Generator) -> torch.Tensor:
    return torch.rand(768, 2, dtype=DTYPE, generator=cpu_generator).to(DEVICE)


@pytest.fixture()
def weak_dep_copula_obs() -> torch.Tensor:
    return gaussian_copula(num_obs=1_024, rho=0.2).to(DEVICE)


@pytest.fixture()
def strong_dep_copula_obs() -> torch.Tensor:
    return gaussian_copula(num_obs=1_500, rho=0.65).to(DEVICE)


@pytest.fixture()
def boundary_heavy_copula_obs(cpu_generator: torch.Generator) -> torch.Tensor:
    obs = torch.rand(1_024, 2, dtype=DTYPE, generator=cpu_generator)
    obs[:512, 0] = 0.95 + 0.05 * obs[:512, 0]
    obs[:512, 1] = 0.01 + 0.05 * obs[:512, 1]
    obs[512:, 0] = 0.01 + 0.05 * obs[512:, 0]
    obs[512:, 1] = 0.95 + 0.05 * obs[512:, 1]
    return obs.to(DEVICE)


@pytest.fixture()
def tie_heavy_obs() -> torch.Tensor:
    obs = gaussian_copula(num_obs=256, rho=0.45, dim=4)
    obs = (obs * 10.0).round() / 10.0
    return obs.to(DEVICE)


@pytest.fixture()
def raw_scale_obs() -> torch.Tensor:
    return correlated_raw(384, 0.55, 5).to(DEVICE)


@pytest.fixture()
def copula_scale_obs() -> torch.Tensor:
    return gaussian_copula(num_obs=320, rho=0.55, dim=5).to(DEVICE)


@pytest.fixture()
def fitted_bicop(strong_dep_copula_obs: torch.Tensor) -> tvc.BiCop:
    cop = tvc.BiCop(num_step_grid=65).to(DEVICE)
    cop.fit(
        strong_dep_copula_obs,
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
    )
    return cop


@pytest.fixture()
def fitted_vine_raw(raw_scale_obs: torch.Tensor) -> tuple[torch.Tensor, tvc.VineCop]:
    vc = tvc.VineCop(num_dim=5, is_cop_scale=False, num_step_grid=65).to(DEVICE)
    vc.fit(
        raw_scale_obs,
        marginal_kwargs={"bandwidth": "isj"},
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
        mtd_bidep="kendall_tau",
        thresh_trunc=None,
    )
    return raw_scale_obs, vc


@pytest.fixture()
def fitted_vine_copula(copula_scale_obs: torch.Tensor) -> tuple[torch.Tensor, tvc.VineCop]:
    vc = tvc.VineCop(num_dim=5, is_cop_scale=True, num_step_grid=65).to(DEVICE)
    vc.fit(
        copula_scale_obs,
        bicop_backend="grid_reflect",
        bicop_kwargs={"bandwidth": "silverman"},
        mtd_bidep="kendall_tau",
        thresh_trunc=None,
    )
    return copula_scale_obs, vc
