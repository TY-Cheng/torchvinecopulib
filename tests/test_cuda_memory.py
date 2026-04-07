import os

import pytest
import torch

import torchvinecopulib as tvc

from . import gaussian_copula


RUN_CUDA_TESTS = os.getenv("TVC_RUN_CUDA_TESTS") == "1"
pytestmark = pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available() or not RUN_CUDA_TESTS, reason="CUDA memory suite is opt-in")
def test_query_peak_memory_is_insensitive_to_fit_sample_size():
    peaks = []
    for num_obs in (2_048, 8_192):
        obs_fit = gaussian_copula(num_obs=num_obs, rho=0.5).cuda()
        obs = gaussian_copula(num_obs=512, rho=0.25).cuda()
        cop = tvc.BiCop(num_step_grid=129).cuda()
        cop.fit(obs_fit, bicop_backend="grid_reflect")
        torch.cuda.reset_peak_memory_stats()
        _ = cop.log_pdf(obs)
        peaks.append(torch.cuda.max_memory_allocated())
    assert max(peaks) <= min(peaks) * 1.25


@pytest.mark.skipif(not torch.cuda.is_available() or not RUN_CUDA_TESTS, reason="CUDA memory suite is opt-in")
def test_representation_memory_scales_with_grid_size():
    obs_fit = gaussian_copula(num_obs=4096, rho=0.5).cuda()
    sizes = []
    for grid_size in (65, 129):
        cop = tvc.BiCop(num_step_grid=grid_size).cuda()
        cop.fit(obs_fit, bicop_backend="grid_reflect")
        sizes.append(sum(buf.numel() * buf.element_size() for buf in [cop._pdf_grid, cop._cdf_grid, cop._hfunc_l_grid, cop._hfunc_r_grid]))
    assert sizes[1] > sizes[0]
