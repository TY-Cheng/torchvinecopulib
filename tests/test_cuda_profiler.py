import os

import pytest
import torch

import torchvinecopulib as tvc

from . import gaussian_copula


RUN_CUDA_TESTS = os.getenv("TVC_RUN_CUDA_TESTS") == "1"
pytestmark = pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available() or not RUN_CUDA_TESTS, reason="CUDA profiler suite is opt-in")
@pytest.mark.parametrize("backend_name", ["grid_reflect", "grid_probit"])
def test_cuda_query_paths_avoid_host_sync(backend_name):
    obs_fit = gaussian_copula(num_obs=4096, rho=0.6).cuda()
    obs = gaussian_copula(num_obs=512, rho=0.3).cuda().requires_grad_(True)
    cop = tvc.BiCop(num_step_grid=129).cuda()
    cop.fit(obs_fit, bicop_backend=backend_name, bicop_kwargs={"bandwidth": "silverman"})
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        loss = cop.log_pdf(obs).sum()
        loss.backward()
    event_names = {evt.key for evt in prof.key_averages()}
    assert not any("Memcpy DtoH" in name or "Memcpy HtoD" in name for name in event_names)
    assert not any("aten::to" == name for name in event_names)
